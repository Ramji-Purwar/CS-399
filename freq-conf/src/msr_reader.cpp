// msr_reader.cpp
// APERF/MPERF reading with automatic fallback to sysfs and dry-run modes.

#include "msr_reader.hpp"
#include "timer.hpp"
#include "common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <string>

// MSR register addresses (IA32_MPERF = 0xE7, IA32_APERF = 0xE8)
static constexpr uint32_t MSR_IA32_MPERF = 0xE7;
static constexpr uint32_t MSR_IA32_APERF = 0xE8;

namespace msr {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static uint64_t read_msr_reg(int fd, uint32_t reg) {
    uint64_t val = 0;
    if (pread(fd, &val, sizeof(val), (off_t)reg) != sizeof(val)) {
        perror("[msr] pread");
        // Return 0 silently — let the caller deal with zero deltas
        return 0;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

std::vector<CoreReader> init_readers(const std::vector<int>& cpu_ids, bool dry_run) {
    std::vector<CoreReader> readers;
    readers.reserve(cpu_ids.size());

    for (int cpu : cpu_ids) {
        CoreReader r;
        r.cpu_id = cpu;
        r.msr_fd = -1;
        r.dry_run_base_ghz = hw::P1_GHZ;

        if (dry_run) {
            r.mode = Mode::DRYRUN;
        } else {
            // Attempt to open /dev/cpu/N/msr
            char path[64];
            snprintf(path, sizeof(path), "/dev/cpu/%d/msr", cpu);
            int fd = open(path, O_RDONLY | O_CLOEXEC);
            if (fd >= 0) {
                r.mode   = Mode::DIRECT;
                r.msr_fd = fd;
            } else {
                // Fall back to sysfs
                r.mode = Mode::SYSFS;
                if (cpu == cpu_ids[0]) {
                    fprintf(stderr, "[msr] /dev/cpu/*/msr not available (need `sudo modprobe msr`), "
                            "falling back to sysfs scaling_cur_freq (less accurate)\n");
                }
            }
        }
        readers.push_back(r);
    }
    return readers;
}

void close_readers(std::vector<CoreReader>& readers) {
    for (auto& r : readers) {
        if (r.msr_fd >= 0) {
            close(r.msr_fd);
            r.msr_fd = -1;
        }
    }
}

Mode active_mode(const std::vector<CoreReader>& readers) {
    if (readers.empty()) return Mode::DRYRUN;
    return readers[0].mode;
}

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::DIRECT: return "DIRECT (APERF/MPERF MSR)";
        case Mode::SYSFS:  return "SYSFS (scaling_cur_freq)";
        case Mode::DRYRUN: return "DRY-RUN (synthetic)";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Snapshot reading
// ---------------------------------------------------------------------------

ApmSnapshot read_snapshot(const CoreReader& r) {
    ApmSnapshot snap{0, 0};

    switch (r.mode) {
        case Mode::DIRECT:
            snap.mperf = read_msr_reg(r.msr_fd, MSR_IA32_MPERF);
            snap.aperf = read_msr_reg(r.msr_fd, MSR_IA32_APERF);
            break;

        case Mode::SYSFS: {
            // For SYSFS mode we can't really accumulate APERF/MPERF — instead
            // we encode the current frequency as a synthetic ratio.
            // We set mperf = 1 and aperf = freq_ghz / tsc_ghz, so that
            // compute_eff_freq_ghz returns the current sysfs frequency.
            double f = read_sysfs_freq_ghz(r.cpu_id);
            if (f <= 0.0) f = hw::P1_GHZ; // safe fallback
            // Encode: aperf/mperf * tsc_hz = f_ghz * 1e9
            // We use mperf = 1e6 for some resolution
            snap.mperf = 1'000'000ULL;
            double tsc_ghz = (double)timer::g_tsc_hz / 1e9;
            // aperf = (f_ghz / tsc_ghz) * mperf
            snap.aperf = (uint64_t)((f / tsc_ghz) * (double)snap.mperf);
            break;
        }

        case Mode::DRYRUN:
            // Synthetic: return zeros — caller will use dry_run_freq_ghz directly
            snap.aperf = 0;
            snap.mperf = 0;
            break;
    }
    return snap;
}

double compute_eff_freq_ghz(const ApmSnapshot& before,
                             const ApmSnapshot& after,
                             uint64_t tsc_hz) {
    uint64_t da = after.aperf - before.aperf;
    uint64_t dm = after.mperf - before.mperf;
    if (dm == 0) return -1.0;
    return ((double)da / (double)dm) * ((double)tsc_hz / 1e9);
}

// ---------------------------------------------------------------------------
// Sysfs instantaneous read
// ---------------------------------------------------------------------------

double read_sysfs_freq_ghz(int cpu_id) {
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu_id);
    std::ifstream f(path);
    if (!f.is_open()) return -1.0;
    uint64_t khz = 0;
    f >> khz;
    if (f.fail() || khz == 0) return -1.0;
    return (double)khz / 1e6; // kHz → GHz
}

// ---------------------------------------------------------------------------
// Dry-run simulation
// ---------------------------------------------------------------------------
// Simulates the following throttling profile:
//
//   1 core active:
//     D < 0.30 → P0 (3.2 GHz)
//     D < 0.60 → P1 (2.1 GHz)
//     D >= 0.60→ P2 (2.0 GHz)
//
//   N > 1 cores: threshold shifts left by 0.05 per additional core,
//     and turbo ceiling drops by 0.1 GHz per additional core.
//
// This is purely synthetic — it exists to allow the full search pipeline
// to be tested end-to-end without hardware access.

double dry_run_freq_ghz(int /*cpu_id*/, int active_cores, double duty_cycle) {
    // Turbo ceiling drops as more cores become active
    double turbo_ghz = hw::P_MAX_GHZ - (active_cores - 1) * 0.10;
    if (turbo_ghz < hw::P1_GHZ) turbo_ghz = hw::P1_GHZ;

    // Thresholds shift left as more cores are active
    double n_factor = (active_cores - 1) * 0.05;
    double thresh_p0_to_p1 = 0.30 - n_factor;
    double thresh_p1_to_p2 = 0.60 - n_factor;
    if (thresh_p0_to_p1 < 0.0) thresh_p0_to_p1 = 0.0;
    if (thresh_p1_to_p2 < thresh_p0_to_p1 + 0.05)
        thresh_p1_to_p2 = thresh_p0_to_p1 + 0.05;

    // Add small noise to simulate realistic measurements
    // (deterministic noise based on cpu_id to keep tests reproducible,
    //  but intentionally commented out — use pure synthetic for cleaner tests)
    double noise = 0.0; // could add: (cpu_id % 3) * 0.01

    if (duty_cycle < thresh_p0_to_p1) {
        return turbo_ghz + noise;
    } else if (duty_cycle < thresh_p1_to_p2) {
        return hw::P1_GHZ + noise;
    } else {
        return hw::P1_GHZ - hw::P_STEP_GHZ + noise;  // P2
    }
}

} // namespace msr
