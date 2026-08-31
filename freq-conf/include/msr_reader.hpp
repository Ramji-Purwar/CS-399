// msr_reader.hpp
// APERF/MPERF MSR interface for effective frequency measurement.
//
// The measurement uses IA32_APERF (MSR 0xE8) and IA32_MPERF (MSR 0xE7):
//
//   effective_freq = (ΔAPERF / ΔMPERF) × TSC_frequency
//
// These counters reflect the CPU's actual operating frequency as seen by
// hardware, not the OS-requested frequency — critical for capturing
// throttling behavior accurately.
//
// Three operating modes:
//
//   MODE_DIRECT: Read /dev/cpu/N/msr directly (requires `msr` kernel module
//                loaded and read permission — typically needs root or
//                `sudo modprobe msr && chmod +r /dev/cpu/*/msr`).
//
//   MODE_SYSFS:  Use /sys/devices/system/cpu/cpuN/cpufreq/scaling_cur_freq
//                as a fallback when MSR device is unavailable. Less precise
//                (OS-reported, not hardware-reported), but better than nothing.
//
//   MODE_DRYRUN: Return synthetic frequencies for testing/development.
//                Simulates P-state transitions at configurable duty cycles.

#pragma once

#include <cstdint>
#include <vector>

namespace msr {

// Reading mode — determined at initialization based on available access.
enum class Mode {
    DIRECT,  // /dev/cpu/N/msr (most accurate)
    SYSFS,   // /sys/...cpufreq/scaling_cur_freq (fallback)
    DRYRUN   // synthetic, for testing
};

// Per-core reader state
struct CoreReader {
    int     cpu_id;
    Mode    mode;
    int     msr_fd;   // file descriptor for /dev/cpu/N/msr, or -1
    double  dry_run_base_ghz; // only used in DRYRUN mode
};

// Initialize readers for a list of logical CPU ids.
// Automatically selects the best available mode:
//   1. Try to open /dev/cpu/N/msr → DIRECT
//   2. Fall back to sysfs read    → SYSFS
//   3. If dry_run = true          → DRYRUN
// Returns vector of CoreReaders (one per cpu_id).
std::vector<CoreReader> init_readers(const std::vector<int>& cpu_ids, bool dry_run = false);

// Close all open file descriptors.
void close_readers(std::vector<CoreReader>& readers);

// Report the active mode (all cores should share the same mode).
Mode active_mode(const std::vector<CoreReader>& readers);
const char* mode_name(Mode m);

// ---------------------------------------------------------------------------
// APERF / MPERF snapshot (used for delta computation)
// ---------------------------------------------------------------------------

struct ApmSnapshot {
    uint64_t aperf;
    uint64_t mperf;
};

// Read APERF and MPERF for one core (or sysfs snapshot for SYSFS mode).
ApmSnapshot read_snapshot(const CoreReader& r);

// Compute effective frequency in GHz from two snapshots and TSC freq.
// eff_freq = (ΔAPERF / ΔMPERF) × tsc_hz / 1e9
// Returns -1.0 if delta_mperf == 0 (no elapsed reference cycles).
double compute_eff_freq_ghz(const ApmSnapshot& before,
                             const ApmSnapshot& after,
                             uint64_t tsc_hz);

// Convenience: read a single instantaneous frequency from sysfs (Hz → GHz).
// Returns -1.0 if unavailable.
double read_sysfs_freq_ghz(int cpu_id);

// ---------------------------------------------------------------------------
// Dry-run simulation (tunable for test scenarios)
// ---------------------------------------------------------------------------

// Configure dry-run synthetic response:
//   At the given duty cycle for n active cores, returns an effective freq
//   that simulates P-state transitions.
double dry_run_freq_ghz(int cpu_id, int active_cores, double duty_cycle);

} // namespace msr
