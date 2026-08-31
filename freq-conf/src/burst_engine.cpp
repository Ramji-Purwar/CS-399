#include "burst_engine.hpp"
#include "workload_kernel.hpp"
#include "timer.hpp"
#include "common.hpp"
#include <unistd.h>

namespace burst {

RunResult run(const msr::CoreReader& reader, double burst_billions, const RunConfig& cfg, uint64_t tsc_hz) {
    (void)tsc_hz;  // Not needed for throughput-based measurement
    // 1. Cool down to reset thermal/power state
    if (cfg.cooldown_ms > 0) {
        usleep((uint64_t)(cfg.cooldown_ms * 1000.0));
    }

    if (burst_billions <= 0.0) {
        // For zero burst, just read sysfs/MSR idle frequency
        double f = msr::read_sysfs_freq_ghz(reader.cpu_id);
        if (f <= 0.0) f = hw::P_MAX_GHZ;
        return RunResult{ f };
    }

    uint64_t insn_count = (uint64_t)(burst_billions * 1e9);

    // ---------- THROUGHPUT-BASED MEASUREMENT ----------
    // Instead of relying on sysfs (which shows OS-requested freq, not actual),
    // we measure the REAL silicon frequency by timing the burst:
    //
    //   actual_freq = (insn_count / fma_ports_per_cycle) / elapsed_seconds
    //
    // On Emerald Rapids: 2 AVX-512 FMA ports → 2 FMAs retire per cycle at peak.
    // So: cycles_needed = insn_count / 2
    //     actual_freq_ghz = cycles_needed / elapsed_ns
    //
    // This works because the wall clock is independent of CPU frequency.
    // If the CPU throttles from 3.0 GHz to 2.0 GHz, the same instructions
    // take 50% longer, and we see it directly.

    constexpr double FMA_PORTS = 2.0;  // Emerald Rapids has 2 AVX-512 FMA ports

    if (cfg.dry_run) {
        // Simulate: throttle kicks in above 10 billion instructions
        double fake_freq = (burst_billions > 10.0) ? 2.1 : hw::P_MAX_GHZ;
        double fake_cycles = (double)insn_count / FMA_PORTS;
        double fake_seconds = fake_cycles / (fake_freq * 1e9);
        usleep((uint64_t)(fake_seconds * 1e6));
        return RunResult{ fake_freq };
    }

    // REAL measurement: time the burst
    uint64_t start_ns = timer::now_ns();
    kernel::run_avx512_instructions(insn_count);
    uint64_t end_ns = timer::now_ns();

    double elapsed_ns = (double)(end_ns - start_ns);
    double elapsed_s  = elapsed_ns * 1e-9;

    // Compute actual frequency from throughput
    // Each FMA instruction takes 1/FMA_PORTS cycles at peak throughput
    // (8-chain dependency ensures we saturate both ports)
    double cycles_executed = (double)insn_count / FMA_PORTS;
    double eff_freq_ghz = cycles_executed / elapsed_s / 1e9;

    return RunResult{ eff_freq_ghz };
}

} // namespace burst
