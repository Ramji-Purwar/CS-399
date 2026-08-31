#pragma once

#include "msr_reader.hpp"
#include <cstdint>

namespace burst {

    struct RunConfig {
        double cooldown_ms;
        bool   dry_run;
    };

    struct RunResult {
        double eff_freq_ghz;
    };

    // Run a single burst of AVX-512 instructions
    // First sleeps for cooldown_ms to reset thermal state, then measures freq over the burst.
    RunResult run(const msr::CoreReader& reader, double burst_billions, const RunConfig& cfg, uint64_t tsc_hz);

} // namespace burst
