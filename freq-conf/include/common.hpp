// common.hpp
// Shared constants, structures, and configuration defaults for the
// AVX-512 P-State Throttling Characterization experiment.
//
// Intel Xeon Gold 5512U (Emerald Rapids)
//   28 physical cores / 56 logical CPUs (HT)
//   P1 guaranteed all-core: 2.1 GHz
//   Max turbo (single core): 3.7 GHz
//   Min P-state (assumed): 0.8 GHz (to be verified empirically)

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Hardware constants — Xeon Gold 5512U
// ---------------------------------------------------------------------------
namespace hw {
    constexpr int    NUM_PHYSICAL_CORES = 28;
    constexpr double P_MAX_GHZ          = 3.7;   // max single-core turbo
    constexpr double P1_GHZ             = 3.6;   // First step down
    constexpr double P_MIN_GHZ          = 2.0;   // Lowest we care to track
    constexpr double P_STEP_GHZ         = 0.1;   // 100 MHz steps between P-states
    constexpr double PSTATE_TOL_GHZ     = 0.05;  // ±50 MHz tolerance for P-state binning
}

// ---------------------------------------------------------------------------
// Experiment defaults (all overridable from CLI)
// ---------------------------------------------------------------------------
namespace defaults {
    // Burst engine
    constexpr double  COOLDOWN_MS         = 2000.0;  // 2 seconds thermal recovery
    constexpr double  MAX_BURST_BILLIONS  = 50.0;    // up to 50 billion instructions

    // Coarse sweep
    constexpr double  COARSE_STEP         = 2.0;     // 2 billion instruction jumps

    // Binary search resolution
    constexpr double  BINARY_RES          = 0.1;     // 100 million instruction target resolution

    // TSC calibration
    constexpr int     TSC_CALIB_MS        = 100;     // calibrate over 100 ms

    // Default output file
    constexpr const char* DEFAULT_CSV     = "results/measurements.csv";
}

// ---------------------------------------------------------------------------
// P-state ladder
// ---------------------------------------------------------------------------
struct PState {
    int    index;          // 0 = turbo, 1 = P1, 2..N = sub-base
    double freq_ghz;       // representative frequency (lower bound of bin)
    std::string label;
};

// ---------------------------------------------------------------------------
// Per-cycle measurement record
// ---------------------------------------------------------------------------
struct CycleMeasurement {
    double   timestamp_s;      // wall time at measurement start
    int      active_cores;     // number of simultaneously active cores
    int      core_id;          // logical CPU id (physical core representative)
    double   burst_billions;   // AVX-512 instruction burst size (billions)
    uint64_t delta_aperf;      // APERF counter delta
    uint64_t delta_mperf;      // MPERF counter delta
    double   eff_freq_ghz;     // (ΔAPERF/ΔMPERF) × tsc_freq_ghz
    int      pstate_index;     // classified P-state index
};

// ---------------------------------------------------------------------------
// Aggregate result for one (core_count, burst_billions) sample
// ---------------------------------------------------------------------------
struct SampleResult {
    int    active_cores;
    double burst_billions;
    double min_freq_ghz;        // minimum across active cores (used for P-state decision)
    double mean_freq_ghz;       // mean across active cores
    int    classified_pstate;   // based on min_freq
    std::vector<double> per_core_freq_ghz; // raw per-core values
};

// ---------------------------------------------------------------------------
// Threshold result for one (core_count, transition) pair
// ---------------------------------------------------------------------------
struct ThresholdResult {
    int    active_cores;
    int    from_pstate;         // e.g. 0 (P0/turbo)
    int    to_pstate;           // e.g. 1 (P1/base)
    double threshold_burst_billions; // minimum burst size to trigger this transition
    bool   found;               // true if transition was observed
};

// ---------------------------------------------------------------------------
// Experiment configuration (populated by CLI parser)
// ---------------------------------------------------------------------------
struct Config {
    std::vector<int> core_counts;
    double max_burst_billions = defaults::MAX_BURST_BILLIONS;
    double cooldown_ms        = defaults::COOLDOWN_MS;
    double coarse_step        = defaults::COARSE_STEP;
    double binary_res         = defaults::BINARY_RES;
    bool   dry_run            = false;
    bool   verbose            = false;
    std::string out_csv;
    std::string out_log;
};
