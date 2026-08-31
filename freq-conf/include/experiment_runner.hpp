#pragma once

#include "msr_reader.hpp"
#include "common.hpp"
#include <vector>
#include <string>
#include <functional>

namespace experiment {

// Progress callback: active_cores, burst_billions, freq_ghz
using ProgressCb = std::function<void(int, double, double)>;

// Evaluate a single point (spawns threads, runs workload, returns mean frequency)
double evaluate_sample(
    int active_cores,
    double burst_billions,
    const Config& config,
    const std::vector<msr::CoreReader>& readers,
    uint64_t tsc_hz);

// Run the sweep across all core counts and burst sizes
void run_sweep(
    const Config& config,
    const std::string& out_csv,
    ProgressCb progress_cb = nullptr);

} // namespace experiment
