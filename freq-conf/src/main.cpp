#include "common.hpp"
#include "topology.hpp"
#include "timer.hpp"
#include "msr_reader.hpp"
#include "experiment_runner.hpp"
#include "workload_kernel.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [OPTIONS]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --cores <list>      Comma-separated core counts (default: 1,2,4,8,14,28)\n");
    fprintf(stderr, "  --all-cores         Sweep all physical cores (1 to N)\n");
    fprintf(stderr, "  --max-billions <N>  Max AVX-512 instruction burst (default: 50.0)\n");
    fprintf(stderr, "  --cooldown-ms <ms>  Thermal recovery time between bursts (default: 2000.0)\n");
    fprintf(stderr, "  --step <N>          Search step in billions (default: 2.0)\n");
    fprintf(stderr, "  --dry-run           Simulate frequencies (no AVX-512/MSR required)\n");
    fprintf(stderr, "  --output-csv <f>    Output sweep data (default: results/sweep_data.csv)\n");
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> res;
    size_t start = 0;
    while (start < s.length()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) end = s.length();
        res.push_back(std::stoi(s.substr(start, end - start)));
        start = end + 1;
    }
    return res;
}

int main(int argc, char* argv[]) {
    Config config;
    config.core_counts  = {1, 2, 4, 8, 14, 28};
    config.core_counts  = {1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28};
    config.out_csv      = "results/sweep_data.csv";
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dry-run") {
            config.dry_run = true;
        } else if (arg == "--all-cores") {
            config.core_counts.clear();
            for (int c = 1; c <= hw::NUM_PHYSICAL_CORES; ++c)
                config.core_counts.push_back(c);
        } else if (arg == "--cores" && i + 1 < argc) {
            config.core_counts = parse_int_list(argv[++i]);
        } else if (arg == "--max-billions" && i + 1 < argc) {
            config.max_burst_billions = std::stod(argv[++i]);
        } else if (arg == "--cooldown-ms" && i + 1 < argc) {
            config.cooldown_ms = std::stod(argv[++i]);
        } else if (arg == "--step" && i + 1 < argc) {
            config.coarse_step = std::stod(argv[++i]);
        } else if (arg == "--output-csv" && i + 1 < argc) {
            config.out_csv = argv[++i];
        }
    }

    fprintf(stderr, "══════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  AVX-512 Frequency Sweep\n");
    fprintf(stderr, "══════════════════════════════════════════════════════════════\n\n");

    if (!kernel::avx512_available() && !config.dry_run) {
        fprintf(stderr, "[main] ERROR: AVX-512 not available on this CPU. Use --dry-run for testing.\n");
        return 1;
    }

    auto cores = topology::detect_physical_cores();
    fprintf(stderr, "[main] Detected %zu physical cores\n", cores.size());

    // Calibrate TSC
    fprintf(stderr, "[main] Calibrating TSC frequency...\n");
    timer::calibrate_tsc_hz(100);
    fprintf(stderr, "[main] TSC frequency: %.4f GHz\n", timer::g_tsc_hz / 1e9);

    // Show mode
    {
        std::vector<msr::CoreReader> probe = msr::init_readers({cores[0].primary_cpu()}, config.dry_run);
        fprintf(stderr, "[main] Measurement mode: %s\n", msr::mode_name(msr::active_mode(probe)));
        msr::close_readers(probe);
    }

    fprintf(stderr, "\n[main] Sweeping bursts up to %.1f Billion instructions (step %.1f B)\n", 
            config.max_burst_billions, config.coarse_step);
    fprintf(stderr, "[main] Cooldown between bursts: %.0f ms\n", config.cooldown_ms);
    fprintf(stderr, "[main] Saving data to: %s\n\n", config.out_csv.c_str());

    auto progress = [&](int n_cores, double burst_billions, double freq_ghz) {
        if (burst_billions < 0) {
            fprintf(stderr, "\n── Active cores: %d ──\n", n_cores);
        } else {
            fprintf(stderr, "  burst=%5.1f B   freq=%.3f GHz\n", burst_billions, freq_ghz);
        }
    };

    experiment::run_sweep(config, config.out_csv, progress);

    fprintf(stderr, "\n[main] Done. Data saved to %s\n", config.out_csv.c_str());
    return 0;
}
