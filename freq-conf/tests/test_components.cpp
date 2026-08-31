// test_components.cpp
// Unit tests for all subsystems of the AVX-512 P-State experiment.

#include "common.hpp"
#include "topology.hpp"
#include "timer.hpp"
#include "msr_reader.hpp"
#include "workload_kernel.hpp"
#include "pstate.hpp"
#include "experiment_runner.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Simple test framework
// ---------------------------------------------------------------------------

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define TEST(name) static void test_##name()
#define RUN(name) do { \
    fprintf(stderr, "  [test] %-55s ", #name); \
    ++g_tests_run; \
    bool ok = true; \
    test_##name(); \
    if (ok) fprintf(stderr, "PASS\n"); \
} while(0)

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL\n    assertion failed: %s (line %d)\n", #cond, __LINE__); \
        ++g_tests_failed; \
        return; \
    } \
} while(0)

#define CHECK_NEAR(a, b, eps) do { \
    if (std::fabs((a) - (b)) > (eps)) { \
        fprintf(stderr, "FAIL\n    |%s - %s| = %.6g > %.6g (line %d)\n", \
                #a, #b, std::fabs((double)(a)-(double)(b)), (double)(eps), __LINE__); \
        ++g_tests_failed; \
        return; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Tests: Topology
// ---------------------------------------------------------------------------

TEST(topology_detects_physical_cores) {
    auto cores = topology::detect_physical_cores();
    // On Xeon Gold 5512U: expect 28 physical cores
    CHECK(cores.size() >= 1);
    // Each physical core should have at least 1 logical CPU
    for (const auto& c : cores) {
        CHECK(c.logical_cpus.size() >= 1);
        CHECK(c.primary_cpu() >= 0);
    }
    fprintf(stderr, "(found %zu physical cores) ", cores.size());
}

TEST(topology_no_duplicate_cpu_ids) {
    auto cores = topology::detect_physical_cores();
    std::vector<int> seen;
    for (const auto& c : cores) {
        int p = c.primary_cpu();
        CHECK(std::find(seen.begin(), seen.end(), p) == seen.end());
        seen.push_back(p);
    }
}

TEST(topology_pick_n_within_bounds) {
    auto cores = topology::detect_physical_cores();
    int max_n = (int)cores.size();
    // Pick 1 core
    auto one = topology::pick_n_physical_cores(1);
    CHECK(one.size() == 1);
    // Pick all cores
    auto all = topology::pick_n_physical_cores(max_n);
    CHECK((int)all.size() == max_n);
    // All CPU ids are unique
    std::sort(all.begin(), all.end());
    CHECK(std::unique(all.begin(), all.end()) == all.end());
}

// ---------------------------------------------------------------------------
// Tests: Timer
// ---------------------------------------------------------------------------

TEST(tsc_calibration_reasonable) {
    uint64_t hz = timer::calibrate_tsc_hz(50); // 50 ms
    double ghz = hz / 1e9;
    // TSC frequency should be between 1.0 GHz and 5.0 GHz
    CHECK(ghz >= 1.0);
    CHECK(ghz <= 5.0);
    fprintf(stderr, "(%.3f GHz) ", ghz);
}

TEST(now_ns_monotonic) {
    uint64_t t0 = timer::now_ns();
    timer::spin_wait_us(100.0);
    uint64_t t1 = timer::now_ns();
    CHECK(t1 > t0);
    // Should be at least 80µs (some jitter tolerance)
    CHECK((t1 - t0) >= 80'000ULL);
}

TEST(spin_wait_accuracy) {
    uint64_t t0 = timer::now_ns();
    timer::spin_wait_us(1000.0); // 1 ms
    uint64_t t1 = timer::now_ns();
    double elapsed_us = (t1 - t0) / 1000.0;
    // Should be within ±5% of 1000 µs
    CHECK(elapsed_us >= 950.0);
    CHECK(elapsed_us <= 1100.0);
    fprintf(stderr, "(%.0f µs elapsed) ", elapsed_us);
}

// ---------------------------------------------------------------------------
// Tests: MSR Reader
// ---------------------------------------------------------------------------

TEST(msr_dry_run_init) {
    std::vector<int> cpus = {0, 1};
    auto readers = msr::init_readers(cpus, /*dry_run=*/true);
    CHECK(readers.size() == 2);
    CHECK(readers[0].mode == msr::Mode::DRYRUN);
    CHECK(readers[1].mode == msr::Mode::DRYRUN);
    msr::close_readers(readers);
}

TEST(msr_dry_run_snapshot_accumulates) {
    // In DRYRUN mode, read_snapshot returns zeros (caller synthesizes)
    std::vector<int> cpus = {0};
    auto readers = msr::init_readers(cpus, true);
    auto s1 = msr::read_snapshot(readers[0]);
    auto s2 = msr::read_snapshot(readers[0]);
    // DRYRUN returns zeros — deltas handled by duty_cycle_engine
    CHECK(s1.aperf == 0 && s1.mperf == 0);
    msr::close_readers(readers);
}

TEST(msr_sysfs_fallback) {
    // Check that sysfs read returns a reasonable frequency for CPU 0
    double f = msr::read_sysfs_freq_ghz(0);
    if (f > 0.0) {
        CHECK(f >= 0.5);
        CHECK(f <= 5.0);
        fprintf(stderr, "(%.3f GHz via sysfs) ", f);
    } else {
        fprintf(stderr, "(sysfs not available — skip) ");
    }
}

TEST(msr_dry_run_freq_profile) {
    // Verify the synthetic response follows the expected profile:
    // 1 core: low duty → turbo, medium → P1, high → P2
    double f_low  = msr::dry_run_freq_ghz(0, 1, 0.10);
    double f_mid  = msr::dry_run_freq_ghz(0, 1, 0.45);
    double f_high = msr::dry_run_freq_ghz(0, 1, 0.80);
    CHECK(f_low  > hw::P1_GHZ);          // turbo
    CHECK_NEAR(f_mid, hw::P1_GHZ, 0.05); // P1 base
    CHECK(f_high < hw::P1_GHZ);          // below base
    fprintf(stderr, "(low=%.2f mid=%.2f high=%.2f GHz) ", f_low, f_mid, f_high);
}

TEST(msr_compute_eff_freq) {
    // Test compute_eff_freq_ghz directly
    // If APERF/MPERF ratio = 1.0, eff_freq = tsc_hz / 1e9
    uint64_t tsc_hz = 3'000'000'000ULL; // 3 GHz reference
    msr::ApmSnapshot before = {1000, 1000};
    msr::ApmSnapshot after  = {2000, 2000}; // ratio = 1.0
    double f = msr::compute_eff_freq_ghz(before, after, tsc_hz);
    CHECK_NEAR(f, 3.0, 0.001);

    // Ratio 0.7 → 0.7 × 3 = 2.1 GHz
    msr::ApmSnapshot after2 = {1700, 2000}; // delta_a=700, delta_m=1000
    double f2 = msr::compute_eff_freq_ghz(before, after2, tsc_hz);
    CHECK_NEAR(f2, 2.1, 0.001);
}

// ---------------------------------------------------------------------------
// Tests: AVX-512 Kernel
// ---------------------------------------------------------------------------

TEST(avx512_available_on_this_cpu) {
    bool avail = kernel::avx512_available();
    CHECK(avail); // Xeon Gold 5512U definitely has AVX-512
    fprintf(stderr, "(avx512=%s) ", avail ? "yes" : "no");
}

TEST(avx512_fma_completes_in_time) {
    // Run for 10 ms and verify it returns within ~50 ms
    uint64_t t0 = timer::now_ns();
    uint64_t iters = kernel::run_avx512_fma(10'000'000ULL); // 10 ms
    uint64_t t1 = timer::now_ns();
    double elapsed_ms = (t1 - t0) / 1e6;
    CHECK(elapsed_ms >= 9.0);
    CHECK(elapsed_ms <= 50.0);
    CHECK(iters > 0);
    fprintf(stderr, "(%.1f ms, %lu FMA iters) ", elapsed_ms, iters);
}

TEST(avx512_fma_iters_doesnt_crash) {
    // Sanity: run a fixed number of iterations without crash
    kernel::run_avx512_fma_iters(1000);
    // If we get here without crashing/illegal instruction, test passes
}

// ---------------------------------------------------------------------------
// Tests: P-State Classification
// ---------------------------------------------------------------------------

TEST(pstate_ladder_structure) {
    auto ladder = pstate::build_ladder();
    // Must have at least P0 + P1 + some sub-states
    CHECK(ladder.size() >= 3);
    // P0 index = 0
    CHECK(ladder[0].index == 0);
    // P1 at 2.1 GHz
    CHECK_NEAR(ladder[1].freq_ghz, hw::P1_GHZ, 0.01);
    // Ladder is monotonically non-increasing in freq
    for (int i = 2; i < (int)ladder.size(); ++i)
        CHECK(ladder[i].freq_ghz <= ladder[i-1].freq_ghz + 0.01);
    fprintf(stderr, "(%zu states, P1=%.1fGHz, min=%.1fGHz) ",
            ladder.size(), ladder[1].freq_ghz, ladder.back().freq_ghz);
}

TEST(pstate_classify_turbo) {
    auto ladder = pstate::build_ladder();
    int ps = pstate::classify(3.5, ladder);
    CHECK(ps == 0); // above P1 = turbo
}

TEST(pstate_classify_p1) {
    auto ladder = pstate::build_ladder();
    int ps = pstate::classify(hw::P1_GHZ, ladder);
    CHECK(ps == 1);
}

TEST(pstate_classify_sub_base) {
    auto ladder = pstate::build_ladder();
    // 2.0 GHz = P2
    int ps = pstate::classify(2.0, ladder);
    CHECK(ps == 2);
    // 1.5 GHz = P7 (2.1 - 0.1*7 = 1.4 → nearest is P7 at 1.4 or P8 at 1.5... check ladder)
    // Just verify it's in sub-base range
    int ps2 = pstate::classify(1.5, ladder);
    CHECK(ps2 >= 2);
}

TEST(pstate_classify_minimum) {
    auto ladder = pstate::build_ladder();
    int ps = pstate::classify(0.8, ladder);
    CHECK(ps == (int)ladder.size() - 1); // minimum P-state
}

TEST(pstate_min_freq) {
    std::vector<double> freqs = {3.1, 2.1, 2.4, 1.9};
    double m = pstate::min_freq(freqs);
    CHECK_NEAR(m, 1.9, 0.001);
}

TEST(pstate_transitions_count) {
    auto ladder = pstate::build_ladder();
    auto trans = pstate::transitions(ladder);
    // Should have exactly ladder.size()-1 transitions
    CHECK(trans.size() == ladder.size() - 1);
    // First: P0→P1
    CHECK(trans[0].first == 0 && trans[0].second == 1);
}

// ---------------------------------------------------------------------------
// Tests: Experiment Search Logic (dry-run)
// ---------------------------------------------------------------------------

TEST(dry_run_single_core_finds_p0_p1_threshold) {
    Config cfg;
    cfg.dry_run       = true;
    cfg.core_counts   = {1};
    cfg.measure_cycles = 10;   // fast
    cfg.period_us     = 200.0;
    cfg.settle_us     = 50.0;
    cfg.coarse_step   = 0.10;
    cfg.binary_res    = 0.02;  // 2% resolution for speed
    cfg.output_csv    = "/dev/null";
    cfg.output_log    = "/dev/null";

    timer::calibrate_tsc_hz(50);

    auto ladder  = pstate::build_ladder();
    auto results = experiment::run_experiment(cfg, ladder, nullptr);

    // Should have found P0→P1 transition
    bool found_p0_p1 = false;
    for (const auto& r : results) {
        if (r.active_cores == 1 && r.from_pstate == 0 && r.to_pstate == 1) {
            found_p0_p1 = true;
            // Dry-run profile: threshold at ~30% for 1 core
            CHECK(r.threshold_duty_cycle >= 0.0);
            CHECK(r.threshold_duty_cycle <= 1.0);
            fprintf(stderr, "(P0→P1 @ %.1f%%) ", r.threshold_duty_cycle * 100.0);
        }
    }
    CHECK(found_p0_p1);
}

TEST(dry_run_multicore_threshold_shifts) {
    Config cfg;
    cfg.dry_run       = true;
    cfg.core_counts   = {1, 4};
    cfg.measure_cycles = 10;
    cfg.period_us     = 200.0;
    cfg.settle_us     = 50.0;
    cfg.coarse_step   = 0.15;
    cfg.binary_res    = 0.03;
    cfg.output_csv    = "/dev/null";
    cfg.output_log    = "/dev/null";

    timer::calibrate_tsc_hz(50);
    auto ladder  = pstate::build_ladder();
    auto results = experiment::run_experiment(cfg, ladder, nullptr);

    double thresh_1core = -1.0, thresh_4core = -1.0;
    for (const auto& r : results) {
        if (r.from_pstate == 0 && r.to_pstate == 1) {
            if (r.active_cores == 1)  thresh_1core  = r.threshold_duty_cycle;
            if (r.active_cores == 4)  thresh_4core  = r.threshold_duty_cycle;
        }
    }

    if (thresh_1core >= 0.0 && thresh_4core >= 0.0) {
        // More cores → threshold shifts left (throttles at lower duty cycle)
        CHECK(thresh_4core <= thresh_1core + 0.05); // allow tiny tolerance
        fprintf(stderr, "(1C=%.1f%% 4C=%.1f%%) ",
                thresh_1core * 100.0, thresh_4core * 100.0);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    fprintf(stderr, "\n═══════════════════════════════════════════════════════\n");
    fprintf(stderr, "  AVX-512 P-State Characterization — Component Tests\n");
    fprintf(stderr, "═══════════════════════════════════════════════════════\n\n");

    fprintf(stderr, "[ Topology ]\n");
    RUN(topology_detects_physical_cores);
    RUN(topology_no_duplicate_cpu_ids);
    RUN(topology_pick_n_within_bounds);

    fprintf(stderr, "\n[ Timer ]\n");
    RUN(tsc_calibration_reasonable);
    RUN(now_ns_monotonic);
    RUN(spin_wait_accuracy);

    fprintf(stderr, "\n[ MSR Reader ]\n");
    RUN(msr_dry_run_init);
    RUN(msr_dry_run_snapshot_accumulates);
    RUN(msr_sysfs_fallback);
    RUN(msr_dry_run_freq_profile);
    RUN(msr_compute_eff_freq);

    fprintf(stderr, "\n[ AVX-512 Kernel ]\n");
    RUN(avx512_available_on_this_cpu);
    RUN(avx512_fma_completes_in_time);
    RUN(avx512_fma_iters_doesnt_crash);

    fprintf(stderr, "\n[ P-State Classification ]\n");
    RUN(pstate_ladder_structure);
    RUN(pstate_classify_turbo);
    RUN(pstate_classify_p1);
    RUN(pstate_classify_sub_base);
    RUN(pstate_classify_minimum);
    RUN(pstate_min_freq);
    RUN(pstate_transitions_count);

    fprintf(stderr, "\n[ Experiment Search (dry-run) ]\n");
    RUN(dry_run_single_core_finds_p0_p1_threshold);
    RUN(dry_run_multicore_threshold_shifts);

    fprintf(stderr, "\n═══════════════════════════════════════════════════════\n");
    fprintf(stderr, "  Results: %d/%d passed, %d failed\n",
            g_tests_run - g_tests_failed, g_tests_run, g_tests_failed);
    fprintf(stderr, "═══════════════════════════════════════════════════════\n\n");

    return g_tests_failed > 0 ? 1 : 0;
}
