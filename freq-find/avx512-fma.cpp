#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <vector>
#include <immintrin.h>   // AVX-512 / FMA intrinsics

using namespace std;

// -----------------------------------------------------------------------
// Config — Gold 5512U: 28 physical cores (0-27), HT siblings (28-55) offline
// -----------------------------------------------------------------------

constexpr int      NUM_CORES         = 28;
constexpr uint64_t ITERATIONS        = 2800000000ULL;
constexpr int      DEFAULT_TRIALS    = 30;   // kept small on purpose: this
                                              // binary gets invoked many
                                              // times per binary-search run
constexpr int      DEFAULT_WARMUPS   = 3;

constexpr const char *RAPL_PKG_FILE =
    "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj";
constexpr const char *RAPL_DRAM_FILE =
    "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj";

uint64_t read_energy_uj(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(EXIT_FAILURE); }
    uint64_t val = 0;
    if (fscanf(f, "%lu", &val) != 1) {
        fprintf(stderr, "failed to parse %s\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(f);
    return val;
}

uint64_t read_max_energy_range_uj(const char *dir_path) {
    string p(dir_path);
    size_t pos = p.find_last_of('/');
    string range_path = p.substr(0, pos + 1) + "max_energy_range_uj";
    FILE *f = fopen(range_path.c_str(), "r");
    if (!f) return 0;
    uint64_t val = 0;
    fscanf(f, "%lu", &val);
    fclose(f);
    return val;
}

inline double energy_diff_uj(uint64_t before, uint64_t after, uint64_t max_range) {
    uint64_t diff;
    if (after >= before) diff = after - before;
    else if (max_range > 0) diff = (max_range - before) + after;
    else diff = 0;
    return diff / 1e6;
}

void pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        exit(EXIT_FAILURE);
    }
}

inline double elapsed_seconds(const timespec &s, const timespec &e) {
    return (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) * 1e-9;
}

// -----------------------------------------------------------------------
// avx512_fma kernel — 512-bit (16 x float32), 8 independent accumulator
// chains. Same non-convergence design as the avx2_fma version:
//   operand 0x00: fa=1.0f, fb=1.0f, init=1.0f       -> bit-static every iter
//   operand 0xAA: fa=1.3333334f, fb=0.75f, init=1.234567f
//                 -> oscillates 1.234567f <-> 1.646089f (10/32 bits/lane)
// -----------------------------------------------------------------------
__attribute__((noinline))
void foo_avx512_fma(uint64_t data) {
    float fa, fb, init_val;
    if (data == 0) {
        fa = 1.0f; fb = 1.0f; init_val = 1.0f;
    } else {
        fa = 1.3333334f; fb = 0.75f; init_val = 1.234567f;
    }
    __m512 mul_a = _mm512_set1_ps(fa);
    __m512 mul_b = _mm512_set1_ps(fb);
    __m512 zero  = _mm512_setzero_ps();

    __m512 x0 = _mm512_set1_ps(init_val), x1 = _mm512_set1_ps(init_val);
    __m512 x2 = _mm512_set1_ps(init_val), x3 = _mm512_set1_ps(init_val);
    __m512 x4 = _mm512_set1_ps(init_val), x5 = _mm512_set1_ps(init_val);
    __m512 x6 = _mm512_set1_ps(init_val), x7 = _mm512_set1_ps(init_val);

    for (uint64_t i = 0; i < ITERATIONS / 16; ++i) {
        x0 = _mm512_fmadd_ps(x0, mul_a, zero);
        x1 = _mm512_fmadd_ps(x1, mul_a, zero);
        x2 = _mm512_fmadd_ps(x2, mul_a, zero);
        x3 = _mm512_fmadd_ps(x3, mul_a, zero);
        x4 = _mm512_fmadd_ps(x4, mul_a, zero);
        x5 = _mm512_fmadd_ps(x5, mul_a, zero);
        x6 = _mm512_fmadd_ps(x6, mul_a, zero);
        x7 = _mm512_fmadd_ps(x7, mul_a, zero);

        x0 = _mm512_fmadd_ps(x0, mul_b, zero);
        x1 = _mm512_fmadd_ps(x1, mul_b, zero);
        x2 = _mm512_fmadd_ps(x2, mul_b, zero);
        x3 = _mm512_fmadd_ps(x3, mul_b, zero);
        x4 = _mm512_fmadd_ps(x4, mul_b, zero);
        x5 = _mm512_fmadd_ps(x5, mul_b, zero);
        x6 = _mm512_fmadd_ps(x6, mul_b, zero);
        x7 = _mm512_fmadd_ps(x7, mul_b, zero);
    }

    volatile float sink =
        _mm_cvtss_f32(_mm512_extractf32x4_ps(x0, 0)) +
        _mm_cvtss_f32(_mm512_extractf32x4_ps(x7, 0));
    (void)sink;
}

struct ThreadArg { int core_id; uint64_t operand; };

pthread_barrier_t g_start_barrier;
pthread_barrier_t g_end_barrier;
vector<double> g_thread_time_s(NUM_CORES, 0.0);

void *worker(void *arg_) {
    ThreadArg *arg = (ThreadArg *)arg_;
    pin_to_core(arg->core_id);
    pthread_barrier_wait(&g_start_barrier);

    timespec t0{}, t1{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    foo_avx512_fma(arg->operand);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    g_thread_time_s[arg->core_id] = elapsed_seconds(t0, t1);
    pthread_barrier_wait(&g_end_barrier);
    return nullptr;
}

struct Result {
    double mean_time_s, std_time_s;
    double pkg_J, dram_J, cores_J;
    double pkg_W, cores_W;
};

Result run_trial(uint64_t operand) {
    pthread_barrier_init(&g_start_barrier, nullptr, NUM_CORES + 1);
    pthread_barrier_init(&g_end_barrier,   nullptr, NUM_CORES + 1);

    vector<pthread_t> threads(NUM_CORES);
    vector<ThreadArg> args(NUM_CORES);
    for (int c = 0; c < NUM_CORES; ++c) {
        args[c] = {c, operand};
        pthread_create(&threads[c], nullptr, worker, &args[c]);
    }

    uint64_t max_pkg_range  = read_max_energy_range_uj(RAPL_PKG_FILE);
    uint64_t max_dram_range = read_max_energy_range_uj(RAPL_DRAM_FILE);
    uint64_t pkg_before  = read_energy_uj(RAPL_PKG_FILE);
    uint64_t dram_before = read_energy_uj(RAPL_DRAM_FILE);

    pthread_barrier_wait(&g_start_barrier);
    pthread_barrier_wait(&g_end_barrier);

    uint64_t pkg_after  = read_energy_uj(RAPL_PKG_FILE);
    uint64_t dram_after = read_energy_uj(RAPL_DRAM_FILE);

    for (auto &t : threads) pthread_join(t, nullptr);
    pthread_barrier_destroy(&g_start_barrier);
    pthread_barrier_destroy(&g_end_barrier);

    double pkg_J   = energy_diff_uj(pkg_before, pkg_after, max_pkg_range);
    double dram_J  = energy_diff_uj(dram_before, dram_after, max_dram_range);
    double cores_J = pkg_J - dram_J;

    double sum = 0, sumsq = 0;
    for (double t : g_thread_time_s) { sum += t; sumsq += t * t; }
    double mean_t = sum / NUM_CORES;
    double var_t  = sumsq / NUM_CORES - mean_t * mean_t;
    double std_t  = sqrt(max(0.0, var_t));

    double pkg_W   = pkg_J / mean_t;
    double cores_W = cores_J / mean_t;

    return {mean_t, std_t, pkg_J, dram_J, cores_J, pkg_W, cores_W};
}

int main(int argc, char *argv[]) {
    // Usage: <0x00|0xAA> <freq_mhz_label> <output_csv>
    // freq_mhz_label is NOT used to set the CPU frequency — the caller
    // (a driver script doing the binary search) must set
    // scaling_min_freq/scaling_max_freq via sysfs BEFORE invoking this
    // binary. It's accepted here purely so it can be stamped into the CSV.
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <0x00|0xAA> <freq_mhz_label> <output_csv>\n", argv[0]);
        fprintf(stderr, "NOTE: this binary does NOT set CPU frequency itself.\n"
                        "      Set scaling_min_freq/scaling_max_freq via sysfs\n"
                        "      before invoking it.\n");
        return EXIT_FAILURE;
    }
    string label    = argv[1];
    int    freq_mhz = atoi(argv[2]);
    string csv_name = argv[3];

    if (label != "0x00" && label != "0xAA") {
        fprintf(stderr, "operand must be 0x00 or 0xAA\n");
        return EXIT_FAILURE;
    }
    uint64_t operand = (label == "0x00") ? 0x0ULL : 0xAAAAAAAAAAAAAAABULL;

    int trials = DEFAULT_TRIALS;
    if (const char *e = getenv("TRIALS")) { trials = atoi(e); if (trials <= 0) trials = 1; }
    int warmups = DEFAULT_WARMUPS;
    if (const char *e = getenv("WARMUP_RUNS")) { warmups = atoi(e); if (warmups < 0) warmups = 0; }

    read_energy_uj(RAPL_PKG_FILE);
    read_energy_uj(RAPL_DRAM_FILE);

    for (int w = 0; w < warmups; ++w) run_trial(operand);

    ofstream out(csv_name);
    if (!out.is_open()) {
        fprintf(stderr, "failed to open %s\n", csv_name.c_str());
        return EXIT_FAILURE;
    }
    out << "kernel,freq_mhz_label,operand,trial,mean_time_s,std_time_s,"
           "pkg_J,dram_J,cores_J,pkg_W,cores_W\n";

    double sum_cores_W = 0.0;
    for (int t = 0; t < trials; ++t) {
        Result r = run_trial(operand);
        out << fixed << setprecision(9)
            << "avx512_fma" << "," << freq_mhz << "," << label << "," << t << ","
            << r.mean_time_s << "," << r.std_time_s << ","
            << r.pkg_J << "," << r.dram_J << "," << r.cores_J << ","
            << r.pkg_W << "," << r.cores_W << "\n";
        sum_cores_W += r.cores_W;
    }
    out.close();

    // Print the mean cores_W as the LAST line on stdout so a driver script
    // can capture it directly with `tail -n1`.
    printf("%.9f\n", sum_cores_W / trials);
    return 0;
}