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
#include <immintrin.h>   // AVX2 / AVX-512 / FMA intrinsics

using namespace std;

// -----------------------------------------------------------------------
// Config — Gold 5512U: 28 physical cores (0-27), HT siblings (28-55) offline
// -----------------------------------------------------------------------

constexpr int      NUM_CORES    = 28;          // cores 0..27
constexpr uint64_t ITERATIONS   = 2800000000ULL;
constexpr int      DEFAULT_TRIALS       = 100;
constexpr int      DEFAULT_WARMUP_RUNS  = 5;

// Package RAPL sysfs — Intel Xeon Gold 5512U (Emerald Rapids / SPR)
// Package: intel-rapl:0, DRAM: intel-rapl:0:0
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
    if (after >= before) {
        diff = after - before;
    } else if (max_range > 0) {
        diff = (max_range - before) + after;   // wrapped
    } else {
        diff = 0;
    }
    return diff / 1e6; // uJ -> J
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
// Kernels
//
// Evaluated instructions:
//   - imul        : 64-bit integer multiply (latency 3, throughput 1) -> 4 chains
//   - avx2_add    : 256-bit AVX2 packed single add (latency 4, throughput 0.5) -> 8 chains
//   - avx2_mul    : 256-bit AVX2 packed single mul (latency 4, throughput 0.5) -> 8 chains
//   - avx2_fma    : 256-bit AVX2 packed single fma (latency 4, throughput 0.5) -> 8 chains
//   - avx512_add  : 512-bit AVX-512 packed single add (latency 4, throughput 0.5) -> 8 chains
//   - avx512_mul  : 512-bit AVX-512 packed single mul (latency 4, throughput 0.5) -> 8 chains
//   - avx512_fma  : 512-bit AVX-512 packed single fma (latency 4, throughput 0.5) -> 8 chains
//
// All kernels execute exactly ITERATIONS instructions across the loop.
// Non-convergence design:
//   - imul: odd multiplier 0xAAAAAAAAAAAAAAAB has multiplicative order 2^62 mod 2^64.
//   - mul/fma: alternating reciprocal multiplier pair (1.3333334f and 0.75f) oscillates
//     between 1.234567f and 1.646089f without drift or overflow.
//   - add: alternating add pair (+0.20833333f and -0.20833333f) oscillates
//     between 0x3FAAAAAA and 0x3FC55555, flipping 22 bits per lane with zero drift.
// -----------------------------------------------------------------------

// ── imul ─────────────────────────────────────────────────────────────────────
__attribute__((noinline))
void foo_imul(uint64_t data) {
    uint64_t x0 = 0xDEADBEEFCAFEBABEULL;
    uint64_t x1 = 0x0123456789ABCDEFULL;
    uint64_t x2 = 0xFEDCBA9876543210ULL;
    uint64_t x3 = 0xA5A5A5A5A5A5A5A5ULL;

    for (uint64_t i = 0; i < ITERATIONS / 4; ++i) {
        asm volatile(
            "imulq %[d], %[a]\n\t"
            "imulq %[d], %[b]\n\t"
            "imulq %[d], %[c]\n\t"
            "imulq %[d], %[e]\n\t"
            : [a] "+r"(x0), [b] "+r"(x1), [c] "+r"(x2), [e] "+r"(x3)
            : [d] "r"(data)
            : "cc");
    }
    asm volatile("" : : "r"(x0), "r"(x1), "r"(x2), "r"(x3));
}

// ── avx2_add ─────────────────────────────────────────────────────────────────
__attribute__((noinline))
void foo_avx2_add(uint64_t data) {
    float fa, fb, init_val;
    if (data == 0) {
        fa = 0.0f;
        fb = 0.0f;
        init_val = 0.0f;
    } else {
        fa = 0.20833333f;
        fb = -0.20833333f;
        init_val = 1.3333333f;
    }
    __m256 add_a = _mm256_set1_ps(fa);
    __m256 add_b = _mm256_set1_ps(fb);

    __m256 x0 = _mm256_set1_ps(init_val), x1 = _mm256_set1_ps(init_val);
    __m256 x2 = _mm256_set1_ps(init_val), x3 = _mm256_set1_ps(init_val);
    __m256 x4 = _mm256_set1_ps(init_val), x5 = _mm256_set1_ps(init_val);
    __m256 x6 = _mm256_set1_ps(init_val), x7 = _mm256_set1_ps(init_val);

    for (uint64_t i = 0; i < ITERATIONS / 16; ++i) {
        x0 = _mm256_add_ps(x0, add_a);
        x1 = _mm256_add_ps(x1, add_a);
        x2 = _mm256_add_ps(x2, add_a);
        x3 = _mm256_add_ps(x3, add_a);
        x4 = _mm256_add_ps(x4, add_a);
        x5 = _mm256_add_ps(x5, add_a);
        x6 = _mm256_add_ps(x6, add_a);
        x7 = _mm256_add_ps(x7, add_a);

        x0 = _mm256_add_ps(x0, add_b);
        x1 = _mm256_add_ps(x1, add_b);
        x2 = _mm256_add_ps(x2, add_b);
        x3 = _mm256_add_ps(x3, add_b);
        x4 = _mm256_add_ps(x4, add_b);
        x5 = _mm256_add_ps(x5, add_b);
        x6 = _mm256_add_ps(x6, add_b);
        x7 = _mm256_add_ps(x7, add_b);
    }

    volatile float sink = _mm256_cvtss_f32(x0) + _mm256_cvtss_f32(x7);
    (void)sink;
}

// ── avx2_mul ─────────────────────────────────────────────────────────────────
__attribute__((noinline))
void foo_avx2_mul(uint64_t data) {
    float fa, fb;
    if (data == 0) {
        fa = 1.0f;
        fb = 1.0f;
    } else {
        fa = 1.3333334f;
        fb = 0.75f;
    }
    __m256 mul_a = _mm256_set1_ps(fa);
    __m256 mul_b = _mm256_set1_ps(fb);

    __m256 x0 = _mm256_set1_ps(1.234567f), x1 = _mm256_set1_ps(1.234567f);
    __m256 x2 = _mm256_set1_ps(1.234567f), x3 = _mm256_set1_ps(1.234567f);
    __m256 x4 = _mm256_set1_ps(1.234567f), x5 = _mm256_set1_ps(1.234567f);
    __m256 x6 = _mm256_set1_ps(1.234567f), x7 = _mm256_set1_ps(1.234567f);

    for (uint64_t i = 0; i < ITERATIONS / 16; ++i) {
        x0 = _mm256_mul_ps(x0, mul_a);
        x1 = _mm256_mul_ps(x1, mul_a);
        x2 = _mm256_mul_ps(x2, mul_a);
        x3 = _mm256_mul_ps(x3, mul_a);
        x4 = _mm256_mul_ps(x4, mul_a);
        x5 = _mm256_mul_ps(x5, mul_a);
        x6 = _mm256_mul_ps(x6, mul_a);
        x7 = _mm256_mul_ps(x7, mul_a);

        x0 = _mm256_mul_ps(x0, mul_b);
        x1 = _mm256_mul_ps(x1, mul_b);
        x2 = _mm256_mul_ps(x2, mul_b);
        x3 = _mm256_mul_ps(x3, mul_b);
        x4 = _mm256_mul_ps(x4, mul_b);
        x5 = _mm256_mul_ps(x5, mul_b);
        x6 = _mm256_mul_ps(x6, mul_b);
        x7 = _mm256_mul_ps(x7, mul_b);
    }

    volatile float sink = _mm256_cvtss_f32(x0) + _mm256_cvtss_f32(x7);
    (void)sink;
}

// ── avx2_fma ─────────────────────────────────────────────────────────────────
__attribute__((noinline))
void foo_avx2_fma(uint64_t data) {
    float fa, fb, init_val;
    if (data == 0) {
        fa = 1.0f;
        fb = 1.0f;
        init_val = 1.0f;
    } else {
        fa = 1.3333334f;
        fb = 0.75f;
        init_val = 1.234567f;
    }
    __m256 mul_a = _mm256_set1_ps(fa);
    __m256 mul_b = _mm256_set1_ps(fb);
    __m256 zero  = _mm256_setzero_ps();

    __m256 x0 = _mm256_set1_ps(init_val), x1 = _mm256_set1_ps(init_val);
    __m256 x2 = _mm256_set1_ps(init_val), x3 = _mm256_set1_ps(init_val);
    __m256 x4 = _mm256_set1_ps(init_val), x5 = _mm256_set1_ps(init_val);
    __m256 x6 = _mm256_set1_ps(init_val), x7 = _mm256_set1_ps(init_val);

    for (uint64_t i = 0; i < ITERATIONS / 16; ++i) {
        x0 = _mm256_fmadd_ps(x0, mul_a, zero);
        x1 = _mm256_fmadd_ps(x1, mul_a, zero);
        x2 = _mm256_fmadd_ps(x2, mul_a, zero);
        x3 = _mm256_fmadd_ps(x3, mul_a, zero);
        x4 = _mm256_fmadd_ps(x4, mul_a, zero);
        x5 = _mm256_fmadd_ps(x5, mul_a, zero);
        x6 = _mm256_fmadd_ps(x6, mul_a, zero);
        x7 = _mm256_fmadd_ps(x7, mul_a, zero);

        x0 = _mm256_fmadd_ps(x0, mul_b, zero);
        x1 = _mm256_fmadd_ps(x1, mul_b, zero);
        x2 = _mm256_fmadd_ps(x2, mul_b, zero);
        x3 = _mm256_fmadd_ps(x3, mul_b, zero);
        x4 = _mm256_fmadd_ps(x4, mul_b, zero);
        x5 = _mm256_fmadd_ps(x5, mul_b, zero);
        x6 = _mm256_fmadd_ps(x6, mul_b, zero);
        x7 = _mm256_fmadd_ps(x7, mul_b, zero);
    }

    volatile float sink = _mm256_cvtss_f32(x0) + _mm256_cvtss_f32(x7);
    (void)sink;
}

// ── avx512_add ───────────────────────────────────────────────────────────────
__attribute__((noinline))
void foo_avx512_add(uint64_t data) {
    float fa, fb, init_val;
    if (data == 0) {
        fa = 0.0f;
        fb = 0.0f;
        init_val = 0.0f;
    } else {
        fa = 0.20833333f;
        fb = -0.20833333f;
        init_val = 1.3333333f;
    }
    __m512 add_a = _mm512_set1_ps(fa);
    __m512 add_b = _mm512_set1_ps(fb);

    __m512 x0 = _mm512_set1_ps(init_val), x1 = _mm512_set1_ps(init_val);
    __m512 x2 = _mm512_set1_ps(init_val), x3 = _mm512_set1_ps(init_val);
    __m512 x4 = _mm512_set1_ps(init_val), x5 = _mm512_set1_ps(init_val);
    __m512 x6 = _mm512_set1_ps(init_val), x7 = _mm512_set1_ps(init_val);

    for (uint64_t i = 0; i < ITERATIONS / 16; ++i) {
        x0 = _mm512_add_ps(x0, add_a);
        x1 = _mm512_add_ps(x1, add_a);
        x2 = _mm512_add_ps(x2, add_a);
        x3 = _mm512_add_ps(x3, add_a);
        x4 = _mm512_add_ps(x4, add_a);
        x5 = _mm512_add_ps(x5, add_a);
        x6 = _mm512_add_ps(x6, add_a);
        x7 = _mm512_add_ps(x7, add_a);

        x0 = _mm512_add_ps(x0, add_b);
        x1 = _mm512_add_ps(x1, add_b);
        x2 = _mm512_add_ps(x2, add_b);
        x3 = _mm512_add_ps(x3, add_b);
        x4 = _mm512_add_ps(x4, add_b);
        x5 = _mm512_add_ps(x5, add_b);
        x6 = _mm512_add_ps(x6, add_b);
        x7 = _mm512_add_ps(x7, add_b);
    }

    volatile float sink =
        _mm_cvtss_f32(_mm512_extractf32x4_ps(x0, 0)) +
        _mm_cvtss_f32(_mm512_extractf32x4_ps(x7, 0));
    (void)sink;
}

// ── avx512_mul ───────────────────────────────────────────────────────────────
__attribute__((noinline))
void foo_avx512_mul(uint64_t data) {
    float fa, fb;
    if (data == 0) {
        fa = 1.0f;
        fb = 1.0f;
    } else {
        fa = 1.3333334f;
        fb = 0.75f;
    }
    __m512 mul_a = _mm512_set1_ps(fa);
    __m512 mul_b = _mm512_set1_ps(fb);

    __m512 x0 = _mm512_set1_ps(1.234567f), x1 = _mm512_set1_ps(1.234567f);
    __m512 x2 = _mm512_set1_ps(1.234567f), x3 = _mm512_set1_ps(1.234567f);
    __m512 x4 = _mm512_set1_ps(1.234567f), x5 = _mm512_set1_ps(1.234567f);
    __m512 x6 = _mm512_set1_ps(1.234567f), x7 = _mm512_set1_ps(1.234567f);

    for (uint64_t i = 0; i < ITERATIONS / 16; ++i) {
        x0 = _mm512_mul_ps(x0, mul_a);
        x1 = _mm512_mul_ps(x1, mul_a);
        x2 = _mm512_mul_ps(x2, mul_a);
        x3 = _mm512_mul_ps(x3, mul_a);
        x4 = _mm512_mul_ps(x4, mul_a);
        x5 = _mm512_mul_ps(x5, mul_a);
        x6 = _mm512_mul_ps(x6, mul_a);
        x7 = _mm512_mul_ps(x7, mul_a);

        x0 = _mm512_mul_ps(x0, mul_b);
        x1 = _mm512_mul_ps(x1, mul_b);
        x2 = _mm512_mul_ps(x2, mul_b);
        x3 = _mm512_mul_ps(x3, mul_b);
        x4 = _mm512_mul_ps(x4, mul_b);
        x5 = _mm512_mul_ps(x5, mul_b);
        x6 = _mm512_mul_ps(x6, mul_b);
        x7 = _mm512_mul_ps(x7, mul_b);
    }

    volatile float sink =
        _mm_cvtss_f32(_mm512_extractf32x4_ps(x0, 0)) +
        _mm_cvtss_f32(_mm512_extractf32x4_ps(x7, 0));
    (void)sink;
}

// ── avx512_fma ───────────────────────────────────────────────────────────────
__attribute__((noinline))
void foo_avx512_fma(uint64_t data) {
    float fa, fb, init_val;
    if (data == 0) {
        fa = 1.0f;
        fb = 1.0f;
        init_val = 1.0f;
    } else {
        fa = 1.3333334f;
        fb = 0.75f;
        init_val = 1.234567f;
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

// -----------------------------------------------------------------------
// Kernel dispatch & mapping
// -----------------------------------------------------------------------
enum KernelType {
    K_IMUL,
    K_AVX2_ADD,
    K_AVX2_MUL,
    K_AVX2_FMA,
    K_AVX512_ADD,
    K_AVX512_MUL,
    K_AVX512_FMA,
    K_UNKNOWN
};

KernelType parse_kernel(const string &k) {
    if (k == "imul" || k == "imulq") return K_IMUL;
    if (k == "avx2_add" || k == "avx2_addps") return K_AVX2_ADD;
    if (k == "avx2_mul" || k == "avx2_mulps") return K_AVX2_MUL;
    if (k == "avx2_fma" || k == "avx2_fmaps" || k == "avx2_fmadd" || k == "avx2_fmaddps") return K_AVX2_FMA;
    if (k == "avx512_add" || k == "avx512_addps") return K_AVX512_ADD;
    if (k == "avx512_mul" || k == "avx512_mulps" || k == "avx512") return K_AVX512_MUL;
    if (k == "avx512_fma" || k == "avx512_fmaps" || k == "avx512_fmadd" || k == "avx512_fmaddps") return K_AVX512_FMA;
    return K_UNKNOWN;
}

// -----------------------------------------------------------------------
// Threading infrastructure
// -----------------------------------------------------------------------

struct ThreadArg {
    int         core_id;
    uint64_t    operand;
    KernelType  ktype;
};

pthread_barrier_t g_start_barrier;
pthread_barrier_t g_end_barrier;
vector<double> g_thread_time_s(NUM_CORES, 0.0);

void *worker(void *arg_) {
    ThreadArg *arg = (ThreadArg *)arg_;
    pin_to_core(arg->core_id);

    pthread_barrier_wait(&g_start_barrier);

    timespec t0{}, t1{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    switch (arg->ktype) {
        case K_IMUL:       foo_imul(arg->operand); break;
        case K_AVX2_ADD:   foo_avx2_add(arg->operand); break;
        case K_AVX2_MUL:   foo_avx2_mul(arg->operand); break;
        case K_AVX2_FMA:   foo_avx2_fma(arg->operand); break;
        case K_AVX512_ADD: foo_avx512_add(arg->operand); break;
        case K_AVX512_MUL: foo_avx512_mul(arg->operand); break;
        case K_AVX512_FMA: foo_avx512_fma(arg->operand); break;
        default: break;
    }

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

Result run_trial(uint64_t operand, KernelType ktype) {
    pthread_barrier_init(&g_start_barrier, nullptr, NUM_CORES + 1);
    pthread_barrier_init(&g_end_barrier,   nullptr, NUM_CORES + 1);

    vector<pthread_t> threads(NUM_CORES);
    vector<ThreadArg> args(NUM_CORES);
    for (int c = 0; c < NUM_CORES; ++c) {
        args[c] = {c, operand, ktype};
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

void benchmark(const string &label, uint64_t operand,
               const char *kernel, KernelType ktype, int freq_mhz,
               int num_trials, int num_warmups, ostream &out) {
    fprintf(stderr, "Kernel: %-12s | %d cores | operand %s @ %d MHz (Trials: %d, Warmups: %d)\n",
            kernel, NUM_CORES, label.c_str(), freq_mhz, num_trials, num_warmups);
    if (num_warmups > 0) {
        fprintf(stderr, "Warming up (%d runs)...\n", num_warmups);
        for (int w = 0; w < num_warmups; ++w) run_trial(operand, ktype);
    }

    out << "kernel,freq_mhz,operand,trial,mean_time_s,std_time_s,"
           "pkg_J,dram_J,cores_J,pkg_W,cores_W\n";
    fprintf(stderr, "Running %d trials...\n", num_trials);

    for (int trial = 0; trial < num_trials; ++trial) {
        if (num_trials >= 10 && trial % 10 == 0) {
            fprintf(stderr, "  trial %d/%d\n", trial, num_trials);
        }
        Result r = run_trial(operand, ktype);
        out << fixed << setprecision(9)
            << kernel   << "," << freq_mhz << "," << label   << "," << trial       << ","
            << r.mean_time_s  << "," << r.std_time_s << ","
            << r.pkg_J        << "," << r.dram_J     << "," << r.cores_J << ","
            << r.pkg_W        << "," << r.cores_W    << "\n";
        out.flush();
    }
}

int main(int argc, char *argv[]) {
    // Usage: <0x00|0xAA> <freq_mhz> <output_csv> [imul|avx2_add|avx2_mul|avx2_fma|avx512_add|avx512_mul|avx512_fma]
    if (argc < 4 || argc > 5 ||
        (string(argv[1]) != "0x00" && string(argv[1]) != "0xAA")) {
        fprintf(stderr,
            "Usage: %s <0x00|0xAA> <freq_mhz> <output_csv> [kernel]\n"
            "Supported kernels:\n"
            "  imul       (or imulq)\n"
            "  avx2_add   (or avx2_addps)\n"
            "  avx2_mul   (or avx2_mulps)\n"
            "  avx2_fma   (or avx2_fmaps)\n"
            "  avx512_add (or avx512_addps)\n"
            "  avx512_mul (or avx512_mulps)\n"
            "  avx512_fma (or avx512_fmaps)\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    string      label    = argv[1];
    int         freq_mhz = atoi(argv[2]);
    string      csv_name = argv[3];
    const char *kernel   = (argc == 5) ? argv[4] : "imul";

    KernelType ktype = parse_kernel(kernel);
    if (ktype == K_UNKNOWN) {
        fprintf(stderr, "ERROR: unknown kernel '%s'.\n"
                        "Valid kernels: imul, avx2_add, avx2_mul, avx2_fma, avx512_add, avx512_mul, avx512_fma\n",
                kernel);
        return EXIT_FAILURE;
    }

    int trials = DEFAULT_TRIALS;
    if (const char *env_t = getenv("TRIALS")) {
        trials = atoi(env_t);
        if (trials <= 0) trials = 1;
    }
    int warmups = DEFAULT_WARMUP_RUNS;
    if (const char *env_w = getenv("WARMUP_RUNS")) {
        warmups = atoi(env_w);
        if (warmups < 0) warmups = 0;
    }

    uint64_t operand = (label == "0x00")
                           ? 0x0000000000000000ULL
                           : 0xAAAAAAAAAAAAAAABULL;

    // Sanity-check RAPL paths before starting
    read_energy_uj(RAPL_PKG_FILE);
    read_energy_uj(RAPL_DRAM_FILE);

    fprintf(stderr, "Expected SNR improvement: ~%dx vs single core\n", NUM_CORES);
    fprintf(stderr, "Frequency: %d MHz | Operand: %s | Kernel: %s | Output: %s\n",
            freq_mhz, label.c_str(), kernel, csv_name.c_str());

    ofstream out(csv_name);
    if (!out.is_open()) {
        fprintf(stderr, "failed to open %s for writing\n", csv_name.c_str());
        return EXIT_FAILURE;
    }

    benchmark(label, operand, kernel, ktype, freq_mhz, trials, warmups, out);

    out.close();
    fprintf(stderr, "Results written to %s\n", csv_name.c_str());
    return 0;
}
