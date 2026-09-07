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
#include <immintrin.h>   // SSE2 / AVX-512 intrinsics

using namespace std;

// -----------------------------------------------------------------------
// Config — Gold 5512U: 28 physical cores (0-27), HT siblings (28-55) offline
// -----------------------------------------------------------------------

constexpr int      NUM_CORES    = 28;          // cores 0..27
constexpr uint64_t ITERATIONS   = 2800000000ULL;
constexpr int      TRIALS       = 100;
constexpr int      WARMUP_RUNS  = 5;

// Package RAPL sysfs — this chip has no PP0/"core" domain (confirmed via
// rdmsr -p 0 0x639 == 0), so we use package energy the same as the
// Broadwell node. Domains found: intel-rapl:0 = package-0, intel-rapl:1 = psys,
// intel-rapl:0:0 = dram.
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
// All kernels run exactly ITERATIONS instruction-executions across
// independent chains to saturate the relevant execution port(s).
//
// 'data' carries the operand bit pattern (0x0000...0000 or 0xAAAA...AAAB).
// The lower bits of 'data' drive the multiplier/addend so that 0x00 vs 0xAA
// creates different switching activity in the execution units — same idea
// as the original imulq experiment.
//
// Chain counts chosen to saturate throughput without stalling on latency:
//   imulq : latency 3,  throughput 1    → 4 chains  (ITERATIONS/4  iters)
//   addq  : latency 1,  throughput 0.25 → 4 chains  (ITERATIONS/4  iters)
//   mulps : latency 4,  throughput 0.5  → 8 chains  (ITERATIONS/8  iters)
//   avx512: latency 4,  throughput 0.5  → 8 chains  (ITERATIONS/8  iters)
// -----------------------------------------------------------------------

// ── imulq ────────────────────────────────────────────────────────────────────
// 64-bit integer multiply.  Fixed 3-cycle latency, never data-dependent.
__attribute__((noinline))
void foo_imulq(uint64_t data) {
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

// ── addq ─────────────────────────────────────────────────────────────────────
// 64-bit integer add.  Fixed 1-cycle latency (lowest-power baseline).
__attribute__((noinline))
void foo_addq(uint64_t data) {
    uint64_t x0 = 0xDEADBEEFCAFEBABEULL;
    uint64_t x1 = 0x0123456789ABCDEFULL;
    uint64_t x2 = 0xFEDCBA9876543210ULL;
    uint64_t x3 = 0xA5A5A5A5A5A5A5A5ULL;

    for (uint64_t i = 0; i < ITERATIONS / 4; ++i) {
        asm volatile(
            "addq %[d], %[a]\n\t"
            "addq %[d], %[b]\n\t"
            "addq %[d], %[c]\n\t"
            "addq %[d], %[e]\n\t"
            : [a] "+r"(x0), [b] "+r"(x1), [c] "+r"(x2), [e] "+r"(x3)
            : [d] "r"(data)
            : "cc");
    }
    asm volatile("" : : "r"(x0), "r"(x1), "r"(x2), "r"(x3));
}

// ── mulps ────────────────────────────────────────────────────────────────────
// SSE 128-bit packed single-precision float multiply (4 float lanes).
// Fixed 4-cycle latency. 8 independent XMM chains unrolled by 2 (16 insns/iter)
// so the total retired instructions across the loop is exactly ITERATIONS.
//
// Guaranteed Non-Converging Design:
//   Repeated floating-point multiplication by a single constant > 1.0 overflows
//   to +Inf within ~320 cycles; multiplying by 0.0 collapses to 0 immediately.
//   To guarantee values NEVER converge to 0, Inf, or NaN, and NEVER trigger
//   subnormal microcode assists, we use exact paired multipliers:
//
//   0x00 operand:
//     mul_a = 1.0f (0x3F800000: all-zero mantissa)
//     mul_b = 1.0f (0x3F800000: all-zero mantissa)
//     Steady state: 1.0f * 1.0f = 1.0f with 0 mantissa bit toggles.
//
//   0xAA operand:
//     mul_a = 1.3333334f (0x3FAAAAAB: alternating mantissa bits: 10101010101010101010101)
//     mul_b = 0.75f      (0x3F400000: exact IEEE-754 inverse)
//     On each cycle, values alternate dynamically between 1.234567f (0x3F9E064B)
//     and 1.646089f (0x3FD2B30F). It NEVER converges, NEVER overflows, and
//     actively flips 10+ bits per float lane on EVERY instruction.
__attribute__((noinline))
void foo_mulps(uint64_t data) {
    float fa, fb;
    if (data == 0) {
        fa = 1.0f;
        fb = 1.0f;
    } else {
        fa = 1.3333334f;
        fb = 0.75f;
    }
    __m128 mul_a = _mm_set1_ps(fa);
    __m128 mul_b = _mm_set1_ps(fb);

    __m128 x0 = _mm_set1_ps(1.234567f), x1 = _mm_set1_ps(1.234567f);
    __m128 x2 = _mm_set1_ps(1.234567f), x3 = _mm_set1_ps(1.234567f);
    __m128 x4 = _mm_set1_ps(1.234567f), x5 = _mm_set1_ps(1.234567f);
    __m128 x6 = _mm_set1_ps(1.234567f), x7 = _mm_set1_ps(1.234567f);

    for (uint64_t i = 0; i < ITERATIONS / 16; ++i) {
        // Step 1: multiply by mul_a (8 instructions)
        x0 = _mm_mul_ps(x0, mul_a);
        x1 = _mm_mul_ps(x1, mul_a);
        x2 = _mm_mul_ps(x2, mul_a);
        x3 = _mm_mul_ps(x3, mul_a);
        x4 = _mm_mul_ps(x4, mul_a);
        x5 = _mm_mul_ps(x5, mul_a);
        x6 = _mm_mul_ps(x6, mul_a);
        x7 = _mm_mul_ps(x7, mul_a);

        // Step 2: multiply by mul_b (8 instructions)
        x0 = _mm_mul_ps(x0, mul_b);
        x1 = _mm_mul_ps(x1, mul_b);
        x2 = _mm_mul_ps(x2, mul_b);
        x3 = _mm_mul_ps(x3, mul_b);
        x4 = _mm_mul_ps(x4, mul_b);
        x5 = _mm_mul_ps(x5, mul_b);
        x6 = _mm_mul_ps(x6, mul_b);
        x7 = _mm_mul_ps(x7, mul_b);
    }

    volatile float sink = _mm_cvtss_f32(x0) + _mm_cvtss_f32(x7);
    (void)sink;
}

// ── avx512 ───────────────────────────────────────────────────────────────────
// AVX-512 512-bit packed single-precision float multiply (16 float lanes).
// Fixed 4-cycle latency. 8 independent ZMM chains unrolled by 2 (16 insns/iter)
// Same alternating non-converging pairs as foo_mulps.
//
// *** WARNING ***: AVX-512 triggers PCU hardware frequency throttle on SPR
// that OVERRIDES intel_pstate settings. Actual silicon frequency will be lower
// than the pinned value.
__attribute__((noinline))
void foo_avx512(uint64_t data) {
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
        // Step 1: multiply by mul_a (8 instructions)
        x0 = _mm512_mul_ps(x0, mul_a);
        x1 = _mm512_mul_ps(x1, mul_a);
        x2 = _mm512_mul_ps(x2, mul_a);
        x3 = _mm512_mul_ps(x3, mul_a);
        x4 = _mm512_mul_ps(x4, mul_a);
        x5 = _mm512_mul_ps(x5, mul_a);
        x6 = _mm512_mul_ps(x6, mul_a);
        x7 = _mm512_mul_ps(x7, mul_a);

        // Step 2: multiply by mul_b (8 instructions)
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

// -----------------------------------------------------------------------
// Threading infrastructure
// -----------------------------------------------------------------------

struct ThreadArg {
    int         core_id;
    uint64_t    operand;
    const char *kernel;   // "imulq" | "addq" | "mulps" | "avx512"
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

    if      (!strcmp(arg->kernel, "imulq"))  foo_imulq(arg->operand);
    else if (!strcmp(arg->kernel, "addq"))   foo_addq (arg->operand);
    else if (!strcmp(arg->kernel, "mulps"))  foo_mulps (arg->operand);
    else if (!strcmp(arg->kernel, "avx512")) foo_avx512(arg->operand);

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

Result run_trial(uint64_t operand, const char *kernel) {
    pthread_barrier_init(&g_start_barrier, nullptr, NUM_CORES + 1);
    pthread_barrier_init(&g_end_barrier,   nullptr, NUM_CORES + 1);

    vector<pthread_t> threads(NUM_CORES);
    vector<ThreadArg> args(NUM_CORES);
    for (int c = 0; c < NUM_CORES; ++c) {
        args[c] = {c, operand, kernel};
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
               const char *kernel, int freq_mhz, ostream &out) {
    fprintf(stderr, "Kernel: %-8s | %d cores | operand %s @ %d MHz\n",
            kernel, NUM_CORES, label.c_str(), freq_mhz);
    fprintf(stderr, "Warming up (%d runs)...\n", WARMUP_RUNS);
    for (int w = 0; w < WARMUP_RUNS; ++w) run_trial(operand, kernel);

    out << "kernel,freq_mhz,operand,trial,mean_time_s,std_time_s,"
           "pkg_J,dram_J,cores_J,pkg_W,cores_W\n";
    fprintf(stderr, "Running %d trials...\n", TRIALS);

    for (int trial = 0; trial < TRIALS; ++trial) {
        if (trial % 10 == 0) fprintf(stderr, "  trial %d/%d\n", trial, TRIALS);
        Result r = run_trial(operand, kernel);
        out << fixed << setprecision(9)
            << kernel   << "," << freq_mhz << "," << label   << "," << trial       << ","
            << r.mean_time_s  << "," << r.std_time_s << ","
            << r.pkg_J        << "," << r.dram_J     << "," << r.cores_J << ","
            << r.pkg_W        << "," << r.cores_W    << "\n";
        out.flush();
    }
}

int main(int argc, char *argv[]) {
    // Usage: <0x00|0xAA> <freq_mhz> <output_csv> [imulq|addq|mulps|avx512]
    //   kernel defaults to "imulq" so existing shell script works unchanged.
    if (argc < 4 || argc > 5 ||
        (string(argv[1]) != "0x00" && string(argv[1]) != "0xAA")) {
        fprintf(stderr,
            "Usage: %s <0x00|0xAA> <freq_mhz> <output_csv> [imulq|addq|mulps|avx512]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    string      label    = argv[1];
    int         freq_mhz = atoi(argv[2]);
    string      csv_name = argv[3];
    const char *kernel   = (argc == 5) ? argv[4] : "imulq";

    if (strcmp(kernel, "imulq") && strcmp(kernel, "addq") &&
        strcmp(kernel, "mulps") && strcmp(kernel, "avx512")) {
        fprintf(stderr, "ERROR: unknown kernel '%s'. Valid: imulq  addq  mulps  avx512\n", kernel);
        return EXIT_FAILURE;
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

    benchmark(label, operand, kernel, freq_mhz, out);

    out.close();
    fprintf(stderr, "Results written to %s\n", csv_name.c_str());
    return 0;
}
