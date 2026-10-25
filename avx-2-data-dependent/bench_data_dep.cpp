// bench_data_dep.cpp  (v2: + CPU cycles via perf_event_open, + TSC ticks via RDTSC/RDTSCP)
//
// Tests whether the power/energy of vector multiply instructions depends on
// operand VALUES. 28-core pinned RAPL harness; the kernel is a fixed-operand,
// independent-op inline-asm loop.
//
// ONE condition per run: you give two operands (data1, data2). The program
// measures TRIALS runs of that single condition and APPENDS the rows (plus
// the pre-computed bit-flip statistics) to one shared CSV.
//
// NEW in v2 (from https://iitgn-fuss.github.io/interesting-reads/measure-your-code/):
//   * cycles    : per-thread CPU_CYCLES hardware counter (perf_event_open),
//                 enabled/disabled right around the kernel loop.
//   * tsc_ticks : per-thread TSC ticks, CPUID+RDTSC at start, RDTSCP+CPUID at end
//                 (the serialised Intel-whitepaper sequence from the article).
//   Both are averaged over the 28 threads. Derived columns:
//   cycles_per_op, core_ghz (= cycles / time), tsc_ghz (= tsc_ticks / time).
//   core_ghz is the real core clock (reveals throttling); tsc_ghz is the
//   constant reference clock.
//   perf_event_open normally needs root (or perf_event_paranoid <= 1). If it
//   cannot be opened, cycles are written as 0 and a warning is printed.
//
// Ops:
//   mul      vmulps       ymm (AVX2)
//   fma      vfmadd231ps  ymm (AVX2/FMA)
//   ifma_lo  vpmadd52luq  ymm (AVX-IFMA / AVX512-IFMA+VL)
//   ifma_hi  vpmadd52huq  ymm
//
// Build:
//   g++ -O2 -mavx2 -mfma -pthread bench_data_dep.cpp -o bench
//
// Usage:
//   sudo ./bench <mul|fma|ifma_lo|ifma_hi> <data1_hex> <data2_hex> [toggle 0|1]
//
// Optional environment variables:
//   BENCH_OUT      output CSV (appended; header written if file is empty)  [bench_results.csv]
//   BENCH_TRIALS   measured trials per run                                 [20]
//   BENCH_WARMUP   unmeasured warm-up runs                                 [5]
//   BENCH_ITERS    ops per thread per trial                                [3000000000]
//   BENCH_ROUND    sweep round number, copied into the CSV                 [0]
//   BENCH_D1_NAME / BENCH_D2_NAME  human-readable operand labels           [""]
//   BENCH_FREQ_KHZ frequency the sweep script pinned, copied into the CSV  [0]
//   BENCH_TOGGLE_A 1 = with toggle=1, data1 ALSO alternates (a, a^mask)    [0]
//   BENCH_C        addend hex value for FMA/IFMA                           [0]
//   BENCH_STATS_ONLY 1 = print "in,res,dest,total" flip counts and exit    [0]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cerrno>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <vector>
#include <algorithm>

using namespace std;

// ----------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------
constexpr int      NUM_CORES   = 28;
uint64_t ITERATIONS  = 3000000000ULL;   // ops per thread per run
int      TRIALS      = 20;
int      WARMUP_RUNS = 5;

constexpr const char *RAPL_PKG_FILE =
    "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj";

// ----------------------------------------------------------------------
// RAPL / timing / pinning helpers
// ----------------------------------------------------------------------
uint64_t read_energy_uj(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(EXIT_FAILURE); }
    unsigned long long val = 0;
    if (fscanf(f, "%llu", &val) != 1) {
        fprintf(stderr, "failed to parse %s\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(f);
    return (uint64_t)val;
}

uint64_t read_max_energy_range_uj(const char *energy_path) {
    string p(energy_path);
    size_t pos = p.find_last_of('/');
    string range_path = p.substr(0, pos + 1) + "max_energy_range_uj";
    FILE *f = fopen(range_path.c_str(), "r");
    if (!f) return 0;
    unsigned long long val = 0;
    if (fscanf(f, "%llu", &val) != 1) val = 0;
    fclose(f);
    return (uint64_t)val;
}

inline double energy_diff_j(uint64_t before, uint64_t after, uint64_t max_range) {
    uint64_t diff;
    if (after >= before)      diff = after - before;
    else if (max_range > 0)   diff = (max_range - before) + after;  // wrapped
    else                      diff = 0;
    return diff / 1e6;  // uJ -> J
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

// ----------------------------------------------------------------------
// TSC ticks (from the FUSS "measure your code" article):
//   CPUID+RDTSC to start, RDTSCP+CPUID to stop (serialises the pipeline).
// ----------------------------------------------------------------------
static inline unsigned long long measure_rdtsc_start() {
    unsigned cycles_low, cycles_high;
    asm volatile("CPUID\n\t"
                 "RDTSC\n\t"
                 "mov %%edx, %0\n\t"
                 "mov %%eax, %1\n\t"
                 : "=r"(cycles_high), "=r"(cycles_low)
                 :: "%rax", "%rbx", "%rcx", "%rdx");
    return ((unsigned long long)cycles_high << 32) | cycles_low;
}

static inline unsigned long long measure_rdtscp_end() {
    unsigned cycles_low, cycles_high;
    asm volatile("RDTSCP\n\t"
                 "mov %%edx, %0\n\t"
                 "mov %%eax, %1\n\t"
                 "CPUID\n\t"
                 : "=r"(cycles_high), "=r"(cycles_low)
                 :: "%rax", "%rbx", "%rcx", "%rdx");
    return ((unsigned long long)cycles_high << 32) | cycles_low;
}

// ----------------------------------------------------------------------
// CPU cycles via perf_event_open (adapted from the article's perf_utils).
// The article's version uses global fds; here every worker thread opens its
// own counter (pid=0, cpu=-1 => this thread, wherever it runs; threads are
// pinned anyway), so counts are per-thread.
// ----------------------------------------------------------------------
static long perf_event_open_sys(struct perf_event_attr *hw_event, pid_t pid,
                                int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int open_cycles_counter() {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size           = sizeof(pe);
    pe.type           = PERF_TYPE_HARDWARE;
    pe.config         = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled       = 1;
    pe.exclude_kernel = 0;   // as in the article: user + kernel
    pe.exclude_hv     = 1;
    pe.exclude_idle   = 1;
    pe.exclude_user   = 0;
    return (int)perf_event_open_sys(&pe, 0, -1, -1, 0);
}

static inline void perf_start(int fd) {
    if (fd < 0) return;
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
}

static inline void perf_stop(int fd) {
    if (fd < 0) return;
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
}

static inline uint64_t perf_read(int fd) {
    if (fd < 0) return 0;
    uint64_t v = 0;
    if (read(fd, &v, sizeof(v)) != (ssize_t)sizeof(v)) return 0;
    return v;
}

// ----------------------------------------------------------------------
// Kernel  (register-only ymm FMA; see v1 notes)
//
//   Register map:  ymm0 = a0   ymm1 = a1 (== a0 unless BENCH_TOGGLE_A)
//                  ymm2 = b0   ymm3 = b1 (== b0 unless toggle)
//                  ymm4 = c    ymm5..ymm12 = destinations
//
// Operand order per 8-op iteration is PAIRED (b0 b0 b1 b1 b0 b0 b1 b1) so
// that each of the two FMA ports sees b0,b1,b0,b1 and flips on every op.
// ----------------------------------------------------------------------
enum Op { OP_MUL = 0, OP_FMA, OP_IFMA_LO, OP_IFMA_HI };

alignas(32) uint64_t g_a[4], g_a1[4], g_b0[4], g_b1[4], g_c[4];
int g_op = OP_FMA;
int g_toggle_a = 0;

#define MUL_ONE(a, b, d)                           \
    "vmulps %%" b ", %%" a ", %%" d "\n\t"

#define MUL_BODY                                   \
    MUL_ONE("ymm0", "ymm2", "ymm5")                \
    MUL_ONE("ymm0", "ymm2", "ymm6")                \
    MUL_ONE("ymm1", "ymm3", "ymm7")                \
    MUL_ONE("ymm1", "ymm3", "ymm8")                \
    MUL_ONE("ymm0", "ymm2", "ymm9")                \
    MUL_ONE("ymm0", "ymm2", "ymm10")               \
    MUL_ONE("ymm1", "ymm3", "ymm11")               \
    MUL_ONE("ymm1", "ymm3", "ymm12")

#define ACC_ONE(insn, a, b, d)                     \
    "vmovdqa %%ymm4, %%" d "\n\t"                  \
    insn " %%" b ", %%" a ", %%" d "\n\t"

#define ACC_BODY(insn)                             \
    ACC_ONE(insn, "ymm0", "ymm2", "ymm5")          \
    ACC_ONE(insn, "ymm0", "ymm2", "ymm6")          \
    ACC_ONE(insn, "ymm1", "ymm3", "ymm7")          \
    ACC_ONE(insn, "ymm1", "ymm3", "ymm8")          \
    ACC_ONE(insn, "ymm0", "ymm2", "ymm9")          \
    ACC_ONE(insn, "ymm0", "ymm2", "ymm10")         \
    ACC_ONE(insn, "ymm1", "ymm3", "ymm11")         \
    ACC_ONE(insn, "ymm1", "ymm3", "ymm12")

#define LOAD_REGS                                  \
    "vmovdqu (%[a]),  %%ymm0\n\t"                  \
    "vmovdqu (%[a1]), %%ymm1\n\t"                  \
    "vmovdqu (%[b0]), %%ymm2\n\t"                  \
    "vmovdqu (%[b1]), %%ymm3\n\t"                  \
    "vmovdqu (%[c]),  %%ymm4\n\t"

#define CLOBBERS "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",  \
                 "ymm8","ymm9","ymm10","ymm11","ymm12","cc","memory"

#define RUN_LOOP(BODY)                                                    \
    asm volatile(                                                         \
        LOAD_REGS                                                         \
        "1:\n\t"                                                          \
        BODY                                                              \
        "dec %[n]\n\t"                                                    \
        "jnz 1b\n\t"                                                      \
        "vzeroupper\n\t"                                                  \
        : [n] "+r"(n)                                                     \
        : [a] "r"(g_a), [a1] "r"(g_a1), [b0] "r"(g_b0), [b1] "r"(g_b1),   \
          [c] "r"(g_c)                                                    \
        : CLOBBERS)

__attribute__((noinline))
void kernel(int op) {
    uint64_t n = ITERATIONS / 8;   // 8 ops per iteration
    switch (op) {
    case OP_MUL:      RUN_LOOP(MUL_BODY);                     break;
    case OP_FMA:      RUN_LOOP(ACC_BODY("vfmadd231ps"));      break;
    case OP_IFMA_LO:  RUN_LOOP(ACC_BODY("vpmadd52luq"));      break;
    case OP_IFMA_HI:  RUN_LOOP(ACC_BODY("vpmadd52huq"));      break;
    }
}

// ----------------------------------------------------------------------
// Threading
// ----------------------------------------------------------------------
struct ThreadArg { int core_id; };

pthread_barrier_t g_start_barrier, g_end_barrier;
vector<double>   g_thread_time_s(NUM_CORES, 0.0);
vector<uint64_t> g_thread_cycles(NUM_CORES, 0);
vector<uint64_t> g_thread_tsc(NUM_CORES, 0);
volatile int     g_perf_failed = 0;

void *worker(void *arg_) {
    ThreadArg *arg = (ThreadArg *)arg_;
    pin_to_core(arg->core_id);

    int pfd = open_cycles_counter();       // per-thread cycles counter
    if (pfd < 0) g_perf_failed = 1;

    pthread_barrier_wait(&g_start_barrier);

    timespec t0{}, t1{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    perf_start(pfd);
    unsigned long long tsc0 = measure_rdtsc_start();
    kernel(g_op);
    unsigned long long tsc1 = measure_rdtscp_end();
    perf_stop(pfd);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    g_thread_time_s[arg->core_id] = elapsed_seconds(t0, t1);
    g_thread_tsc[arg->core_id]    = tsc1 - tsc0;
    g_thread_cycles[arg->core_id] = perf_read(pfd);
    if (pfd >= 0) close(pfd);

    pthread_barrier_wait(&g_end_barrier);
    return nullptr;
}

// ----------------------------------------------------------------------
// Conditions and trials
// ----------------------------------------------------------------------
struct Cond {
    uint64_t a;       // data1 (fixed operand)
    uint64_t b;       // data2
    uint64_t c;       // addend
};

struct Result {
    double mean_time_s, std_time_s, pkg_J, pkg_W, nJ_per_op;
    double mean_cycles, mean_tsc, cycles_per_op, core_ghz, tsc_ghz;
};

static uint64_t lane_value(uint64_t v, int op) {
    if (op == OP_MUL || op == OP_FMA) {
        uint64_t lo = v & 0xFFFFFFFFULL;
        return (lo << 32) | lo;
    }
    return v;
}

static uint64_t toggle_mask(int op) {
    if (op == OP_MUL || op == OP_FMA) return 0x007FFFFF007FFFFFULL;
    return 0x000FFFFFFFFFFFFFULL;   // 52-bit IFMA operand
}

// ----------------------------------------------------------------------
// Bit-flip statistics (unchanged from v1)
// ----------------------------------------------------------------------
struct BitStats {
    int    a_weight, b0_weight, b1_weight, r0_weight, r1_weight;
    int    input_flips_per_op;
    int    result_flips_per_op;
    double dest_flips_per_op;
    double total_flips_per_op;
};

static uint64_t lane_result(int op, uint64_t a, uint64_t b, uint64_t c) {
    if (op == OP_MUL || op == OP_FMA) {
        uint64_t out = 0;
        for (int h = 0; h < 2; ++h) {
            uint32_t ai = (uint32_t)(a >> (32 * h));
            uint32_t bi = (uint32_t)(b >> (32 * h));
            uint32_t ci = (uint32_t)(c >> (32 * h));
            float fa, fb, fc;
            memcpy(&fa, &ai, 4); memcpy(&fb, &bi, 4); memcpy(&fc, &ci, 4);
            float r = (op == OP_MUL) ? fa * fb : fmaf(fa, fb, fc);
            uint32_t ri; memcpy(&ri, &r, 4);
            out |= (uint64_t)ri << (32 * h);
        }
        return out;
    }
    const uint64_t M = (1ULL << 52) - 1;
    unsigned __int128 p = (unsigned __int128)(a & M) * (b & M);
    uint64_t t = (op == OP_IFMA_LO) ? (uint64_t)(p & M) : (uint64_t)(p >> 52);
    return c + t;
}

static uint64_t a1_value(uint64_t a0, int toggle) {
    return (toggle && g_toggle_a) ? (a0 ^ toggle_mask(g_op)) : a0;
}

static BitStats compute_bitstats(const Cond &c, int toggle) {
    uint64_t a0 = lane_value(c.a, g_op);
    uint64_t a1 = a1_value(a0, toggle);
    uint64_t b0 = lane_value(c.b, g_op);
    uint64_t b1 = toggle ? (b0 ^ toggle_mask(g_op)) : b0;
    uint64_t cc = lane_value(c.c, g_op);
    uint64_t r0 = lane_result(g_op, a0, b0, cc);
    uint64_t r1 = lane_result(g_op, a1, b1, cc);

    auto pc = [](uint64_t x) { return __builtin_popcountll(x); };
    const int L = 4;

    BitStats s;
    s.a_weight  = L * pc(a0);
    s.b0_weight = L * pc(b0);
    s.b1_weight = L * pc(b1);
    s.r0_weight = L * pc(r0);
    s.r1_weight = L * pc(r1);
    s.input_flips_per_op  = L * (pc(b0 ^ b1) + pc(a0 ^ a1));
    s.result_flips_per_op = L * pc(r0 ^ r1);
    if (g_op == OP_MUL)
        s.dest_flips_per_op = 0.0;
    else
        s.dest_flips_per_op = L * (2.0 * pc(r0 ^ cc) + 2.0 * pc(r1 ^ cc)) / 2.0;
    s.total_flips_per_op = s.input_flips_per_op + s.result_flips_per_op
                         + s.dest_flips_per_op;
    return s;
}

// ----------------------------------------------------------------------
// One timed run (28 pinned threads + RAPL read + cycles + TSC)
// ----------------------------------------------------------------------
Result run_trial(const Cond &c, int toggle) {
    uint64_t a  = lane_value(c.a, g_op);
    uint64_t a1 = a1_value(a, toggle);
    uint64_t b0 = lane_value(c.b, g_op);
    uint64_t b1 = toggle ? (b0 ^ toggle_mask(g_op)) : b0;
    uint64_t cc = lane_value(c.c, g_op);
    for (int i = 0; i < 4; ++i) {
        g_a[i] = a; g_a1[i] = a1; g_b0[i] = b0; g_b1[i] = b1; g_c[i] = cc;
    }

    pthread_barrier_init(&g_start_barrier, nullptr, NUM_CORES + 1);
    pthread_barrier_init(&g_end_barrier,   nullptr, NUM_CORES + 1);

    vector<pthread_t> threads(NUM_CORES);
    vector<ThreadArg> args(NUM_CORES);
    for (int i = 0; i < NUM_CORES; ++i) {
        args[i] = {i};
        pthread_create(&threads[i], nullptr, worker, &args[i]);
    }

    uint64_t max_pkg = read_max_energy_range_uj(RAPL_PKG_FILE);
    uint64_t before  = read_energy_uj(RAPL_PKG_FILE);
    pthread_barrier_wait(&g_start_barrier);
    pthread_barrier_wait(&g_end_barrier);
    uint64_t after   = read_energy_uj(RAPL_PKG_FILE);

    for (auto &t : threads) pthread_join(t, nullptr);
    pthread_barrier_destroy(&g_start_barrier);
    pthread_barrier_destroy(&g_end_barrier);

    double pkg_J = energy_diff_j(before, after, max_pkg);

    double sum = 0, sumsq = 0, csum = 0, tsum = 0;
    for (int i = 0; i < NUM_CORES; ++i) {
        double t = g_thread_time_s[i];
        sum += t; sumsq += t * t;
        csum += (double)g_thread_cycles[i];
        tsum += (double)g_thread_tsc[i];
    }
    double mean_t = sum / NUM_CORES;
    double std_t  = sqrt(max(0.0, sumsq / NUM_CORES - mean_t * mean_t));
    double mean_c = csum / NUM_CORES;
    double mean_tsc = tsum / NUM_CORES;

    double total_ops = (double)ITERATIONS * NUM_CORES;
    Result r;
    r.mean_time_s   = mean_t;
    r.std_time_s    = std_t;
    r.pkg_J         = pkg_J;
    r.pkg_W         = pkg_J / mean_t;
    r.nJ_per_op     = pkg_J / total_ops * 1e9;
    r.mean_cycles   = mean_c;
    r.mean_tsc      = mean_tsc;
    r.cycles_per_op = mean_c / (double)ITERATIONS;
    r.core_ghz      = mean_c / mean_t / 1e9;
    r.tsc_ghz       = mean_tsc / mean_t / 1e9;
    return r;
}

// ----------------------------------------------------------------------
// main
// ----------------------------------------------------------------------
static int parse_op(const string &s) {
    if (s == "mul")     return OP_MUL;
    if (s == "fma")     return OP_FMA;
    if (s == "ifma_lo") return OP_IFMA_LO;
    if (s == "ifma_hi") return OP_IFMA_HI;
    return -1;
}

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5 || parse_op(argv[1]) < 0) {
        fprintf(stderr,
            "Usage: %s <mul|fma|ifma_lo|ifma_hi> <data1_hex> <data2_hex> [toggle 0|1]\n"
            "  e.g. %s fma 0x3fa00000 0x3fb504f3 0\n",
            argv[0], argv[0]);
        return EXIT_FAILURE;
    }
    g_op = parse_op(argv[1]);
    uint64_t d1 = strtoull(argv[2], nullptr, 16);
    uint64_t d2 = strtoull(argv[3], nullptr, 16);
    int toggle  = (argc == 5) ? atoi(argv[4]) : 0;

    if (const char *e = getenv("BENCH_ITERS"))  ITERATIONS  = strtoull(e, nullptr, 10);
    if (const char *e = getenv("BENCH_TRIALS")) TRIALS      = atoi(e);
    if (const char *e = getenv("BENCH_WARMUP")) WARMUP_RUNS = atoi(e);
    if (const char *e = getenv("BENCH_TOGGLE_A")) g_toggle_a = atoi(e) != 0;
    uint64_t dc = 0;
    if (const char *e = getenv("BENCH_C")) dc = strtoull(e, nullptr, 16);
    const char *out_path = getenv("BENCH_OUT")      ? getenv("BENCH_OUT")      : "bench_results.csv";
    const char *round_s  = getenv("BENCH_ROUND")    ? getenv("BENCH_ROUND")    : "0";
    const char *d1_name  = getenv("BENCH_D1_NAME")  ? getenv("BENCH_D1_NAME")  : "";
    const char *d2_name  = getenv("BENCH_D2_NAME")  ? getenv("BENCH_D2_NAME")  : "";
    const char *freq_s   = getenv("BENCH_FREQ_KHZ") ? getenv("BENCH_FREQ_KHZ") : "0";

    Cond cond{d1, d2, dc};

    BitStats s = compute_bitstats(cond, toggle);
    fprintf(stderr, "op=%s d1=0x%llx d2=0x%llx c=0x%llx toggle=%d toggle_a=%d | in_flips=%d res_flips=%d "
                    "dest_flips=%.1f total=%.1f w(a,b0,b1,r0,r1)=%d,%d,%d,%d,%d\n",
            argv[1], (unsigned long long)d1, (unsigned long long)d2,
            (unsigned long long)dc, toggle, g_toggle_a,
            s.input_flips_per_op, s.result_flips_per_op, s.dest_flips_per_op,
            s.total_flips_per_op, s.a_weight, s.b0_weight, s.b1_weight,
            s.r0_weight, s.r1_weight);

    if (const char *e = getenv("BENCH_STATS_ONLY")) {
        if (atoi(e) != 0) {
            printf("%d,%d,%.1f,%.1f\n", s.input_flips_per_op,
                   s.result_flips_per_op, s.dest_flips_per_op,
                   s.total_flips_per_op);
            return 0;
        }
    }

    read_energy_uj(RAPL_PKG_FILE);   // sanity check that RAPL is readable

    for (int w = 0; w < WARMUP_RUNS; ++w) run_trial(cond, toggle);

    char hex1[32], hex2[32], hexc[32];
    snprintf(hex1, sizeof hex1, "0x%llx", (unsigned long long)d1);
    snprintf(hex2, sizeof hex2, "0x%llx", (unsigned long long)d2);
    snprintf(hexc, sizeof hexc, "0x%llx", (unsigned long long)dc);

    g_perf_failed = 0;
    ostringstream rows;
    for (int trial = 0; trial < TRIALS; ++trial) {
        Result r = run_trial(cond, toggle);
        rows << fixed << setprecision(9)
             << argv[1] << "," << round_s << "," << d1_name << "," << d2_name << ","
             << hex1 << "," << hex2 << "," << hexc << ","
             << freq_s << "," << toggle << "," << trial << ","
             << r.mean_time_s << "," << r.std_time_s << ","
             << r.pkg_J << "," << r.pkg_W << "," << r.nJ_per_op << ","
             << s.a_weight << "," << s.b0_weight << "," << s.b1_weight << ","
             << s.r0_weight << "," << s.r1_weight << ","
             << s.input_flips_per_op << "," << s.result_flips_per_op << ","
             << s.dest_flips_per_op << "," << s.total_flips_per_op << ","
             << g_toggle_a << ","
             << setprecision(1) << r.mean_cycles << "," << r.mean_tsc << ","
             << setprecision(9) << r.cycles_per_op << ","
             << r.core_ghz << "," << r.tsc_ghz << "\n";
    }
    if (g_perf_failed)
        fprintf(stderr, "WARNING: perf_event_open failed (run as root or set "
                        "kernel.perf_event_paranoid<=1); cycles column is 0.\n");

    bool need_header = true;
    {
        ifstream probe(out_path, ios::ate);
        if (probe.good() && (long long)probe.tellg() > 0) need_header = false;
    }
    ofstream out(out_path, ios::app);
    if (!out.is_open()) {
        fprintf(stderr, "failed to open %s\n", out_path);
        return EXIT_FAILURE;
    }
    if (need_header)
        out << "op,round,d1_name,d2_name,data1,data2,datac,freq_khz,toggle,trial,"
               "mean_time_s,std_time_s,pkg_J,pkg_W,nJ_per_op,"
               "a_weight,b0_weight,b1_weight,r0_weight,r1_weight,"
               "input_flips_per_op,result_flips_per_op,dest_flips_per_op,total_flips_per_op,"
               "toggle_a,cycles,tsc_ticks,cycles_per_op,core_ghz,tsc_ghz\n";
    out << rows.str();
    out.flush();
    fprintf(stderr, "appended %d rows to %s\n", TRIALS, out_path);
    return 0;
}
