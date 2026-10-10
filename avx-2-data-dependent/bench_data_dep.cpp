// bench_data_dep.cpp
//
// Tests whether the power/energy of vector multiply instructions depends on
// operand VALUES. 28-core pinned RAPL harness; the kernel is a fixed-operand,
// independent-op inline-asm loop.
//
// ONE condition per run: you give two operands (data1, data2). The program
// measures TRIALS runs of that single condition and APPENDS the rows (plus
// the pre-computed bit-flip statistics) to one shared CSV. run_sweep.sh calls
// this binary for every (data1, data2) combination, so all results end up in
// the same file.
//
// Ops:
//   mul      vmulps       ymm (AVX2)
//   fma      vfmadd231ps  ymm (AVX2/FMA)
//   ifma_lo  vpmadd52luq  ymm (AVX-IFMA / AVX512-IFMA+VL)  <- used by OpenSSL
//   ifma_hi  vpmadd52huq  ymm                                 rsaz_exp_x2.c
//
// Build:
//   g++ -O2 -mavx2 -mfma -pthread bench_data_dep.cpp -o bench
//   (needs binutils >= 2.35 for vpmadd52 mnemonics)
//
// Usage:
//   ./bench <mul|fma|ifma_lo|ifma_hi> <data1_hex> <data2_hex> [toggle 0|1]
//     mul/fma : lower 32 bits of each value is an IEEE float bit pattern
//     ifma_*  : value is a 64-bit lane (only the low 52 bits are multiplied)
//     toggle=1: second operand alternates between data2 and data2^mantissa_mask
//               on consecutive ops, forcing bit transitions.
//
// Optional environment variables:
//   BENCH_OUT      output CSV (appended; header written if file is empty)  [bench_results.csv]
//   BENCH_TRIALS   measured trials per run                                 [100]
//   BENCH_WARMUP   unmeasured warm-up runs                                 [5]
//   BENCH_ITERS    ops per thread per trial                                [3000000000]
//   BENCH_ROUND    sweep round number, copied into the CSV                 [0]
//   BENCH_D1_NAME / BENCH_D2_NAME  human-readable operand labels           [""]
//   BENCH_FREQ_KHZ frequency the sweep script pinned, copied into the CSV  [0]
//
//   --- load knobs (raise package power so the CPU itself throttles) ---
//   BENCH_TOGGLE_A 1 = with toggle=1, data1 ALSO alternates (a, a^mask) in
//                      step with data2, so both multiplier inputs flip.
//                      Has no effect when toggle=0 (LOW condition).         [0]
//   BENCH_C        addend hex value for FMA/IFMA (the value each destination
//                      is reset to before every op). A non-zero c raises the
//                      dest-register flip count, because the dest toggles
//                      c -> result -> c instead of 0 -> result -> 0.        [0]
//   BENCH_STATS_ONLY 1 = print "in,res,dest,total" flip counts on stdout and
//                      exit without measuring (operand screening).          [0]
//
// NOTE: a memory-operand ("heavy") kernel was tried and removed. Sourcing the
// data2 operand from L1 pushed the part into a fixed 2600 MHz ceiling for BOTH
// operand sets, which destroys the comparison. Everything here stays
// register-only ymm FMA so the 3.0 GHz all-core AVX ceiling is kept.
//
// Example:
//   sudo ./bench fma 0x3fa00000 0x3fb504f3 0
//
// Fix the frequency first (run_sweep.sh does this for you), e.g.:
//   sudo cpupower frequency-set -g performance
//   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <vector>
#include <algorithm>

using namespace std;

// ----------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------
constexpr int      NUM_CORES   = 56;   // 28 physical + 28 HT siblings; all threads active
// Defaults; override at runtime with env vars BENCH_ITERS / BENCH_TRIALS /
// BENCH_WARMUP (used by run_sweep.sh). Output CSV path: BENCH_OUT.
uint64_t ITERATIONS  = 3000000000ULL;   // ops per thread per run
int      TRIALS      = 100;
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
// Kernel  (register-only ymm FMA: no memory operands, so the part stays at
//          the 3.0 GHz all-core AVX ceiling instead of the 2600 MHz ceiling
//          that load-port traffic triggers)
//
//   Register map:  ymm0 = a0   ymm1 = a1 (== a0 unless BENCH_TOGGLE_A)
//                  ymm2 = b0   ymm3 = b1 (== b0 unless toggle)
//                  ymm4 = c    ymm5..ymm12 = destinations
//
// Operand order per 8-op iteration is PAIRED, not alternating:
//
//   op index   0   1   2   3   4   5   6   7
//   operand   b0  b0  b1  b1  b0  b0  b1  b1   (and a0 a0 a1 a1 ...)
//
// The two FMA ports usually take alternate ops, so port A gets ops 0,2,4,6 and
// port B gets ops 1,3,5,7. With this paired order each port then sees
// b0,b1,b0,b1 -- its operand flips on EVERY op it executes.
//
// The previous order (b0,b1,b0,b1,...) looks like it toggles, but handing
// alternate ops to alternate ports leaves port A with b0,b0,b0,b0 and port B
// with b1,b1,b1,b1: each multiplier sees a CONSTANT operand and never pays the
// switching energy. That is why toggling only ever bought ~1.5 W.
//
// Inputs never change and results are never fed back, so there is no
// data-dependent dependency chain, and the compiler cannot hoist the work
// (it is inline asm).
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

// FMA / IFMA accumulate into the destination, so reset it from c each time
// (reg-reg move; identical cost in A and B, so it cancels in the comparison).
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
vector<double> g_thread_time_s(NUM_CORES, 0.0);

void *worker(void *arg_) {
    ThreadArg *arg = (ThreadArg *)arg_;
    pin_to_core(arg->core_id);
    pthread_barrier_wait(&g_start_barrier);

    timespec t0{}, t1{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    kernel(g_op);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    g_thread_time_s[arg->core_id] = elapsed_seconds(t0, t1);

    pthread_barrier_wait(&g_end_barrier);
    return nullptr;
}

// ----------------------------------------------------------------------
// Conditions and trials
// ----------------------------------------------------------------------
struct Cond {
    uint64_t a;       // data1 (fixed operand)
    uint64_t b;       // data2
    uint64_t c;       // addend (0)
};

struct Result {
    double mean_time_s, std_time_s, pkg_J, pkg_W, nJ_per_op;
};

// Replicate/pack operand into all lanes. For float ops the low 32 bits are
// an IEEE float pattern repeated in both halves of each 64-bit lane.
static uint64_t lane_value(uint64_t v, int op) {
    if (op == OP_MUL || op == OP_FMA) {
        uint64_t lo = v & 0xFFFFFFFFULL;
        return (lo << 32) | lo;
    }
    return v;
}

static uint64_t toggle_mask(int op) {
    // Flip mantissa only (keeps sign/exponent, so no denormal/Inf/NaN).
    if (op == OP_MUL || op == OP_FMA) return 0x007FFFFF007FFFFFULL;
    return 0x000FFFFFFFFFFFFFULL;   // 52-bit IFMA operand
}

// ----------------------------------------------------------------------
// Bit-flip statistics: computed ONCE per condition, before the experiment.
// Valid because the same ops repeat with fixed operands. Assumes the two
// toggle states (b0/a0 and b1/a1) alternate at the execution-unit level.
//
//   *_weight            popcount over the full 256-bit register
//   input_flips_per_op  HD(b0,b1) (+ HD(a0,a1) with BENCH_TOGGLE_A) between
//                       consecutive ops
//   result_flips_per_op HD(r0,r1) between consecutive results
//   dest_flips_per_op   dest-register overwrites: result -> c (vmovdqa reset)
//                       and c -> result (the op); 0 for mul (each dest reg
//                       always receives the same value)
//   total_flips_per_op  sum of the three above
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
        for (int h = 0; h < 2; ++h) {                 // two floats per 64-bit lane
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
    return c + t;                                      // accumulate into c (=0)
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
    const int L = 4;                                   // 4 x 64-bit lanes per ymm

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
    else   // dest: r -> c (vmovdqa) then c -> r (op); averaged over r0/r1
        s.dest_flips_per_op = L * (2.0 * pc(r0 ^ cc) + 2.0 * pc(r1 ^ cc)) / 2.0;
    s.total_flips_per_op = s.input_flips_per_op + s.result_flips_per_op
                         + s.dest_flips_per_op;
    return s;
}

// ----------------------------------------------------------------------
// One timed run (28 pinned threads + RAPL read)
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

    double sum = 0, sumsq = 0;
    for (double t : g_thread_time_s) { sum += t; sumsq += t * t; }
    double mean_t = sum / NUM_CORES;
    double std_t  = sqrt(max(0.0, sumsq / NUM_CORES - mean_t * mean_t));

    double total_ops = (double)ITERATIONS * NUM_CORES;
    return { mean_t, std_t, pkg_J, pkg_J / mean_t, pkg_J / total_ops * 1e9 };
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

    // Bit-flip stats: computed once, outside the measured experiment.
    BitStats s = compute_bitstats(cond, toggle);
    fprintf(stderr, "op=%s d1=0x%llx d2=0x%llx c=0x%llx toggle=%d toggle_a=%d | in_flips=%d res_flips=%d "
                    "dest_flips=%.1f total=%.1f w(a,b0,b1,r0,r1)=%d,%d,%d,%d,%d\n",
            argv[1], (unsigned long long)d1, (unsigned long long)d2,
            (unsigned long long)dc, toggle, g_toggle_a,
            s.input_flips_per_op, s.result_flips_per_op, s.dest_flips_per_op,
            s.total_flips_per_op, s.a_weight, s.b0_weight, s.b1_weight,
            s.r0_weight, s.r1_weight);

    // Screening mode: print the flip counts and exit without measuring. Lets a
    // search script rank thousands of candidate operands in a second, then
    // spend real trials only on the best ones.
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

    // Rows are buffered and written in one append at the end, so an
    // interrupted run never leaves partial data in the shared CSV.
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
             << g_toggle_a << "\n";
    }

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
               "toggle_a\n";
    out << rows.str();
    out.flush();
    fprintf(stderr, "appended %d rows to %s\n", TRIALS, out_path);
    return 0;
}