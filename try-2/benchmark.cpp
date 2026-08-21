#include <cstdint>
#include <ctime>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <unistd.h>

using namespace std;

// MSR addresses
constexpr uint32_t MSR_RAPL_POWER_UNIT    = 0x606;  // energy unit
constexpr uint32_t MSR_PP0_ENERGY_STATUS  = 0x639;  // cores only ← this replaces sysfs
constexpr uint32_t MSR_PKG_ENERGY_STATUS  = 0x611;  // full package (for reference)

constexpr int      BENCH_CPU    = 1;
constexpr uint64_t ITERATIONS   = 2800000000ULL;
constexpr int      TRIALS       = 300;
constexpr int      WARMUP_RUNS  = 5;

// energy unit from MSR_RAPL_POWER_UNIT bits 12:8
// actual joules = raw_count × 2^(-energy_unit)
double energy_unit = 0.0;

// -----------------------------------------------------------------------
// MSR helpers
// -----------------------------------------------------------------------

static int msr_fd = -1;

void open_msr(int cpu) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/cpu/%d/msr", cpu);
    msr_fd = open(path, O_RDONLY);
    if (msr_fd < 0) {
        perror("open /dev/cpu/N/msr — run as root and check: modprobe msr");
        exit(EXIT_FAILURE);
    }
}

uint64_t read_msr(uint32_t reg) {
    uint64_t val;
    if (pread(msr_fd, &val, sizeof(val), reg) != sizeof(val)) {
        perror("pread msr");
        exit(EXIT_FAILURE);
    }
    return val;
}

void init_energy_unit() {
    uint64_t unit = read_msr(MSR_RAPL_POWER_UNIT);
    // bits 12:8 = energy status units
    uint32_t eu = (unit >> 8) & 0x1F;
    energy_unit = 1.0 / (1ULL << eu);  // joules per count
}

// PP0 counter is 32-bit and wraps
inline double energy_diff_msr(uint32_t before, uint32_t after) {
    uint32_t diff = after - before;   // wraps correctly in uint32
    return diff * energy_unit;
}

// -----------------------------------------------------------------------
// CPU setup
// -----------------------------------------------------------------------

void pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        exit(EXIT_FAILURE);
    }
}

void enable_realtime() {
    sched_param sp{};
    sp.sched_priority = 99;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
        perror("sched_setscheduler");
    }
}

inline double elapsed_seconds(const timespec &s, const timespec &e) {
    return (e.tv_sec - s.tv_sec) + (e.tv_nsec - s.tv_nsec) * 1e-9;
}

// -----------------------------------------------------------------------
// foo() — 4 independent chains, data is fixed source operand
// -----------------------------------------------------------------------

__attribute__((noinline))
void foo(uint64_t data) {
    uint64_t x0 = 0xDEADBEEFCAFEBABEULL;
    uint64_t x1 = 0x123456789ABCDEFULL;
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

// -----------------------------------------------------------------------
// Single trial
// -----------------------------------------------------------------------

struct Result { double time_s, energy_J, power_W; };

Result run_trial(uint64_t operand) {
    timespec start{}, end{};

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    uint32_t e_before = (uint32_t)read_msr(MSR_PP0_ENERGY_STATUS);

    foo(operand);

    uint32_t e_after = (uint32_t)read_msr(MSR_PP0_ENERGY_STATUS);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double time   = elapsed_seconds(start, end);
    double energy = energy_diff_msr(e_before, e_after);
    double power  = energy / time;

    return {time, energy, power};
}

// -----------------------------------------------------------------------
// Benchmark — interleaved trials
// -----------------------------------------------------------------------

void benchmark() {
    const uint64_t OP_ZERO = 0x0000000000000000ULL;
    const uint64_t OP_AA   = 0xAAAAAAAAAAAAAAAAULL;

    for (int w = 0; w < WARMUP_RUNS; ++w) foo(OP_ZERO);
    for (int w = 0; w < WARMUP_RUNS; ++w) foo(OP_AA);

    cout << "operand,trial,time_s,energy_J,power_W\n";

    for (int trial = 0; trial < TRIALS; ++trial) {
        Result r0 = run_trial(OP_ZERO);
        Result rA = run_trial(OP_AA);

        cout << fixed << setprecision(9)
             << "0x00," << trial << ","
             << r0.time_s << "," << r0.energy_J << "," << r0.power_W << "\n"
             << "0xAA," << trial << ","
             << rA.time_s << "," << rA.energy_J << "," << rA.power_W << "\n";
    }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main() {
    pin_to_core(BENCH_CPU);   // cpu1, sibling cpu11 is offline
    enable_realtime();
    open_msr(BENCH_CPU);      // open /dev/cpu/1/msr
    init_energy_unit();       // read energy unit from MSR_RAPL_POWER_UNIT

    // Print what energy unit we got — sanity check
    fprintf(stderr, "energy_unit = %.10f J/count\n", energy_unit);
    // Expected: ~6.1e-5 J/count (61 microjoules) for Broadwell

    benchmark();

    close(msr_fd);
    return 0;
}