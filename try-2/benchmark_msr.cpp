#include <cstdint>
#include <ctime>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <cstdio>

using namespace std;

// MSR addresses
constexpr uint32_t MSR_RAPL_POWER_UNIT   = 0x606;
constexpr uint32_t MSR_PKG_ENERGY_STATUS = 0x611;  // full package
constexpr uint32_t MSR_DRAM_ENERGY_STATUS= 0x619;  // DRAM only
// PP0 (0x639) is hardwired to 0 on E5-2640 v4 — server Xeon limitation
// cores ≈ PKG − DRAM (removes DRAM noise, still includes uncore)

constexpr int      BENCH_CPU   = 1;       // cpu11 (HT sibling) is offline
constexpr uint64_t ITERATIONS  = 2800000000ULL;
constexpr int      TRIALS      = 300;
constexpr int      WARMUP_RUNS = 5;

double energy_unit = 0.0;   // joules per count, from MSR_RAPL_POWER_UNIT
static int msr_fd  = -1;

// -----------------------------------------------------------------------
// MSR helpers
// -----------------------------------------------------------------------

void open_msr(int cpu) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/cpu/%d/msr", cpu);
    msr_fd = open(path, O_RDONLY);
    if (msr_fd < 0) {
        perror("open /dev/cpu/N/msr — did you run: sudo modprobe msr ?");
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
    energy_unit = 1.0 / (double)(1ULL << eu);
    fprintf(stderr, "MSR_RAPL_POWER_UNIT = 0x%lx\n", unit);
    fprintf(stderr, "energy_unit = %.10f J/count (~%.1f µJ/count)\n",
            energy_unit, energy_unit * 1e6);
}

// Both PKG and DRAM counters are 32-bit and wrap
inline double energy_diff_msr(uint32_t before, uint32_t after) {
    return (uint32_t)(after - before) * energy_unit;
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
// data (0x00 or 0xAA..AA) never changes — it is the source in every IMUL
// x0-x3 are accumulators — their values are not the operand under test
// Total IMULs = (ITERATIONS/4) * 4 = 2.8B
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
// Reads PKG and DRAM counters separately
// cores ≈ PKG − DRAM
// -----------------------------------------------------------------------

struct Result {
    double time_s;
    double pkg_J;       // full package energy
    double dram_J;      // DRAM only energy
    double cores_J;     // PKG - DRAM (approximate cores + uncore)
    double pkg_W;       // package power
    double cores_W;     // approximate core power
};

Result run_trial(uint64_t operand) {
    timespec start{}, end{};

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    uint32_t pkg_before  = (uint32_t)read_msr(MSR_PKG_ENERGY_STATUS);
    uint32_t dram_before = (uint32_t)read_msr(MSR_DRAM_ENERGY_STATUS);

    foo(operand);

    uint32_t pkg_after   = (uint32_t)read_msr(MSR_PKG_ENERGY_STATUS);
    uint32_t dram_after  = (uint32_t)read_msr(MSR_DRAM_ENERGY_STATUS);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double time    = elapsed_seconds(start, end);
    double pkg_J   = energy_diff_msr(pkg_before,  pkg_after);
    double dram_J  = energy_diff_msr(dram_before, dram_after);
    double cores_J = pkg_J - dram_J;   // approximate cores + uncore

    return {time, pkg_J, dram_J, cores_J, pkg_J / time, cores_J / time};
}

// -----------------------------------------------------------------------
// Benchmark — interleaved trials
// -----------------------------------------------------------------------

void benchmark() {
    const uint64_t OP_ZERO = 0x0000000000000000ULL;
    const uint64_t OP_AA   = 0xAAAAAAAAAAAAAAAAULL;

    // Warmup — not measured
    for (int w = 0; w < WARMUP_RUNS; ++w) foo(OP_ZERO);
    for (int w = 0; w < WARMUP_RUNS; ++w) foo(OP_AA);

    cout << "operand,trial,time_s,pkg_J,dram_J,cores_J,pkg_W,cores_W\n";

    for (int trial = 0; trial < TRIALS; ++trial) {
        Result r0 = run_trial(OP_ZERO);
        Result rA = run_trial(OP_AA);

        cout << fixed << setprecision(9)
             << "0x00," << trial << ","
             << r0.time_s  << "," << r0.pkg_J  << "," << r0.dram_J  << ","
             << r0.cores_J << "," << r0.pkg_W  << "," << r0.cores_W << "\n"
             << "0xAA," << trial << ","
             << rA.time_s  << "," << rA.pkg_J  << "," << rA.dram_J  << ","
             << rA.cores_J << "," << rA.pkg_W  << "," << rA.cores_W << "\n";
    }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main() {
    pin_to_core(BENCH_CPU);
    enable_realtime();
    open_msr(BENCH_CPU);
    init_energy_unit();

    benchmark();

    close(msr_fd);
    return 0;
}