#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <unistd.h>

using namespace std;

constexpr const char *RAPL_FILE =
    "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj";

constexpr const char *RAPL_MAX =
    "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj";

constexpr uint64_t ITERATIONS = 2800000000ULL;
constexpr int TRIALS = 100;

uint64_t max_energy_range = 0;

uint64_t read_energy() {
    ifstream fin(RAPL_FILE);
    uint64_t energy;
    fin >> energy;
    return energy;
}

void init_energy_range() {
    ifstream fin(RAPL_MAX);
    fin >> max_energy_range;
}

inline uint64_t energy_diff(uint64_t before, uint64_t after) {
    if (after >= before)
        return after - before;
    return (max_energy_range - before) + after;
}

void pin_to_core(int core = 0) {
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

inline double elapsed_seconds(const timespec &start,
                              const timespec &end) {
    return (end.tv_sec - start.tv_sec) +
           (end.tv_nsec - start.tv_nsec) * 1e-9;
}

__attribute__((noinline))
void foo(uint64_t data) {

    register uint64_t x = data;
    constexpr uint64_t FIXED = 0x123456789ABCDEFULL;

    for (uint64_t i = 0; i < ITERATIONS; ++i) {
        asm volatile(
            "imulq %[fixed], %[val]"
            : [val] "+r"(x)
            : [fixed] "r"(FIXED)
            : "cc");
    }

    asm volatile("" : : "r"(x));
}

void benchmark(uint64_t operand, const string &label) {

    // Warmup (discard result)
    foo(operand);

    for (int trial = 0; trial < TRIALS; ++trial) {

        uint64_t energy_before = read_energy();

        timespec start{}, end{};
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);

        foo(operand);

        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        uint64_t energy_after = read_energy();

        double time =
            elapsed_seconds(start, end);

        double energy =
            energy_diff(energy_before, energy_after) / 1e6;

        double power = energy / time;

        cout << label << ","
             << trial << ","
             << fixed << setprecision(9)
             << time << ","
             << energy << ","
             << power << '\n';
    }
}

int main() {

    pin_to_core(0);

    enable_realtime();

    init_energy_range();

    cout << "operand,trial,time_s,energy_J,power_W\n";

    benchmark(0x0000000000000000ULL, "0x00");
    benchmark(0xAAAAAAAAAAAAAAAAULL, "0xAA");

    return 0;
}