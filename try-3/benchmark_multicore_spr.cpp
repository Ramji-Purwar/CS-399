#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <vector>

using namespace std;

// -----------------------------------------------------------------------
// Config — Gold 5512U: 28 physical cores (0-27), HT siblings (28-55) offline
// -----------------------------------------------------------------------

constexpr int      NUM_CORES    = 28;          // cores 0..27
constexpr uint64_t ITERATIONS   = 2800000000ULL;
constexpr int      TRIALS       = 300;
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
    if (!f) {
        perror(path);
        exit(EXIT_FAILURE);
    }
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
    if (!f) return 0; // not fatal, just disables wrap correction
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

__attribute__((noinline))
void foo(uint64_t data) {
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

struct ThreadArg {
    int core_id;
    uint64_t operand;
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
    foo(arg->operand);
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
    pthread_barrier_init(&g_end_barrier, nullptr, NUM_CORES + 1);

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

void benchmark() {
    const uint64_t OP_ZERO = 0x0000000000000000ULL;
    const uint64_t OP_AA   = 0xAAAAAAAAAAAAAAAAULL;

    fprintf(stderr, "Running on %d cores simultaneously\n", NUM_CORES);
    fprintf(stderr, "Warming up 0x00...\n");
    for (int w = 0; w < WARMUP_RUNS; ++w) run_trial(OP_ZERO);
    fprintf(stderr, "Warming up 0xAA...\n");
    for (int w = 0; w < WARMUP_RUNS; ++w) run_trial(OP_AA);

    cout << "operand,trial,mean_time_s,std_time_s,pkg_J,dram_J,cores_J,pkg_W,cores_W\n";
    fprintf(stderr, "Starting benchmark (%d trials)...\n", TRIALS);

    for (int trial = 0; trial < TRIALS; ++trial) {
        if (trial % 10 == 0) fprintf(stderr, "trial %d/%d\n", trial, TRIALS);

        Result r0 = run_trial(OP_ZERO);
        Result rA = run_trial(OP_AA);

        cout << fixed << setprecision(9)
             << "0x00," << trial << "," << r0.mean_time_s << "," << r0.std_time_s << ","
             << r0.pkg_J << "," << r0.dram_J << "," << r0.cores_J << ","
             << r0.pkg_W << "," << r0.cores_W << "\n"
             << "0xAA," << trial << "," << rA.mean_time_s << "," << rA.std_time_s << ","
             << rA.pkg_J << "," << rA.dram_J << "," << rA.cores_J << ","
             << rA.pkg_W << "," << rA.cores_W << "\n";
        cout.flush();
    }
}

int main() {
    read_energy_uj(RAPL_PKG_FILE);
    read_energy_uj(RAPL_DRAM_FILE);

    fprintf(stderr, "Expected SNR improvement: ~%dx vs single core\n", NUM_CORES);
    benchmark();
    return 0;
}