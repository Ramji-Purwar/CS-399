#include "experiment_runner.hpp"
#include "burst_engine.hpp"
#include "timer.hpp"
#include "topology.hpp"
#include "common.hpp"

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>

namespace experiment {

struct WorkerArg {
    int core_id;
    int active_cores;
    double burst_billions;
    const Config* config;
    uint64_t tsc_hz;
    
    msr::CoreReader reader;
    pthread_barrier_t* barrier_start;
    pthread_barrier_t* barrier_done;

    double result_eff_freq;
};

static void* worker_thread(void* arg_void) {
    WorkerArg* arg = (WorkerArg*)arg_void;

    // Pin to core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(arg->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // Give kernel time to migrate thread
    usleep(1000);

    msr::CoreReader reader = arg->reader;
    if (reader.mode == msr::Mode::DIRECT && reader.msr_fd < 0) {
        char path[64];
        snprintf(path, sizeof(path), "/dev/cpu/%d/msr", arg->core_id);
        int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd >= 0) reader.msr_fd = fd;
        else reader.mode = msr::Mode::SYSFS;
    }

    burst::RunConfig rcfg;
    rcfg.cooldown_ms = arg->config->cooldown_ms;
    rcfg.dry_run     = arg->config->dry_run;

    // Sync start
    pthread_barrier_wait(arg->barrier_start);

    // Run burst
    burst::RunResult r = burst::run(reader, arg->burst_billions, rcfg, arg->tsc_hz);
    arg->result_eff_freq = r.eff_freq_ghz;

    // Sync end
    pthread_barrier_wait(arg->barrier_done);

    if (reader.msr_fd >= 0) close(reader.msr_fd);
    return nullptr;
}

double evaluate_sample(
    int active_cores,
    double burst_billions,
    const Config& config,
    const std::vector<msr::CoreReader>& readers,
    uint64_t tsc_hz)
{
    pthread_barrier_t barrier_start;
    pthread_barrier_t barrier_done;
    pthread_barrier_init(&barrier_start, nullptr, active_cores + 1);
    pthread_barrier_init(&barrier_done, nullptr, active_cores + 1);

    std::vector<pthread_t> threads(active_cores);
    std::vector<WorkerArg> args(active_cores);

    for (int i = 0; i < active_cores; ++i) {
        args[i].core_id = readers[i].cpu_id;
        args[i].active_cores = active_cores;
        args[i].burst_billions = burst_billions;
        args[i].config = &config;
        args[i].tsc_hz = tsc_hz;
        args[i].reader = readers[i];
        args[i].barrier_start = &barrier_start;
        args[i].barrier_done = &barrier_done;

        pthread_create(&threads[i], nullptr, worker_thread, &args[i]);
    }

    // Release workers
    pthread_barrier_wait(&barrier_start);

    // Wait for completion
    pthread_barrier_wait(&barrier_done);

    double sum_f = 0.0;
    for (int i = 0; i < active_cores; ++i) {
        pthread_join(threads[i], nullptr);
        sum_f += args[i].result_eff_freq;
    }

    pthread_barrier_destroy(&barrier_start);
    pthread_barrier_destroy(&barrier_done);

    return sum_f / active_cores;
}

void run_sweep(
    const Config& config,
    const std::string& out_csv,
    ProgressCb progress_cb)
{
    // Make sure directory exists
    auto slash = out_csv.rfind('/');
    if (slash != std::string::npos) {
        std::string dir = out_csv.substr(0, slash);
        mkdir(dir.c_str(), 0755);
    }

    std::ofstream f(out_csv);
    if (f.is_open()) {
        f << "active_cores,burst_billions,freq_ghz\n";
    }

    for (int n_cores : config.core_counts) {
        if (progress_cb) progress_cb(n_cores, -1.0, 0.0); // signal new core block

        std::vector<msr::CoreReader> readers;
        auto phys_cores = topology::pick_n_physical_cores(n_cores);
        for (int cid : phys_cores) {
            msr::CoreReader r;
            r.cpu_id = cid;
            r.msr_fd = -1;
            r.mode = config.dry_run ? msr::Mode::DRYRUN : msr::Mode::DIRECT;
            readers.push_back(r);
        }

        // Linear sweep
        for (double b = 0.0; b <= config.max_burst_billions + 1e-9; b += config.coarse_step) {
            double freq = evaluate_sample(n_cores, b, config, readers, timer::g_tsc_hz);
            
            if (f.is_open()) {
                f << n_cores << "," << b << "," << freq << "\n";
                f.flush();
            }

            if (progress_cb) progress_cb(n_cores, b, freq);
        }
    }
}

} // namespace experiment
