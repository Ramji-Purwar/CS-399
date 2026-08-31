// topology.cpp
// Enumerate physical cores from Linux sysfs topology files.

#include "topology.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace topology {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool read_int_from_file(const std::string& path, int& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    f >> out;
    return !f.fail();
}

// Parse a cpulist string like "0,2,4-7,10" into a sorted vector of ints.
static std::vector<int> parse_cpulist(const std::string& s) {
    std::vector<int> result;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        auto dash = token.find('-');
        if (dash == std::string::npos) {
            result.push_back(std::stoi(token));
        } else {
            int lo = std::stoi(token.substr(0, dash));
            int hi = std::stoi(token.substr(dash + 1));
            for (int i = lo; i <= hi; ++i) result.push_back(i);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

static std::vector<int> get_online_cpus() {
    std::ifstream f("/sys/devices/system/cpu/online");
    if (!f.is_open()) {
        // Fallback: scan /sys/devices/system/cpu/cpuN directories
        std::vector<int> cpus;
        for (int i = 0; i < 512; ++i) {
            std::string p = "/sys/devices/system/cpu/cpu" + std::to_string(i)
                            + "/topology/core_id";
            std::ifstream probe(p);
            if (probe.is_open()) cpus.push_back(i);
        }
        return cpus;
    }
    std::string line;
    std::getline(f, line);
    // trim trailing whitespace/newline
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
        line.pop_back();
    return parse_cpulist(line);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<PhysicalCore> detect_physical_cores() {
    std::vector<int> online = get_online_cpus();
    if (online.empty()) {
        fprintf(stderr, "[topology] ERROR: no online CPUs found\n");
        exit(EXIT_FAILURE);
    }

    // Group logical CPUs by physical core_id
    // Key: physical_package_id * 10000 + core_id (handles multi-socket, though
    // this chip is single socket)
    std::map<int, std::vector<int>> core_map;

    for (int cpu : online) {
        int core_id = -1;
        int package_id = 0;
        std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";

        if (!read_int_from_file(base + "core_id", core_id)) {
            fprintf(stderr, "[topology] WARNING: cannot read core_id for cpu%d, skipping\n", cpu);
            continue;
        }
        read_int_from_file(base + "physical_package_id", package_id);

        int key = package_id * 100000 + core_id;
        core_map[key].push_back(cpu);
    }

    if (core_map.empty()) {
        fprintf(stderr, "[topology] ERROR: could not determine any physical cores\n");
        exit(EXIT_FAILURE);
    }

    std::vector<PhysicalCore> result;
    result.reserve(core_map.size());
    for (auto& [key, cpus] : core_map) {
        std::sort(cpus.begin(), cpus.end());
        int core_id = key % 100000;
        result.push_back({core_id, cpus});
    }

    // Sort by primary CPU for determinism
    std::sort(result.begin(), result.end(),
              [](const PhysicalCore& a, const PhysicalCore& b) {
                  return a.primary_cpu() < b.primary_cpu();
              });

    return result;
}

std::vector<int> pick_n_physical_cores(int n) {
    auto cores = detect_physical_cores();
    if (n > (int)cores.size()) {
        fprintf(stderr, "[topology] ERROR: requested %d cores but only %d physical cores found\n",
                n, (int)cores.size());
        exit(EXIT_FAILURE);
    }
    std::vector<int> cpu_ids;
    cpu_ids.reserve(n);
    for (int i = 0; i < n; ++i)
        cpu_ids.push_back(cores[i].primary_cpu());
    return cpu_ids;
}

void print_topology_info(const std::vector<PhysicalCore>& cores) {
    fprintf(stderr, "[topology] Detected %zu physical cores:\n", cores.size());
    for (const auto& c : cores) {
        fprintf(stderr, "  PhysCore %3d → logical CPUs: ", c.core_id);
        for (int cpu : c.logical_cpus) fprintf(stderr, "%d ", cpu);
        fprintf(stderr, "(pinning to cpu%d)\n", c.primary_cpu());
    }
}

} // namespace topology
