// topology.hpp
// CPU physical core enumeration for the AVX-512 P-State experiment.
//
// Purpose: identify the set of N *distinct physical cores* (not HT siblings)
// for the multi-core sweep, and provide logical CPU IDs for thread pinning.
//
// On Linux, each logical CPU's physical core ID is at:
//   /sys/devices/system/cpu/cpuN/topology/core_id
// Its HT sibling list is at:
//   /sys/devices/system/cpu/cpuN/topology/thread_siblings_list
//
// Strategy: enumerate all online logical CPUs, group them by core_id,
// and take the lowest-numbered logical CPU from each unique physical core.

#pragma once

#include <vector>
#include <string>

namespace topology {

// One physical core: its physical core_id and the logical CPUs that share it.
struct PhysicalCore {
    int core_id;                        // /sys topology core_id (physical)
    std::vector<int> logical_cpus;      // all logical CPU ids in this core (including HT siblings)
    int primary_cpu() const { return logical_cpus[0]; } // lowest-numbered, used for pinning
};

// Detect all online physical cores and return them sorted by core_id.
// Dies with an error message if no cores can be detected.
std::vector<PhysicalCore> detect_physical_cores();

// Return the N representative logical CPU ids to pin N threads to.
// Selects the first N entries from detect_physical_cores().
// Aborts if N > available physical cores.
std::vector<int> pick_n_physical_cores(int n);

// Print topology info to stderr (for verbose/diagnostic mode).
void print_topology_info(const std::vector<PhysicalCore>& cores);

} // namespace topology
