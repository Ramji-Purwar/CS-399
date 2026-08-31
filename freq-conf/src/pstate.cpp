// pstate.cpp
// P-state ladder construction, frequency binning, and classification.

#include "pstate.hpp"
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace pstate {

std::vector<PState> build_ladder() {
    std::vector<PState> ladder;

    // P0 — Turbo bucket (anything above P1)
    ladder.push_back({0, hw::P1_GHZ, "P0 (Turbo)"});

    // P1 — guaranteed all-core base frequency
    ladder.push_back({1, hw::P1_GHZ, "P1 (" + std::to_string((int)(hw::P1_GHZ * 10) / 10) + "." +
                      std::to_string((int)(hw::P1_GHZ * 10) % 10) + " GHz)"});

    // P2..Pn — sub-base states at 100 MHz steps down to P_MIN_GHZ
    int idx = 2;
    for (double f = hw::P1_GHZ - hw::P_STEP_GHZ; f >= hw::P_MIN_GHZ - 1e-6; f -= hw::P_STEP_GHZ) {
        char lbl[32];
        // Round to nearest 0.1 GHz to avoid floating-point drift in label
        double f_rounded = std::round(f * 10.0) / 10.0;
        snprintf(lbl, sizeof(lbl), "P%d (%.1f GHz)", idx, f_rounded);
        ladder.push_back({idx, f_rounded, std::string(lbl)});
        ++idx;
    }

    return ladder;
}

int classify(double freq_ghz, const std::vector<PState>& ladder) {
    if (ladder.empty()) return -1;
    if (freq_ghz <= 0.0) return (int)ladder.size() - 1;

    // P0: above P1 threshold
    if (freq_ghz > hw::P1_GHZ + hw::PSTATE_TOL_GHZ) {
        return 0; // Turbo
    }

    // Search P1..Pn for nearest frequency
    // ladder[0] = P0 (turbo, same freq reference as P1)
    // ladder[1] = P1
    // ladder[2..] = sub-base, decreasing
    int best_idx = 1;
    double best_dist = std::fabs(freq_ghz - ladder[1].freq_ghz);

    for (int i = 2; i < (int)ladder.size(); ++i) {
        double dist = std::fabs(freq_ghz - ladder[i].freq_ghz);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx  = i;
        }
    }

    return best_idx;
}

double min_freq(const std::vector<double>& per_core_ghz) {
    if (per_core_ghz.empty()) return -1.0;
    double m = per_core_ghz[0];
    for (double f : per_core_ghz)
        if (f > 0.0 && f < m) m = f;
    return m;
}

std::string label(int pstate_idx, const std::vector<PState>& ladder) {
    if (pstate_idx < 0 || pstate_idx >= (int)ladder.size())
        return "P? (unknown)";
    return ladder[pstate_idx].label;
}

std::vector<std::pair<int,int>> transitions(const std::vector<PState>& ladder) {
    std::vector<std::pair<int,int>> result;
    for (int i = 0; i + 1 < (int)ladder.size(); ++i) {
        result.push_back({i, i + 1});
    }
    return result;
}

} // namespace pstate
