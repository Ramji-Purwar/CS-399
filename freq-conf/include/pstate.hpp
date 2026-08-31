// pstate.hpp
// P-state table, binning, and classification logic.
//
// The P-state ladder for Xeon Gold 5512U (Emerald Rapids):
//
//   P0  | > P1_GHZ (2.1 GHz) | Turbo — upper bound measured empirically
//   P1  | 2.1 GHz             | Guaranteed all-core sustained frequency
//   P2  | 2.0 GHz             | Sub-base, 100 MHz steps
//   P3  | 1.9 GHz             |
//   ... | ...                 |
//   P14 | 0.8 GHz             | Assumed minimum (verify vs hardware)
//
// Classification:
//   Given a measured effective frequency `f`:
//   - If f > P1 − tolerance   → P0 (turbo)
//   - If |f − Pk| < tolerance → Pk (for k = 1..14)
//   - Otherwise               → nearest Pk
//
// The minimum across all active cores is used for multi-core classification,
// because the most-throttled core defines whether the system crossed a boundary.

#pragma once

#include "common.hpp"
#include <vector>
#include <string>

namespace pstate {

// Build the P-state ladder for this SKU.
// Returns states from P0 (turbo) through Pmax (0.8 GHz).
std::vector<PState> build_ladder();

// Classify a measured effective frequency into a P-state index.
// Returns 0 for turbo (P0), 1 for P1 base, 2..N for sub-base.
// `ladder` should come from build_ladder().
int classify(double freq_ghz, const std::vector<PState>& ladder);

// Return the minimum frequency across a set of per-core measurements.
double min_freq(const std::vector<double>& per_core_ghz);

// Human-readable label for a P-state index.
std::string label(int pstate_idx, const std::vector<PState>& ladder);

// Transitions to characterize: returns (from, to) pairs representing
// each downward transition in the ladder (P0→P1, P1→P2, ...).
std::vector<std::pair<int,int>> transitions(const std::vector<PState>& ladder);

} // namespace pstate
