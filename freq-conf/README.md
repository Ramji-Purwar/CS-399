# AVX-512 P-State Throttling Characterization

Characterizes CPU frequency throttling behavior under AVX-512 FMA workloads on the **Intel Xeon Gold 5512U** (Emerald Rapids, 28 cores / 56 threads).

**Research question:** At what per-core AVX-512 load (duty cycle) does the CPU transition between hardware P-states, and how does that threshold shift as the number of simultaneously active cores increases?

---

## Quick Start

```bash
# Build everything
make all

# Run component tests (always works, no root needed)
make test

# Dry-run — synthetic data, verifies the full pipeline
./bin/avx512_freq_characterizer --dry-run --cores 1,2,4,8 --cycles 50

# Full experiment (requires msr kernel module)
sudo modprobe msr && sudo chmod +r /dev/cpu/*/msr
./bin/avx512_freq_characterizer --cores 1,2,4,8,14,28 --cycles 500

# Analyze results
python3 analyze_results.py
```

---

## Prerequisites

| Requirement | Purpose | How to check |
|---|---|---|
| `g++ ≥ 11` | C++17 + AVX-512 support | `g++ --version` |
| `msr` kernel module | APERF/MPERF hardware reads | `sudo modprobe msr` |
| Read access to `/dev/cpu/*/msr` | No-root MSR reads | `sudo chmod +r /dev/cpu/*/msr` |
| SST-BF disabled (BIOS) | Prevents static tier bias | `intel-speed-select info` |

If `/dev/cpu/*/msr` is unavailable, the tool automatically falls back to
`/sys/.../scaling_cur_freq` (OS-reported frequency, less accurate). Use `--dry-run`
for development and pipeline validation without any hardware access.

---

## Build

```bash
make all       # builds bin/avx512_freq_characterizer
make test      # builds and runs bin/test_components (23 tests)
make clean     # removes bin/
```

Compiler flags: `-std=c++17 -O3 -march=native -mavx512f -mavx512dq -pthread`

---

## Usage

```
./bin/avx512_freq_characterizer [OPTIONS]

  --help                    Show help
  --dry-run                 Synthetic simulation (no hardware needed)
  --cores N[,N,...]         Active core counts to sweep  [default: 1,2,4,8,14,28]
  --all-cores               Sweep all counts 1..28
  --period-us FLOAT         Duty-cycle period µs         [default: 1000]
  --settle-us FLOAT         Settle phase µs per ON       [default: 300]
  --cycles INT              Measurement cycles/point     [default: 500]
  --coarse-step FLOAT       Coarse sweep step (0–1)     [default: 0.10]
  --resolution FLOAT        Binary search resolution     [default: 0.005]
  --output-csv FILE         Threshold summary CSV        [default: results/thresholds.csv]
  --output-log FILE         Raw measurement log CSV      [default: results/raw_measurements.csv]
  --verbose                 Verbose per-sample output
```

---

## How It Works

### Workload Kernel
Eight independent accumulator chains of **`vfmadd231ps`** (512-bit FMA) running on `zmm0–zmm7` with multipliers in `zmm8–zmm15`. Zero memory access — pure register compute. This saturates both AVX-512 FMA execution ports (port 0 and port 5 on Emerald Rapids) and produces the maximum per-core power draw needed to trigger throttling.

### Duty Cycle
Each 1 ms period is split into:

```
|<────── T_on = D × T ──────>|<─── T_off = (1−D) × T ───>|
|  settle (300 µs, unmeas.)  | measured  |   PAUSE spin   |
```

The settle phase (300 µs) allows the frequency to stabilize after the idle→loaded transition before any measurement begins. The idle phase uses `_mm_pause()` busy-spin (not `sleep`) to keep the core in C0, avoiding deep C-state wake-up latency.

### Frequency Measurement
APERF/MPERF MSR deltas, accumulated across all measured sub-intervals:

```
eff_freq = (ΣΔAPERF / ΣΔMPERF) × TSC_freq
```

TSC frequency is empirically calibrated against `CLOCK_MONOTONIC_RAW` at startup.

Falls back to `/sys/.../scaling_cur_freq` when MSR device is unavailable.

### Multi-Core Coordination
N worker threads are pinned to N *distinct physical cores* (no HT sibling colocation) using `sched_setaffinity`. A pthread barrier ensures all threads begin the duty-cycle engine simultaneously. The **minimum** effective frequency across all active cores is used for P-state classification.

### Search Strategy
1. **Coarse sweep** (10% steps): bracket the transition interval `[D_low, D_high]`
2. **Binary search** (0.5% resolution): narrow to the precise threshold

---

## Output Files

| File | Contents |
|---|---|
| `results/thresholds.csv` | Summary: `active_cores, from_pstate, to_pstate, transition, threshold_pct, found` |
| `results/raw_measurements.csv` | Every sample: `timestamp_s, active_cores, core_id, duty_cycle, delta_aperf, delta_mperf, eff_freq_ghz, pstate_index` |
| `results/plots/transition_curves.svg` | Threshold vs core count curves per transition |
| `results/plots/raw_freq_scatter.svg` | Per-core frequency scatter vs duty cycle |

---

## P-State Ladder (Xeon Gold 5512U)

| State | Frequency | Note |
|---|---|---|
| P0 | > 2.1 GHz | Turbo (ceiling measured empirically per core count) |
| P1 | 2.1 GHz | Guaranteed all-core sustained base |
| P2 | 2.0 GHz | Sub-base, 100 MHz steps |
| … | … | … |
| P14 | 0.8 GHz | Assumed minimum (verify with `turbostat` at idle) |

---

## Source Layout

```
freq-conf/
├── Makefile
├── README.md
├── analyze_results.py         Post-processing and plots
├── include/
│   ├── common.hpp             Shared structs, hardware constants, config
│   ├── topology.hpp           Physical core enumeration (no HT siblings)
│   ├── timer.hpp              TSC calibration, spin-wait
│   ├── msr_reader.hpp         APERF/MPERF MSR interface + fallbacks
│   ├── workload_kernel.hpp    8-chain AVX-512 FMA kernel
│   ├── duty_cycle_engine.hpp  On/settle/idle timing engine
│   ├── pstate.hpp             P-state ladder, classification, binning
│   └── experiment_runner.hpp  Coarse sweep + binary search + multi-core
├── src/
│   ├── topology.cpp
│   ├── timer.cpp
│   ├── msr_reader.cpp
│   ├── workload_kernel.cpp
│   ├── duty_cycle_engine.cpp
│   ├── pstate.cpp
│   ├── experiment_runner.cpp
│   └── main.cpp               CLI entry point
├── tests/
│   └── test_components.cpp    23 unit/integration tests
└── results/                   (created at runtime)
    ├── thresholds.csv
    ├── raw_measurements.csv
    └── plots/
```

---

## Pre-Run Checklist

- [ ] `make test` passes all 23 tests
- [ ] Confirm AVX-512 is available: `grep avx512 /proc/cpuinfo | head -1`
- [ ] Disable SST-BF in BIOS or via `intel-speed-select`
- [ ] Load MSR module: `sudo modprobe msr`
- [ ] Grant read access: `sudo chmod +r /dev/cpu/*/msr`
- [ ] Optionally verify minimum P-state: `sudo turbostat --show Core,Pkg_MHz --interval 1` while system is idle
- [ ] Set CPU governor to `performance` for cleaner results: `cpupower frequency-set -g performance`
