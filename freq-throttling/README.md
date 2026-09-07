# CPU Frequency Scaling & Data-Dependent Power Characterization

## 1. Overview
This experiment characterizes the relationship between **CPU Operating Frequency (DVFS)**, **Execution Time**, and **Core Power Consumption** across different microprocessor execution units on modern Intel Xeon hardware.

Additionally, this testbed evaluates **data-dependent power leakage** by comparing the power signature of operands with minimal bit-switching activity (`0x0000000000000000`) against operands with maximal alternating bit-switching activity (`0xAAAAAAAAAAAAAAAB`).

---

## 2. Hardware Testbed

* **Processor:** Intel(R) Xeon(R) Gold 5512U (Emerald Rapids / Sapphire Rapids family)
* **Cores:** 28 Physical Cores (HT sibling threads 28–55 disabled/offline)
* **Base Frequency:** 2.1 GHz (2100 MHz)
* **Min Frequency:** 800 MHz (800000 kHz)
* **Max Turbo:** 3.7 GHz (Turbo Boost explicitly disabled during sweep)
* **Power Measurement:** Hardware Intel Running Average Power Limit (RAPL) sysfs interface:
  * Package Energy: `/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj`
  * DRAM Energy: `/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj`
  * Compute Cores Energy: $E_{\text{cores}} = E_{\text{pkg}} - E_{\text{dram}}$

---

## 3. Evaluated Instructions & Latency Characteristics

All benchmark kernels are strictly selected to exhibit **fixed execution latency** (independent of data values) to prevent microarchitectural stalls or variable execution times:

| Kernel | Instruction Type | Vector Width | Fixed Latency | Execution Ports / Chains |
|---|---|---|---|---|
| **`addq`** | 64-bit Integer Addition | 64-bit | **1 cycle** | 4 independent chains |
| **`imulq`** | 64-bit Integer Multiplication | 64-bit | **3 cycles** | 4 independent chains |
| **`mulps`** | SSE Packed Single-Precision Float Mul | 128-bit (4 floats) | **4 cycles** | 8 independent chains |
| **`avx512`** | AVX-512 Packed Single-Precision Float Mul | 512-bit (16 floats) | **4 cycles** | 8 independent chains |

> **Note on AVX-512 Frequency Throttling:** Executing 512-bit wide vector instructions engages Intel PCU dynamic power budgeting, which may cause hardware-level downclocking below the requested OS frequency limit.

---

## 4. Mathematical Non-Convergence Guarantee

In data-dependent power analysis, execution units must continuously toggle bits throughout the entire benchmark (2.8 billion operations). Naive arithmetic loops can quickly converge to static states, freezing registers and destroying dynamic power signals.

### Integer Arithmetic (`addq` & `imulq`)
* **`0x00`**: Neutral operand (`x + 0` or `x * 0`). Mantissa/bits remain zero with minimal dynamic toggling.
* **`0xAA`**: The operand `0xAAAAAAAAAAAAAAABULL` is odd (coprime to $2^{64}$):
  * In `addq`: Adding an odd constant modulo $2^{64}$ produces a maximal cyclic group of length $2^{64}$.
  * In `imulq`: Multiplicative order modulo $2^{64}$ is $2^{62}$.
  * **Result:** Values continuously pseudo-randomize across all 64 bits with high Hamming distance on every single cycle without ever collapsing.

### Floating-Point Arithmetic (`mulps` & `avx512`)
Repeatedly multiplying a float by a constant $> 1.0$ (e.g. $1.333f$) causes exponential growth that overflows to `+Inf` in ~320 iterations. In IEEE-754, `+Inf` has an all-zero mantissa, which would freeze the registers for 99.999% of the run.

To solve this, the kernels use an **alternating reciprocal multiplier pair**:
* **`0x00` Operand:** Multiplies by $1.0f$ (`0x3F800000`, all-zero mantissa). Stays in a stable normal range with zero dynamic bit flips.
* **`0xAA` Operand:**
  * **Step 1:** Multiplies by $A = 1.3333334f$ (`0x3FAAAAAB`, alternating mantissa bits: `10101010101010101010101`).
  * **Step 2:** Multiplies by $B = 0.75f$ (`0x3F400000`, exact float reciprocal $\frac{3}{4}$).
  * **Result:** Values dynamically oscillate between $1.234567f$ (`0x3F9E064B`) and $1.646089f$ (`0x3FD2B30F`). It flips 10+ bits per lane on every instruction, never overflows to $\infty$, never underflows to $0$, and never encounters subnormals or NaNs.

---

## 5. File Structure

```text
freq-throttling/
├── benchmark_multicore_spr.cpp  # Multithreaded C++ benchmark with RAPL sampling
├── run_freq_sweep.sh            # Automated sweep harness (frequency, kernel, operand)
├── plot_time_vs_power.py        # Plots execution time (X) vs power (Y)
├── results/                     # Output directory for CSV sweep results
│   ├── frequency_sweep_addq.csv
│   ├── frequency_sweep_imulq.csv
│   ├── frequency_sweep_mulps.csv
│   └── frequency_sweep_avx512.csv
├── run-1/                       # Archive of initial frequency sweep run
└── run-2/                       # Archive of multi-kernel sweep run & plots
```

---

## 6. How to Run

### Step 1: Compile the Benchmark Binary
```bash
g++ -O3 -march=native -mavx512f -o benchmark benchmark_multicore_spr.cpp -lpthread
```

### Step 2: Run the Automated Frequency Sweep
The sweep script pins frequencies (800 MHz – 2100 MHz), disables Turbo Boost, runs warmups and trials across all 28 physical cores, and streams data into dedicated CSV files per instruction:
```bash
sudo ./run_freq_sweep.sh
```

### Step 3: Generate Plots
To generate Time vs Power distribution plots for all instructions:
```bash
python3 plot_time_vs_power.py
```
This automatically produces individual visual plots in the working directory:
* `time_vs_power_addq.png`
* `time_vs_power_imulq.png`
* `time_vs_power_mulps.png`
* `time_vs_power_avx512.png`

---

## 7. CSV Data Schema

Each row in `frequency_sweep_<kernel>.csv` corresponds to a single completed multi-core trial:

| Column | Description | Units |
|---|---|---|
| `kernel` | Instruction under test (`addq`, `imulq`, `mulps`, `avx512`) | string |
| `freq_mhz` | Pinned hardware CPU frequency | MHz |
| `operand` | Test data pattern (`0x00` or `0xAA`) | hex string |
| `trial` | Trial iteration index (0–99) | integer |
| `mean_time_s` | Mean execution wall-clock time across 28 cores | seconds |
| `std_time_s` | Standard deviation of thread execution times | seconds |
| `pkg_J` | Total energy consumed by CPU package during trial | Joules |
| `dram_J` | Total energy consumed by DRAM during trial | Joules |
| `cores_J` | Net energy consumed by compute cores (`pkg_J - dram_J`) | Joules |
| `pkg_W` | Average package power draw (`pkg_J / mean_time_s`) | Watts |
| `cores_W` | Average compute core power draw (`cores_J / mean_time_s`) | Watts |
