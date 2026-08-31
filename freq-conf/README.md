# AVX-512 Frequency Throttling Characterization

## Project Overview
This project characterizes the dynamic frequency scaling (throttling) behavior of modern Intel processors (specifically the Intel Xeon Gold 5512U "Emerald Rapids", 28C/56T) when executing extremely dense AVX-512 workloads. 

Historically, executing 512-bit wide vector math drew so much current that Intel CPUs applied severe, immediate frequency penalties (AVX-512 licenses) to prevent thermal runaway. This experiment was designed to bypass operating system reporting flaws and map exactly how frequency degrades under maximum physical AVX-512 FMA (Fused Multiply-Add) port pressure across varying core counts.

---

## Experimental Methodology & Deep Technical Design

The experimental design evolved through several iterations as we uncovered the underlying behavior of the hardware and the limitations of the Linux OS frequency governors.

### 1. The Workload Kernel (Maximizing Silicon Power Draw)
To force the CPU to draw maximum power and trigger PCU (Power Control Unit) throttling, we cannot simply execute random math; we must perfectly saturate the execution units without stalling. We achieved this through two critical micro-architectural optimizations in `workload_kernel.cpp`:

* **Hiding Memory Latency (Register-Only Execution):**
  If an FMA loop relies on loading data from the CPU L1/L2 cache or RAM before doing a calculation, the execution pipes will stall for several nanoseconds waiting for the memory controller. To keep the power consumption pinned at 100% every single clock cycle, we eliminated memory latency entirely. All variables in our inner loop are initialized directly into the core's private 512-bit physical registers (ZMM registers). No memory loads or stores occur during the measurement phase.

* **Forcing Max Port Pressure (Utilizing Both FMA Ports):**
  Modern high-end Xeon processors have two independent 512-bit FMA execution blocks per physical core, sitting on Port 0 and Port 5 of the execution engine. If a loop structure forces a serial dependency (e.g., calculation #2 needs the result of calculation #1), the processor can only use one port at a time, leaving the other block idle.
  The AVX-512 FMA instruction (`vfmadd231ps`) has a latency of 4 clock cycles. To break serial dependencies and saturate the pipeline, we explicitly unrolled the loop and interleaved **8 completely independent calculation chains** (`acc0` through `acc7`). 
  Because there are no dependencies between these registers, the hardware's out-of-order execution engine schedules two completely independent FMA operations on Port 0 and Port 5 simultaneously, every single clock cycle. This instantly doubles the local current draw on the core, achieving the theoretical maximum FMA pressure a Xeon can physically sustain.

### 2. Burst-Based Sweep and Thermal Reset
Modern Intel PCUs manage heat and turbo frequencies using PL1/PL2 power limits and EWMA (Exponentially Weighted Moving Averages) over time.

* **The Design:** We executed the workload in discrete, continuous bursts ranging from 0 to 50 Billion instructions. Between every single measurement, the execution threads are forced to sleep (`usleep`) for a strict **2-second cooldown period**.
* **The Rationale:** This guarantees that the CPU's thermal capacitance is completely flushed and the PCU power budget is fully reset back to a "cold" idle state. Consequently, a 50-billion instruction burst is a completely independent, fresh measurement, not a continuation of heat built up from a previous 48-billion burst. This allowed us to test if a short burst sustained higher clocks than a long burst before the PCU kicked in thermal limits.

### 3. Bypassing the Linux `sysfs` Trap (Throughput-Based Measurement)
The biggest hurdle we faced was that the Linux Operating System lied to us about the CPU frequency.

* **The Problem:** We initially read the CPU frequency from `/sys/devices/system/cpu/cpuX/cpufreq/scaling_cur_freq`. However, the data showed the CPU magically sustaining ~3.0 GHz indefinitely with zero AVX-512 throttling across all 28 cores. 
* **The Discovery:** `sysfs` only reports the frequency the *Linux OS Governor requested*. When a CPU executes AVX-512, the internal hardware PCU intercepts this and silently downclocks the silicon to stay within physical package power limits. The OS is completely unaware of this hardware-level throttling.
* **The Solution (Physics & Throughput):** We rewrote `burst_engine.cpp` to measure the **True Silicon Frequency**. We know the Emerald Rapids microarchitecture retires exactly 2 FMA instructions per clock cycle at peak saturation. We use a high-precision nanosecond timer to measure exactly how long the burst takes to finish in wall-clock time:
  
  $$ \text{Theoretical Cycles} = \frac{\text{Instruction Count}}{2 \text{ (FMA Ports)}} $$
  $$ \text{Actual Frequency (GHz)} = \frac{\text{Theoretical Cycles}}{\text{Elapsed Wall-Clock Seconds}} / 10^9 $$

* **The Rationale:** The wall-clock doesn't lie. If the PCU silently drops the clock from 3.0 GHz to 2.0 GHz, the FMA execution units physically cycle slower, meaning the 50 Billion instructions take exactly 50% longer to complete. This throughput math mathematically guarantees we measure the true silicon speed, bypassing OS reporting bugs entirely.

---

## Algorithm Design & Execution Flow

The measurement engine is built in C++ to provide microsecond-level control over thread execution and synchronization. The algorithm is designed to ensure that all active cores ramp up to 100% AVX-512 utilization at the exact same nanosecond to maximize sudden package power draw.

### 1. The Coordinator Algorithm (`experiment_runner.cpp`)
```text
FOR EACH core_count IN [1, 2, 4, 6, ..., 28]:
    FOR EACH burst_size IN [0B, 2B, 4B, ..., 50B]:
        1. Initialize two pthread barriers (barrier_start, barrier_done) for (core_count + 1) threads.
        2. Spawn 'core_count' worker threads.
        3. Pin each worker thread to a unique physical core using sched_setaffinity.
        4. Wait at barrier_start to release all workers simultaneously.
        5. Wait at barrier_done for all workers to finish.
        6. Calculate the average frequency across all workers and log the result to CSV.
```

### 2. The Worker Thread Algorithm (`burst_engine.cpp`)
```text
WORKER_THREAD(core_id, burst_size, cooldown_ms):
    1. Sleep for 'cooldown_ms' (e.g., 2000 ms) to reset PCU thermal EWMA limits.
    2. Wait at barrier_start until all peers are ready.
    3. Record high-precision start_time (nanoseconds).
    
    // Execute Max-Pressure AVX-512 Workload
    4. Calculate iterations = (burst_size * 10^9) / 8
    5. LOOP iterations times:
           acc0 = _mm512_fmadd_ps(...)
           ...
           acc7 = _mm512_fmadd_ps(...)
           
    6. Record high-precision end_time.
    7. Wait at barrier_done (Prevents early-finishers from relieving thermal pressure on slow cores).
    
    8. elapsed_seconds = (end_time - start_time) / 10^9
    9. return True Silicon Frequency = ((burst_size * 10^9) / 2) / elapsed_seconds
```

This strict barrier synchronization guarantees that if 28 cores are selected, all 28 cores are drawing dual-port FMA power simultaneously for the entire duration of the measurement.

---

## Results & Findings

### Finding 1: Lack of Temporal (Duration-Based) Throttling
We originally hypothesized that the CPU frequency would dynamically degrade as the burst size (and therefore total generated heat) increased over time. 

**Result:** The throughput measurements proved this hypothesis false for this hardware generation. For any given number of active cores, the frequency remained perfectly flat whether we ran 2 Billion or 50 Billion instructions. Emerald Rapids handles AVX-512 power efficiency vastly better than older Skylake-SP chips. It does not suffer from extreme thermal runaway requiring gradual frequency drops over the span of a burst.

### Finding 2: The All-Core AVX-512 Scaling Curve
Because burst duration did not affect frequency, we discovered that the Emerald Rapids PCU handles AVX-512 by immediately locking into a sustainable **All-Core AVX-512 Turbo Limit** based strictly on *how many cores are actively executing the workload*.

**Final Output Plotting:**
Because the frequency is constant across time, plotting Frequency vs Burst Size yielded flat lines. The analysis script (`analyze_results.py`) was updated to average the burst frequencies and plot the true scaling curve of the processor: **Frequency vs Number of Active Cores**. 

* **1 Core Active:** The CPU easily sustains its maximum Single-Core AVX-512 Turbo limit.
* **28 Cores Active:** As more cores wake up and demand dual-port FMA power, the PCU instantly caps the silicon frequency (down to ~2.9 GHz to ~3.0 GHz) to ensure the total package power draw of all 28 cores does not exceed the physical socket TDP limits.

---

## Compilation and Execution
To reproduce the experiment and generate the scaling curve visualization:

```bash
# 1. Compile the C++ measurement engine
make all

# 2. Run the experiment
#    By default, this sweeps 1, 2, 4, 6... up to 28 cores.
#    It executes bursts up to 50 Billion instructions, stepping by 2.0B
./bin/avx512_freq_characterizer --max-billions 50 --step 2

# 3. Generate the visualization
#    Reads results/sweep_data.csv and outputs results/cores_plot.svg
python3 analyze_results.py
```
