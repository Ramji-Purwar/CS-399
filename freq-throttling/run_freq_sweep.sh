#!/usr/bin/env bash
# =============================================================================
# run_freq_sweep.sh  —  IMUL power/time vs CPU frequency sweep
#
# Hardware: Intel Xeon Gold 5512U (Emerald Rapids), 28 physical cores, SPR
# Driver:   intel_pstate (active), min=800 MHz, base=2100 MHz, max=3700 MHz
#
# What it does:
#   1. Disables Turbo Boost so the PCU won't override our frequency setting.
#   2. For each target frequency (kHz):
#       a. Locks scaling_min_freq = scaling_max_freq = target to pin the P-state.
#       b. Waits for the hardware to settle.
#       c. Runs the benchmark binary for operand 0x00, then 0xAA.
#       d. Uses a temp file per run; strips its header and appends to master.
#       e. Deletes the temp file immediately after appending.
#   3. Restores the system to its original state via a trap on EXIT/INT/TERM.
#
# Usage:
#   sudo ./run_freq_sweep.sh
#
# Output:
#   results/frequency_sweep_results.csv  — single CSV with ALL data
#   Columns: freq_mhz, operand, trial, mean_time_s, std_time_s,
#            pkg_J, dram_J, cores_J, pkg_W, cores_W
# =============================================================================

set -euo pipefail

# ── Privilege check ──────────────────────────────────────────────────────────
if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo ./run_freq_sweep.sh)."
    echo "       It needs to write to /sys/devices/system/cpu/*/cpufreq/."
    exit 1
fi

# ── Configuration ─────────────────────────────────────────────────────────────
BINARY="./benchmark"               # compiled binary (must be in same dir or PATH)
RESULTS_DIR="./results"
SETTLE_SECONDS=3                   # seconds to wait after changing frequency
COOL_BETWEEN_OPERANDS=2            # seconds between the two operand runs

# Frequencies to test (in kHz).
# Range is 800–2100 MHz (no turbo). Adjust step size as needed.
FREQS_KHZ=(
    800000    #  800 MHz
    1000000   # 1000 MHz
    1200000   # 1200 MHz
    1400000   # 1400 MHz
    1600000   # 1600 MHz
    1800000   # 1800 MHz
    2000000   # 2000 MHz
    2100000   # 2100 MHz  (base clock)
)

OPERANDS=("0x00" "0xAA")

MASTER_CSV="${RESULTS_DIR}/frequency_sweep_results.csv"
MASTER_HEADER="freq_mhz,operand,trial,mean_time_s,std_time_s,pkg_J,dram_J,cores_J,pkg_W,cores_W"

# ── Restore original system state on exit ────────────────────────────────────
# Save current values so cleanup restores exactly what was there before.
ORIG_NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
ORIG_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
ORIG_EPP=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)
ORIG_MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)
ORIG_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)

cleanup() {
    echo ""
    echo "==> Restoring CPU to original state..."
    # Unlock min/max BEFORE restoring turbo/governor so there are no
    # ordering conflicts (e.g. min > max during intermediate states).
    echo "${ORIG_MIN}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq > /dev/null
    echo "${ORIG_MAX}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null
    echo "${ORIG_NO_TURBO}" | tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
    echo "${ORIG_GOV}"  | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    echo "${ORIG_EPP}"  | tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference > /dev/null
    echo "==> Restored: governor=${ORIG_GOV}, no_turbo=${ORIG_NO_TURBO}, min=${ORIG_MIN}, max=${ORIG_MAX}, epp=${ORIG_EPP}"
}
trap cleanup EXIT INT TERM

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Benchmark binary '${BINARY}' not found or not executable."
    echo "       Compile first with:"
    echo "         g++ -O3 -march=native -o benchmark benchmark_multicore_spr.cpp -lpthread"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

# ── Prepare system for benchmarking ───────────────────────────────────────────
echo "==> Configuring CPU for sweep..."
# 1. Disable Turbo so the PCU won't override our target via HWP autonomous mode
echo 1 | tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
# 2. Use 'performance' governor — disables OS-level DVFS
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
# 3. Set EPP to 'performance' — hints HWP to stay at our pinned P-state
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference > /dev/null

echo "==> Turbo disabled. Governor: performance. EPP: performance."

# ── Master CSV ────────────────────────────────────────────────────────────────
echo "${MASTER_HEADER}" > "${MASTER_CSV}"

# ── Main sweep loop ───────────────────────────────────────────────────────────
TOTAL_FREQS=${#FREQS_KHZ[@]}
IDX=0

for FREQ in "${FREQS_KHZ[@]}"; do
    IDX=$((IDX + 1))
    FREQ_MHZ=$((FREQ / 1000))

    echo ""
    echo "========================================================"
    echo "  [${IDX}/${TOTAL_FREQS}] Setting frequency to ${FREQ_MHZ} MHz"
    echo "========================================================"

    # Pin the P-state: min = max = target.
    # With intel_pstate + performance governor this forces HWP to operate
    # at exactly this P-state (both frequency and voltage).
    echo "${FREQ}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq > /dev/null
    echo "${FREQ}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null

    echo "  Waiting ${SETTLE_SECONDS}s for frequency and thermal settling..."
    sleep "${SETTLE_SECONDS}"

    # Informational: sysfs-reported frequency (may lag ~1 s under HWP)
    ACTUAL_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
    echo "  Requested: ${FREQ} kHz  |  sysfs reports: ${ACTUAL_FREQ} kHz"

    for OP in "${OPERANDS[@]}"; do
        TMP_CSV=$(mktemp /tmp/bench_tmp.XXXXXX.csv)

        echo "  --> Running operand=${OP} ..."
        "${BINARY}" "${OP}" "${FREQ_MHZ}" "${TMP_CSV}"

        # Strip the header from this run and append data rows to master CSV
        tail -n +2 "${TMP_CSV}" >> "${MASTER_CSV}"
        rm -f "${TMP_CSV}"

        echo "  --> Done (rows appended to master CSV)."
        sleep "${COOL_BETWEEN_OPERANDS}"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Sweep complete!"
echo "  Master CSV      : ${MASTER_CSV}"
ROW_COUNT=$(wc -l < "${MASTER_CSV}")
echo "  Total data rows : $((ROW_COUNT - 1))  (excluding header)"
echo "========================================================"
