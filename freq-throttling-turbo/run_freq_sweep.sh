#!/usr/bin/env bash
# =============================================================================
# run_freq_sweep.sh  —  Multi-Kernel Power & Time vs CPU Frequency Sweep (Turbo Mode Enabled)
#
# Hardware: Intel Xeon Gold 5512U (Emerald Rapids / SPR), 28 physical cores
# Driver:   intel_pstate (active), min=800 MHz, base=2100 MHz, max=3700 MHz
#
# Turbo Mode:
#   intel_pstate/no_turbo = 0 (Turbo Boost ENABLED)
#   Governor = performance
#   EPP = performance
#
# Evaluated Kernels:
#   - imul       -> results/frequency_sweep_imul.csv
#   - avx2_add   -> results/frequency_sweep_avx2_add.csv
#   - avx2_mul   -> results/frequency_sweep_avx2_mul.csv
#   - avx2_fma   -> results/frequency_sweep_avx2_fma.csv
#   - avx512_add -> results/frequency_sweep_avx512_add.csv
#   - avx512_mul -> results/frequency_sweep_avx512_mul.csv
#   - avx512_fma -> results/frequency_sweep_avx512_fma.csv
#
# Operands swept:
#   - 0x00    : all-zero bit pattern (minimal ALU toggles)
#   - 0xAA    : alternating bit pattern (maximal ALU toggles)
#
# Usage:
#   sudo ./run_freq_sweep.sh
# =============================================================================

set -euo pipefail

# ── Privilege check ──────────────────────────────────────────────────────────
if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo ./run_freq_sweep.sh)."
    echo "       It needs root privileges to configure /sys/devices/system/cpu/*/cpufreq/."
    exit 1
fi

# ── Paths strictly within freq-throttling-turbo directory ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/benchmark"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# Ensure anyone can read/write results inside workspace
chmod 777 "${RESULTS_DIR}" 2>/dev/null || true

SETTLE_SECONDS=3                   # seconds to wait after changing frequency
COOL_BETWEEN_RUNS=2                # seconds to wait between benchmark runs

# ── Sweep Parameters ─────────────────────────────────────────────────────────
KERNELS=(
    "imul"
    "avx2_add"
    "avx2_mul"
    "avx2_fma"
    "avx512_add"
    "avx512_mul"
    "avx512_fma"
)

OPERANDS=(
    "0x00"
    "0xAA"
)

# Frequencies to sweep (in kHz): Base range (800 MHz - 2100 MHz) + Turbo range (2300 MHz - 3700 MHz)
FREQS_KHZ=(
    800000    #  800 MHz (min freq)
    1000000   # 1000 MHz
    1200000   # 1200 MHz
    1400000   # 1400 MHz
    1600000   # 1600 MHz
    1800000   # 1800 MHz
    2000000   # 2000 MHz
    2100000   # 2100 MHz (base clock)
    2300000   # 2300 MHz (turbo)
    2500000   # 2500 MHz (turbo)
    2700000   # 2700 MHz (turbo)
    2900000   # 2900 MHz (turbo)
    3100000   # 3100 MHz (turbo)
    3300000   # 3300 MHz (turbo)
    3500000   # 3500 MHz (turbo)
    3700000   # 3700 MHz (max turbo)
)

CSV_HEADER="kernel,freq_mhz,operand,trial,mean_time_s,std_time_s,pkg_J,dram_J,cores_J,pkg_W,cores_W"

# ── Restore original system state on exit ────────────────────────────────────
ORIG_NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
ORIG_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
ORIG_EPP=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)
ORIG_MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)
ORIG_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)

cleanup() {
    echo ""
    echo "==> Restoring CPU configuration to default state..."
    echo "${ORIG_MIN}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq > /dev/null
    echo "${ORIG_MAX}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null
    echo "${ORIG_NO_TURBO}" | tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
    echo "${ORIG_GOV}"  | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    echo "${ORIG_EPP}"  | tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference > /dev/null
    
    # Fix ownership and permissions for non-root user
    if [[ -n "${SUDO_USER:-}" ]]; then
        chown -R "${SUDO_USER}:${SUDO_USER}" "${RESULTS_DIR}" 2>/dev/null || true
    fi
    chmod -R a+rw "${RESULTS_DIR}" 2>/dev/null || true
    echo "==> Restored original CPU configuration and permissions."
}
trap cleanup EXIT INT TERM

# ── Compilation check ────────────────────────────────────────────────────────
if [[ ! -x "${BINARY}" ]]; then
    echo "==> Benchmark binary not found. Compiling with g++..."
    g++ -O3 -march=native -mavx2 -mfma -mavx512f -o "${BINARY}" "${SCRIPT_DIR}/benchmark_multicore_spr.cpp" -lpthread
fi

# ── Prepare system for benchmarking (Turbo Mode Enabled) ─────────────────────
echo "==> Configuring CPU for sweep (Turbo Mode Enabled)..."
# Setting no_turbo to 0 enables Turbo frequencies up to 3700 MHz
echo 0 | tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference > /dev/null

echo "==> Turbo enabled (intel_pstate/no_turbo=0). Governor: performance. EPP: performance."

# ── Setup separate CSV for every instruction ──────────────────────────────────
echo "==> Initializing separate CSV for each instruction in: ${RESULTS_DIR}/"
for KERNEL in "${KERNELS[@]}"; do
    KERNEL_CSV="${RESULTS_DIR}/frequency_sweep_${KERNEL}.csv"
    if [[ -f "${KERNEL_CSV}" ]]; then
        BACKUP_CSV="${RESULTS_DIR}/frequency_sweep_${KERNEL}_backup_$(date +%Y%m%d_%H%M%S).csv"
        cp "${KERNEL_CSV}" "${BACKUP_CSV}"
        chmod 666 "${BACKUP_CSV}" 2>/dev/null || true
        echo "  [${KERNEL}] Existing CSV backed up to: $(basename "${BACKUP_CSV}")"
    fi
    echo "${CSV_HEADER}" > "${KERNEL_CSV}"
    chmod 666 "${KERNEL_CSV}" 2>/dev/null || true
    echo "  [${KERNEL}] Ready: frequency_sweep_${KERNEL}.csv"
done

# ── Main sweep loop ───────────────────────────────────────────────────────────
TOTAL_RUNS=$(( ${#FREQS_KHZ[@]} * ${#KERNELS[@]} * ${#OPERANDS[@]} ))
RUN_COUNT=0

echo ""
echo "========================================================"
echo " Starting sweep: ${#FREQS_KHZ[@]} freqs × ${#KERNELS[@]} kernels × ${#OPERANDS[@]} operands = ${TOTAL_RUNS} total runs"
echo "========================================================"

for FREQ in "${FREQS_KHZ[@]}"; do
    FREQ_MHZ=$((FREQ / 1000))

    echo ""
    echo "========================================================"
    echo " [FREQ] Setting CPU Frequency Target to ${FREQ_MHZ} MHz (${FREQ} kHz)"
    echo "========================================================"

    echo "${FREQ}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq > /dev/null
    echo "${FREQ}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null

    echo " Waiting ${SETTLE_SECONDS}s for frequency & thermal settling..."
    sleep "${SETTLE_SECONDS}"

    ACTUAL_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
    echo " Target: ${FREQ} kHz  |  sysfs reports: ${ACTUAL_FREQ} kHz"

    for KERNEL in "${KERNELS[@]}"; do
        KERNEL_CSV="${RESULTS_DIR}/frequency_sweep_${KERNEL}.csv"

        for OP in "${OPERANDS[@]}"; do
            RUN_COUNT=$((RUN_COUNT + 1))

            TMP_CSV="${RESULTS_DIR}/.tmp_bench_${KERNEL}_${OP}_${FREQ_MHZ}.csv"

            echo "  --> [Run ${RUN_COUNT}/${TOTAL_RUNS}] Kernel: ${KERNEL} | Operand: ${OP} @ ${FREQ_MHZ} MHz"
            "${BINARY}" "${OP}" "${FREQ_MHZ}" "${TMP_CSV}" "${KERNEL}"

            # Append trial data rows to this kernel's dedicated CSV
            if [[ -f "${TMP_CSV}" ]]; then
                tail -n +2 "${TMP_CSV}" >> "${KERNEL_CSV}"
                rm -f "${TMP_CSV}"
            fi

            chmod 666 "${KERNEL_CSV}" 2>/dev/null || true

            echo "  --> Completed. (Rows appended to frequency_sweep_${KERNEL}.csv)"
            sleep "${COOL_BETWEEN_RUNS}"
        done
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Sweep complete! Results saved in separate CSVs:"
for KERNEL in "${KERNELS[@]}"; do
    KERNEL_CSV="${RESULTS_DIR}/frequency_sweep_${KERNEL}.csv"
    ROW_COUNT=$(wc -l < "${KERNEL_CSV}")
    echo "  - ${KERNEL_CSV} ($((ROW_COUNT - 1)) rows)"
done
echo "========================================================"
