#!/usr/bin/env bash
# =============================================================================
# freq_binary_search.sh — Find the frequency at which 0xAA (avx512_fma)
# draws the SAME core power as 0x00 does at a fixed reference frequency.
#
# Idea:
#   1. Pin CPU to REF_FREQ_MHZ, run operand 0x00, record its mean cores_W
#      as the TARGET power.
#   2. Binary search over frequency in [MIN_FREQ_MHZ, REF_FREQ_MHZ] for the
#      frequency at which operand 0xAA's mean cores_W matches TARGET,
#      within TOLERANCE_W.
#
#   Frequency can only be changed BETWEEN runs (via sysfs), never mid-run —
#   each candidate frequency gets its own full benchmark invocation.
#
# Usage:
#   sudo ./freq_binary_search.sh [ref_freq_mhz] [min_freq_mhz]
#
# Requires: benchmark_avx512_fma (built from benchmark_avx512_fma.cpp)
# =============================================================================

set -euo pipefail

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: must be run as root (sudo ./freq_binary_search.sh)."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/benchmark_avx512_fma"
RESULTS_DIR="${SCRIPT_DIR}/results_binsearch"
mkdir -p "${RESULTS_DIR}"
chmod 777 "${RESULTS_DIR}" 2>/dev/null || true

# ── Parameters ────────────────────────────────────────────────────────────
REF_FREQ_MHZ="${1:-2100}"     # frequency at which 0x00 is measured (target)
MIN_FREQ_MHZ="${2:-800}"      # lower bound of the search for 0xAA
STEP_MHZ=10                   # snap all candidate freqs to this grid
TOLERANCE_W=0.5                # stop when |measured - target| <= this
MAX_ITERS=12                   # binary search iteration cap
SETTLE_SECONDS=3               # wait after changing frequency
COOL_SECONDS=2                 # wait between benchmark invocations

TRIALS="${TRIALS:-20}"         # trials per candidate freq (env override)
WARMUP_RUNS="${WARMUP_RUNS:-3}"
export TRIALS WARMUP_RUNS

# ── Sanity checks ────────────────────────────────────────────────────────
if [[ ! -x "${BINARY}" ]]; then
    echo "==> Binary not found, compiling..."
    g++ -O3 -march=native -mavx512f -mfma -o "${BINARY}" \
        "${SCRIPT_DIR}/benchmark_avx512_fma.cpp" -lpthread
fi

HW_MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq)  # kHz
HW_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)  # kHz

# ── Restore original state on exit ──────────────────────────────────────
ORIG_NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
ORIG_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
ORIG_MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)
ORIG_MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)

cleanup() {
    echo ""
    echo "==> Restoring original CPU configuration..."
    echo "${ORIG_MIN}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq > /dev/null
    echo "${ORIG_MAX}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null
    echo "${ORIG_NO_TURBO}" | tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
    echo "${ORIG_GOV}"  | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    if [[ -n "${SUDO_USER:-}" ]]; then
        chown -R "${SUDO_USER}:${SUDO_USER}" "${RESULTS_DIR}" 2>/dev/null || true
    fi
    chmod -R a+rw "${RESULTS_DIR}" 2>/dev/null || true
    echo "==> Done."
}
trap cleanup EXIT INT TERM

echo "==> Configuring: turbo OFF, governor=performance"
echo 1 | tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

# ── Helper: set frequency (MHz) on all cores, wait to settle ────────────
set_freq() {
    local freq_mhz=$1
    local freq_khz=$((freq_mhz * 1000))
    echo "${freq_khz}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq > /dev/null
    echo "${freq_khz}" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null
    sleep "${SETTLE_SECONDS}"
}

# ── Helper: run benchmark, return mean cores_W (last stdout line) ───────
run_and_measure() {
    local operand=$1
    local freq_mhz=$2
    local tag=$3
    local csv="${RESULTS_DIR}/avx512_fma_${operand}_${tag}_${freq_mhz}MHz.csv"
    local mean_w
    mean_w=$("${BINARY}" "${operand}" "${freq_mhz}" "${csv}" | tail -n1)
    sleep "${COOL_SECONDS}"
    echo "${mean_w}"
}

round_to_step() {
    # round $1 (MHz) to nearest STEP_MHZ, clamp to [MIN_FREQ_MHZ, REF_FREQ_MHZ]
    local f=$1
    local rounded=$(( ( (f + STEP_MHZ/2) / STEP_MHZ ) * STEP_MHZ ))
    if (( rounded < MIN_FREQ_MHZ )); then rounded=$MIN_FREQ_MHZ; fi
    if (( rounded > REF_FREQ_MHZ )); then rounded=$REF_FREQ_MHZ; fi
    echo "${rounded}"
}

# ── Step 1: measure 0x00 target power at REF_FREQ_MHZ ────────────────────
echo ""
echo "========================================================"
echo " Step 1: baseline — 0x00 @ ${REF_FREQ_MHZ} MHz"
echo "========================================================"
set_freq "${REF_FREQ_MHZ}"
TARGET_W=$(run_and_measure "0x00" "${REF_FREQ_MHZ}" "baseline")
echo " Target power (0x00 @ ${REF_FREQ_MHZ} MHz): ${TARGET_W} W"

# ── Step 2: binary search over frequency for 0xAA ────────────────────────
LO=${MIN_FREQ_MHZ}
HI=${REF_FREQ_MHZ}
BEST_FREQ=${HI}
BEST_DIFF="999999"

echo ""
echo "========================================================"
echo " Step 2: binary search for 0xAA freq matching ${TARGET_W} W"
echo " Search range: [${LO}, ${HI}] MHz, tolerance ${TOLERANCE_W} W"
echo "========================================================"

for (( iter=1; iter<=MAX_ITERS; iter++ )); do
    MID=$(( (LO + HI) / 2 ))
    MID=$(round_to_step "${MID}")

    echo ""
    echo "--- Iteration ${iter}: trying ${MID} MHz (range [${LO}, ${HI}]) ---"
    set_freq "${MID}"
    MEAS_W=$(run_and_measure "0xAA" "${MID}" "iter${iter}")

    # compare using awk (bash has no float arithmetic)
    DIFF=$(awk -v a="${MEAS_W}" -v b="${TARGET_W}" 'BEGIN{d=a-b; if(d<0)d=-d; print d}')
    SIGN=$(awk -v a="${MEAS_W}" -v b="${TARGET_W}" 'BEGIN{print (a>b)?1:0}')

    echo " Measured 0xAA power: ${MEAS_W} W  |  target: ${TARGET_W} W  |  |diff|=${DIFF} W"

    IS_BEST=$(awk -v d="${DIFF}" -v b="${BEST_DIFF}" 'BEGIN{print (d<b)?1:0}')
    if [[ "${IS_BEST}" == "1" ]]; then
        BEST_DIFF="${DIFF}"
        BEST_FREQ="${MID}"
    fi

    WITHIN_TOL=$(awk -v d="${DIFF}" -v t="${TOLERANCE_W}" 'BEGIN{print (d<=t)?1:0}')
    if [[ "${WITHIN_TOL}" == "1" ]]; then
        echo ""
        echo " Converged: ${MID} MHz gives 0xAA power within ${TOLERANCE_W} W of target."
        BEST_FREQ="${MID}"
        break
    fi

    if [[ "${SIGN}" == "1" ]]; then
        # 0xAA measured power still higher than target -> need lower freq
        HI=$(( MID - STEP_MHZ ))
    else
        # 0xAA measured power lower than target -> need higher freq
        LO=$(( MID + STEP_MHZ ))
    fi

    if (( LO > HI )); then
        echo " Search bounds crossed (LO=${LO} > HI=${HI}); stopping."
        break
    fi
done

echo ""
echo "========================================================"
echo " RESULT"
echo "   0x00  @ ${REF_FREQ_MHZ} MHz -> ${TARGET_W} W"
echo "   0xAA  @ ${BEST_FREQ} MHz    -> matched within ${BEST_DIFF} W"
echo "   Implied frequency 'cost' of the 0xAA data pattern:"
echo "     ${REF_FREQ_MHZ} MHz - ${BEST_FREQ} MHz = $((REF_FREQ_MHZ - BEST_FREQ)) MHz"
echo "========================================================"
echo ""
echo "All per-candidate CSVs saved under: ${RESULTS_DIR}/"