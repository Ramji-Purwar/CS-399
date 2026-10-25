#!/usr/bin/env bash
# run_experiment_2.sh
#
# Sweep for bench_data_dep.cpp (v2: also records CPU cycles + TSC ticks).
#   * only the FIRST TWO instruction types: mul, fma  (ifma_lo / ifma_hi dropped)
#   * 20 measured trials per (op, operands, toggle) condition
#   * all results (CSV, log, stats) go to the run_2/ folder
#
# Usage:  sudo ./run_experiment_2.sh
# Optional env: FREQ_KHZ (default 3000000), ROUNDS (default 3), ITERS, WARMUP

set -euo pipefail

OUTDIR="run_2"
CSV="$OUTDIR/bench_results.csv"
LOG="$OUTDIR/sweep.log"

OPS=(mul fma)                       # first two instruction types only
TRIALS=20                           # 20 trials per condition
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-3000000000}"
ROUNDS="${ROUNDS:-3}"               # repeat whole sweep, order shuffled each round
FREQ_KHZ="${FREQ_KHZ:-3000000}"     # 3.0 GHz all-core AVX ceiling

# Conditions: "name1 hex1 name2 hex2 toggle toggle_a"
#   LOW  : identical operands, no toggling (minimal bit activity)
#   MID  : the example operands from the bench header, no toggling
#   HIGH : all-ones mantissa, both multiplier inputs toggle every op
CONDS=(
  "one 0x3f800000 one 0x3f800000 0 0"
  "1.25 0x3fa00000 sqrt2 0x3fb504f3 0 0"
  "1.25 0x3fa00000 sqrt2 0x3fb504f3 1 1"
  "max_mant 0x3fffffff max_mant 0x3fffffff 1 1"
)

mkdir -p "$OUTDIR"
if [[ -s "$CSV" ]]; then
  echo "ERROR: $CSV already exists. Move it away (or delete run_2/) to avoid mixing runs." >&2
  exit 1
fi

# ---------- machine setup ----------
echo "[setup] pinning governor/frequency to ${FREQ_KHZ} kHz, turbo off" | tee -a "$LOG"
cpupower frequency-set -g performance          >>"$LOG" 2>&1 || true
cpupower frequency-set -d "$FREQ_KHZ" -u "$FREQ_KHZ" >>"$LOG" 2>&1 || true
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>>"$LOG" || true
# allow perf_event_open for cycles (we run as root, but this helps if not)
sysctl -q kernel.perf_event_paranoid=1 2>>"$LOG" || true

# ---------- build ----------
echo "[build]" | tee -a "$LOG"
g++ -O2 -mavx2 -mfma -pthread bench_data_dep.cpp -o "$OUTDIR/bench"

# ---------- sweep ----------
total=$(( ROUNDS * ${#OPS[@]} * ${#CONDS[@]} ))
n=0
for round in $(seq 1 "$ROUNDS"); do
  # shuffle condition order each round to average out thermal drift
  mapfile -t ORDER < <(printf '%s\n' "${CONDS[@]}" | shuf)
  for op in "${OPS[@]}"; do
    for cond in "${ORDER[@]}"; do
      read -r n1 h1 n2 h2 tog toga <<<"$cond"
      n=$((n+1))
      echo "[$n/$total] round=$round op=$op d1=$h1 d2=$h2 toggle=$tog toggle_a=$toga" | tee -a "$LOG"
      BENCH_OUT="$CSV" \
      BENCH_TRIALS="$TRIALS" \
      BENCH_WARMUP="$WARMUP" \
      BENCH_ITERS="$ITERS" \
      BENCH_ROUND="$round" \
      BENCH_D1_NAME="$n1" BENCH_D2_NAME="$n2" \
      BENCH_FREQ_KHZ="$FREQ_KHZ" \
      BENCH_TOGGLE_A="$toga" \
        "$OUTDIR/bench" "$op" "$h1" "$h2" "$tog" 2>>"$LOG"
      sleep 5     # short cool-down between conditions
    done
  done
done

# ---------- restore ----------
cpupower frequency-set -g powersave >>"$LOG" 2>&1 || true
echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>>"$LOG" || true

