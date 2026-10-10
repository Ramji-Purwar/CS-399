#!/usr/bin/env bash
#
# run_experiment.sh -- demonstrate that CPU clock throttling depends on the
# BIT PATTERNS in AVX2/FMA/IFMA operands, sweeping all supported instruction
# types: fma (vfmadd231ps), mul (vmulps), ifma_lo (vpmadd52luq),
# ifma_hi (vpmadd52huq).
#
#   sudo ./run_experiment.sh
#
# For each instruction type the script runs a two-phase experiment
# (following Wang et al., CCS 2022 §3.1):
#
#   Phase 1 — BASELINE (stock PL1 = 185 W):
#     LOW and HIGH operands run at the same clock and take the same time.
#     Proves the difference in Phase 2 is NOT from any software setup.
#
#   Phase 2 — SWEEP (PL1 lowered to SET_PL1_W):
#     For each D2 pattern, LOW (zeros, toggle=0) is compared against HIGH
#     (that pattern, toggle=1). Patterns that exceed PL1 get hardware-
#     throttled; patterns below PL1 do not.
#
# Instruction types:
#   fma      vfmadd231ps  (AVX2/FMA)          -- float32 operands
#   mul      vmulps       (AVX2)               -- float32 operands
#   ifma_lo  vpmadd52luq  (AVX512IFMA+AVX512VL)-- uint52 operands
#   ifma_hi  vpmadd52huq  (AVX512IFMA+AVX512VL)-- uint52 operands
#   ifma_lo/ifma_hi are skipped if avx512ifma is absent from /proc/cpuinfo.
#
# The OS governor is "performance", turbo on, full frequency range open.
# No frequency is ever pinned — every clock difference is hardware's own doing.
#
# Tunables (env vars):
#   BENCH_C     float addend for fma/mul (raises dest-register flips) [0xbfaaaaab]
#   TRIALS      measured trials per run                               [5]
#   WARMUP      warm-up runs per run                                  [1]
#   ITERS       ops per thread per trial                              [12000000000]
#   ROUNDS      LOW/HIGH round-trips per D2 pattern                   [3]
#   COOLDOWN    seconds between runs                                  [10]
#   MAX_TEMP_C  wait until pkg temp drops below this                  [0 = skip]
#   SET_PL1_W   PL1 for Phase 2 (between LOW and HIGH)               [165]
#   D1          fixed float multiplier operand (fma/mul)              [0x40b33333]
#   THRESH_PCT  timing gap (%) to flag as significant                 [0.5]
#   FREQ_TOL    clock gap fraction to flag                            [0.005]
#   OUT_DIR     directory for result CSVs                             [results]
#
set -u
cd "$(dirname "$(readlink -f "$0")")"

BENCH_C=${BENCH_C:-0xbfaaaaab}
TRIALS=${TRIALS:-5}
WARMUP=${WARMUP:-1}
ITERS=${ITERS:-12000000000}
ROUNDS=${ROUNDS:-3}
COOLDOWN=${COOLDOWN:-10}
MAX_TEMP_C=${MAX_TEMP_C:-0}
SET_PL1_W=${SET_PL1_W:-165}
THRESH_PCT=${THRESH_PCT:-0.5}
FREQ_TOL=${FREQ_TOL:-0.005}
OUT_DIR=${OUT_DIR:-results}
NUM_CORES=28          # must match bench_data_dep.cpp

D1=${D1:-0x40b33333}              # best float D1 (fma/mul)
D1_IFMA=0xAAAAAAAAAAAAA           # alternating-bit D1 for ifma_lo/ifma_hi

SRC=bench_data_dep.cpp
BIN=./bench

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root: sudo $0"
command -v python3 >/dev/null || die "python3 needed"
command -v awk    >/dev/null || die "awk needed"
mkdir -p "$OUT_DIR"

# Detect supported instruction types
ALL_OPS=("fma" "mul")
BUILD_EXTRA=""
if grep -q "avx512ifma" /proc/cpuinfo 2>/dev/null; then
  ALL_OPS+=("ifma_lo" "ifma_hi")
  BUILD_EXTRA="-mavx512ifma -mavx512vl"
  log "avx512ifma detected — will run: fma, mul, ifma_lo, ifma_hi"
else
  log "avx512ifma NOT detected — running fma and mul only"
fi

# Build
if [[ ! -x $BIN || $SRC -nt $BIN ]]; then
  log "compiling $SRC"
  # shellcheck disable=SC2086
  g++ -O2 -mavx2 -mfma $BUILD_EXTRA -pthread "$SRC" -o "$BIN" || die "compile failed"
fi

CPUFREQ_DIRS=(/sys/devices/system/cpu/cpu[0-9]*/cpufreq)
[[ -d ${CPUFREQ_DIRS[0]} ]] || die "no cpufreq sysfs found"
CPU0=/sys/devices/system/cpu/cpu0/cpufreq
START_KHZ=$(<"$CPU0/cpuinfo_max_freq")

# ---------------- read RAPL power limits ----------------
RAPL=/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0
STOCK_PL1_W=0; SAVED_PL1_UW=""; RAPL_PL1_FILE=""
for c in 0 1; do
  [[ -f $RAPL/constraint_${c}_power_limit_uw ]] || continue
  nm=$(<"$RAPL/constraint_${c}_name")
  lim=$(( $(<"$RAPL/constraint_${c}_power_limit_uw") / 1000000 ))
  log "RAPL $nm limit = $lim W"
  if [[ $nm == long_term ]]; then
    STOCK_PL1_W=$lim
    SAVED_PL1_UW=$(<"$RAPL/constraint_${c}_power_limit_uw")
    RAPL_PL1_FILE="$RAPL/constraint_${c}_power_limit_uw"
  fi
done

# ---------------- cpufreq + RAPL save/restore ----------------
NO_TURBO=/sys/devices/system/cpu/intel_pstate/no_turbo
BOOST=/sys/devices/system/cpu/cpufreq/boost
declare -a SAVED_FREQ=()
SAVED_NO_TURBO=""; SAVED_BOOST=""
for d in "${CPUFREQ_DIRS[@]}"; do
  SAVED_FREQ+=("$d|$(<"$d/scaling_governor")|$(<"$d/scaling_min_freq")|$(<"$d/scaling_max_freq")")
done
[[ -f $NO_TURBO ]] && SAVED_NO_TURBO=$(<"$NO_TURBO")
[[ -f $BOOST    ]] && SAVED_BOOST=$(<"$BOOST")

SAMPLER_PID=""
restore_all() {
  [[ -n $SAMPLER_PID ]] && kill "$SAMPLER_PID" 2>/dev/null
  log "restoring cpufreq + RAPL settings"
  local e d gov mn mx
  for e in "${SAVED_FREQ[@]}"; do
    IFS='|' read -r d gov mn mx <<<"$e"
    cat "$d/cpuinfo_min_freq" > "$d/scaling_min_freq" 2>/dev/null
    echo "$mx"  > "$d/scaling_max_freq" 2>/dev/null
    echo "$mn"  > "$d/scaling_min_freq" 2>/dev/null
    echo "$gov" > "$d/scaling_governor" 2>/dev/null
  done
  [[ -n $SAVED_NO_TURBO ]] && echo "$SAVED_NO_TURBO" > "$NO_TURBO"
  [[ -n $SAVED_BOOST    ]] && echo "$SAVED_BOOST"    > "$BOOST"
  if [[ -n $SAVED_PL1_UW && -n $RAPL_PL1_FILE ]]; then
    echo "$SAVED_PL1_UW" > "$RAPL_PL1_FILE"
    log "restored PL1 to ${STOCK_PL1_W} W"
  fi
}
trap restore_all EXIT
trap 'exit 130' INT TERM

[[ -f $NO_TURBO ]] && echo 0 > "$NO_TURBO"
[[ -f $BOOST    ]] && echo 1 > "$BOOST"

release_freq() {
  local d
  for d in "${CPUFREQ_DIRS[@]}"; do
    cat "$d/cpuinfo_min_freq" > "$d/scaling_min_freq"
    cat "$d/cpuinfo_max_freq" > "$d/scaling_max_freq"
    echo performance > "$d/scaling_governor"
  done
}

pkg_temp_c() {
  local z
  for z in /sys/class/thermal/thermal_zone*; do
    [[ $(<"$z/type") == x86_pkg_temp ]] && { echo $(( $(<"$z/temp") / 1000 )); return; }
  done
  echo 0
}

wait_cool() {
  (( MAX_TEMP_C > 0 )) || return 0
  local t waited=0
  while :; do
    t=$(pkg_temp_c)
    (( t == 0 || t <= MAX_TEMP_C )) && return 0
    (( waited % 30 == 0 )) && log "  cooling: pkg ${t}C > ${MAX_TEMP_C}C"
    sleep 5; waited=$((waited+5))
    (( waited > 600 )) && { log "  cooldown timeout, continuing"; return 0; }
  done
}

SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.05}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-4}
NCPU=$(nproc --all)
SAMPLER_CPU=$(( NCPU - 1 ))

sample_loop() {
  local out=$1 c s n f
  : > "$out"
  while :; do
    s=0; n=0
    for (( c=0; c<NUM_CORES; c+=SAMPLE_STRIDE )); do
      f=$(<"/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq") 2>/dev/null || continue
      s=$((s+f)); n=$((n+1))
    done
    (( n > 0 )) && echo $(( s / n )) >> "$out"
    sleep "$SAMPLE_INTERVAL"
  done
}

start_sampler() {
  if (( SAMPLER_CPU >= NUM_CORES )); then
    taskset -c "$SAMPLER_CPU" bash -c "$(declare -f sample_loop); \
      SAMPLE_INTERVAL=$SAMPLE_INTERVAL SAMPLE_STRIDE=$SAMPLE_STRIDE \
      NUM_CORES=$NUM_CORES sample_loop '$1'" &
  else
    sample_loop "$1" &
  fi
  SAMPLER_PID=$!
}

# ---------------------------------------------------------------
# run_combo <d2_name> <d2_high_hex> <out_csv> <freq_csv> <logf>
#           <op> <d1_hex> <bench_c> <d2_low_hex>
#   Runs ROUNDS round-trips of LOW vs HIGH for one D2 pattern.
# ---------------------------------------------------------------
run_combo() {
  local d2_name="$1" d2_high="$2" out="$3" fcsv="$4" lf="$5"
  local op="$6" d1="$7" bc="$8" d2_low="$9"
  local TMP
  TMP=$(mktemp)

  local LABELS=(low high)
  local D2S=("$d2_low" "$d2_high")
  local TOGS=(0 1)

  log "  ── D2=${d2_name} (${d2_high})"
  local order r i lbl d2 tog
  for (( r=1; r<=ROUNDS; r++ )); do
    if (( r % 2 )); then order=(0 1); else order=(1 0); fi
    for i in "${order[@]}"; do
      lbl=${LABELS[$i]}; d2=${D2S[$i]}; tog=${TOGS[$i]}
      wait_cool
      log "    r=$r $lbl"
      start_sampler "$TMP"
      BENCH_OUT="$out" BENCH_TRIALS="$TRIALS" BENCH_WARMUP="$WARMUP" \
      BENCH_ITERS="$ITERS" BENCH_ROUND="$r" \
      BENCH_D1_NAME="$d2_name" BENCH_D2_NAME="$lbl" BENCH_FREQ_KHZ=0 \
      BENCH_TOGGLE_A=1 BENCH_C="$bc" \
        "$BIN" "$op" "$d1" "$d2" "$tog" 2>>"$lf" \
        || log "    FAILED (see $lf)"
      kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
      awk -v c="$d2_name" -v f=0 -v l="$lbl" -v r="$r" '
        {a[NR]=$1}
        END{ if(NR==0){exit}
             s=(NR<4) ? 1 : int(NR/2)+1; n=0; sum=0; mn=a[s]
             for(i=s;i<=NR;i++){sum+=a[i]; n++; if(a[i]<mn)mn=a[i]}
             printf "%s,%s,%s,%s,%.0f,%d\n", c,f,l,r,sum/n,mn }' "$TMP" >> "$fcsv"
      sleep "$COOLDOWN"
    done
  done
  rm -f "$TMP"
}

# ---------------------------------------------------------------
# analyse <phase_label> <out_csv> <freq_csv> <pl1_w>
# ---------------------------------------------------------------
analyse() {
  local phase="$1" out="$2" fcsv="$3" pl1="$4"
  log "  analysing $phase"
  python3 - "$out" "$fcsv" "$THRESH_PCT" "$FREQ_TOL" "$pl1" "$phase" <<'PY'
import csv, sys, statistics as st
from collections import defaultdict
bench, fcsv, phase = sys.argv[1], sys.argv[2], sys.argv[6]
thr, tol, pl1 = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

T = defaultdict(list); W = defaultdict(list); A = defaultdict(list); cfgs = []
for r in csv.DictReader(open(bench)):
    k = (r['d1_name'], r['d2_name'])
    T[k].append(float(r['mean_time_s'])); W[k].append(float(r['pkg_W']))
    if r['d1_name'] not in cfgs: cfgs.append(r['d1_name'])
for r in csv.DictReader(open(fcsv)):
    A[(r['cfg'], r['label'])].append(float(r['actual_mean_khz']))

print(f"\n{'='*80}")
print(f"  {phase}   (PL1 = {pl1:.0f} W)")
print(f"{'='*80}")
hdr = (f"{'D2 pattern':>12} | {'clk L':>7} {'clk H':>7} {'dclk %':>7} | "
       f"{'time L (s)':>10} {'time H (s)':>10} {'dt %':>6} | "
       f"{'W low':>6} {'W high':>6} | result")
print(hdr); print("-" * len(hdr))

for cfg in cfgs:
    kl, kh = (cfg, 'low'), (cfg, 'high')
    if kl not in T or kh not in T: continue
    tl, th = st.mean(T[kl]), st.mean(T[kh])
    wl, wh = st.mean(W[kl]), st.mean(W[kh])
    diff = (th - tl) / tl * 100
    sem  = ((st.pstdev(T[kl])**2 + st.pstdev(T[kh])**2)**0.5 / len(T[kl])**0.5) / tl * 100
    al = st.mean(A[kl]) if A[kl] else float('nan')
    ah = st.mean(A[kh]) if A[kh] else float('nan')
    dclk = (ah - al) / al * 100 if al else float('nan')
    clock_gap = ah < al * (1 - tol)
    td = diff > thr and diff > 3 * sem
    flags = []
    if pl1 > 0 and wl > pl1: flags.append("LOW>PL1")
    if pl1 > 0 and wh > pl1: flags.append("HIGH>PL1")
    if clock_gap and td: flags.append("THROTTLED ✓")
    elif clock_gap:      flags.append("clock gap")
    elif td:             flags.append("timing gap")
    else:                flags.append("—")
    print(f"{cfg:>12} | {al/1000:7.0f} {ah/1000:7.0f} {dclk:7.2f} | "
          f"{tl:10.4f} {th:10.4f} {diff:6.2f} | {wl:6.1f} {wh:6.1f} | {', '.join(flags)}")
print()
PY
}

# ================================================================
# 13-pattern D2 sweep  (raw 52-bit mantissa values)
# fma/mul: OR with 0x3F800000 to get a valid float in [1.0, 2.0)
# ifma_*:  use raw 52-bit integer values directly
# ================================================================
_raw_patterns=(
  "zero:0x0"
  "ones:0xFFFFFFFFFFFFF"
  "alt10:0xAAAAAAAAAAAAA"
  "alt01:0x5555555555555"
  "pair0011:0x3333333333333"
  "pair1100:0xCCCCCCCCCCCCC"
  "nib0F:0x0F0F0F0F0F0F0"
  "nibF0:0xF0F0F0F0F0F0F"
  "byte00FF:0x00FF00FF00FF0"
  "low26:0x0000003FFFFFF"
  "high26:0xFFFFFFC000000"
  "lsb:0x0000000000001"
  "msb:0x8000000000000"
)

release_freq; sleep 2
log "frequency NOT pinned: hardware range $(<"$CPU0/cpuinfo_min_freq")..${START_KHZ} kHz"

# ================================================================
# Main loop: Phase 1 + Phase 2 for each instruction type
# ================================================================
# Total run_combo calls = ops × patterns × 2 phases
COMBO_TOTAL=$(( ${#ALL_OPS[@]} * ${#_raw_patterns[@]} * 2 ))
COMBO_DONE=0
log "total combos to run: ${COMBO_TOTAL}  (${#ALL_OPS[@]} ops × ${#_raw_patterns[@]} patterns × 2 phases)"

for CUR_OP in "${ALL_OPS[@]}"; do
  log ""
  log "╔══════════════════════════════════════════════════════════════════╗"
  log "║  INSTRUCTION TYPE: $CUR_OP"
  log "╚══════════════════════════════════════════════════════════════════╝"

  # Per-op operand parameters and D2 sweep computation
  if [[ "$CUR_OP" == "fma" || "$CUR_OP" == "mul" ]]; then
    CUR_D1="$D1"
    CUR_BENCH_C="$BENCH_C"
    CUR_D2_LOW=0x00000000
    # Float patterns: 0x3F800000 | (raw52 & 0x7FFFFF) → value in [1.0, 2.0)
    D2_SWEEP=()
    for _e in "${_raw_patterns[@]}"; do
      _n="${_e%%:*}"; _r="${_e#*:}"
      _hex=$(printf "0x%08x" $(( 0x3F800000 | (_r & 0x7FFFFF) )))
      D2_SWEEP+=("${_n}:${_hex}")
    done
  else
    # ifma_lo / ifma_hi: raw 52-bit unsigned integer patterns
    CUR_D1="$D1_IFMA"
    CUR_BENCH_C=0xAAAAAAAAAAAAA
    CUR_D2_LOW=0x0000000000000
    D2_SWEEP=()
    for _e in "${_raw_patterns[@]}"; do
      _n="${_e%%:*}"; _r="${_e#*:}"
      _hex=$(printf "0x%013x" $(( _r & 0xFFFFFFFFFFFFF )))
      D2_SWEEP+=("${_n}:${_hex}")
    done
  fi
  unset _e _n _r _hex

  # ── Phase 1: BASELINE (stock PL1, no throttling expected) ────────────────
  # Restore PL1 to stock in case the previous op left it lowered.
  if [[ -n $RAPL_PL1_FILE ]]; then
    echo "$SAVED_PL1_UW" > "$RAPL_PL1_FILE"
  fi
  release_freq; sleep 2

  log "════════════════════════════════════════════════════════════════"
  log "  PHASE 1: BASELINE  —  stock PL1 = ${STOCK_PL1_W} W  |  op = $CUR_OP"
  log "  D1 = $CUR_D1  |  LOW: D2=$CUR_D2_LOW toggle=0  vs  HIGH: D2=<pattern> toggle=1"
  log "════════════════════════════════════════════════════════════════"

  BASE_CSV="$OUT_DIR/baseline_${CUR_OP}.csv"
  BASE_FREQ="$OUT_DIR/baseline_${CUR_OP}_freq.csv"
  BASE_LOG="$OUT_DIR/baseline_${CUR_OP}.log"
  rm -f "$BASE_CSV" "$BASE_FREQ" "$BASE_LOG"
  echo "op,round,d1_name,d2_name,data1,data2,datac,freq_khz,toggle,trial,mean_time_s,std_time_s,pkg_J,pkg_W,nJ_per_op,a_weight,b0_weight,b1_weight,r0_weight,r1_weight,input_flips_per_op,result_flips_per_op,dest_flips_per_op,total_flips_per_op,toggle_a" > "$BASE_CSV"
  echo "cfg,freq_khz,label,round,actual_mean_khz,actual_min_khz" > "$BASE_FREQ"

  for entry in "${D2_SWEEP[@]}"; do
    COMBO_DONE=$(( COMBO_DONE + 1 ))
    d2_name="${entry%%:*}"; d2_high="${entry#*:}"
    log "[${COMBO_DONE}/${COMBO_TOTAL}] op=${CUR_OP} | phase=1 (baseline) | pattern=${d2_name}"
    run_combo "$d2_name" "$d2_high" "$BASE_CSV" "$BASE_FREQ" "$BASE_LOG" \
              "$CUR_OP" "$CUR_D1" "$CUR_BENCH_C" "$CUR_D2_LOW"
  done
  analyse "PHASE 1: BASELINE (stock PL1=${STOCK_PL1_W} W) | op=$CUR_OP" \
          "$BASE_CSV" "$BASE_FREQ" "$STOCK_PL1_W"

  # ── Phase 2: SWEEP with reduced PL1 ──────────────────────────────────────
  if [[ -n $RAPL_PL1_FILE ]]; then
    echo $(( SET_PL1_W * 1000000 )) > "$RAPL_PL1_FILE"
    log "PL1 lowered to ${SET_PL1_W} W (will be restored before next op)"
  fi

  log ""
  log "════════════════════════════════════════════════════════════════"
  log "  PHASE 2: EXPERIMENT  —  PL1 = ${SET_PL1_W} W  |  op = $CUR_OP"
  log "  D1 = $CUR_D1  |  LOW: D2=$CUR_D2_LOW toggle=0  vs  HIGH: D2=<pattern> toggle=1"
  log "════════════════════════════════════════════════════════════════"

  EXP_CSV="$OUT_DIR/experiment_${CUR_OP}.csv"
  EXP_FREQ="$OUT_DIR/experiment_${CUR_OP}_freq.csv"
  EXP_LOG="$OUT_DIR/experiment_${CUR_OP}.log"
  rm -f "$EXP_CSV" "$EXP_FREQ" "$EXP_LOG"
  echo "op,round,d1_name,d2_name,data1,data2,datac,freq_khz,toggle,trial,mean_time_s,std_time_s,pkg_J,pkg_W,nJ_per_op,a_weight,b0_weight,b1_weight,r0_weight,r1_weight,input_flips_per_op,result_flips_per_op,dest_flips_per_op,total_flips_per_op,toggle_a" > "$EXP_CSV"
  echo "cfg,freq_khz,label,round,actual_mean_khz,actual_min_khz" > "$EXP_FREQ"

  for entry in "${D2_SWEEP[@]}"; do
    COMBO_DONE=$(( COMBO_DONE + 1 ))
    d2_name="${entry%%:*}"; d2_high="${entry#*:}"
    log "[${COMBO_DONE}/${COMBO_TOTAL}] op=${CUR_OP} | phase=2 (experiment PL1=${SET_PL1_W}W) | pattern=${d2_name}"
    run_combo "$d2_name" "$d2_high" "$EXP_CSV" "$EXP_FREQ" "$EXP_LOG" \
              "$CUR_OP" "$CUR_D1" "$CUR_BENCH_C" "$CUR_D2_LOW"
  done
  analyse "PHASE 2: EXPERIMENT (PL1=${SET_PL1_W} W) | op=$CUR_OP" \
          "$EXP_CSV" "$EXP_FREQ" "$SET_PL1_W"

done

log ""
log "done.  results in $OUT_DIR/"
