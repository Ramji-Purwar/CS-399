#!/usr/bin/env bash
#
# run_experiment.sh -- measure how AVX2/FMA/IFMA operand values affect
# power draw and clock frequency, sweeping all (D1, D2) combinations.
#
#   sudo ./run_experiment.sh
#
# For every (D1, D2) pair the benchmark runs with those fixed operands and
# records TRIALS measurements of (power, time, clock).
#
# Two phases:
#   Phase 1 — stock PL1 = 185 W  : no throttling expected for any combo
#   Phase 2 — PL1 = SET_PL1_W    : high-power (D1,D2) pairs get throttled
#
# Instruction types run:
#   fma      vfmadd231ps  (AVX2/FMA)           float32
#   mul      vmulps       (AVX2)                float32
#   ifma_lo  vpmadd52luq  (AVX512IFMA+AVX512VL) uint52
#   ifma_hi  vpmadd52huq  (AVX512IFMA+AVX512VL) uint52
#   ifma_lo/hi skipped if avx512ifma absent from /proc/cpuinfo.
#
# Tunables (env vars):
#   BENCH_C     addend for fma/mul (raises dest-reg flips)  [0xbfaaaaab]
#   TRIALS      measurements per (D1,D2) run                [100]
#   WARMUP      warm-up runs                                 [1]
#   ITERS       ops per thread per trial                     [12000000000]
#   COOLDOWN    seconds between runs                         [10]
#   MAX_TEMP_C  wait until pkg temp drops below this         [0 = skip]
#   SET_PL1_W   PL1 for Phase 2                             [165]
#   THRESH_PCT  timing gap (%) to flag as significant        [0.5]
#   FREQ_TOL    clock gap fraction to flag                   [0.005]
#   OUT_DIR     directory for result CSVs                    [results]
#
set -u
cd "$(dirname "$(readlink -f "$0")")"

BENCH_C=${BENCH_C:-0xbfaaaaab}
TRIALS=${TRIALS:-25}
WARMUP=${WARMUP:-1}
ITERS=${ITERS:-12000000000}
COOLDOWN=${COOLDOWN:-10}
MAX_TEMP_C=${MAX_TEMP_C:-0}
SET_PL1_W=${SET_PL1_W:-165}
THRESH_PCT=${THRESH_PCT:-0.5}
FREQ_TOL=${FREQ_TOL:-0.005}
OUT_DIR=${OUT_DIR:-results}
NUM_CORES=28          # must match bench_data_dep.cpp

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
# run_single <d1_name> <d2_name> <d2_hex> <out_csv> <freq_csv>
#            <logf> <op> <d1_hex> <bench_c>
#   Runs TRIALS measurements for one fixed (D1, D2) pair.
# ---------------------------------------------------------------
run_single() {
  local d1_name="$1" d2_name="$2" d2_hex="$3" out="$4" fcsv="$5" lf="$6"
  local op="$7" d1_hex="$8" bc="$9"
  local combo_key="${d1_name}:${d2_name}"
  local TMP
  TMP=$(mktemp)

  wait_cool
  start_sampler "$TMP"
  BENCH_OUT="$out" BENCH_TRIALS="$TRIALS" BENCH_WARMUP="$WARMUP" \
  BENCH_ITERS="$ITERS" BENCH_ROUND="1" \
  BENCH_D1_NAME="$combo_key" BENCH_D2_NAME="result" BENCH_FREQ_KHZ=0 \
  BENCH_TOGGLE_A=0 BENCH_C="$bc" \
    "$BIN" "$op" "$d1_hex" "$d2_hex" "0" 2>>"$lf" \
    || log "    FAILED (see $lf)"
  kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
  awk -v c="$combo_key" -v f=0 -v l="result" -v r="1" '
    {a[NR]=$1}
    END{ if(NR==0){exit}
         s=(NR<4) ? 1 : int(NR/2)+1; n=0; sum=0; mn=a[s]
         for(i=s;i<=NR;i++){sum+=a[i]; n++; if(a[i]<mn)mn=a[i]}
         printf "%s,%s,%s,%s,%.0f,%d\n", c,f,l,r,sum/n,mn }' "$TMP" >> "$fcsv"
  sleep "$COOLDOWN"
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
    k = r['d1_name']
    T[k].append(float(r['mean_time_s'])); W[k].append(float(r['pkg_W']))
    if k not in cfgs: cfgs.append(k)
for r in csv.DictReader(open(fcsv)):
    A[r['cfg']].append(float(r['actual_mean_khz']))

print(f"\n{'='*80}")
print(f"  {phase}   (PL1 = {pl1:.0f} W)")
print(f"{'='*80}")
hdr = (f"{'D1:D2 combo':>22} | {'clk MHz':>7} | "
       f"{'time (s)':>10} | {'power W':>8} | result")
print(hdr); print("-" * len(hdr))

for cfg in cfgs:
    if cfg not in T: continue
    t = st.mean(T[cfg])
    w = st.mean(W[cfg])
    al = st.mean(A[cfg]) if A[cfg] else float('nan')
    flags = []
    if pl1 > 0 and w > pl1: flags.append("OVER PL1")
    else: flags.append("—")
    print(f"{cfg:>22} | {al/1000:7.0f} | {t:10.4f} | {w:8.1f} | {', '.join(flags)}")
print()
PY
}

# ================================================================
# D1 values: 0, 1, small primes, large primes, large numbers, random
# Edit these lists to change what D1 values are tried.
# ================================================================
_d1_float_vals=(
  "zero:0.0"
  "one:1.0"
  "prime2:2.0"
  "prime3:3.0"
  "prime5:5.0"
  "prime7:7.0"
  "prime97:97.0"
  "prime1009:1009.0"
  "large1:1000000.0"
  "large2:1.0e10"
  "rand1:3.14159265"
  "rand2:1234567.89"
)
_d1_int_vals=(
  "zero:0"
  "one:1"
  "prime2:2"
  "prime3:3"
  "prime5:5"
  "prime7:7"
  "prime97:97"
  "prime1009:1009"
  "large1:1000000"
  "large2:999999937"
  "rand1:3141592653"
  "rand2:2718281828"
)

# ================================================================
# D2 patterns: explicit hex values for each op type.
# float ops: includes 0x00000000 (actual zero) plus mantissa patterns.
# ifma ops:  includes 0 (integer zero) plus 52-bit patterns.
# ================================================================
_d2_float_patterns=(
  "zero:0x00000000"
  "one:0x3f800000"
  "alt10:0x3faaaaaa"
  "alt01:0x3fd55555"
  "pair0011:0x3fb33333"
  "pair1100:0x3fcccccc"
  "nib0F:0x3ff0f0f0"
  "nibF0:0x3f8f0f0f"
  "byte00FF:0x3ff00ff0"
  "allbits:0x3fffffff"
  "lsb:0x3f800001"
  "large:0x4f800000"
  "neg_one:0xbf800000"
)
_d2_int_patterns=(
  "zero:0x0000000000000"
  "ones:0xfffffffffffff"
  "alt10:0xaaaaaaaaaaaaa"
  "alt01:0x5555555555555"
  "pair0011:0x3333333333333"
  "pair1100:0xccccccccccccc"
  "nib0F:0x0f0f0f0f0f0f0"
  "nibF0:0xf0f0f0f0f0f0f"
  "byte00FF:0x00ff00ff00ff0"
  "low26:0x0000003ffffff"
  "high26:0xffffffc000000"
  "lsb:0x0000000000001"
  "msb:0x8000000000000"
)

# ================================================================
# Main loop: Phase 1 + Phase 2 for each instruction type
# ================================================================
_D1_COUNT=${#_d1_float_vals[@]}
_D2_COUNT=${#_d2_float_patterns[@]}
COMBO_TOTAL=$(( ${#ALL_OPS[@]} * _D1_COUNT * _D2_COUNT * 2 ))
COMBO_DONE=0

release_freq; sleep 2
log "frequency NOT pinned: hardware range $(<"$CPU0/cpuinfo_min_freq")..${START_KHZ} kHz"
log "total combos to run: ${COMBO_TOTAL}  (${#ALL_OPS[@]} ops × ${_D1_COUNT} D1 × ${_D2_COUNT} D2 × 2 phases)"

for CUR_OP in "${ALL_OPS[@]}"; do
  log ""
  log "╔══════════════════════════════════════════════════════════════════╗"
  log "║  INSTRUCTION TYPE: $CUR_OP"
  log "╚══════════════════════════════════════════════════════════════════╝"

  # Build D1_SWEEP and D2_SWEEP for this op type
  if [[ "$CUR_OP" == "fma" || "$CUR_OP" == "mul" ]]; then
    CUR_BENCH_C="$BENCH_C"
    # D1: actual float values → convert to IEEE float32 hex
    D1_SWEEP=()
    for _e in "${_d1_float_vals[@]}"; do
      _n="${_e%%:*}"; _v="${_e#*:}"
      _hex=$(python3 -c "import struct; print(hex(struct.unpack('I', struct.pack('f', $_v))[0]))")
      D1_SWEEP+=("${_n}:${_hex}")
    done
    # D2: explicit float hex patterns (includes actual 0x00000000)
    D2_SWEEP=("${_d2_float_patterns[@]}")
  else
    # ifma_lo / ifma_hi: 52-bit unsigned integer operands
    CUR_BENCH_C=0xAAAAAAAAAAAAA
    # D1: actual integer values
    D1_SWEEP=()
    for _e in "${_d1_int_vals[@]}"; do
      _n="${_e%%:*}"; _v="${_e#*:}"
      _hex=$(printf "0x%013x" $(( _v )))
      D1_SWEEP+=("${_n}:${_hex}")
    done
    # D2: explicit 52-bit integer hex patterns (includes actual 0)
    D2_SWEEP=("${_d2_int_patterns[@]}")
  fi
  unset _e _n _v _hex

  # ── Phase 1: BASELINE (stock PL1, no throttling expected) ────────────────
  if [[ -n $RAPL_PL1_FILE ]]; then
    echo "$SAVED_PL1_UW" > "$RAPL_PL1_FILE"
  fi
  release_freq; sleep 2

  log "════════════════════════════════════════════════════════════════"
  log "  PHASE 1: BASELINE  —  stock PL1 = ${STOCK_PL1_W} W  |  op = $CUR_OP"
  log "  D1: ${#D1_SWEEP[@]} values × D2: ${#D2_SWEEP[@]} patterns = $(( ${#D1_SWEEP[@]} * ${#D2_SWEEP[@]} )) combos"
  log "════════════════════════════════════════════════════════════════"

  BASE_CSV="$OUT_DIR/baseline_${CUR_OP}.csv"
  BASE_FREQ="$OUT_DIR/baseline_${CUR_OP}_freq.csv"
  BASE_LOG="$OUT_DIR/baseline_${CUR_OP}.log"
  rm -f "$BASE_CSV" "$BASE_FREQ" "$BASE_LOG"
  echo "op,round,d1_name,d2_name,data1,data2,datac,freq_khz,toggle,trial,mean_time_s,std_time_s,pkg_J,pkg_W,nJ_per_op,a_weight,b0_weight,b1_weight,r0_weight,r1_weight,input_flips_per_op,result_flips_per_op,dest_flips_per_op,total_flips_per_op,toggle_a" > "$BASE_CSV"
  echo "cfg,freq_khz,label,round,actual_mean_khz,actual_min_khz" > "$BASE_FREQ"

  for d1_entry in "${D1_SWEEP[@]}"; do
    d1_pat="${d1_entry%%:*}"; d1_hex="${d1_entry#*:}"
    for d2_entry in "${D2_SWEEP[@]}"; do
      COMBO_DONE=$(( COMBO_DONE + 1 ))
      d2_name="${d2_entry%%:*}"; d2_hex="${d2_entry#*:}"
      log "[${COMBO_DONE}/${COMBO_TOTAL}] op=${CUR_OP} phase=1 | D1=${d1_pat} D2=${d2_name}"
      run_single "$d1_pat" "$d2_name" "$d2_hex" "$BASE_CSV" "$BASE_FREQ" "$BASE_LOG" \
                 "$CUR_OP" "$d1_hex" "$CUR_BENCH_C"
    done
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
  log "  D1: ${#D1_SWEEP[@]} values × D2: ${#D2_SWEEP[@]} patterns = $(( ${#D1_SWEEP[@]} * ${#D2_SWEEP[@]} )) combos"
  log "════════════════════════════════════════════════════════════════"

  EXP_CSV="$OUT_DIR/experiment_${CUR_OP}.csv"
  EXP_FREQ="$OUT_DIR/experiment_${CUR_OP}_freq.csv"
  EXP_LOG="$OUT_DIR/experiment_${CUR_OP}.log"
  rm -f "$EXP_CSV" "$EXP_FREQ" "$EXP_LOG"
  echo "op,round,d1_name,d2_name,data1,data2,datac,freq_khz,toggle,trial,mean_time_s,std_time_s,pkg_J,pkg_W,nJ_per_op,a_weight,b0_weight,b1_weight,r0_weight,r1_weight,input_flips_per_op,result_flips_per_op,dest_flips_per_op,total_flips_per_op,toggle_a" > "$EXP_CSV"
  echo "cfg,freq_khz,label,round,actual_mean_khz,actual_min_khz" > "$EXP_FREQ"

  for d1_entry in "${D1_SWEEP[@]}"; do
    d1_pat="${d1_entry%%:*}"; d1_hex="${d1_entry#*:}"
    for d2_entry in "${D2_SWEEP[@]}"; do
      COMBO_DONE=$(( COMBO_DONE + 1 ))
      d2_name="${d2_entry%%:*}"; d2_hex="${d2_entry#*:}"
      log "[${COMBO_DONE}/${COMBO_TOTAL}] op=${CUR_OP} phase=2 PL1=${SET_PL1_W}W | D1=${d1_pat} D2=${d2_name}"
      run_single "$d1_pat" "$d2_name" "$d2_hex" "$EXP_CSV" "$EXP_FREQ" "$EXP_LOG" \
                 "$CUR_OP" "$d1_hex" "$CUR_BENCH_C"
    done
  done
  analyse "PHASE 2: EXPERIMENT (PL1=${SET_PL1_W} W) | op=$CUR_OP" \
          "$EXP_CSV" "$EXP_FREQ" "$SET_PL1_W"

done

log ""
log "done.  results in $OUT_DIR/"
