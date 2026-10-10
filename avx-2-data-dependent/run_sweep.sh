#!/usr/bin/env bash
#
# run_sweep.sh  --  sweep (data1 x data2) for bench_data_dep.cpp at a FIXED CPU
# frequency; every result row goes into ONE csv.
#
#   sudo ./run_sweep.sh
#
# Tunables (environment variables, e.g.  sudo FREQ_KHZ=2400000 TRIALS=20 ./run_sweep.sh):
#   FREQ_KHZ   pinned frequency in kHz              [2400000 = 2.4 GHz]
#   OPS        ops to test                          ["mul fma ifma_lo ifma_hi"]
#   TOGGLES    toggle modes                         ["0 1"]
#   TRIALS     measured trials per combination      [20]
#   WARMUP     warm-up runs per combination         [2]
#   ITERS      ops per thread per trial             [3000000000]
#   ROUNDS     repeat the whole sweep N times       [1]   (>=2 recommended: lets you
#                                                         separate data effect from drift)
#   ORDER      shuffle | sequential                 [shuffle]  (shuffle spreads thermal
#                                                         / drift effects over all combos)
#   COOLDOWN   seconds to sleep between combos      [5]
#   OUT_CSV    the single results file              [results/sweep_all.csv]
#   DRY_RUN=1  only print the combinations, touch nothing
#
# Resume: finished combinations are listed in <OUT_CSV>.done and are skipped on re-run.

set -u
cd "$(dirname "$(readlink -f "$0")")"

FREQ_KHZ=${FREQ_KHZ:-2400000}
OPS=${OPS:-"mul fma ifma_lo ifma_hi"}
TOGGLES=${TOGGLES:-"0 1"}
TRIALS=${TRIALS:-20}
WARMUP=${WARMUP:-2}
ITERS=${ITERS:-3000000000}
ROUNDS=${ROUNDS:-1}
ORDER=${ORDER:-shuffle}
COOLDOWN=${COOLDOWN:-5}
OUT_CSV=${OUT_CSV:-results/sweep_all.csv}
DRY_RUN=${DRY_RUN:-0}
NUM_CORES=28                       # must match NUM_CORES in bench_data_dep.cpp

SRC=bench_data_dep.cpp
BIN=./bench
DONE_FILE="${OUT_CSV}.done"
LOG_FILE="${OUT_CSV%.csv}.log"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# ----------------------------------------------------------------------
# Operand lists
#   data1 = the fixed operand (outer loop)
#   data2 = the varied operand (inner loop)
# ----------------------------------------------------------------------

# name:float-value   (converted to IEEE-754 single bit patterns below)
D1_FLOAT=("zero:0" "one:1" "prime3:3" "prime97:97" "prime1000003:1000003" "large1e30:1e30")

# name:hex           (52-bit IFMA operands)
D1_INT=("zero:0x0" "one:0x1" "prime3:0x3" "prime65537:0x10001"
        "prime2p31m1:0x7FFFFFFF" "prime2p52m47:0xFFFFFFFFFFFD1" "allones:0xFFFFFFFFFFFFF")

# data2 bit patterns (52-bit). For float ops the low 23 bits become the
# mantissa with sign=0 and exponent=127 (value in [1,2)), so there are no
# NaN / Inf / denormal side effects. "zero" is the literal 0 / 0.0.
D2_PATTERNS=(
  "ones:0xFFFFFFFFFFFFF"       # 1111...
  "alt10:0xAAAAAAAAAAAAA"      # 1010...
  "alt01:0x5555555555555"      # 0101...
  "pair0011:0x3333333333333"   # 0011 0011...
  "pair1100:0xCCCCCCCCCCCCC"   # 1100 1100...
  "nib0F:0x0F0F0F0F0F0F0"      # 00001111...
  "nibF0:0xF0F0F0F0F0F0F"      # 11110000...
  "byte00FF:0x00FF00FF00FF0"
  "low26:0x0000003FFFFFF"      # lower half ones
  "high26:0xFFFFFFC000000"     # upper half ones
  "lsb:0x0000000000001"        # single bit
  "msb:0x8000000000000"        # single bit (bit 51)
)

is_float_op() { [[ $1 == mul || $1 == fma ]]; }

f2hex() {
  python3 -c 'import struct,sys;print("0x%08x"%struct.unpack("<I",struct.pack("<f",float(sys.argv[1])))[0])' "$1"
}

# fills D1_NAMES / D1_VALS for op $1
build_d1() {
  D1_NAMES=(); D1_VALS=()
  local e n v
  if is_float_op "$1"; then
    for e in "${D1_FLOAT[@]}"; do n=${e%%:*}; v=${e#*:}; D1_NAMES+=("$n"); D1_VALS+=("$(f2hex "$v")"); done
  else
    for e in "${D1_INT[@]}"; do n=${e%%:*}; v=${e#*:}; D1_NAMES+=("$n"); D1_VALS+=("$v"); done
  fi
}

# fills D2_NAMES / D2_VALS for op $1 (duplicates removed)
build_d2() {
  D2_NAMES=(); D2_VALS=()
  local -A seen=()
  add() { [[ -n ${seen[$2]:-} ]] && return; seen[$2]=1; D2_NAMES+=("$1"); D2_VALS+=("$2"); }
  local e n v
  if is_float_op "$1"; then
    add zero 0x00000000
    add one_1.0 0x3f800000
    for e in "${D2_PATTERNS[@]}"; do
      n=${e%%:*}; v=${e#*:}
      add "$n" "$(printf '0x%08x' $(( 0x3F800000 | (v & 0x7FFFFF) )))"
    done
  else
    add zero 0x0
    for e in "${D2_PATTERNS[@]}"; do
      n=${e%%:*}; v=${e#*:}
      add "$n" "$(printf '0x%x' $((v)))"
    done
  fi
}

# ----------------------------------------------------------------------
# Build the list of combinations
# ----------------------------------------------------------------------
HAS_IFMA=0
grep -qE 'avx512ifma|avx_ifma' /proc/cpuinfo 2>/dev/null && HAS_IFMA=1

COMBOS=()
for round in $(seq 1 "$ROUNDS"); do
  for op in $OPS; do
    if [[ $op == ifma_* && $HAS_IFMA -eq 0 ]]; then
      log "skipping $op (no avx512ifma / avx_ifma in /proc/cpuinfo)"; continue
    fi
    build_d1 "$op"; build_d2 "$op"
    for tog in $TOGGLES; do
      for i in "${!D1_NAMES[@]}"; do
        for j in "${!D2_NAMES[@]}"; do
          COMBOS+=("$round|$op|${D1_NAMES[$i]}|${D1_VALS[$i]}|${D2_NAMES[$j]}|${D2_VALS[$j]}|$tog")
        done
      done
    done
  done
done

if [[ $ORDER == shuffle ]]; then
  mapfile -t COMBOS < <(printf '%s\n' "${COMBOS[@]}" | shuf)
fi

TOTAL=${#COMBOS[@]}
SECS_PER=$(( (TRIALS + WARMUP) * 2 + COOLDOWN ))   # ~1-2 s per 3e9-op trial; rough
log "$TOTAL combinations, ~$(( TOTAL * SECS_PER / 60 )) min (rough), freq=${FREQ_KHZ} kHz, out=$OUT_CSV"

if [[ $DRY_RUN == 1 ]]; then
  printf '%s\n' "${COMBOS[@]}"
  exit 0
fi

[[ $EUID -eq 0 ]] || die "run as root (RAPL + cpufreq need it): sudo $0"
command -v python3 >/dev/null || die "python3 needed for float -> hex conversion"
mkdir -p "$(dirname "$OUT_CSV")"

# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
if [[ ! -x $BIN || $SRC -nt $BIN ]]; then
  log "compiling $SRC"
  g++ -O2 -mavx2 -mfma -pthread "$SRC" -o "$BIN" || die "compile failed"
fi
(( $(nproc --all) >= NUM_CORES )) || log "WARNING: fewer than $NUM_CORES CPUs; bench pins threads to cpu0..cpu$((NUM_CORES-1))"

# ----------------------------------------------------------------------
# Fix the CPU frequency (saved + restored on exit)
# ----------------------------------------------------------------------
CPUFREQ_DIRS=(/sys/devices/system/cpu/cpu[0-9]*/cpufreq)
NO_TURBO=/sys/devices/system/cpu/intel_pstate/no_turbo
BOOST=/sys/devices/system/cpu/cpufreq/boost
declare -a SAVED=()
SAVED_NO_TURBO=""; SAVED_BOOST=""

save_state() {
  local d
  for d in "${CPUFREQ_DIRS[@]}"; do
    SAVED+=("$d|$(<"$d/scaling_governor")|$(<"$d/scaling_min_freq")|$(<"$d/scaling_max_freq")")
  done
  [[ -f $NO_TURBO ]] && SAVED_NO_TURBO=$(<"$NO_TURBO")
  [[ -f $BOOST    ]] && SAVED_BOOST=$(<"$BOOST")
}

restore_state() {
  log "restoring cpufreq settings"
  local e d gov mn mx
  for e in "${SAVED[@]}"; do
    IFS='|' read -r d gov mn mx <<<"$e"
    cat "$d/cpuinfo_min_freq" > "$d/scaling_min_freq" 2>/dev/null
    echo "$mx"  > "$d/scaling_max_freq" 2>/dev/null
    echo "$mn"  > "$d/scaling_min_freq" 2>/dev/null
    echo "$gov" > "$d/scaling_governor" 2>/dev/null
  done
  [[ -n $SAVED_NO_TURBO ]] && echo "$SAVED_NO_TURBO" > "$NO_TURBO"
  [[ -n $SAVED_BOOST    ]] && echo "$SAVED_BOOST"    > "$BOOST"
}

set_fixed_freq() {
  local d
  [[ -f $NO_TURBO ]] && echo 1 > "$NO_TURBO"
  [[ -f $BOOST    ]] && echo 0 > "$BOOST"
  for d in "${CPUFREQ_DIRS[@]}"; do
    echo performance > "$d/scaling_governor"
    cat "$d/cpuinfo_min_freq" > "$d/scaling_min_freq"     # make room first
    echo "$FREQ_KHZ" > "$d/scaling_max_freq"
    echo "$FREQ_KHZ" > "$d/scaling_min_freq"
  done
}

verify_freq() {
  # spin on every bench core for 3 s and read the resulting frequency
  local c worst=0 f dev
  for c in $(seq 0 $((NUM_CORES-1))); do
    taskset -c "$c" timeout 3 bash -c 'while :; do :; done' &
  done
  sleep 2
  for c in $(seq 0 $((NUM_CORES-1))); do
    f=$(<"/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq")
    dev=$(( f > FREQ_KHZ ? f - FREQ_KHZ : FREQ_KHZ - f ))
    (( dev > worst )) && worst=$dev
  done
  wait 2>/dev/null
  if (( worst > 50000 )); then
    log "WARNING: a core is ${worst} kHz away from the target ${FREQ_KHZ} kHz - check cpufreq driver/HWP/power limits"
  else
    log "frequency verified: all cores within $((worst/1000)) MHz of target"
  fi
}

[[ -d ${CPUFREQ_DIRS[0]} ]] || die "no cpufreq sysfs found (VM or cpufreq driver disabled?)"
save_state
trap restore_state EXIT
trap 'exit 130' INT TERM
set_fixed_freq
verify_freq

# ----------------------------------------------------------------------
# Run the sweep
# ----------------------------------------------------------------------
touch "$DONE_FILE"
n=0
for combo in "${COMBOS[@]}"; do
  n=$((n+1))
  IFS='|' read -r round op d1n d1v d2n d2v tog <<<"$combo"

  if grep -qxF "$combo" "$DONE_FILE"; then
    continue
  fi

  log "[$n/$TOTAL] r$round $op toggle=$tog data1=$d1n($d1v) data2=$d2n($d2v)"
  if BENCH_OUT="$OUT_CSV" BENCH_TRIALS="$TRIALS" BENCH_WARMUP="$WARMUP" \
     BENCH_ITERS="$ITERS" BENCH_ROUND="$round" BENCH_D1_NAME="$d1n" \
     BENCH_D2_NAME="$d2n" BENCH_FREQ_KHZ="$FREQ_KHZ" \
     "$BIN" "$op" "$d1v" "$d2v" "$tog" 2>>"$LOG_FILE"; then
    echo "$combo" >> "$DONE_FILE"
  else
    log "FAILED: $combo (see $LOG_FILE)"
  fi
  sleep "$COOLDOWN"
done

log "done. results: $OUT_CSV"