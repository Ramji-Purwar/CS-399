#!/usr/bin/env bash
#
# find_throttle_freq.sh -- show that the HARDWARE lowers its own clock for the
# HIGH-power operands but not for the LOW-power ones, at stock power limits.
#
#   sudo ./find_throttle_freq.sh
#
# Idea
#   The frequency is never pinned. Both operand sets are run under exactly the
#   same OS request -- governor "performance", turbo enabled, the full hardware
#   min..max range left open -- so the OS asks for maximum speed in both cases
#   and every difference in the achieved clock comes from the CPU's own power
#   management reacting to how much power the operands actually burn:
#     LOW  = data2 = 0,        toggle 0   (draws little power)
#     HIGH = data2 = 1010..,   toggle 1   (draws more power)
#   If HIGH needs more than the package limit (PL1) allows, the PCU drops the
#   clock for HIGH only -> HIGH runs slower. That slowdown is the result.
#
#   Pinning min=max to a fixed value (what this script used to do) is the one
#   thing that must NOT happen: it makes US choose the operating point, so the
#   runtime is fixed by construction and the hardware's own decision -- the
#   whole effect being measured -- is hidden.
#
#   Load configurations (CONFIGS) are BENCH_TOGGLE_A values:
#     0  only data2 toggles
#     1  data1 toggles in step with data2 (both multiplier inputs flip)
#
#   The memory-operand ("heavy") kernel was removed: L1 load traffic pinned the
#   part to a fixed 2600 MHz for BOTH operand sets, which destroys the
#   comparison. Everything is register-only ymm FMA.
#
# Tunables (env vars):
#   OP          fma | mul | ifma_lo | ifma_hi                  [fma]
#   CONFIGS     space-separated BENCH_TOGGLE_A values          ["0 1"]
#   BENCH_C     addend hex; non-zero raises dest-register flips [0]
#   TRIALS      measured trials per run                        [5]
#   WARMUP      warm-up runs per run                           [1]
#   ITERS       ops per thread per trial                       [3000000000]
#   ROUNDS      repeats of LOW/HIGH per config                 [3]
#   COOLDOWN    seconds between runs                           [10]
#   MAX_TEMP_C  wait until package temp is below this          [60]  (0 = skip)
#   PREHEAT_TEMP_C  warm the package to this temp before       [0]   (0 = skip)
#                   measuring; leakage rises with temperature, so a cold
#                   package understates power by several watts. Keep it well
#                   under TJmax, or thermal throttling hits BOTH sides and
#                   the comparison is lost. Set MAX_TEMP_C=0 when using this.
#   PREHEAT_MAX_S   give up preheating after this long         [180]
#   D1 / D2_LOW / D2_HIGH  override the operands (e.g. a rank_operands.sh winner)
#   THRESH_PCT  timing difference (%) to flag                  [1.0]
#   FREQ_TOL    flag a clock gap when H < (1-tol)*L            [0.01]
#   OUT_CSV     bench results                                  [results/find_throttle.csv]
#   KEEP=1      keep old CSVs instead of starting fresh
#
set -u
cd "$(dirname "$(readlink -f "$0")")"

OP=${OP:-fma}
CONFIGS=${CONFIGS:-"0 1"}
BENCH_C=${BENCH_C:-0}
TRIALS=${TRIALS:-5}
WARMUP=${WARMUP:-1}
ITERS=${ITERS:-3000000000}
ROUNDS=${ROUNDS:-3}
COOLDOWN=${COOLDOWN:-10}
MAX_TEMP_C=${MAX_TEMP_C:-60}
PREHEAT_TEMP_C=${PREHEAT_TEMP_C:-0}
PREHEAT_MAX_S=${PREHEAT_MAX_S:-180}
THRESH_PCT=${THRESH_PCT:-1.0}
FREQ_TOL=${FREQ_TOL:-0.01}
OUT_CSV=${OUT_CSV:-results/find_throttle.csv}
KEEP=${KEEP:-0}
NUM_CORES=56                       # must match bench_data_dep.cpp

SRC=bench_data_dep.cpp
BIN=./bench
FREQ_CSV="${OUT_CSV%.csv}_actualfreq.csv"
LOG_FILE="${OUT_CSV%.csv}.log"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root: sudo $0"
command -v python3 >/dev/null || die "python3 needed"
command -v awk >/dev/null || die "awk needed"
mkdir -p "$(dirname "$OUT_CSV")"

CPUFREQ_DIRS=(/sys/devices/system/cpu/cpu[0-9]*/cpufreq)
[[ -d ${CPUFREQ_DIRS[0]} ]] || die "no cpufreq sysfs found"
CPU0=/sys/devices/system/cpu/cpu0/cpufreq
START_KHZ=$(<"$CPU0/cpuinfo_max_freq")

# an old bench_data_dep.cpp has no load knobs -> only the baseline config works
if ! grep -q BENCH_TOGGLE_A "$SRC" 2>/dev/null; then
  log "WARNING: $SRC has no BENCH_TOGGLE_A - using the baseline config only"
  CONFIGS="0"
fi

# ---------------- operand sets ----------------
# D1 / D2_HIGH / BENCH_C can be overridden to feed in a winner from
# rank_operands.sh, which searches for the operands that burn the most power.
if [[ $OP == mul || $OP == fma ]]; then
  D1=${D1:-$(python3 -c 'import struct;print("0x%08x"%struct.unpack("<I",struct.pack("<f",123.456))[0])')}
  D2_LOW=${D2_LOW:-0x00000000}
  D2_HIGH=${D2_HIGH:-$(printf '0x%08x' $(( 0x3F800000 | (0xAAAAAAAAAAAAA & 0x7FFFFF) )))}
else
  D1=${D1:-0x12345678}
  D2_LOW=${D2_LOW:-0x0}
  D2_HIGH=${D2_HIGH:-0xAAAAAAAAAAAAA}
fi
LABELS=(low high)
D2S=("$D2_LOW" "$D2_HIGH")
TOGS=(0 1)

# ---------------- power limit (info only; never modified) ----------------
RAPL=/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0
PL1_W=0
for c in 0 1; do
  if [[ -f $RAPL/constraint_${c}_power_limit_uw ]]; then
    nm=$(<"$RAPL/constraint_${c}_name"); lim=$(( $(<"$RAPL/constraint_${c}_power_limit_uw") / 1000000 ))
    log "RAPL $nm limit = $lim W"
    [[ $nm == long_term ]] && PL1_W=$lim
  fi
done

# ---------------- build ----------------
if [[ ! -x $BIN || $SRC -nt $BIN ]]; then
  log "compiling $SRC"
  g++ -O2 -mavx2 -mfma -pthread "$SRC" -o "$BIN" || die "compile failed"
fi

# ---------------- cpufreq save/restore ----------------
NO_TURBO=/sys/devices/system/cpu/intel_pstate/no_turbo
BOOST=/sys/devices/system/cpu/cpufreq/boost
declare -a SAVED=()
SAVED_NO_TURBO=""; SAVED_BOOST=""
for d in "${CPUFREQ_DIRS[@]}"; do
  SAVED+=("$d|$(<"$d/scaling_governor")|$(<"$d/scaling_min_freq")|$(<"$d/scaling_max_freq")")
done
[[ -f $NO_TURBO ]] && SAVED_NO_TURBO=$(<"$NO_TURBO")
[[ -f $BOOST    ]] && SAVED_BOOST=$(<"$BOOST")

SAMPLER_PID=""
restore_state() {
  [[ -n $SAMPLER_PID ]] && kill "$SAMPLER_PID" 2>/dev/null
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
trap restore_state EXIT
trap 'exit 130' INT TERM

# Turbo must be allowed, otherwise frequencies above base cannot be requested.
[[ -f $NO_TURBO ]] && echo 0 > "$NO_TURBO"
[[ -f $BOOST    ]] && echo 1 > "$BOOST"

# Ask for maximum speed and leave the whole hardware range open, so the CPU --
# not us -- picks the operating point. governor=performance makes the OS request
# the top P-state; scaling_min stays at the hardware minimum so the PCU is free
# to drop the clock as far as the power limit requires. Identical for LOW and
# HIGH, so any clock difference between them is the hardware's own doing.
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
    if [[ $(<"$z/type") == x86_pkg_temp ]]; then
      echo $(( $(<"$z/temp") / 1000 )); return
    fi
  done
  echo 0
}

wait_cool() {
  (( MAX_TEMP_C > 0 )) || return 0
  local t waited=0
  while :; do
    t=$(pkg_temp_c)
    (( t == 0 || t <= MAX_TEMP_C )) && return 0
    (( waited % 30 == 0 )) && log "  waiting for cooldown: pkg ${t}C > ${MAX_TEMP_C}C"
    sleep 5; waited=$((waited+5))
    (( waited > 600 )) && { log "  cooldown timeout, continuing"; return 0; }
  done
}

# Warm the package up to a hot steady state before measuring.
#
# Transistor leakage rises with temperature, so a cold package draws measurably
# less than the same work does once hot -- measuring from cold understates the
# power by several watts and can hide a crossing of PL1 entirely. Preheating
# with the HIGH operands puts the part in the state the real comparison needs.
#
# PREHEAT_TEMP_C has to stay well under TJmax: if the package reaches its
# THERMAL limit then both operand sets throttle and the comparison is lost. We
# want the power limit to be the only thing that ever throttles.
preheat() {
  (( PREHEAT_TEMP_C > 0 )) || return 0
  local t waited=0 t0
  t0=$(pkg_temp_c)
  (( t0 == 0 )) && { log "no package temperature sensor, skipping preheat"; return 0; }
  log "preheating to ${PREHEAT_TEMP_C}C (from ${t0}C, cap ${PREHEAT_MAX_S}s)"
  while :; do
    t=$(pkg_temp_c)
    (( t >= PREHEAT_TEMP_C )) && { log "  preheat done: pkg ${t}C"; return 0; }
    (( waited >= PREHEAT_MAX_S )) && { log "  preheat timeout at ${t}C, continuing"; return 0; }
    BENCH_OUT=/dev/null BENCH_TRIALS=1 BENCH_WARMUP=0 BENCH_ITERS="$ITERS" \
    BENCH_TOGGLE_A=1 BENCH_C="$BENCH_C" \
      "$BIN" "$OP" "$D1" "$D2_HIGH" 1 >/dev/null 2>&1
    waited=$((waited+2))
    (( waited % 10 == 0 )) && log "  preheating: pkg ${t}C"
  done
}

# Background sampler: average scaling_cur_freq across the bench cores.
#
# Two things make this fiddly. The bench occupies every physical core, so an
# unpinned sampler is starved and can emit nothing at all; it is therefore
# pinned to a hyperthread sibling outside the bench set (cpu>=NUM_CORES), which
# shares a core but costs it almost nothing, and equally for LOW and HIGH so it
# cancels. And the interval has to stay well under one trial, so a short run
# still yields samples -- hence the stride, which keeps one pass cheap.
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

# run the sampler off the bench cores when there is a spare sibling to use
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

# ---------------- fresh output ----------------
if [[ $KEEP != 1 ]]; then rm -f "$OUT_CSV" "$FREQ_CSV" "$LOG_FILE"; fi
[[ -s $FREQ_CSV ]] || echo "cfg,freq_khz,label,round,actual_mean_khz,actual_min_khz" > "$FREQ_CSV"

# ---------------- open the clock range, then measure ----------------
release_freq
sleep 2
log "frequency NOT pinned: governor=performance, turbo on, range left at the"
log "hardware min..max ($(<"$CPU0/cpuinfo_min_freq")..${START_KHZ} kHz). The CPU picks its own clock."
log "op=$OP configs(toggle_a): $CONFIGS  c=$BENCH_C  PL1=${PL1_W}W"
log "LOW : data1=$D1 data2=$D2_LOW toggle=0"
log "HIGH: data1=$D1 data2=$D2_HIGH toggle=1"

# ---------------- sweep ----------------
TMP=$(mktemp)
for cfg in $CONFIGS; do
  TA=$cfg
  tag="a${TA}"
  log "##### config $tag  (BENCH_TOGGLE_A=$TA) #####"
  preheat
    for (( r=1; r<=ROUNDS; r++ )); do
      # alternate order each round so thermal carry-over doesn't favour one side
      if (( r % 2 )); then order=(0 1); else order=(1 0); fi
      for i in "${order[@]}"; do
        lbl=${LABELS[$i]}; d2=${D2S[$i]}; tog=${TOGS[$i]}
        wait_cool
        log "  $tag r=$r $lbl data2=$d2 toggle=$tog"
        start_sampler "$TMP"
        BENCH_OUT="$OUT_CSV" BENCH_TRIALS="$TRIALS" BENCH_WARMUP="$WARMUP" \
        BENCH_ITERS="$ITERS" BENCH_ROUND="$r" BENCH_D1_NAME="$tag" \
        BENCH_D2_NAME="$lbl" BENCH_FREQ_KHZ=0 \
        BENCH_TOGGLE_A="$TA" BENCH_C="$BENCH_C" \
          "$BIN" "$OP" "$D1" "$d2" "$tog" 2>>"$LOG_FILE" \
          || log "  FAILED (see $LOG_FILE)"
        kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
        # use the second half of the samples (skip warm-up), report mean and min
        # Second half of the samples only, so the clock ramp is excluded -- but
        # with very few samples use them all rather than discarding half.
        awk -v c="$tag" -v f=0 -v l="$lbl" -v r="$r" '
          {a[NR]=$1}
          END{ if(NR==0){exit}
               s=(NR<4) ? 1 : int(NR/2)+1; n=0; sum=0; mn=a[s]
               for(i=s;i<=NR;i++){sum+=a[i]; n++; if(a[i]<mn)mn=a[i]}
               printf "%s,%s,%s,%s,%.0f,%d\n", c,f,l,r,sum/n,mn }' "$TMP" >> "$FREQ_CSV"
        sleep "$COOLDOWN"
      done
    done
done
rm -f "$TMP"

# ---------------- analysis ----------------
log "analysing"
python3 - "$OUT_CSV" "$FREQ_CSV" "$THRESH_PCT" "$FREQ_TOL" "$PL1_W" <<'PY'
import csv, sys, statistics as st
from collections import defaultdict
bench, fcsv = sys.argv[1], sys.argv[2]
thr, tol, pl1 = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

T = defaultdict(list); W = defaultdict(list); A = defaultdict(list); cfgs = []
for r in csv.DictReader(open(bench)):
    k = (r['d1_name'], r['d2_name'])
    T[k].append(float(r['mean_time_s'])); W[k].append(float(r['pkg_W']))
    if r['d1_name'] not in cfgs: cfgs.append(r['d1_name'])
for r in csv.DictReader(open(fcsv)):
    A[(r['cfg'], r['label'])].append(float(r['actual_mean_khz']))

# Nothing was pinned, so there is no "requested" clock to compare against: the
# reference for each config is the clock the hardware chose for LOW, and the
# question is whether it chose a lower one for HIGH.
print(f"\nPL1 (long-term limit) = {pl1:.0f} W   (power limits and frequency BOTH left at stock)")
print("Both sides ran under an identical OS request, so any clock gap is the CPU's own doing.\n")
hdr = (f"{'cfg':>5} | {'clk L':>7} {'clk H':>7} {'dclk %':>7} | {'time L (s)':>10} {'time H (s)':>10} "
       f"{'dt %':>6} | {'W low':>6} {'W high':>6} | flags")
print(hdr); print("-" * len(hdr))

good = []; all_wh = []; all_wl = []
for cfg in cfgs:
    kl, kh = (cfg, 'low'), (cfg, 'high')
    if kl not in T or kh not in T: continue
    tl, th = st.mean(T[kl]), st.mean(T[kh])
    wl, wh = st.mean(W[kl]), st.mean(W[kh])
    all_wl.append(wl); all_wh.append(wh)
    diff = (th - tl) / tl * 100
    sem = ((st.pstdev(T[kl]) ** 2 + st.pstdev(T[kh]) ** 2) ** 0.5 / (len(T[kl]) ** 0.5)) / tl * 100
    al = st.mean(A[kl]) if A[kl] else float('nan')
    ah = st.mean(A[kh]) if A[kh] else float('nan')
    dclk = (ah - al) / al * 100 if al else float('nan')
    clock_gap = ah < al * (1 - tol)          # hardware chose a lower clock for HIGH
    td = diff > thr and diff > 3 * sem
    flags = []
    if pl1 > 0 and wl > pl1: flags.append("LOW>PL1")
    if pl1 > 0 and wh > pl1: flags.append("HIGH>PL1")
    if clock_gap: flags.append("HW DOWNCLOCKED HIGH")
    if td: flags.append("TIMING DIFF")
    if clock_gap and td:
        flags.append("<== GOOD"); good.append((cfg, diff, dclk))
    print(f"{cfg:>5} | {al/1000:7.0f} {ah/1000:7.0f} {dclk:7.2f} | {tl:10.4f} {th:10.4f} {diff:6.2f} | "
          f"{wl:6.1f} {wh:6.1f} | {', '.join(flags)}")

print()
if good:
    cfg, d, dclk = max(good, key=lambda x: x[1])
    print(f"The hardware throttled itself on the HIGH operands: config {cfg} ran "
          f"{dclk:.2f}% slower in clock and {d:.2f}% longer in time, with nothing pinned.")
    print(f"Run the main sweep with:  sudo BENCH_TOGGLE_A={cfg[1]} ./run_sweep_throttling.sh")
else:
    print("The hardware picked the same clock for both operand sets, so nothing throttled.")
    if pl1 > 0 and all_wh and max(all_wh) <= pl1:
        print(f"HIGH never exceeded PL1 ({pl1:.0f} W; max seen {max(all_wh):.1f} W): "
              f"short by {pl1 - max(all_wh):.1f} W, so the PCU had no reason to downclock.")
        print("  -> ./rank_operands.sh picks HIGH operands that flip more bits per op,")
        print("     and BENCH_C=<hex> raises the dest-register flip count.")
        print("  -> ITERS=12000000000 and MAX_TEMP_C=0 also help: PL1 is a ~1 s rolling")
        print("     average, so a short run can finish before the limit even engages.")
    elif pl1 > 0 and all_wl and min(all_wl) > pl1:
        print(f"LOW already exceeds PL1 ({pl1:.0f} W) everywhere, so both sides throttle "
              f"and the comparison is lost.")
        print("  -> use CONFIGS=0, or a lower-power LOW operand.")
    else:
        print("HIGH crosses PL1 but the clock did not move: raise ITERS so the ~1 s PL1")
        print("average fills, and MAX_TEMP_C=0 to keep the package hot.")
PY
log "done. bench: $OUT_CSV  actual-freq: $FREQ_CSV"