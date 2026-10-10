#!/usr/bin/env bash
#
# find_throttle_power.sh -- find the RAPL package power limit at which the
# HIGH-power operands get throttled (clock drops -> slower) while the
# LOW-power operands do not.
#
#   sudo ./find_throttle_power.sh
#
# Why: on this machine every request >= 3.0 GHz runs at 3.0 GHz (AVX all-core
# ceiling). At 3.0 GHz LOW draws ~175 W and HIGH ~186 W, and the default
# limits (PL1=185 W, PL2=222 W) never bite. So the frequency is fixed at the
# ceiling and the POWER LIMIT is swept instead, in the window between the
# LOW and HIGH power draw.
#
# Tunables (env vars):
#   OP          fma | mul | ifma_lo | ifma_hi                 [fma]
#   FREQ_KHZ    pinned frequency                              [3000000]
#   PL_LIST     power limits (W) to try, space separated      ["186 184 182 180 178 176 174"]
#   WINDOW_US   long-term time window written to RAPL         [1000000]  (0 = leave unchanged)
#   TRIALS / WARMUP / ITERS / ROUNDS / COOLDOWN               [5 / 1 / 3e9 / 3 / 10]
#   MAX_TEMP_C  wait until package temp is below this         [60] (0 = skip)
#   THRESH_PCT  timing difference (%) to flag                 [1.0]
#   FREQ_TOL    flag throttling if actual < (1-tol)*pinned    [0.01]
#   OUT_CSV     bench results                                 [results/find_throttle_power.csv]
#   KEEP=1      keep old CSVs
#
set -u
cd "$(dirname "$(readlink -f "$0")")"

OP=${OP:-fma}
FREQ_KHZ=${FREQ_KHZ:-3000000}
PL_LIST=${PL_LIST:-"186 184 182 180 178 176 174"}
WINDOW_US=${WINDOW_US:-1000000}
TRIALS=${TRIALS:-5}
WARMUP=${WARMUP:-1}
ITERS=${ITERS:-3000000000}
ROUNDS=${ROUNDS:-3}
COOLDOWN=${COOLDOWN:-10}
MAX_TEMP_C=${MAX_TEMP_C:-60}
THRESH_PCT=${THRESH_PCT:-1.0}
FREQ_TOL=${FREQ_TOL:-0.01}
OUT_CSV=${OUT_CSV:-results/find_throttle_power.csv}
KEEP=${KEEP:-0}
NUM_CORES=28                       # must match bench_data_dep.cpp

SRC=bench_data_dep.cpp
BIN=./bench
FREQ_CSV="${OUT_CSV%.csv}_actualfreq.csv"
LOG_FILE="${OUT_CSV%.csv}.log"
RAPL=/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root: sudo $0"
command -v python3 >/dev/null || die "python3 needed"
command -v awk >/dev/null || die "awk needed"
[[ -f $RAPL/constraint_0_power_limit_uw ]] || die "no RAPL powercap at $RAPL"
mkdir -p "$(dirname "$OUT_CSV")"

CPUFREQ_DIRS=(/sys/devices/system/cpu/cpu[0-9]*/cpufreq)
[[ -d ${CPUFREQ_DIRS[0]} ]] || die "no cpufreq sysfs found"

# ---------------- operand sets ----------------
if [[ $OP == mul || $OP == fma ]]; then
  D1=$(python3 -c 'import struct;print("0x%08x"%struct.unpack("<I",struct.pack("<f",123.456))[0])')
  D2_LOW=0x00000000
  D2_HIGH=$(printf '0x%08x' $(( 0x3F800000 | (0xAAAAAAAAAAAAA & 0x7FFFFF) )))
else
  D1=0x12345678
  D2_LOW=0x0
  D2_HIGH=0xAAAAAAAAAAAAA
fi
LABELS=(low high)
D2S=("$D2_LOW" "$D2_HIGH")
TOGS=(0 1)

log "op=$OP  freq=$FREQ_KHZ kHz  power limits (W): $PL_LIST"
log "LOW : data2=$D2_LOW toggle=0   HIGH: data2=$D2_HIGH toggle=1"

# ---------------- build ----------------
if [[ ! -x $BIN || $SRC -nt $BIN ]]; then
  log "compiling $SRC"
  g++ -O2 -mavx2 -mfma -pthread "$SRC" -o "$BIN" || die "compile failed"
fi

# ---------------- save / restore (cpufreq + RAPL) ----------------
NO_TURBO=/sys/devices/system/cpu/intel_pstate/no_turbo
BOOST=/sys/devices/system/cpu/cpufreq/boost
declare -a SAVED=()
SAVED_NO_TURBO=""; SAVED_BOOST=""
for d in "${CPUFREQ_DIRS[@]}"; do
  SAVED+=("$d|$(<"$d/scaling_governor")|$(<"$d/scaling_min_freq")|$(<"$d/scaling_max_freq")")
done
[[ -f $NO_TURBO ]] && SAVED_NO_TURBO=$(<"$NO_TURBO")
[[ -f $BOOST    ]] && SAVED_BOOST=$(<"$BOOST")

declare -A PL_SAVED=() TW_SAVED=()
for c in 0 1; do
  [[ -f $RAPL/constraint_${c}_power_limit_uw ]] || continue
  PL_SAVED[$c]=$(<"$RAPL/constraint_${c}_power_limit_uw")
  [[ -f $RAPL/constraint_${c}_time_window_us ]] && TW_SAVED[$c]=$(<"$RAPL/constraint_${c}_time_window_us")
  log "RAPL $(<"$RAPL/constraint_${c}_name"): ${PL_SAVED[$c]} uW, window ${TW_SAVED[$c]:-?} us (saved)"
done

SAMPLER_PID=""
restore_state() {
  [[ -n $SAMPLER_PID ]] && kill "$SAMPLER_PID" 2>/dev/null
  log "restoring RAPL + cpufreq settings"
  local c e d gov mn mx
  for c in "${!PL_SAVED[@]}"; do
    echo "${PL_SAVED[$c]}" > "$RAPL/constraint_${c}_power_limit_uw" 2>/dev/null
    [[ -n ${TW_SAVED[$c]:-} ]] && echo "${TW_SAVED[$c]}" > "$RAPL/constraint_${c}_time_window_us" 2>/dev/null
  done
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

[[ -f $NO_TURBO ]] && echo 0 > "$NO_TURBO"
[[ -f $BOOST    ]] && echo 1 > "$BOOST"
for d in "${CPUFREQ_DIRS[@]}"; do
  echo performance > "$d/scaling_governor"
  cat "$d/cpuinfo_min_freq" > "$d/scaling_min_freq"
  echo "$FREQ_KHZ" > "$d/scaling_max_freq"
  echo "$FREQ_KHZ" > "$d/scaling_min_freq"
done

set_pl() {   # $1 = watts; sets long- AND short-term limits so PL2 can't hide it
  local uw=$(( $1 * 1000000 )) c got
  for c in 0 1; do
    [[ -f $RAPL/constraint_${c}_power_limit_uw ]] || continue
    echo "$uw" > "$RAPL/constraint_${c}_power_limit_uw" 2>/dev/null
    got=$(<"$RAPL/constraint_${c}_power_limit_uw")
    if [[ $got != "$uw" ]]; then
      log "  WARNING: constraint_$c not set (wanted $uw, got $got) - limit may be locked by BIOS"
      return 1
    fi
  done
  if (( WINDOW_US > 0 )) && [[ -f $RAPL/constraint_0_time_window_us ]]; then
    echo "$WINDOW_US" > "$RAPL/constraint_0_time_window_us" 2>/dev/null
  fi
  return 0
}

pkg_temp_c() {
  local z
  for z in /sys/class/thermal/thermal_zone*; do
    if [[ $(<"$z/type") == x86_pkg_temp ]]; then echo $(( $(<"$z/temp") / 1000 )); return; fi
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

sample_loop() {
  local out=$1 c s n f
  : > "$out"
  while :; do
    s=0; n=0
    for (( c=0; c<NUM_CORES; c++ )); do
      f=$(<"/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq") 2>/dev/null || continue
      s=$((s+f)); n=$((n+1))
    done
    (( n > 0 )) && echo $(( s / n )) >> "$out"
    sleep 0.5
  done
}

if [[ $KEEP != 1 ]]; then rm -f "$OUT_CSV" "$FREQ_CSV" "$LOG_FILE"; fi
[[ -s $FREQ_CSV ]] || echo "pl_w,label,round,actual_mean_khz,actual_min_khz" > "$FREQ_CSV"

# ---------------- sweep ----------------
TMP=$(mktemp)
for pl in $PL_LIST; do
  log "=== power limit ${pl} W (freq pinned at $FREQ_KHZ kHz) ==="
  set_pl "$pl" || log "  continuing anyway"
  sleep 2
  for (( r=1; r<=ROUNDS; r++ )); do
    if (( r % 2 )); then order=(0 1); else order=(1 0); fi
    for i in "${order[@]}"; do
      lbl=${LABELS[$i]}; d2=${D2S[$i]}; tog=${TOGS[$i]}
      wait_cool
      log "  PL=${pl}W r=$r $lbl data2=$d2 toggle=$tog"
      sample_loop "$TMP" & SAMPLER_PID=$!
      BENCH_OUT="$OUT_CSV" BENCH_TRIALS="$TRIALS" BENCH_WARMUP="$WARMUP" \
      BENCH_ITERS="$ITERS" BENCH_ROUND="$r" BENCH_D1_NAME="pl_${pl}" \
      BENCH_D2_NAME="$lbl" BENCH_FREQ_KHZ="$FREQ_KHZ" \
        "$BIN" "$OP" "$D1" "$d2" "$tog" 2>>"$LOG_FILE" \
        || log "  FAILED (see $LOG_FILE)"
      kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
      awk -v p="$pl" -v l="$lbl" -v r="$r" '
        {a[NR]=$1}
        END{ if(NR==0){exit}
             s=int(NR/2)+1; n=0; sum=0; mn=a[s]
             for(i=s;i<=NR;i++){sum+=a[i]; n++; if(a[i]<mn)mn=a[i]}
             printf "%s,%s,%s,%.0f,%d\n", p,l,r,sum/n,mn }' "$TMP" >> "$FREQ_CSV"
      sleep "$COOLDOWN"
    done
  done
done
rm -f "$TMP"

# ---------------- analysis ----------------
log "analysing"
python3 - "$OUT_CSV" "$FREQ_CSV" "$THRESH_PCT" "$FREQ_TOL" "$FREQ_KHZ" <<'PY'
import csv, sys, statistics as st
from collections import defaultdict
bench, fcsv, thr, tol, fpin = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

T = defaultdict(list); W = defaultdict(list); A = defaultdict(list)
for r in csv.DictReader(open(bench)):
    k = (int(r['d1_name'].split('_')[1]), r['d2_name'])
    T[k].append(float(r['mean_time_s'])); W[k].append(float(r['pkg_W']))
for r in csv.DictReader(open(fcsv)):
    A[(int(r['pl_w']), r['label'])].append(float(r['actual_mean_khz']))

pls = sorted({k[0] for k in T}, reverse=True)
hdr = f"{'PL (W)':>6} | {'freq L':>7} {'freq H':>7} | {'time L (s)':>10} {'time H (s)':>10} {'diff %':>7} | {'W low':>6} {'W high':>6} | flags"
print("\n" + hdr); print("-" * len(hdr))
good = []
for p in pls:
    kl, kh = (p, 'low'), (p, 'high')
    if kl not in T or kh not in T: continue
    tl, th = st.mean(T[kl]), st.mean(T[kh])
    diff = (th - tl) / tl * 100
    sem = ((st.pstdev(T[kl]) ** 2 + st.pstdev(T[kh]) ** 2) ** 0.5 / (len(T[kl]) ** 0.5)) / tl * 100
    al = st.mean(A[kl]) if A[kl] else float('nan')
    ah = st.mean(A[kh]) if A[kh] else float('nan')
    lt = al < fpin * (1 - tol); ht = ah < fpin * (1 - tol)
    flags = []
    if lt: flags.append("LOW throttled")
    if ht: flags.append("HIGH throttled")
    td = diff > thr and diff > 3 * sem
    if td: flags.append("TIMING DIFF")
    if ht and not lt and td:
        flags.append("<== GOOD: only HIGH throttled"); good.append((p, diff))
    print(f"{p:6d} | {al/1000:7.0f} {ah/1000:7.0f} | {tl:10.4f} {th:10.4f} {diff:7.2f} | "
          f"{st.mean(W[kl]):6.1f} {st.mean(W[kh]):6.1f} | {', '.join(flags)}")

print()
if good:
    best = max(good, key=lambda x: x[1])
    print("Power limits where only HIGH throttles (W):", ", ".join(str(g[0]) for g in good))
    print(f"Largest timing difference: {best[1]:.2f}% at PL = {best[0]} W")
    print(f"Use PL_LIST=\"{best[0]}\" (long- and short-term) and FREQ_KHZ={int(fpin)} for the main sweep.")
else:
    print("No power limit gave a HIGH-only timing difference.")
    print("Try a finer PL_LIST (e.g. \"184 183 182 181 180 179 178\"), more ROUNDS, or check the 'limit may be locked' warning.")
PY
log "done. bench: $OUT_CSV  actual-freq: $FREQ_CSV"