#!/usr/bin/env bash
#
# rank_operands.sh -- find the HIGH operand triple (data1, data2, c) that burns
# the most package power, so HIGH crosses PL1 (185 W) while LOW stays under it.
#
#   sudo ./rank_operands.sh
#
# Why this exists
#   find_throttle_freq.sh showed LOW ~171 W and HIGH ~184 W against a 185 W
#   PL1: HIGH is only ~1 W short, and LOW has ~13 W of headroom. So the job is
#   not "more load" (the memory-operand kernel did that and pinned BOTH sides
#   to 2600 MHz, destroying the comparison) but "operands that switch more bits
#   per op", which raises HIGH only.
#
#   The flip budget per op splits three ways:
#     input_flips   HD(b0,b1) + HD(a0,a1).  FIXED at 184 (or 368 with
#                   BENCH_TOGGLE_A=1): b1 is b0 ^ mantissa_mask, so exactly the
#                   23 mantissa bits flip whatever b0 is. No headroom here.
#     result_flips  HD(r0,r1).   Currently 96 of a possible 256.
#     dest_flips    2*(HD(r0,c)+HD(r1,c)).  Currently 256 of a possible 512,
#                   and it is the largest single term. A non-zero c is what
#                   buys this: the destination swings c -> r -> c each op
#                   instead of 0 -> r -> 0.
#   So result_flips and dest_flips are the levers, and both are set purely by
#   the choice of (data1, data2, c).
#
# Stage 1 (seconds, no root)  screen a grid of candidates via BENCH_STATS_ONLY
#                             and rank by total flips per op.
# Stage 2 (minutes, root)     measure the top TOPN for real at the pinned
#                             all-core AVX ceiling, report pkg W beside LOW.
#
# Tunables (env vars):
#   OP        fma | mul | ifma_lo | ifma_hi          [fma]
#   TOPN      candidates to measure in stage 2       [8]
#   TOGGLE_A  BENCH_TOGGLE_A for the HIGH runs       [1]
#   TRIALS    measured trials per candidate          [5]
#   WARMUP    warm-up runs per candidate             [1]
#   ITERS     ops per thread per trial               [6000000000]
#   COOLDOWN  seconds between measured runs          [8]
#   OUT_CSV   stage-2 bench rows                     [results/rank_operands.csv]
#   STAGE1_ONLY=1   screen and print the table, measure nothing (no root)
#
set -u
cd "$(dirname "$(readlink -f "$0")")"

OP=${OP:-fma}
TOPN=${TOPN:-8}
TOGGLE_A=${TOGGLE_A:-1}
TRIALS=${TRIALS:-5}
WARMUP=${WARMUP:-1}
ITERS=${ITERS:-6000000000}
COOLDOWN=${COOLDOWN:-8}
OUT_CSV=${OUT_CSV:-results/rank_operands.csv}
STAGE1_ONLY=${STAGE1_ONLY:-0}
NUM_CORES=28

SRC=bench_data_dep.cpp
BIN=./bench
LOG_FILE="${OUT_CSV%.csv}.log"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

command -v python3 >/dev/null || die "python3 needed"
mkdir -p "$(dirname "$OUT_CSV")"

if [[ ! -x $BIN || $SRC -nt $BIN ]]; then
  log "compiling $SRC"
  g++ -O2 -mavx2 -mfma -pthread "$SRC" -o "$BIN" || die "compile failed"
fi
grep -q BENCH_STATS_ONLY "$SRC" || die "$SRC has no BENCH_STATS_ONLY screening mode"

is_float_op() { [[ $OP == mul || $OP == fma ]]; }

# ----------------------------------------------------------------------
# Candidate grid
#
# float ops: data2 keeps sign=0 exp=127 (value in [1,2)) so the mantissa
#   toggle can never produce a denormal, Inf or NaN. data1 and c vary in both
#   exponent and mantissa, since the exponent bits count toward the Hamming
#   distances just like the mantissa bits do.
# ----------------------------------------------------------------------
MANTISSAS=(0x000000 0x7FFFFF 0x555555 0x2AAAAB 0x333333 0x4CCCCD
           0x0F0F0F 0x70F0F1 0x123457 0x6DCBA9 0x000001 0x400000)
EXPONENTS=(0x3C 0x3E 0x3F 0x40 0x41 0x43)   # biased exp>>1 nibble pairs below

gen_candidates() {
  if is_float_op; then
    python3 - "$@" <<'PY'
import itertools, sys
# data2: exp=127, mantissa in [1,2) -> 0x3F800000 | m
mans = [0x000000, 0x7FFFFF, 0x555555, 0x2AAAAB, 0x333333,
        0x4CCCCD, 0x0F0F0F, 0x123457, 0x6DCBA9, 0x400000]
# data1 / c: vary exponent too. exp byte values kept moderate so a*b+c stays
# finite and normal.
exps = [0x70, 0x78, 0x7D, 0x7F, 0x81, 0x85, 0x8A]
def f32(sign, exp, man): return (sign << 31) | (exp << 23) | man
d1s, cs = [], []
for e in exps:
    for m in mans:
        d1s.append(f32(0, e, m))
        cs.append(f32(0, e, m))
        cs.append(f32(1, e, m))      # negative c: flips the sign bit too
d2s = [0x3F800000 | m for m in mans]
cs.append(0x00000000)
# de-dup, keep order
def uniq(xs):
    seen=set(); out=[]
    for x in xs:
        if x not in seen: seen.add(x); out.append(x)
    return out
for d1, d2, c in itertools.product(uniq(d1s), uniq(d2s), uniq(cs)):
    print(f"0x{d1:08x} 0x{d2:08x} 0x{c:08x}")
PY
  else
    python3 - <<'PY'
import itertools
mans = [0x0000000000000, 0xFFFFFFFFFFFFF, 0xAAAAAAAAAAAAA, 0x5555555555555,
        0x3333333333333, 0xCCCCCCCCCCCCC, 0x0F0F0F0F0F0F0, 0xF0F0F0F0F0F0F,
        0x123456789ABCD, 0xFEDCBA9876543, 0x0000000000001, 0x8000000000000]
for d1, d2, c in itertools.product(mans, mans, mans):
    print(f"0x{d1:x} 0x{d2:x} 0x{c:x}")
PY
  fi
}

# ----------------------------------------------------------------------
# Stage 1: screen (fast, no root, no measurement)
# ----------------------------------------------------------------------
SCREEN=$(mktemp)
log "stage 1: screening candidates with BENCH_STATS_ONLY (op=$OP toggle_a=$TOGGLE_A)"
n=0
while read -r d1 d2 c; do
  [[ -n $d1 ]] || continue
  stats=$(BENCH_STATS_ONLY=1 BENCH_TOGGLE_A="$TOGGLE_A" BENCH_C="$c" \
            "$BIN" "$OP" "$d1" "$d2" 1 2>/dev/null) || continue
  IFS=, read -r inf resf destf totf <<<"$stats"
  echo "$totf $inf $resf $destf $d1 $d2 $c" >> "$SCREEN"
  n=$((n+1))
done < <(gen_candidates)
log "screened $n candidates"
(( n > 0 )) || die "no candidates screened"

# LOW baseline flips, for reference
LOW_D1=$(python3 -c 'import struct;print("0x%08x"%struct.unpack("<I",struct.pack("<f",123.456))[0])')
if is_float_op; then LOW_D2=0x00000000; else LOW_D2=0x0; fi

# The total saturates (input flips are fixed, and res+dest is conserved at 512
# whenever c sits "between" r0 and r1 in every bit), so the top tier is a wide
# tie. Those ties are NOT physically equivalent: a dest flip is a 256-bit
# register write, a result flip is multiplier-output switching, and the two cost
# different energy. So the shortlist takes one candidate per distinct dest value
# to sample that tradeoff, instead of 8 near-identical winners.
SHORT=$(mktemp)
sort -rn -k1,1 -k4,4 "$SCREEN" | awk -v n="$TOPN" '
  { key = $1 "_" $4 }
  !seen[key]++ { print; c++ }
  c >= n { exit }' > "$SHORT"

echo
echo "Shortlist: top total flips, spread across the result/dest split (op=$OP, toggle_a=$TOGGLE_A):"
printf '%-5s %-12s %-12s %-12s %7s %7s %7s %7s\n' \
       rank data1 data2 c total in res dest
awk '{printf "%-5d %-12s %-12s %-12s %7.0f %7d %7d %7.0f\n", NR, $5, $6, $7, $1, $2, $3, $4}' "$SHORT"
echo
echo "for reference, the operands used so far:"
CUR_D2=$(printf '0x%08x' $(( 0x3F800000 | (0xAAAAAAAAAAAAA & 0x7FFFFF) )))
cur=$(BENCH_STATS_ONLY=1 BENCH_TOGGLE_A="$TOGGLE_A" BENCH_C=0 \
        "$BIN" "$OP" "$LOW_D1" "$CUR_D2" 1 2>/dev/null)
IFS=, read -r ci cr cd ct <<<"$cur"
printf '%-5s %-12s %-12s %-12s %7.0f %7d %7d %7.0f\n' cur "$LOW_D1" "$CUR_D2" 0x0 "$ct" "$ci" "$cr" "$cd"
echo

if [[ $STAGE1_ONLY == 1 ]]; then
  log "STAGE1_ONLY=1, stopping before measurement"
  rm -f "$SCREEN" "$SHORT"; exit 0
fi

# ----------------------------------------------------------------------
# Stage 2: measure the top candidates for real
# ----------------------------------------------------------------------
[[ $EUID -eq 0 ]] || die "stage 2 needs root (RAPL + cpufreq): sudo $0  (or STAGE1_ONLY=1)"

RAPL=/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0
PL1_W=0
for ci in 0 1; do
  if [[ -f $RAPL/constraint_${ci}_power_limit_uw ]]; then
    nm=$(<"$RAPL/constraint_${ci}_name")
    lim=$(( $(<"$RAPL/constraint_${ci}_power_limit_uw") / 1000000 ))
    log "RAPL $nm limit = $lim W"
    [[ $nm == long_term ]] && PL1_W=$lim
  fi
done

CPUFREQ_DIRS=(/sys/devices/system/cpu/cpu[0-9]*/cpufreq)
[[ -d ${CPUFREQ_DIRS[0]} ]] || die "no cpufreq sysfs found"
NO_TURBO=/sys/devices/system/cpu/intel_pstate/no_turbo
BOOST=/sys/devices/system/cpu/cpufreq/boost
declare -a SAVED=()
SAVED_NO_TURBO=""; SAVED_BOOST=""
for d in "${CPUFREQ_DIRS[@]}"; do
  SAVED+=("$d|$(<"$d/scaling_governor")|$(<"$d/scaling_min_freq")|$(<"$d/scaling_max_freq")")
done
[[ -f $NO_TURBO ]] && SAVED_NO_TURBO=$(<"$NO_TURBO")
[[ -f $BOOST    ]] && SAVED_BOOST=$(<"$BOOST")

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
trap restore_state EXIT
trap 'exit 130' INT TERM

[[ -f $NO_TURBO ]] && echo 0 > "$NO_TURBO"
[[ -f $BOOST    ]] && echo 1 > "$BOOST"

# Ask for maximum speed and leave the whole hardware range open, so the CPU --
# not us -- picks the operating point. Pinning min=max would fix the runtime by
# construction and hide the throttling this experiment is looking for.
release_freq() {
  local d
  for d in "${CPUFREQ_DIRS[@]}"; do
    cat "$d/cpuinfo_min_freq" > "$d/scaling_min_freq"
    cat "$d/cpuinfo_max_freq" > "$d/scaling_max_freq"
    echo performance > "$d/scaling_governor"
  done
}

release_freq
sleep 2
log "frequency NOT pinned: governor=performance, turbo on, hardware picks its own clock"

# Background sampler: average scaling_cur_freq over the bench cores.
# Needed because power saturates once a candidate crosses PL1 -- the package
# cannot draw more, so the extra demand shows up as a LOWER clock instead. Above
# PL1 the clock drop, not the wattage, is what separates candidates.
#
# The bench occupies every physical core, so an unpinned sampler is starved and
# emits nothing; it runs on a hyperthread sibling outside the bench set instead.
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-0.05}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-4}
SAMPLER_CPU=$(( $(nproc --all) - 1 ))

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
FREQ_CSV="${OUT_CSV%.csv}_actualfreq.csv"
echo "label,actual_mean_khz,actual_min_khz" > "$FREQ_CSV"
SAMPLER_PID=""
FTMP=$(mktemp)

rm -f "$OUT_CSV"
log "stage 2: measuring LOW baseline + top $TOPN candidates (trials=$TRIALS iters=$ITERS)"

run_one() {   # $1=label $2=d1 $3=d2 $4=c $5=toggle $6=toggle_a
  start_sampler "$FTMP"
  BENCH_OUT="$OUT_CSV" BENCH_TRIALS="$TRIALS" BENCH_WARMUP="$WARMUP" \
  BENCH_ITERS="$ITERS" BENCH_ROUND=1 BENCH_D1_NAME="$1" BENCH_D2_NAME="$1" \
  BENCH_FREQ_KHZ=0 BENCH_TOGGLE_A="$6" BENCH_C="$4" \
    "$BIN" "$OP" "$2" "$3" "$5" 2>>"$LOG_FILE" || log "  FAILED $1"
  kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
  # Second half of the samples only, so the clock ramp is excluded -- but with
  # very few samples use them all rather than discarding half.
  awk -v l="$1" '{a[NR]=$1}
    END{ if(NR==0){exit}
         s=(NR<4) ? 1 : int(NR/2)+1; n=0; sum=0; mn=a[s]
         for(i=s;i<=NR;i++){sum+=a[i]; n++; if(a[i]<mn)mn=a[i]}
         printf "%s,%.0f,%d\n", l, sum/n, mn }' "$FTMP" >> "$FREQ_CSV"
  sleep "$COOLDOWN"
}

log "  LOW baseline"
run_one low "$LOW_D1" "$LOW_D2" 0x0 0 0

i=0
while read -r totf inf resf destf d1 d2 c; do
  i=$((i+1))
  log "  cand$i d1=$d1 d2=$d2 c=$c (total flips $totf)"
  run_one "cand$i" "$d1" "$d2" "$c" 1 "$TOGGLE_A"
done < "$SHORT"
rm -f "$SCREEN" "$SHORT" "$FTMP"

# ----------------------------------------------------------------------
log "analysing"
python3 - "$OUT_CSV" "$FREQ_CSV" "$PL1_W" "$TOGGLE_A" <<'PY'
import csv, sys, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
pl1 = float(sys.argv[3]); toggle_a = sys.argv[4]
W = defaultdict(list); T = defaultdict(list); A = defaultdict(list)
meta = {}; order = []
for r in rows:
    k = r['d1_name']
    if k not in meta:
        meta[k] = (r['data1'], r['data2'], r['datac'], r['total_flips_per_op'])
        order.append(k)
    W[k].append(float(r['pkg_W'])); T[k].append(float(r['mean_time_s']))
for r in csv.DictReader(open(sys.argv[2])):
    A[r['label']].append(float(r['actual_mean_khz']))

low_w = st.mean(W['low']) if 'low' in W else float('nan')
low_c = st.mean(A['low']) if A['low'] else float('nan')
hdr = (f"{'cand':>6} {'data1':>12} {'data2':>12} {'c':>12} {'flips':>7} "
       f"{'W':>7} {'vs LOW':>7} {'clk MHz':>8} {'dclk %':>7} {'time s':>8}  flags")
print(f"\nPL1 = {pl1:.0f} W    LOW = {low_w:.1f} W at {low_c/1000:.0f} MHz    "
      f"headroom to PL1 = {pl1-low_w:.1f} W")
print("Nothing is pinned, so a candidate that crosses PL1 shows up as a LOWER clock,")
print("not as higher wattage: the package simply cannot draw more than the limit.\n")
print(hdr); print('-'*len(hdr))
best = None
for k in order:
    d1, d2, c, fl = meta[k]
    w, t = st.mean(W[k]), st.mean(T[k])
    clk = st.mean(A[k]) if A[k] else float('nan')
    dclk = (clk - low_c) / low_c * 100 if low_c else float('nan')
    flags = []
    if pl1 > 0 and w > pl1:
        flags.append("LOW OVER PL1 (bad)" if k == 'low' else "OVER PL1")
    if k != 'low' and clk < low_c * 0.99: flags.append("HW DOWNCLOCKED <==")
    print(f"{k:>6} {d1:>12} {d2:>12} {c:>12} {float(fl):7.0f} {w:7.1f} {w-low_w:+7.1f} "
          f"{clk/1000:8.0f} {dclk:+7.2f} {t:8.3f}  {', '.join(flags)}")
    # rank by clock drop once PL1 is reached, by wattage while still below it
    score = (1, low_c - clk) if clk < low_c * 0.99 else (0, w)
    if k != 'low' and (best is None or score > best[0]): best = (score, k, w, clk, c)
print()
if best:
    score, k, w, clk, c = best
    if score[0] == 1:
        print(f"best: {k} -- the hardware downclocked it to {clk/1000:.0f} MHz vs "
              f"{low_c/1000:.0f} MHz for LOW, on its own, with nothing pinned.")
        print("Confirm the full LOW/HIGH comparison with:")
        print(f"  sudo CONFIGS={toggle_a} BENCH_C={c} ./find_throttle_freq.sh")
    else:
        print(f"best: {k} at {w:.1f} W, still {pl1-w:.1f} W short of PL1 "
              f"({clk/1000:.0f} MHz, no downclock).")
        print("Next levers, in order of expected value:")
        print("  ITERS=12000000000   PL1 is a ~1 s rolling average; a short trial can")
        print("                      finish before the limit engages at all.")
        print("  TRIALS/WARMUP up    a hot package leaks more, which is free wattage.")
        if toggle_a != "1":
            print("  TOGGLE_A=1          both multiplier inputs flip instead of one.")
        print("  find_throttle_power.sh   fallback: lowers the limit to ~178-180 W,")
        print("                      which does produce the asymmetry.")
PY
log "done. rows: $OUT_CSV"
