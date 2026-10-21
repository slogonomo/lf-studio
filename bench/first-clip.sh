#!/usr/bin/env bash
# first-clip.sh LABEL [SECONDS] [ORIENTATION] [TASK] [JSONL]
# Cold-start bench: time from launching a new Sol-H3 session in web-UI mode (infer.py --deliver-first) to the first
# finished MP4, plus a MemAvailable trace every 0.25 s (lowest value and when). docs/PERFORMANCE.md's first-clip
# numbers were measured with the script this one was ported from (same steps and memory guard; paths and settings
# now come from the env file). Defaults: 5 s, landscape, t2va, bench/prompts/bench.jsonl (seed 42).
# Needs the whole box: stop the studio and other GPU apps first. Settings: the same env file as bin/lf-studio.
# Output: $SOL_H3_SPARK_RUNTIME_ROOT/outputs/bench/firstclip-LABEL-*, logs in ${LFSTUDIO_BENCH_LOGS:-$LFSTUDIO_HOME/logs/bench}.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../install/env.sh
source "$HERE/../install/env.sh"
[[ $# -ge 1 && "$1" != -* ]] || { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
L=$1; S=${2:-5}; O=${3:-landscape}; T=${4:-t2va}; J=${5:-$HERE/prompts/bench.jsonl}
LOGDIR=${LFSTUDIO_BENCH_LOGS:-$LFSTUDIO_HOME/logs/bench}
MANIFEST=$(manifest_for "$T")
[[ -f "$MANIFEST" ]] || { echo "no manifest $MANIFEST (install/70-manifests.sh)" >&2; exit 1; }
[[ -f "$J" ]] || { echo "no prompts file $J" >&2; exit 1; }
J=$(realpath "$J")
need=${LFSTUDIO_MIN_FREE_GIB:-110}; have=$(mem_avail_gib)
(( need == 0 || have >= need )) || { echo "only ${have} GiB MemAvailable, need ${need}: stop other apps first" >&2; exit 1; }
mkdir -p "$LOGDIR" "$SOL_H3_SPARK_RUNTIME_ROOT/outputs/bench"
OUT="$SOL_H3_SPARK_RUNTIME_ROOT/outputs/bench/firstclip-$L-$(date +%Y%m%d-%H%M%S)"
MEM="$LOGDIR/mem-$L.tsv"; LOG="$LOGDIR/firstclip-$L.log"
( while :; do echo "$(date +%s.%N) $(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)"; sleep 0.25; done ) > "$MEM" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
sync; t0=$(date +%s.%N)
(cd "$PKG" && "$HERE/memguard-run.sh" "$LOG" -- "$LFSTUDIO_PYTHON" infer.py --deliver-first \
    --paths "$MANIFEST" --task "$T" --seconds "$S" --orientation "$O" --prompts "$J" --output-dir "$OUT") > /dev/null 2>&1
rc=$?; t1=$(date +%s.%N); kill $SAMPLER 2>/dev/null
python3 - "$OUT" "$L" "$t0" "$t1" "$rc" "$MEM" <<'PY'
import json, os, sys
out, label, t0, t1, rc, mem = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
m = lambda p: os.path.getmtime(p) if os.path.exists(p) else float("nan")
r = json.load(open(out + "/results.json")) if os.path.exists(out + "/results.json") else {}
q = (r.get("requests") or [{}])[0]
mp4 = q.get("output") or ""
first = m(mp4) - t0 if mp4 else float("nan")
rows = [line.split() for line in open(mem) if len(line.split()) == 2]
lo = min(rows, key=lambda x: int(x[1])) if rows else ("nan", "0")
ph = {"stage1_ready": m(f"{out}/stage1-worker/ready.json") - t0,
      "qwen_enc": m(f"{out}/warmup-qwen-worker/response-00001.json") - t0,
      "stage2_ready": m(f"{out}/stage2-worker/ready.json") - t0,
      "stage1_done": m(f"{out}/stage1-worker/response-00001.json") - t0}
print(f"FIRSTCLIP {label} rc={rc} first_mp4={first:.1f}s row_e2e={q.get('e2e_s', float('nan')):.1f}s "
      f"status={q.get('status')} wall={t1 - t0:.0f}s | " + " ".join(f"{k}={v:.0f}" for k, v in ph.items())
      + f" | min_avail={int(lo[1]) / 1024:.1f}GiB at t={float(lo[0]) - t0:.0f}s | {r.get('error', '')[:200]}")
PY
grep MEMGUARD "$LOG" | tail -1
exit "$rc"
