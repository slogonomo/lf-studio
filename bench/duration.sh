#!/usr/bin/env bash
# duration.sh SECONDS [ORIENTATION] [JSONL] [TASK]
# Warm-clip bench for one clip length: one infer.py session (upstream flow: a warm-up pass, then every JSONL row
# as a formal request), under the memory guard. Prints each request's end-to-end time, stage times and the
# ffprobe'd output. Defaults: landscape, bench/prompts/bench.jsonl, t2va.
# Needs the whole box: stop the studio and other GPU apps first. Settings: the same env file as bin/lf-studio.
# Output: $SOL_H3_SPARK_RUNTIME_ROOT/outputs/bench/dur-*, logs in ${LFSTUDIO_BENCH_LOGS:-$LFSTUDIO_HOME/logs/bench}.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../install/env.sh
source "$HERE/../install/env.sh"
[[ $# -ge 1 && "$1" != -* ]] || { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
S=$1; O=${2:-landscape}; J=${3:-$HERE/prompts/bench.jsonl}; T=${4:-t2va}
LOGDIR=${LFSTUDIO_BENCH_LOGS:-$LFSTUDIO_HOME/logs/bench}
MANIFEST=$(manifest_for "$T")
[[ -f "$MANIFEST" ]] || { echo "no manifest $MANIFEST (install/70-manifests.sh)" >&2; exit 1; }
[[ -f "$J" ]] || { echo "no prompts file $J" >&2; exit 1; }
J=$(realpath "$J")
need=${LFSTUDIO_MIN_FREE_GIB:-110}; have=$(mem_avail_gib)
(( need == 0 || have >= need )) || { echo "only ${have} GiB MemAvailable, need ${need}: stop other apps first" >&2; exit 1; }
mkdir -p "$LOGDIR" "$SOL_H3_SPARK_RUNTIME_ROOT/outputs/bench"
OUT="$SOL_H3_SPARK_RUNTIME_ROOT/outputs/bench/dur-$T-${S}s-$O-$(date +%Y%m%d-%H%M%S)"
LOG="$LOGDIR/dur-$T-${S}s-$O.log"
t0=$(date +%s)
(cd "$PKG" && "$HERE/memguard-run.sh" "$LOG" -- "$LFSTUDIO_PYTHON" infer.py \
    --paths "$MANIFEST" --task "$T" --orientation "$O" --seconds "$S" --prompts "$J" --output-dir "$OUT")
rc=$?
python3 - "$OUT" "$S" "$(( $(date +%s) - t0 ))" "$rc" <<'PY'
import json, subprocess, sys
out, s, wall, rc = sys.argv[1:]
try:
    r = json.load(open(out + "/results.json"))
    for q in r["requests"]:
        mp4 = q.get("output", "")
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration:stream=codec_type,width,height,nb_frames", "-of", "compact=p=0", mp4],
                               capture_output=True, text=True).stdout.replace("\n", " ") if mp4 else ""
        print(f"RESULT {s}s {q['case_id']} status={q.get('status')} e2e={round(q.get('e2e_s', 0), 1)}s "
              f"stage1={round(q.get('stage1_s', 0), 1)} stage2={round(q.get('stage2_s', 0), 1)} | {probe} | {mp4}")
    print(f"RESULT {s}s rc={rc} session={r.get('status')} startup={round(r.get('startup_and_warmup_s', 0))}s "
          f"wall={wall}s err={(r.get('error') or '')[:400]}")
except Exception as e:
    print(f"RESULT {s}s rc={rc} no results: {e}")
PY
grep MEMGUARD "$LOG" | tail -1
exit "$rc"
