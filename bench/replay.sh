#!/usr/bin/env bash
# replay.sh COMMAND ...: replay Stage 2 on saved Stage-1 drafts under named arms (recipe changes), with per-step
# timing, output hashes and bit-exact checks; CPU quality checks afterwards. docs/PERFORMANCE.md, "Replay bench".
#   add NAME CASE_DIR [--dialogue] [--turns SPEC] [--mouth X:Y:W:H] [--force]
#                     copy a studio clip's draft (<runtime root>/outputs/webui/<session>/<case>) into the store
#   list | arms       the drafts in the store | the arms and what each changes
#   run [--dry] [--drafts A,B] [--arms ARM[:N],...] [--repeats N] [--label L] [--stop-at EPOCH] [--keep-cache]
#                     GPU; --dry prints the processes, shapes, arms and runs and starts nothing
#   report RUN_DIR    hashes, "same as" columns and summary.md for a run (run does this at its end)
#   quality RUN_DIR [--lpips PYTHON] [--syncnet CMD]   CPU only: SSIM/PSNR, sheets, side-by-sides, LPIPS, SyncNet
# run needs the whole box: stop the studio and other GPU apps first. Settings: the same env file as bin/sol-h3-studio.
# Drafts: $SOL_H3_SPARK_RUNTIME_ROOT/outputs/bench/replay-drafts/; runs: .../outputs/bench/replay-<time>[-label]/.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../install/env.sh
source "$HERE/../install/env.sh"
[[ $# -ge 1 && "$1" != -* ]] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
unset SOL_H3_SECONDS SOL_H3_ORIENTATION      # as the web server does: format and length come from each draft
export SOLH3_PKG="$PKG" SOLH3_REPO SOLH3_SANA SOLH3_PATHS SOLH3_HOME
command -v ffmpeg >/dev/null || { echo "replay.sh needs ffmpeg" >&2; exit 1; }
if [[ "$1" == run && " $* " != *" --dry "* ]]; then
    need=${SOLH3_MIN_FREE_GIB:-110}; have=$(mem_avail_gib)
    (( need == 0 || have >= need )) || { echo "only ${have} GiB MemAvailable, need ${need}: stop other apps first" >&2; exit 1; }
    export SOLH3_MIN_FREE_GIB=$need
fi
[[ "$1" == run ]] || export CUDA_VISIBLE_DEVICES=     # every other command is CPU only
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" exec python3 -B -m replay.cli "$@"
