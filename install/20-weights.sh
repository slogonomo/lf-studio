#!/usr/bin/env bash
# 20-weights.sh: download the pinned model weights with NVIDIA's download_checkpoints.py into $LFSTUDIO_CHECKPOINTS,
# for each task in $LFSTUDIO_TASKS (default: t2va fl2va ref2va), with --include-offline --verify-sha256.
# Measured on a DGX Spark: ~224 GB, ~48 min at ~80 MB/s for all three tasks plus the offline weights.
# To reuse weights you already have, set LFSTUDIO_CHECKPOINTS to that directory; the script resumes and verifies.
# The offline weights (INT8 Gemma + INT8 connector, ~37 GB) are only needed once, for 80-prompt-cache.sh.
# RTX 5090 (LFSTUDIO_BACKEND=rtx5090): the same files and revisions from Sol-H3-RTX5090's download_checkpoints.py, but
# without the offline weights (~220 GB for all three tasks), since the prompt cache comes from a DGX Spark there
# (80-prompt-cache.sh).
# Checkpoints a DGX Spark already has can be copied instead (LFSTUDIO_CHECKPOINTS; docs/INSTALL.md, "RTX 5090").
# Needs: a Hugging Face login whose account accepted the Lightricks/LTX-2.5 terms: $LFSTUDIO_HOME/envs/download/bin/hf
# auth login (install/00-check.sh prints how to create that venv first), or HF_TOKEN in the environment.
# Model terms: you download the weights yourself; each model has its own license (README.md, "Model terms"). The
# MiniMax H3 Community License does not cover the EU, the UK, South Korea or the USA (its sections I.5 and V.4).
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
[[ -d "$PKG" || "$DRY_RUN" == 1 ]] || die "no $PKG_NAME package at $PKG; run 10-sana.sh first"
need_host_py "sudo apt install python3.12-venv"
DL="$ENVS/download"
offline=(--include-offline)
[[ "$LFSTUDIO_BACKEND" == rtx5090 ]] && offline=()     # the prompt cache is built on a DGX Spark

step "download tool ($DL, huggingface-hub from requirements/download.txt)"
if [[ ! -x "$DL/bin/python" ]]; then
    run "$HOST_PY" -m venv "$DL"
fi
if [[ "$DRY_RUN" == 1 ]] || ! "$DL/bin/python" -c 'import huggingface_hub' 2>/dev/null; then
    run "$DL/bin/pip" install -q -r "$PKG/requirements/download.txt"
fi
if [[ "$DRY_RUN" != 1 ]] && ! "$DL/bin/hf" auth whoami >/dev/null 2>&1; then
    die "not logged in to Hugging Face: run  $DL/bin/hf auth login  (or export HF_TOKEN), and accept https://huggingface.co/Lightricks/LTX-2.5"
fi

run mkdir -p "$LFSTUDIO_CHECKPOINTS"
for task in $LFSTUDIO_TASKS; do
    step "weights for $task -> $LFSTUDIO_CHECKPOINTS"
    run_in "$PKG" "$DL/bin/python" download_checkpoints.py --task "$task" "${offline[@]}" --verify-sha256 \
        --output-dir "$LFSTUDIO_CHECKPOINTS" || die "download for $task failed; run this script again to resume"
    if [[ -f "$LFSTUDIO_CHECKPOINTS/checkpoint-paths.json" ]]; then
        run cp "$LFSTUDIO_CHECKPOINTS/checkpoint-paths.json" "$LFSTUDIO_CHECKPOINTS/checkpoint-paths-$task.json"
    fi
done
say "done. Next: install/30-sources.sh"
