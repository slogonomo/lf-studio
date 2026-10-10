#!/usr/bin/env bash
# 70-manifests.sh: write one path manifest per task with NVIDIA's prepare.py: $SOLH3_PATHS with {task} = t2va,
# fl2va, ref2va (default $SOLH3_HOME/paths-<task>.json). qwen_python points at qwen_python.sh in the PATCHED
# checkout, so the Qwen worker runs the patched code. The first run also applies NVIDIA's FastVideo loader patch
# to sources/FastVideo (prepare.py does that itself and checks every file hash).
# Before 80-prompt-cache.sh the manifests are written with --cache-pending; they become valid once
# $SOLH3_HOME/fixed-prompt.pt exists, no rewrite needed.
# Idempotent: existing manifests are kept (prepare.py refuses to overwrite); FORCE=1 rewrites them (old copy *.bak).
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
[[ -f "$PKG/prepare.py" || "$DRY_RUN" == 1 ]] || die "no $PKG/prepare.py; run 10-sana.sh first"
need_cmd python3.12
for py in "$ENVS/stage1/bin/python" "$ENVS/stage2/bin/python"; do
    [[ -x "$py" || "$DRY_RUN" == 1 ]] || die "$py missing; run 40-stage1.sh and 50-stage2.sh first"
done
EXISTING="$SOLH3_HOME/existing-paths.json"
[[ -f "$EXISTING" || "$DRY_RUN" == 1 ]] || die "$EXISTING missing (40-stage1.sh writes it)"

pending=()
[[ -f "$PROMPT_CACHE" ]] || pending=(--cache-pending)
for task in $SOLH3_TASKS; do
    out=$(manifest_for "$task")
    step "$task -> $out"
    if [[ -f "$out" ]]; then
        if [[ "$FORCE" != 1 ]]; then say "exists; FORCE=1 to rewrite"; continue; fi
        run mv -f "$out" "$out.bak"
    fi
    run mkdir -p "$(dirname "$out")"
    run_in "$PKG" python3.12 prepare.py --task "$task" \
        --sources "$SOURCES" --checkpoints "$SOLH3_CHECKPOINTS" --paths "$EXISTING" \
        --python-stage1 "$ENVS/stage1/bin/python" --python-stage2 "$ENVS/stage2/bin/python" \
        --python-qwen "$PKG/qwen_python.sh" --prompt-cache "$PROMPT_CACHE" "${pending[@]}" \
        --output "$out" || die "prepare.py failed for $task (are its weights downloaded? SOLH3_TASKS=$SOLH3_TASKS)"
done
[[ ${#pending[@]} -eq 0 ]] || say "prompt cache pending: run install/80-prompt-cache.sh next"
say "done. Next: install/80-prompt-cache.sh"
