#!/usr/bin/env bash
# 80-prompt-cache.sh: build the one-time Stage2 prompt cache $SOLH3_HOME/fixed-prompt.pt with NVIDIA's
# runtime.cache_builder (Stage2 env + the offline-context target). It encodes Sol-H3-Spark's fixed refinement
# prompt with the INT8 Gemma and INT8 connector weights: GPU work, ~37 GB of weights loaded for ~20 s
# (16.7 s measured on a DGX Spark). Nothing else should run on the GPU meanwhile.
# Afterwards the two offline weight files (~37 GB) are not needed for inference; the script prints them so you can
# delete them. Re-download with 20-weights.sh if you ever need to rebuild the cache.
# Idempotent: skipped when the cache exists (FORCE=1 rebuilds; old copy *.bak).
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
MANIFEST=$(manifest_for t2va)          # any task's manifest works; the cache is task-independent
if [[ ! -f "$MANIFEST" ]]; then
    for task in $SOLH3_TASKS; do
        if [[ -f "$(manifest_for "$task")" ]]; then MANIFEST=$(manifest_for "$task"); break; fi
    done
fi
[[ -f "$MANIFEST" || "$DRY_RUN" == 1 ]] || die "no manifest ($SOLH3_PATHS); run 70-manifests.sh first"
[[ -x "$ENVS/stage2/bin/python" || "$DRY_RUN" == 1 ]] || die "no Stage2 env; run 50-stage2.sh first"

step "prompt cache -> $PROMPT_CACHE"
if [[ -f "$PROMPT_CACHE" && "$FORCE" != 1 ]]; then
    say "already built"
else
    if [[ "$DRY_RUN" != 1 ]]; then
        avail=$(mem_avail_gib)
        (( avail >= 48 )) || die "only ${avail} GiB MemAvailable; the cache builder needs ~40 GiB. Stop other GPU apps first."
    fi
    [[ -f "$PROMPT_CACHE" ]] && run mv -f "$PROMPT_CACHE" "$PROMPT_CACHE.bak"
    run mkdir -p "$LOGS"
    run_in "$PKG" env PYTHONPATH="$DEPS/offline-context:$PKG" \
        "$ENVS/stage2/bin/python" -m runtime.cache_builder --paths "$MANIFEST" --output "$PROMPT_CACHE" ||
        die "cache build failed"
fi

step "offline weights (only needed to rebuild this cache)"
if [[ -f "$MANIFEST" ]]; then
    python3 - "$MANIFEST" <<'PY'
import json, os, sys
paths = json.load(open(sys.argv[1]))
files = [paths[k] for k in ("offline_gemma", "offline_connector") if k in paths and os.path.isfile(paths[k])]
if files:
    size = sum(os.path.getsize(f) for f in files) / 1e9
    print(f"These {len(files)} files ({size:.1f} GB) are not used for inference. To free the space:")
    print("  rm " + " ".join(files))
else:
    print("already removed")
PY
fi
say "done. Next: install/90-env.sh"
