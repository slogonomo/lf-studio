#!/usr/bin/env bash
# 80-prompt-cache.sh: build the one-time Stage2 prompt cache $LFSTUDIO_HOME/fixed-prompt.pt with NVIDIA's
# runtime.cache_builder (Stage2 env + the offline-context target). It encodes Sol-H3-Spark's fixed refinement
# prompt with the INT8 Gemma and INT8 connector weights: GPU work, ~37 GB of weights loaded for ~20 s
# (16.7 s measured on a DGX Spark). Nothing else should run on the GPU meanwhile.
# Afterwards the two offline weight files (~37 GB) are not needed for inference; the script prints them so you can
# delete them. Re-download with 20-weights.sh if you ever need to rebuild the cache.
# Idempotent: skipped when the cache exists (FORCE=1 rebuilds; old copy *.bak).
# RTX 5090 (LFSTUDIO_BACKEND=rtx5090): NVIDIA's cache builder requires an SM121 GPU (a DGX Spark), so the cache is taken
# from a Spark install instead: LFSTUDIO_PROMPT_CACHE_FROM names a copy of its fixed-prompt.pt on this machine. The
# cache builder and its recipe check are byte-identical in Sol-H3-Spark and Sol-H3-RTX5090; the file is checked with
# that recipe check (runtime/prompt_cache.py, CPU only) before it is copied to $LFSTUDIO_HOME/fixed-prompt.pt. Without
# a cache this refuses. It needs no manifest, so on the RTX 5090 it runs before 70-manifests.sh. There FORCE=1 keeps
# the old cache as fixed-prompt.pt.<date-time>.bak, so a second FORCE=1 run does not overwrite the first backup.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"

if [[ "$LFSTUDIO_BACKEND" == rtx5090 ]]; then
    PY="$ENVS/stage1/bin/python"
    # check_cache FILE: Sol-H3-RTX5090's own recipe check (prompt, recipe fingerprint, shapes, dtypes, finite values)
    check_cache() {
        if [[ ! -x "$PY" ]]; then
            [[ "$DRY_RUN" == 1 ]] || die "no Stage1 env ($PY); run 40-stage1.sh first"
            echo "+ (cd $PKG && $PY -B -c 'runtime.prompt_cache.load_cache(...)' $1)   (dry run: no Stage1 env yet)"
            return 0
        fi
        (cd "$PKG" && CUDA_VISIBLE_DEVICES='' "$PY" -B -c 'import sys, torch
from runtime.prompt_cache import FIXED_PROMPT, load_cache
cache = load_cache(sys.argv[1], prompt=FIXED_PROMPT, torch_module=torch)
print("  ok    %s: recipe check passed (fingerprint %s)" % (sys.argv[1], cache.payload["fingerprint"][:16]))' "$1")
    }
    step "prompt cache -> $PROMPT_CACHE (from a DGX Spark: NVIDIA's cache builder needs an SM121 GPU)"
    if [[ -f "$PROMPT_CACHE" && "$FORCE" != 1 ]]; then
        check_cache "$PROMPT_CACHE" ||
            die "$PROMPT_CACHE fails Sol-H3-RTX5090's recipe check (above). Move it away, then run this again with LFSTUDIO_PROMPT_CACHE_FROM"
        say "already there"
    else
        src=${LFSTUDIO_PROMPT_CACHE_FROM:-}
        [[ -n "$src" ]] || die "no prompt cache at $PROMPT_CACHE, and LFSTUDIO_PROMPT_CACHE_FROM is not set.
NVIDIA's cache builder needs an SM121 GPU, so the cache comes from a DGX Spark: copy fixed-prompt.pt of a Spark
install (its \$LFSTUDIO_HOME, by default ~/sol-h3; its install/80-prompt-cache.sh builds it) to this machine, then run
  LFSTUDIO_PROMPT_CACHE_FROM=<that copy> $0"
        [[ -f "$src" ]] || die "LFSTUDIO_PROMPT_CACHE_FROM=$src: no such file"
        check_cache "$src" || die "$src fails Sol-H3-RTX5090's recipe check (above); it is not a cache this package can use"
        # FORCE=1 replaces a cache this machine cannot rebuild: the old one is kept under a dated name, never
        # overwritten by a later run
        if [[ -f "$PROMPT_CACHE" ]]; then
            bak="$PROMPT_CACHE.$(date +%Y%m%d-%H%M%S).bak"
            [[ ! -e "$bak" ]] || die "$bak exists already; run this again in a second"
            run mv "$PROMPT_CACHE" "$bak"
        fi
        run mkdir -p "$(dirname "$PROMPT_CACHE")"
        run cp "$src" "$PROMPT_CACHE"
        if [[ "$DRY_RUN" != 1 ]]; then cmp -s "$src" "$PROMPT_CACHE" || die "the copy $PROMPT_CACHE differs from $src"; fi
    fi
    say "done. Next: install/70-manifests.sh"
    exit 0
fi
MANIFEST=$(manifest_for t2va)          # any task's manifest works; the cache is task-independent
if [[ ! -f "$MANIFEST" ]]; then
    for task in $LFSTUDIO_TASKS; do
        if [[ -f "$(manifest_for "$task")" ]]; then MANIFEST=$(manifest_for "$task"); break; fi
    done
fi
[[ -f "$MANIFEST" || "$DRY_RUN" == 1 ]] || die "no manifest ($LFSTUDIO_PATHS); run 70-manifests.sh first"
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
