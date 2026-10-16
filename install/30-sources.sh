#!/usr/bin/env bash
# 30-sources.sh: fetch the pinned source trees (FastVideo, flash-attention, ComfyUI v0.30.0, LTX-2, the H3 latent
# upscaler) into $SOLH3_HOME/sources with NVIDIA's prepare.py --fetch-sources (~770 MB).
# prepare.py stops with "Missing required inputs" at this point because the Python envs do not exist yet;
# that is expected. This script checks the fetched revisions itself. The FastVideo loader patch is applied
# later, by the first prepare.py run in 70-manifests.sh.
# RTX 5090: the same sources and revisions from Sol-H3-RTX5090's prepare.py (~1 min measured on a workstation with
# one RTX 5090), with uv's Python 3.12 (05-python.sh).
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
[[ -d "$PKG" || "$DRY_RUN" == 1 ]] || die "no $PKG_NAME package at $PKG; run 10-sana.sh first"
need_host_py
need_cmd git

step "pinned sources -> $SOURCES"
run mkdir -p "$SOURCES"
probe="$SOLH3_HOME/.paths-probe-$$.json"
run_in "$PKG" "$HOST_PY" prepare.py --fetch-sources --sources "$SOURCES" --checkpoints "$SOLH3_CHECKPOINTS" \
    --output "$probe" || say "prepare.py stopped (expected before the envs exist); checking the sources"
[[ "$DRY_RUN" == 1 ]] && { say "dry run: skipping the revision check"; exit 0; }
rm -f "$probe"

bad=0
while IFS=$'\t' read -r key dir rev; do
    [[ "$key" == fa4_source_root && -d "$SOURCES/fa4-runtime/flash_attn/cute" && ! -d "$SOURCES/$dir" ]] && continue
    have=$(git -C "$SOURCES/$dir" rev-parse HEAD 2>/dev/null || echo missing)
    if [[ "$have" == "$rev" ]]; then echo "  ok    $dir @ ${rev:0:8}"; else echo "  FAIL  $dir: $have (want $rev)"; bad=1; fi
done < <("$HOST_PY" -c '
import json, sys
for s in json.load(open(sys.argv[1]))["sources"]:
    print(s["key"], s["directory"], s["revision"], sep="\t")' "$PKG/configs/dependencies.json")
(( bad == 0 )) || die "some sources are missing or at another revision"
say "done. Next: install/40-stage1.sh"
