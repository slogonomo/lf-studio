#!/usr/bin/env bash
# 60-qwen-image.sh: build the Docker image for the Qwen prompt encoder, in two steps:
#   1. NVIDIA's Dockerfile.qwen (NGC PyTorch 25.11 + ComfyUI v0.30.0)        -> sol-h3-spark-qwen:base
#   2. this repo's qwen-image/ layer (torchaudio stubs, see its Dockerfile)   -> $SOL_H3_SPARK_QWEN_IMAGE (sol-h3-spark-qwen)
# ~20 min measured (mostly the NGC base pull, no docker login needed), ~21 GB.
# The image contains GPL-3.0 ComfyUI and NGC components: build it yourself, never redistribute it.
# Idempotent: existing images are kept (FORCE=1 rebuilds).
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
[[ -f "$PKG/Dockerfile.qwen" || "$DRY_RUN" == 1 ]] || die "no $PKG/Dockerfile.qwen; run 10-sana.sh first"
need_cmd docker
[[ "$DRY_RUN" == 1 ]] || docker info >/dev/null 2>&1 || die "docker does not work as $(id -un) (see 00-check.sh)"
BASE_TAG=${SOL_H3_SPARK_QWEN_IMAGE%%:*}:base
[[ "$SOL_H3_SPARK_QWEN_IMAGE" == "$BASE_TAG" ]] && die "SOL_H3_SPARK_QWEN_IMAGE must not end in :base"

have_image() { [[ "$DRY_RUN" != 1 ]] && docker image inspect "$1" >/dev/null 2>&1; }

step "upstream image $BASE_TAG (Dockerfile.qwen)"
if have_image "$BASE_TAG" && [[ "$FORCE" != 1 ]]; then
    say "already built"
else
    run_in "$PKG" docker build -f Dockerfile.qwen -t "$BASE_TAG" .
fi

step "torchaudio stub layer -> $SOL_H3_SPARK_QWEN_IMAGE"
if have_image "$SOL_H3_SPARK_QWEN_IMAGE" && [[ "$FORCE" != 1 ]]; then
    say "already built"
else
    run docker build --build-arg "BASE=$BASE_TAG" -t "$SOL_H3_SPARK_QWEN_IMAGE" "$REPO/qwen-image"
fi

step "import check (no GPU, no network)"
if [[ "$DRY_RUN" != 1 ]]; then
    docker run --rm --network=none --pull=never --entrypoint /usr/bin/python3 "$SOL_H3_SPARK_QWEN_IMAGE" -c \
        'import torchaudio, torchaudio.functional, torchaudio.transforms; print("torchaudio stub", torchaudio.__version__)' ||
        die "the image does not import the torchaudio stub"
fi
say "done. Next: install/70-manifests.sh"
