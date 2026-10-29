#!/usr/bin/env bash
# 65-regen2k.sh: optional, DGX Spark only. Sets up the experimental Upscale (LFSTUDIO_UPSCALE=1; docs/INSTALL.md,
# "Experimental: Upscale"): NVIDIA's Open-H3-Regenerate-2K in an environment of its own, and its weights.
#  1. uv 0.13.4 (the aarch64 binary from its GitHub release, checked against a pinned SHA-256) into $LFSTUDIO_HOME/bin,
#     and with it CPython 3.11.17 into $LFSTUDIO_HOME/python (UV_PYTHON_INSTALL_DIR; uv's cache in
#     $LFSTUDIO_HOME/cache/uv).
#  2. NVIDIA's code at the pinned commit of NVlabs/Sana (sol-engine 6e4a440): models/Open-H3-Regenerate-2K and
#     techniques/sparse_backends, extracted with git archive from the clone 10-sana.sh made (its checkout is not
#     touched) into $LFSTUDIO_HOME/sources/sana-6e4a440, with LFSTUDIO_UPSCALE_PKG (default
#     $LFSTUDIO_HOME/Open-H3-Regenerate-2K) a link to the first.
#  3. A new venv, $LFSTUDIO_HOME/envs/regen2k (never a copy of another one), with exactly the packages of
#     install/freeze/regen2k-lock.txt, each checked against its SHA-256 (uv pip install --no-deps --require-hashes),
#     then NVIDIA's Sol-Attn (sparse_backends) --no-deps; an import check without the GPU marks it done.
#  4. The weights at the pinned revision of Efficient-Large-Model/Open-H3-Regenerate-2K (5809c33), in
#     LFSTUDIO_UPSCALE_WEIGHTS (default $LFSTUDIO_CHECKPOINTS/Open-H3-Regenerate-2K), with Sharp's AdaLN table read by
#     HTTP range from Comfy-Org/MiniMax-H3 (install/regen2k_weights.py). A copy with NVIDIA's SHA-256 in the Hugging
#     Face cache, or the studio's own H3-upscaler checkpoint, is linked rather than copied, and nothing is written into
#     the cache. The base model is the studio's MiniMax-H3 (LFSTUDIO_UPSCALE_BASE, from 20-weights.sh).
#  5. upscale/regen2k_job.py's own check of the package and the weights.
# Downloads about 3.2 GB of packages and up to 14.4 GB of weights (12.5 GB LoRA, 1.2 GB curve branches, 0.7 GB
# latent upscaler); the venv takes about 5.5 GB. Then set LFSTUDIO_UPSCALE=1 in the settings file and restart.
# Needs 10-sana.sh (the clone) and 20-weights.sh (MiniMax-H3) first. Every step stops the script when it fails (#74).
# Idempotent: what is in place is kept. FORCE=1 builds the venv again (the old one is moved aside, not deleted); a
# venv this script did not make is never touched.
# --env-only: steps 1-3 alone, the environment without the Upscale's weights. The experimental LightX2V Turbo
# (66-h3turbo.sh, which runs this when the environment is not there yet) uses the same environment.
ENV_ONLY=0
args=()
for arg in "$@"; do
    if [[ "$arg" == --env-only ]]; then ENV_ONLY=1; else args+=("$arg"); fi
done
set -- "${args[@]}"                                    # (source without arguments would pass on the script's own)
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
if [[ "$LFSTUDIO_BACKEND" != spark ]]; then
    say "LFSTUDIO_BACKEND=$LFSTUDIO_BACKEND: nothing to do (the experimental Upscale runs on the DGX Spark only)"
    exit 0
fi
need_cmd curl
need_cmd tar
need_cmd git
need_cmd sha256sum
arch=$(uname -m)
if [[ "$arch" != aarch64 ]]; then
    [[ "$DRY_RUN" == 1 ]] || die "uname -m is $arch: the lock and uv's binary here are for the DGX Spark (aarch64)"
    warn "uname -m is $arch, not aarch64 (dry run: going on)"
fi

UV_VERSION=0.13.4        # as 05-python.sh: uv's bundled roots no longer trust the emSign ECC Root CA - G3
UV_ASSET=uv-aarch64-unknown-linux-gnu.tar.gz
UV_SHA256=1ae72b4718729e2ccc4f1ae2c6d1e531ca8467800e0546e9130b3886023a60ab   # the release's own .sha256
PY_VERSION=3.11.17       # NVIDIA's requirements.txt is for Python 3.11
SANA_REV=6e4a44002149418a547434b91d7280a8bbdcace3
LOCK="$INSTALL_DIR/freeze/regen2k-lock.txt"
UV="$LFSTUDIO_HOME/bin/uv"
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$LFSTUDIO_HOME/python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$LFSTUDIO_HOME/cache/uv}
export UV_NO_CONFIG=1                                  # no uv.toml of the user's changes the indexes or the resolution
DOWNLOADS="$LFSTUDIO_HOME/cache/downloads"
SRC="$SOURCES/sana-${SANA_REV:0:7}"
VENV="$ENVS/regen2k"
PY="$VENV/bin/python"
MARK="$VENV/lf-studio-install.txt"                     # "building <lock sha>", then "done <lock sha>"
PKG_DIR=$LFSTUDIO_UPSCALE_PKG                          # install/env.sh: the server's defaults
WEIGHTS=$LFSTUDIO_UPSCALE_WEIGHTS
BASE=$LFSTUDIO_UPSCALE_BASE
UPSCALER_COPY="$LFSTUDIO_CHECKPOINTS/H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors"
LOCK_SHA=$(sha256sum "$LOCK" | cut -d' ' -f1) || die "cannot read $LOCK"
[[ -z "${LFSTUDIO_UPSCALE_PYTHON:-}" || "$LFSTUDIO_UPSCALE_PYTHON" == "$PY" ]] ||
    die "LFSTUDIO_UPSCALE_PYTHON=$LFSTUDIO_UPSCALE_PYTHON: this script builds $PY only; unset it (the default is that one), or set up that environment yourself"
stamp() { date +%Y%m%d-%H%M%S; }

step "uv $UV_VERSION -> $UV"
if [[ -x "$UV" && "$("$UV" --version 2>/dev/null)" == "uv $UV_VERSION"* ]]; then
    say "already there"
else
    fetch_checked "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$UV_ASSET" \
        "$DOWNLOADS/uv-$UV_VERSION/$UV_ASSET" "$UV_SHA256"
    run tar -xzf "$DOWNLOADS/uv-$UV_VERSION/$UV_ASSET" -C "$DOWNLOADS/uv-$UV_VERSION" || die "cannot unpack $UV_ASSET"
    run mkdir -p "$(dirname "$UV")" || die "cannot create $(dirname "$UV")"
    run install -m 755 "$DOWNLOADS/uv-$UV_VERSION/${UV_ASSET%.tar.gz}/uv" "$DOWNLOADS/uv-$UV_VERSION/${UV_ASSET%.tar.gz}/uvx" \
        "$(dirname "$UV")/" || die "cannot install uv into $(dirname "$UV")"
fi

step "CPython $PY_VERSION -> $UV_PYTHON_INSTALL_DIR"
run "$UV" python install --no-bin "$PY_VERSION" || die "uv python install $PY_VERSION failed"

step "NVIDIA's Open-H3-Regenerate-2K and Sol-Attn at NVlabs/Sana ${SANA_REV:0:7} -> $SRC"
if [[ -f "$SRC/.lf-studio-commit" && "$(cat "$SRC/.lf-studio-commit")" == "$SANA_REV" ]]; then
    say "already there"
else
    [[ -e "$SRC" ]] && die "$SRC exists but is not this script's extract of $SANA_REV; move it away and run this again"
    [[ -e "$LFSTUDIO_SANA/.git" || "$DRY_RUN" == 1 ]] || die "no NVlabs/Sana clone at $LFSTUDIO_SANA; run 10-sana.sh first"
    if [[ "$DRY_RUN" == 1 ]] || ! git -C "$LFSTUDIO_SANA" cat-file -e "$SANA_REV^{commit}" 2>/dev/null; then
        run git -C "$LFSTUDIO_SANA" fetch --filter=blob:none origin "$SANA_REV" || die "cannot fetch $SANA_REV into $LFSTUDIO_SANA"
    fi
    run mkdir -p "$SOURCES" || die "cannot create $SOURCES"
    if [[ "$DRY_RUN" == 1 ]]; then
        tmp="$SOURCES/.sana-${SANA_REV:0:7}.XXXXXX"
        echo "+ tmp=\$(mktemp -d $tmp)"
        echo "+ git -C $LFSTUDIO_SANA archive $SANA_REV models/Open-H3-Regenerate-2K techniques/sparse_backends | tar -x -C \$tmp"
        echo "+ echo $SANA_REV > \$tmp/.lf-studio-commit"
        echo "+ chmod 755 \$tmp && mv -T \$tmp $SRC"
    else
        tmp=$(mktemp -d "$SOURCES/.sana-${SANA_REV:0:7}.XXXXXX") || die "cannot create a directory in $SOURCES"
        git -C "$LFSTUDIO_SANA" archive "$SANA_REV" models/Open-H3-Regenerate-2K techniques/sparse_backends |
            tar -x -C "$tmp" || die "git archive of $SANA_REV failed (the partial extract is in $tmp)"
        [[ -f "$tmp/models/Open-H3-Regenerate-2K/open_h3/pipeline.py" ]] ||
            die "the extract of $SANA_REV has no models/Open-H3-Regenerate-2K/open_h3/pipeline.py (see $tmp)"
        echo "$SANA_REV" > "$tmp/.lf-studio-commit" || die "cannot write to $tmp"
        chmod 755 "$tmp" && mv -T "$tmp" "$SRC" || die "cannot move $tmp to $SRC"
    fi
fi
link=
[[ -L "$PKG_DIR" ]] && link=$(readlink -- "$PKG_DIR")
if [[ "$link" == "$SRC/models/Open-H3-Regenerate-2K" ]]; then
    say "LFSTUDIO_UPSCALE_PKG: $PKG_DIR -> $SRC/models/Open-H3-Regenerate-2K"
elif [[ -e "$PKG_DIR" || -L "$PKG_DIR" ]] && [[ "$link" != "$SOURCES"/sana-*/models/Open-H3-Regenerate-2K ]]; then
    die "$PKG_DIR (LFSTUDIO_UPSCALE_PKG) exists and is not a link this script made; move it away, or point LFSTUDIO_UPSCALE_PKG at $SRC/models/Open-H3-Regenerate-2K"
else                                                   # none yet, or this script's link to an earlier extract
    run mkdir -p "$(dirname -- "$PKG_DIR")" || die "cannot create $(dirname -- "$PKG_DIR")"
    run ln -sfn "$SRC/models/Open-H3-Regenerate-2K" "$PKG_DIR" || die "cannot link $PKG_DIR"
fi

step "venv $VENV from install/freeze/regen2k-lock.txt (uv, CPython $PY_VERSION)"
state=
[[ -f "$MARK" ]] && state=$(cat "$MARK")
if [[ "$state" == "done $LOCK_SHA" && "$FORCE" != 1 ]]; then
    say "already there (FORCE=1 builds it again)"
else
    fresh=1
    if [[ -e "$VENV" ]]; then
        if [[ -z "$state" ]]; then
            die "$VENV exists but this script did not make it (no $(basename "$MARK")); move it away and run this again"
        elif [[ "$state" == "building $LOCK_SHA" && "$FORCE" != 1 ]]; then
            say "an earlier run stopped while installing into it: going on"
            fresh=0
        else
            old="$ENVS/regen2k.$(stamp)"
            if [[ "$FORCE" == 1 ]]; then why="FORCE=1"; else why="it was built from another lock"; fi
            say "$why: moving it to $old (delete that when the new one works)"
            run mv -T "$VENV" "$old" || die "cannot move $VENV to $old"
        fi
    fi
    if [[ "$fresh" == 1 ]]; then
        run mkdir -p "$ENVS" || die "cannot create $ENVS"
        run "$UV" venv --python "$PY_VERSION" "$VENV" || die "uv venv failed"   # no pip in it: no bin/pip to misuse
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "+ echo building $LOCK_SHA > $MARK"
        else
            echo "building $LOCK_SHA" > "$MARK" || die "cannot write $MARK"
        fi
    fi
    run "$UV" pip install --python "$PY" --no-deps --require-hashes --only-binary :all: \
        --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match -r "$LOCK" ||
        die "the packages of install/freeze/regen2k-lock.txt did not install into $VENV; run this again to go on"
    # Sol-Attn from the extract, not editable (NVIDIA's pip install -e would write into the source tree); its build
    # uses the lock's setuptools
    run "$UV" pip install --python "$PY" --no-deps --no-build-isolation "$SRC/techniques/sparse_backends" ||
        die "Sol-Attn (techniques/sparse_backends) did not install into $VENV; run this again to go on"
    # without the GPU: the packages import, NVIDIA's models are in this diffusers, and the versions are the lock's
    run env CUDA_VISIBLE_DEVICES= "$PY" -I -B -c 'import torch, triton, diffusers, transformers, PIL, sol_attn
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Transformer3DModel
print("torch", torch.__version__, "cuDNN", torch.backends.cudnn.version(), "triton", triton.__version__, "diffusers",
      diffusers.__version__, "transformers", transformers.__version__, "Pillow", PIL.__version__)
assert torch.__version__ == "2.11.0+cu130" and diffusers.__version__ == "0.40.0", "not the locked versions"' ||
        die "the import check of $VENV failed; run this again, or FORCE=1 to build it anew"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "+ echo done $LOCK_SHA > $MARK"
    else
        echo "done $LOCK_SHA" > "$MARK" || die "cannot write $MARK"
    fi
fi

if [[ "$ENV_ONLY" == 1 ]]; then
    say "done: the environment $VENV (--env-only: the Upscale's weights and check left out; run this again without it for the Upscale)"
    exit 0
fi

step "weights -> $WEIGHTS (Efficient-Large-Model/Open-H3-Regenerate-2K at 5809c33, and the AdaLN table)"
run "$PY" -I -B "$INSTALL_DIR/regen2k_weights.py" --weights "$WEIGHTS" --upscaler-copy "$UPSCALER_COPY" ||
    die "the weights are not complete (above); run this again to go on"

step "check: the package, the weights and the base model $BASE (upscale/regen2k_job.py)"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "+ check_weights of upscale/regen2k_job.py: $PKG_DIR, $WEIGHTS, $BASE   (dry run: not run)"
else
    "$PY" -I -B -c 'import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("regen2k_job", sys.argv[1])
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)
try:
    job.check_weights({"package": Path(sys.argv[2]), "weights": Path(sys.argv[3]), "base_model": Path(sys.argv[4]),
                       "output": Path("/")})
except job.JobError as error:
    sys.exit(f"ERROR: {error}")' "$REPO/upscale/regen2k_job.py" "$PKG_DIR" "$WEIGHTS" "$BASE" ||
        die "the Upscale's files are not complete (above); the base model comes with 20-weights.sh (LFSTUDIO_UPSCALE_BASE)"
fi
say "done. To turn the experimental Upscale on, set LFSTUDIO_UPSCALE=1 in $ENV_FILE and restart the studio (bin/lf-studio --check checks it first)"
