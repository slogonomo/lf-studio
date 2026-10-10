#!/usr/bin/env bash
# 40-stage1.sh: build the Stage1 venv ($SOLH3_HOME/envs/stage1, Python 3.12) and the isolated FA4 target.
# Stage1 = MiniMax-H3 draft: torch 2.12.0+cu130, patched FastVideo (editable, from sources/), fastvideo-kernel
# 0.3.5 built for SM121, cuDNN frontend 1.27.0 + cuDNN 9.20.0.48 for the BSA kernel. ~22 min measured, ~7 GB.
# Pins: install/constraints/stage1-constraints.txt (the observed working set; upstream's requirements/*-observed.txt
# are inventories, not lockfiles). Deviations from upstream's lists:
#   - flashinfer-python is not installed: its release needs cudnn-frontend >= 1.29; the H3 path does not use it.
#   - FA4 target: apache-tvm-ffi 0.1.13 instead of 0.1.13rc1 (not on PyPI). PyPI marks 0.1.13 as yanked; pip prints
#     a warning and installs it anyway because the pin is exact.
# Idempotent: skipped when $ENVS/stage1/.sol-h3-studio-done exists (FORCE=1 rebuilds into the same venv).
# Do not run 40 and 50 at the same time: two nvcc builds at once can exhaust unified memory.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
[[ -d "$SOURCES/FastVideo" || "$DRY_RUN" == 1 ]] || die "no $SOURCES/FastVideo; run 30-sources.sh first"
need_cmd python3.12 "sudo apt install python3.12-venv"
[[ -x /usr/local/cuda/bin/nvcc ]] || die "CUDA 13 toolkit not found at /usr/local/cuda"
V="$ENVS/stage1"
C="$INSTALL_DIR/constraints/stage1-constraints.txt"
TORCH_IDX=https://download.pytorch.org/whl/cu130
DONE="$V/.sol-h3-studio-done"

if [[ -f "$DONE" && "$FORCE" != 1 ]]; then
    say "Stage1 env already built ($V); FORCE=1 to rebuild"
else
    step "Stage1 venv $V"
    [[ -x "$V/bin/python" ]] || run python3.12 -m venv "$V"
    P="$V/bin/pip"
    run "$P" install -U pip==25.2 setuptools wheel
    # 1) torch family (pypi as extra index for the nvidia-* wheels)
    run "$P" install --index-url "$TORCH_IDX" --extra-index-url https://pypi.org/simple \
        torch==2.12.0+cu130 torchvision==0.27.0+cu130 triton==3.7.0
    # 2) torchaudio 2.11 declares torch 2.11; the observed env pairs it with torch 2.12, so no deps
    run "$P" install --no-deps --index-url "$TORCH_IDX" torchaudio==2.11.0+cu130
    # 3) observed pins + FastVideo's runtime deps (minus the torch family, fastvideo-kernel, flashinfer)
    run "$P" install -c "$C" --extra-index-url "$TORCH_IDX" \
        diffusers==0.40.0 transformers==5.15.1 safetensors==0.8.0 accelerate==1.0.1 \
        nvidia-cudnn-frontend==1.27.0 "nvidia-cutlass-dsl[cu13]==4.8.0.dev0" cuda-python==13.3.1 \
        apache-tvm-ffi==0.1.13.post3 huggingface-hub==1.29.0 av==18.1.0 imageio==2.37.4 imageio-ffmpeg==0.6.0 \
        einops==0.9.0.dev0 \
        "scipy>=1.14.1" "six>=1.16.0" "h5py>=3.12.1" "requests>=2.32.2" "tokenizers>=0.20.1,<0.23" "sentencepiece>=0.2.0" \
        "timm>=1.0.11" "peft>=0.15.0" "opencv-python>=4.10.0.84" "pillow>=10.3.0" "wandb>=0.21.0" loguru "test-tube>=0.7.5" \
        tqdm pytest "PyYAML>=6.0.1" "protobuf>=5.28.3" "gradio==5.32.0" "moviepy>=2.0.0" flask flask_restful aiohttp aiofiles \
        cloudpickle omegaconf gpustat watch remote-pdb wheel torchdata pyarrow "datasets>=4.0.0" torchcodec "ray>=2.49.1" \
        "ftfy>=6.3.1" "fastapi>=0.129.0" "uvicorn>=0.41.0" scikit-build-core cmake ninja
    # 4) the cuDNN runtime used with the BSA path (overrides torch's pin)
    run "$P" install --no-deps nvidia-cudnn-cu13==9.20.0.48
    # 5) FastVideo from the pinned source (editable; the runtime also puts it on PYTHONPATH)
    run "$P" install --no-deps -e "$SOURCES/FastVideo"
    # 6) fastvideo-kernel 0.3.5 for SM121 (no ThunderKittens; no build isolation, so it builds against this torch)
    run git -C "$SOURCES/FastVideo" submodule update --init fastvideo-kernel/include/cutlass
    run env CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$V/bin:$PATH" \
        TORCH_CUDA_ARCH_LIST=12.1 CMAKE_BUILD_PARALLEL_LEVEL=6 MAX_JOBS=6 \
        CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=121 -DFASTVIDEO_KERNEL_BUILD_TK=OFF -DGPU_BACKEND=CUDA" \
        "$P" install --no-deps --no-build-isolation -v "$SOURCES/FastVideo/fastvideo-kernel"
    run "$P" check || true          # expected: flashinfer-python missing, nvidia-cusparselt-cu13 "not supported"
    run mkdir -p "$LOGS"
    if [[ "$DRY_RUN" != 1 ]]; then "$P" freeze > "$LOGS/stage1-freeze.txt"; fi
    run touch "$DONE"
fi

step "FA4 dependency target $DEPS/fa4 (kept apart from Stage1: FA4 and BSA use different CuTe versions)"
if [[ -d "$DEPS/fa4/quack" && "$FORCE" != 1 ]]; then
    say "already there"
else
    run "$V/bin/pip" install --no-deps --target "$DEPS/fa4" \
        nvidia-cutlass-dsl==4.6.0.dev0 nvidia-cutlass-dsl-libs-base==4.6.0.dev0 nvidia-cutlass-dsl-libs-cu13==4.6.0.dev0 \
        quack-kernels==0.5.3 apache-tvm-ffi==0.1.13 torch-c-dlpack-ext==0.1.5
fi
EXISTING="$SOLH3_HOME/existing-paths.json"
if [[ ! -f "$EXISTING" || "$FORCE" == 1 ]]; then
    echo "+ write $EXISTING (fa4_dependencies -> $DEPS/fa4)"
    if [[ "$DRY_RUN" != 1 ]]; then
        python3.12 -c 'import json, sys; print(json.dumps({"fa4_dependencies": sys.argv[1]}))' "$DEPS/fa4" > "$EXISTING"
    fi
fi

step "quick import check (CPU only)"
# `import fastvideo` needs a visible GPU (Triton picks its driver at import), so FastVideo is only located here,
# and fastvideo-kernel's compiled extension is loaded (that needs no GPU).
if [[ "$DRY_RUN" != 1 ]]; then
    CUDA_VISIBLE_DEVICES='' "$V/bin/python" -B -c '
import glob, importlib.util, os, torch, torchvision, torchaudio, diffusers, transformers, cudnn
for m in ("fastvideo", "fastvideo_kernel"):
    assert importlib.util.find_spec(m), m + " is not installed"
so = glob.glob(os.path.join(os.path.dirname(importlib.util.find_spec("fastvideo_kernel").origin), "_C", "*.so"))
assert so, "fastvideo-kernel has no compiled extension"
torch.ops.load_library(so[0])
print("torch", torch.__version__, "torchaudio", torchaudio.__version__, "cudnn-frontend", cudnn.__version__)' ||
        die "the Stage1 env does not import"
fi
say "done. Run 10-sana.sh again to run the CPU tests with this env. Next: install/50-stage2.sh"
