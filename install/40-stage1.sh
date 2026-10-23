#!/usr/bin/env bash
# 40-stage1.sh: build the Stage1 venv ($LFSTUDIO_HOME/envs/stage1, Python 3.12) and the isolated FA4 target.
# Stage1 = MiniMax-H3 draft: torch 2.12.0+cu130, patched FastVideo (editable, from sources/), fastvideo-kernel
# 0.3.5 built for SM121, cuDNN frontend 1.27.0 + cuDNN 9.20.0.48 for the BSA kernel. ~22 min measured, ~7 GB.
# Pins: install/constraints/stage1-constraints.txt (the observed working set; upstream's requirements/*-observed.txt
# are inventories, not lockfiles). Deviations from upstream's lists:
#   - flashinfer-python is not installed: its release needs cudnn-frontend >= 1.29; the H3 path does not use it.
#   - FA4 target: apache-tvm-ffi 0.1.13 instead of 0.1.13rc1 (not on PyPI). PyPI marks 0.1.13 as yanked; pip prints
#     a warning and installs it anyway because the pin is exact.
#   - no Gradio or MoviePy (#69): FastVideo declares them, but only its Gradio demos import them, and both cap Pillow
#     below 12. Pillow is 13.0.0 (common.sh), on both backends, and `pip check` lists the two as missing.
# RTX 5090 (LFSTUDIO_BACKEND=rtx5090): the venv comes from uv (Python 3.12.15, 05-python.sh) and also runs the Qwen
# prompt encoder natively (no Docker), so it gets ComfyUI's requirements, comfy-kitchen and comfy-aimdo too. No
# fastvideo-kernel (Sol-H3-RTX5090 does not use it; its PyPI wheel bundles its own libcuda) and no CUDA toolkit.
# Pins: install/constraints/stage1-rtx5090-constraints.txt. ~25 min measured on a workstation with one RTX 5090,
# most of it downloads at ~9 MB/s; ~8 GB.
# Idempotent: skipped when $ENVS/stage1/.lf-studio-done (until 0.6 also .sol-h3-studio-done, its name up to 0.4) exists
# (FORCE=1 rebuilds into the same venv). On the RTX 5090 a venv without that marker whose pip freeze is
# install/freeze/stage1-rtx5090-freeze.txt (one built by hand to the same recipe) is adopted: nothing is installed, and
# the marker is written once the import check passes. Every run (built, adopted or skipped) then removes Gradio and
# MoviePy and brings Pillow to 13.0.0 (pillow_update in common.sh, #69), after pip 26.2.2 and certifi >= 2026.10.19
# (pip_update in common.sh), which a new venv also gets before anything else is installed.
# A venv without the marker (one built before the markers, or by hand) is built into again, so to update an existing
# install use --pillow-only: only the update steps (pip and certifi, then Pillow) on the existing $ENVS/stage1 venv,
# nothing else (docs/INSTALL.md, "From 0.4 to 0.5").
# Do not run 40 and 50 at the same time: two nvcc builds at once can exhaust unified memory.
PILLOW_ONLY=0
args=()
for a in "$@"; do
    if [[ "$a" == --pillow-only ]]; then PILLOW_ONLY=1; else args+=("$a"); fi
done
set -- "${args[@]}"     # (source without arguments would pass the original ones on)
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
if [[ "$PILLOW_ONLY" == 1 ]]; then
    V="$ENVS/stage1"
    [[ -x "$V/bin/python" || "$DRY_RUN" == 1 ]] || die "no venv at $V (--pillow-only needs an existing Stage1 venv)"
    pip_update "$V"
    pillow_update "$V" gradio gradio-client moviepy
    if [[ "$DRY_RUN" != 1 ]]; then
        CUDA_VISIBLE_DEVICES='' "$V/bin/python" -B -c 'import PIL.Image, torchvision; print("Pillow", PIL.__version__, "torchvision", torchvision.__version__)' ||
            die "the Stage1 env does not import Pillow and torchvision"
    fi
    say "done (--pillow-only)"
    exit 0
fi
[[ -d "$SOURCES/FastVideo" || "$DRY_RUN" == 1 ]] || die "no $SOURCES/FastVideo; run 30-sources.sh first"
need_host_py "sudo apt install python3.12-venv"
if [[ "$LFSTUDIO_BACKEND" == spark ]]; then
    [[ -x /usr/local/cuda/bin/nvcc ]] || die "CUDA 13 toolkit not found at /usr/local/cuda"
fi
V="$ENVS/stage1"
C="$INSTALL_DIR/constraints/stage1-constraints.txt"
[[ "$LFSTUDIO_BACKEND" == rtx5090 ]] && C="$INSTALL_DIR/constraints/stage1-rtx5090-constraints.txt"
TORCH_IDX=https://download.pytorch.org/whl/cu130
DONE="$V/.lf-studio-done"
ADOPT=0

if [[ (-f "$DONE" || -f "$V/.sol-h3-studio-done") && "$FORCE" != 1 ]]; then    # (the 0.4 marker: until 0.6)
    say "Stage1 env already built ($V); FORCE=1 to rebuild"
elif [[ "$LFSTUDIO_BACKEND" == rtx5090 && "$FORCE" != 1 ]] &&
        freeze_matches "$V/bin/python" "$INSTALL_DIR/freeze/stage1-rtx5090-freeze.txt"; then
    say "Stage1 env $V has no marker but matches install/freeze/stage1-rtx5090-freeze.txt: adopted as it is (FORCE=1 rebuilds)"
    ADOPT=1
elif [[ "$LFSTUDIO_BACKEND" == rtx5090 ]]; then
    step "Stage1 venv $V (Python $PY312_VERSION from uv; also the Qwen prompt encoder's)"
    [[ -x "$V/bin/python" ]] || run "$UV" venv --seed --python "$HOST_PY" "$V"
    P=("$V/bin/python" -m pip)
    pip_update "$V"
    run "${P[@]}" install -U setuptools wheel || die "setuptools and wheel did not install into $V"
    # 1) torch family (pypi as extra index for the nvidia-* wheels); 2) torchaudio 2.11 without deps, as on the Spark
    run "${P[@]}" install --index-url "$TORCH_IDX" --extra-index-url https://pypi.org/simple \
        torch==2.12.0+cu130 torchvision==0.27.0+cu130 triton==3.7.0
    run "${P[@]}" install --no-deps --index-url "$TORCH_IDX" torchaudio==2.11.0+cu130
    # 3) the observed pins (constraints) + FastVideo's runtime deps + what the native Qwen encoder imports
    run "${P[@]}" install -c "$C" --extra-index-url "$TORCH_IDX" \
        diffusers transformers safetensors accelerate nvidia-cudnn-frontend "nvidia-cutlass-dsl[cu13]" quack-kernels \
        apache-tvm-ffi cuda-python huggingface-hub av imageio imageio-ffmpeg einops comfy-kitchen comfy-aimdo remote-pdb \
        ftfy omegaconf \
        "scipy>=1.14.1" "six>=1.16.0" "h5py>=3.12.1" "requests>=2.32.2" "tokenizers>=0.20.1,<0.23" "sentencepiece>=0.2.0" \
        "timm>=1.0.11" "peft>=0.15.0" "opencv-python>=4.10.0.84" "pillow==$PILLOW" "wandb>=0.21.0" loguru "test-tube>=0.7.5" \
        tqdm pytest "PyYAML>=6.0.1" "protobuf>=5.28.3" flask flask_restful aiohttp aiofiles \
        cloudpickle gpustat watch wheel torchdata pyarrow "datasets>=4.0.0" torchcodec "ray>=2.49.1" \
        "fastapi>=0.129.0" "uvicorn>=0.41.0"
    # 4) ComfyUI's requirements (the Qwen encoder runs ComfyUI's code), without its torch lines
    run mkdir -p "$LOGS"
    echo "+ grep -vE '^(torch|torchvision|torchaudio)([<=>~! ].*)?\$' $SOURCES/ComfyUI/requirements.txt > $LOGS/comfy-requirements.txt"
    if [[ "$DRY_RUN" != 1 ]]; then
        grep -vE '^(torch|torchvision|torchaudio)([<=>~! ].*)?$' "$SOURCES/ComfyUI/requirements.txt" > "$LOGS/comfy-requirements.txt" ||
            die "cannot read $SOURCES/ComfyUI/requirements.txt (30-sources.sh)"
    fi
    run "${P[@]}" install -c "$C" --extra-index-url "$TORCH_IDX" -r "$LOGS/comfy-requirements.txt"
    # 5) the cuDNN runtime used with the BSA path (overrides torch's pin); 6) FastVideo from the pinned source
    run "${P[@]}" install --no-deps nvidia-cudnn-cu13==9.20.0.48
    run "${P[@]}" install --no-deps -e "$SOURCES/FastVideo"
    run "${P[@]}" check || true     # expected: fastvideo-kernel, flashinfer-python, gradio and moviepy missing
    if [[ "$DRY_RUN" != 1 ]]; then "${P[@]}" freeze > "$LOGS/stage1-freeze.txt"; fi
    run touch "$DONE"
else
    step "Stage1 venv $V"
    [[ -x "$V/bin/python" ]] || run python3.12 -m venv "$V"
    P=("$V/bin/python" -m pip)       # not bin/pip, whose shebang names the venv it was made in
    pip_update "$V"
    run "${P[@]}" install -U setuptools wheel || die "setuptools and wheel did not install into $V"
    # 1) torch family (pypi as extra index for the nvidia-* wheels)
    run "${P[@]}" install --index-url "$TORCH_IDX" --extra-index-url https://pypi.org/simple \
        torch==2.12.0+cu130 torchvision==0.27.0+cu130 triton==3.7.0
    # 2) torchaudio 2.11 declares torch 2.11; the observed env pairs it with torch 2.12, so no deps
    run "${P[@]}" install --no-deps --index-url "$TORCH_IDX" torchaudio==2.11.0+cu130
    # 3) observed pins + FastVideo's runtime deps (minus the torch family, fastvideo-kernel, flashinfer)
    run "${P[@]}" install -c "$C" --extra-index-url "$TORCH_IDX" \
        diffusers==0.40.0 transformers==5.15.1 safetensors==0.8.0 accelerate==1.0.1 \
        nvidia-cudnn-frontend==1.27.0 "nvidia-cutlass-dsl[cu13]==4.8.0.dev0" cuda-python==13.3.1 \
        apache-tvm-ffi==0.1.13.post3 huggingface-hub==1.29.0 av==18.1.0 imageio==2.37.4 imageio-ffmpeg==0.6.0 \
        einops==0.9.0.dev0 \
        "scipy>=1.14.1" "six>=1.16.0" "h5py>=3.12.1" "requests>=2.32.2" "tokenizers>=0.20.1,<0.23" "sentencepiece>=0.2.0" \
        "timm>=1.0.11" "peft>=0.15.0" "opencv-python>=4.10.0.84" "pillow==$PILLOW" "wandb>=0.21.0" loguru "test-tube>=0.7.5" \
        tqdm pytest "PyYAML>=6.0.1" "protobuf>=5.28.3" flask flask_restful aiohttp aiofiles \
        cloudpickle omegaconf gpustat watch remote-pdb wheel torchdata pyarrow "datasets>=4.0.0" torchcodec "ray>=2.49.1" \
        "ftfy>=6.3.1" "fastapi>=0.129.0" "uvicorn>=0.41.0" scikit-build-core cmake ninja
    # 4) the cuDNN runtime used with the BSA path (overrides torch's pin)
    run "${P[@]}" install --no-deps nvidia-cudnn-cu13==9.20.0.48
    # 5) FastVideo from the pinned source (editable; the runtime also puts it on PYTHONPATH)
    run "${P[@]}" install --no-deps -e "$SOURCES/FastVideo"
    # 6) fastvideo-kernel 0.3.5 for SM121 (no ThunderKittens; no build isolation, so it builds against this torch)
    run git -C "$SOURCES/FastVideo" submodule update --init fastvideo-kernel/include/cutlass
    run env CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$V/bin:$PATH" \
        TORCH_CUDA_ARCH_LIST=12.1 CMAKE_BUILD_PARALLEL_LEVEL=6 MAX_JOBS=6 \
        CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=121 -DFASTVIDEO_KERNEL_BUILD_TK=OFF -DGPU_BACKEND=CUDA" \
        "${P[@]}" install --no-deps --no-build-isolation -v "$SOURCES/FastVideo/fastvideo-kernel"
    run "${P[@]}" check || true          # expected: flashinfer-python, gradio and moviepy missing; nvidia-cusparselt-cu13 "not supported"
    run mkdir -p "$LOGS"
    if [[ "$DRY_RUN" != 1 ]]; then "${P[@]}" freeze > "$LOGS/stage1-freeze.txt"; fi
    run touch "$DONE"
fi

pip_update "$V"
pillow_update "$V" gradio gradio-client moviepy

step "FA4 dependency target $DEPS/fa4 (kept apart from Stage1: FA4 and BSA use different CuTe versions)"
if [[ -d "$DEPS/fa4/quack" && "$FORCE" != 1 ]]; then
    say "already there"
else
    run "$V/bin/python" -m pip install --no-deps --target "$DEPS/fa4" \
        nvidia-cutlass-dsl==4.6.0.dev0 nvidia-cutlass-dsl-libs-base==4.6.0.dev0 nvidia-cutlass-dsl-libs-cu13==4.6.0.dev0 \
        quack-kernels==0.5.3 apache-tvm-ffi==0.1.13 torch-c-dlpack-ext==0.1.5
fi
EXISTING="$LFSTUDIO_HOME/existing-paths.json"
if [[ ! -f "$EXISTING" || "$FORCE" == 1 ]]; then
    if [[ "$LFSTUDIO_BACKEND" == rtx5090 ]]; then
        # Sol-H3-RTX5090 also names the CuTe DSL packages of the Stage1 env (its docs/setup.md)
        CUTE="$V/lib/python$(cut -d. -f1,2 <<<"$PY312_VERSION")/site-packages/nvidia_cutlass_dsl/dsl_packages"
        echo "+ write $EXISTING (fa4_dependencies -> $DEPS/fa4, stage1_dependencies -> $CUTE)"
        if [[ "$DRY_RUN" != 1 ]]; then
            "$HOST_PY" -c 'import json, sys
print(json.dumps({"fa4_dependencies": sys.argv[1], "stage1_dependencies": sys.argv[2]}, indent=1))' "$DEPS/fa4" "$CUTE" > "$EXISTING"
        fi
    else
        echo "+ write $EXISTING (fa4_dependencies -> $DEPS/fa4)"
        if [[ "$DRY_RUN" != 1 ]]; then
            python3.12 -c 'import json, sys; print(json.dumps({"fa4_dependencies": sys.argv[1]}))' "$DEPS/fa4" > "$EXISTING"
        fi
    fi
fi

step "quick import check (CPU only)"
# `import fastvideo` needs a visible GPU (Triton picks its driver at import), so FastVideo is only located here,
# and fastvideo-kernel's compiled extension is loaded (that needs no GPU).
# RTX 5090: no fastvideo-kernel; comfy_kitchen and ComfyUI's dependencies are checked instead (the Qwen encoder).
if [[ "$DRY_RUN" != 1 && "$LFSTUDIO_BACKEND" == rtx5090 ]]; then
    CUDA_VISIBLE_DEVICES='' "$V/bin/python" -B -c '
import importlib.util, torch, torchvision, torchaudio, diffusers, transformers, cudnn, comfy_kitchen  # noqa: F401
assert importlib.util.find_spec("fastvideo"), "fastvideo is not installed"
print("torch", torch.__version__, "torchaudio", torchaudio.__version__, "cudnn-frontend", cudnn.__version__)' ||
        die "the Stage1 env does not import"
elif [[ "$DRY_RUN" != 1 ]]; then
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
[[ "$ADOPT" == 1 ]] && run touch "$DONE"
say "done. Run 10-sana.sh again to run the CPU tests with this env. Next: install/50-stage2.sh"
