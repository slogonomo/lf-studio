#!/usr/bin/env bash
# 50-stage2.sh: build the Stage2 venv ($SOLH3_HOME/envs/stage2, Python 3.12) and the offline-context target.
# Stage2 = LTX-2.5 refine: torch 2.13.0+cu132, LTX-2 source packages (editable), ltx-kernels with the compiled
# all2all_cpp extension (required even on one GPU) for SM120/121. ~6 GB.
# Fixes over a plain install, all found on a DGX Spark:
#   - nvidia-cuda-nvcc 13.2 pulls unpinned nvidia-nvvm / nvidia-cuda-crt 13.4, whose cicc emits PTX 9.4 that ptxas
#     13.2 rejects ("Unsupported .version 9.4"). They are pinned to 13.2 here before the kernel build.
#   - There is no aarch64 torchaudio wheel for cu132, so the cu130 build is installed without deps. Its import-time
#     check refuses any CUDA minor mismatch with torch; it is relaxed to a major-version check (backup: utils.py.orig).
# Idempotent: skipped when $ENVS/stage2/.sol-h3-studio-done exists (FORCE=1 rebuilds into the same venv).
# --torchaudio-fix-only: only relax the torchaudio check in an existing $ENVS/stage2 venv (for example one you
# built yourself); installs nothing.
# Do not run 40 and 50 at the same time (two nvcc builds can exhaust unified memory).
FIX_ONLY=0
args=()
for a in "$@"; do
    if [[ "$a" == --torchaudio-fix-only ]]; then FIX_ONLY=1; else args+=("$a"); fi
done
set -- "${args[@]}"     # (source without arguments would pass the original ones on)
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
V="$ENVS/stage2"
if [[ "$FIX_ONLY" != 1 ]]; then
    [[ -d "$SOURCES/LTX-2" || "$DRY_RUN" == 1 ]] || die "no $SOURCES/LTX-2; run 30-sources.sh first"
    need_cmd python3.12 "sudo apt install python3.12-venv"
fi
C="$INSTALL_DIR/constraints/stage2-constraints.txt"
IDX132=https://download.pytorch.org/whl/cu132
DONE="$V/.sol-h3-studio-done"

if [[ "$FIX_ONLY" == 1 ]]; then
    [[ -x "$V/bin/python" || "$DRY_RUN" == 1 ]] ||
        die "no venv at $V (--torchaudio-fix-only needs an existing Stage2 venv)"
    say "--torchaudio-fix-only: leaving the rest of $V as it is"
elif [[ -f "$DONE" && "$FORCE" != 1 ]]; then
    say "Stage2 env already built ($V); FORCE=1 to rebuild"
else
    step "Stage2 venv $V"
    [[ -x "$V/bin/python" ]] || run python3.12 -m venv "$V"
    P="$V/bin/pip"
    run "$P" install -U pip==25.2 setuptools wheel
    run "$P" install --index-url "$IDX132" --extra-index-url https://pypi.org/simple \
        torch==2.13.0+cu132 torchvision==0.28.0+cu132 triton==3.7.1
    # no aarch64 torchaudio on the cu132 index: the cu130 2.11.0 wheel (same libcudart.so.13), without deps
    run "$P" install --no-deps --index-url https://download.pytorch.org/whl/cu130 torchaudio==2.11.0+cu130
    run "$P" install -c "$C" --extra-index-url "$IDX132" \
        transformers==5.14.1 safetensors==0.8.0 accelerate==1.14.0 "nvidia-cutlass-dsl[cu13]==4.7.0" cuda-python==13.3.1 \
        huggingface-hub==1.27.0 av==18.0.0 imageio==2.37.4 imageio-ffmpeg==0.6.0 einops==0.8.2 \
        numpy "scipy>=1.14" tokenizers tqdm pillow openimageio "cloudpickle>=3.1" typing_extensions \
        "nvidia-cuda-nvcc==13.2.*" "nvidia-cuda-cccl==13.2.*" "nvidia-nvvm==13.2.*" "nvidia-cuda-crt==13.2.*" ninja
    run "$P" install --no-deps nvidia-cudnn-cu13==9.20.0.48
    # LTX source packages (the runtime also puts packages/*/src on sys.path)
    run "$P" install --no-deps -e "$SOURCES/LTX-2/packages/ltx-core" -e "$SOURCES/LTX-2/packages/ltx-pipelines"
    # ltx-kernels: all2all_cpp (required), ops_cpp, blockwise_cpp and nvfp4_cpp. "12.0;12.1": LTX-2's setup.py maps
    # only 12.0 to the blockwise/nvfp4 arch lists; 12.1 alone would build blockwise for every arch.
    # Uses the venv's pip nvcc 13.2, as LTX-2's setup does (CUDA_HOME unset on purpose).
    run rm -rf "$SOURCES/LTX-2/packages/ltx-kernels/build"
    run env -u CUDA_HOME -u CUDA_PATH PATH="$V/bin:$PATH" TORCH_CUDA_ARCH_LIST="12.0;12.1" MAX_JOBS=6 \
        "$P" install --no-deps --no-build-isolation -v -e "$SOURCES/LTX-2/packages/ltx-kernels"
    run "$P" check || true          # expected: nvidia-cusparselt-cu13 "not supported on this platform"
    run mkdir -p "$LOGS"
    if [[ "$DRY_RUN" != 1 ]]; then "$P" freeze > "$LOGS/stage2-freeze.txt"; fi
    run touch "$DONE"
fi

step "torchaudio CUDA check: major version only"
fix_torchaudio() {
    "$V/bin/python" - "$@" <<'PY'
import pathlib, shutil, sys, sysconfig
path = pathlib.Path(sysconfig.get_paths()["purelib"]) / "torchaudio/_extension/utils.py"
dry = "--dry-run" in sys.argv
if not path.is_file():
    sys.exit(f"{path} not found")
text = path.read_text()
old = "        if ta_version != t_version:\n"
new = ("        # sol-h3-studio: only a CUDA major mismatch is fatal (cu130 torchaudio next to cu132 torch;\n"
       "        # no aarch64 cu132 torchaudio wheel exists)\n"
       "        if ta_version.split('.')[0] != t_version.split('.')[0]:\n")
if new in text:
    print(f"already relaxed: {path}")
elif text.count(old) == 1:
    print(f"relaxing {path} (backup {path.name}.orig)")
    if not dry:
        backup = path.with_name(path.name + ".orig")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(text.replace(old, new))
else:
    sys.exit(f"unexpected torchaudio source in {path}; check it by hand (docs/TROUBLESHOOTING.md)")
PY
}
if [[ -x "$V/bin/python" ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then fix_torchaudio --dry-run; else fix_torchaudio || die "torchaudio fix failed"; fi
else
    echo "+ (relax torchaudio/_extension/utils.py in $V; backup utils.py.orig)"
fi

if [[ "$FIX_ONLY" == 1 ]]; then
    if [[ "$DRY_RUN" != 1 ]]; then
        CUDA_VISIBLE_DEVICES='' "$V/bin/python" -c '
import torch, torchaudio
print("torch", torch.__version__, "torchaudio", torchaudio.__version__)' ||
            die "torch/torchaudio still do not import in $V"
    fi
    say "done"
    exit 0
fi

step "offline-context target $DEPS/offline-context (only for the prompt-cache builder)"
if [[ -d "$DEPS/offline-context/comfy_kitchen" && "$FORCE" != 1 ]]; then
    say "already there"
else
    run "$V/bin/pip" install --no-deps --target "$DEPS/offline-context" -r "$PKG/requirements/offline-context.txt"
fi

step "quick import check (CPU only)"
if [[ "$DRY_RUN" != 1 ]]; then
    (cd "$PKG" && CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PKG:$SOLH3_SANA" "$V/bin/python" -c '
import torch, torchaudio, all2all_cpp  # noqa: F401
print("torch", torch.__version__, "torchaudio", torchaudio.__version__)') ||
        die "the Stage2 env does not import"
fi
say "done. Next: install/60-qwen-image.sh"
