#!/usr/bin/env bash
# 50-stage2.sh: build the Stage2 venv ($SOLH3_HOME/envs/stage2, Python 3.12) and the offline-context target.
# Stage2 = LTX-2.5 refine: torch 2.13.0+cu132, LTX-2 source packages (editable), ltx-kernels with the compiled
# all2all_cpp extension (required even on one GPU) for SM120/121. ~6 GB.
# Fixes over a plain install, all found on a DGX Spark:
#   - nvidia-cuda-nvcc 13.2 pulls unpinned nvidia-nvvm / nvidia-cuda-crt 13.4, whose cicc emits PTX 9.4 that ptxas
#     13.2 rejects ("Unsupported .version 9.4"). They are pinned to 13.2 here before the kernel build.
#   - There is no aarch64 torchaudio wheel for cu132, so the cu130 build is installed without deps. Its import-time
#     check refuses any CUDA minor mismatch with torch; it is relaxed to a major-version check (backup: utils.py.orig).
# RTX 5090 (SOLH3_BACKEND=rtx5090): Python 3.13.16 from uv (05-python.sh), the same packages and pins, ltx-kernels
# for SM120 only (TORCH_CUDA_ARCH_LIST=12.0) with nvcc 13.2 from pip. nvcc 13.2 accepts host compilers up to GCC 15
# (and nvcc 13.4 rejects torch's CUDA 13.2 headers), so with a newer gcc (Arch ships 16) a GCC 14 toolchain from
# conda-forge is installed with micromamba 2.9.0 (static binary, pinned SHA-256) into $SOLH3_HOME/toolchains/gcc14
# (~1.2 GB). Keep it: the built extensions carry an RPATH to its lib/ and load its libstdc++ at run time. FORCE=1
# keeps it too; FORCE=1 FORCE_TOOLCHAIN=1 installs it again (the old one is moved to $SOLH3_HOME/cache first, and
# the extensions are rebuilt against the new one). CFLAGS/CXXFLAGS=-w: glibc 2.44's _POSIX_C_SOURCE clashes with Python's
# pyconfig.h under the extension's -Werror (warnings only). No offline-context target (the prompt cache comes from a
# DGX Spark). ~10 min measured on a workstation with one RTX 5090, the ltx-kernels build ~3 min of it; ~6 GB.
# Idempotent: skipped when $ENVS/stage2/.sol-h3-studio-done exists (FORCE=1 rebuilds into the same venv). On the
# RTX 5090 a venv without that marker whose pip freeze is install/freeze/stage2-rtx5090-freeze.txt (one built by hand
# to the same recipe) is adopted: nothing is installed or rebuilt, and the marker is written once the import check
# (which loads the compiled all2all_cpp) passes.
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
    need_host_py "sudo apt install python3.12-venv"
    if [[ "$SOLH3_BACKEND" == rtx5090 && "$DRY_RUN" != 1 && ! -x "$HOST_PY313" ]]; then
        die "no Python at $HOST_PY313; run install/05-python.sh first"
    fi
fi
C="$INSTALL_DIR/constraints/stage2-constraints.txt"
IDX132=https://download.pytorch.org/whl/cu132
DONE="$V/.sol-h3-studio-done"
ADOPT=0

if [[ "$FIX_ONLY" == 1 ]]; then
    [[ -x "$V/bin/python" || "$DRY_RUN" == 1 ]] ||
        die "no venv at $V (--torchaudio-fix-only needs an existing Stage2 venv)"
    say "--torchaudio-fix-only: leaving the rest of $V as it is"
elif [[ -f "$DONE" && "$FORCE" != 1 ]]; then
    say "Stage2 env already built ($V); FORCE=1 to rebuild"
elif [[ "$SOLH3_BACKEND" == rtx5090 && "$FORCE" != 1 ]] &&
        freeze_matches "$V/bin/python" "$INSTALL_DIR/freeze/stage2-rtx5090-freeze.txt"; then
    say "Stage2 env $V has no marker but matches install/freeze/stage2-rtx5090-freeze.txt: adopted as it is (FORCE=1 rebuilds)"
    ADOPT=1
elif [[ "$SOLH3_BACKEND" == rtx5090 ]]; then
    step "Stage2 venv $V (Python $PY313_VERSION from uv)"
    [[ -x "$V/bin/python" ]] || run "$UV" venv --seed --python "$HOST_PY313" "$V"
    P=("$V/bin/python" -m pip)
    run "${P[@]}" install -U pip==25.2 setuptools wheel
    run "${P[@]}" install --index-url "$IDX132" --extra-index-url https://pypi.org/simple \
        torch==2.13.0+cu132 torchvision==0.28.0+cu132 triton==3.7.1
    # the cu130 torchaudio 2.11.0 without deps, as on the Spark (its CUDA check is relaxed below)
    run "${P[@]}" install --no-deps --index-url https://download.pytorch.org/whl/cu130 torchaudio==2.11.0+cu130
    run "${P[@]}" install -c "$C" --extra-index-url "$IDX132" \
        transformers==5.14.1 safetensors==0.8.0 accelerate==1.14.0 "nvidia-cutlass-dsl[cu13]==4.7.0" cuda-python==13.3.1 \
        huggingface-hub==1.27.0 av==18.0.0 imageio==2.37.4 imageio-ffmpeg==0.6.0 einops==0.8.2 \
        numpy "scipy>=1.14" tokenizers tqdm pillow openimageio "cloudpickle>=3.1" typing_extensions \
        "nvidia-cuda-nvcc==13.2.*" "nvidia-cuda-cccl==13.2.*" "nvidia-nvvm==13.2.*" "nvidia-cuda-crt==13.2.*" ninja
    run "${P[@]}" install --no-deps nvidia-cudnn-cu13==9.20.0.48
    run "${P[@]}" install --no-deps -e "$SOURCES/LTX-2/packages/ltx-core" -e "$SOURCES/LTX-2/packages/ltx-pipelines"
    # the host compiler for nvcc 13.2: the system's gcc up to 15, else GCC 14 from conda-forge
    build_env=(CFLAGS=-w CXXFLAGS=-w)
    tc_path=
    gcc_major=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
    if [[ -z "$gcc_major" ]] || (( gcc_major > 15 )); then
        TC="$SOLH3_HOME/toolchains/gcc14"
        MM="$SOLH3_HOME/bin/micromamba"
        step "GCC 14 toolchain $TC (the system's gcc is ${gcc_major:-missing}; nvcc 13.2 accepts up to 15)"
        if [[ -x "$TC/bin/x86_64-conda-linux-gnu-g++" && "${FORCE_TOOLCHAIN:-0}" != 1 ]]; then
            say "already there (FORCE_TOOLCHAIN=1 installs it again)"
        else
            MM_VERSION=2.9.0-0
            MM_SHA256=366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3   # the release's own .sha256
            fetch_checked "https://github.com/mamba-org/micromamba-releases/releases/download/$MM_VERSION/micromamba-linux-64" \
                "$SOLH3_HOME/cache/downloads/micromamba-$MM_VERSION/micromamba-linux-64" "$MM_SHA256"
            run mkdir -p "$(dirname "$MM")"
            run install -m 755 "$SOLH3_HOME/cache/downloads/micromamba-$MM_VERSION/micromamba-linux-64" "$MM"
            # micromamba create empties an existing prefix: an old toolchain is moved aside, not deleted
            if [[ -e "$TC" ]]; then
                run mkdir -p "$SOLH3_HOME/cache"
                run mv "$TC" "$SOLH3_HOME/cache/gcc14.$(date +%Y%m%d-%H%M%S)"
            fi
            run env MAMBA_ROOT_PREFIX="$SOLH3_HOME/toolchains/mamba" "$MM" create -y -p "$TC" \
                -c conda-forge --override-channels "gxx_linux-64=14" "gcc_linux-64=14" || die "the GCC 14 toolchain did not install"
        fi
        build_env+=(CC="$TC/bin/x86_64-conda-linux-gnu-gcc" CXX="$TC/bin/x86_64-conda-linux-gnu-g++"
                    NVCC_CCBIN="$TC/bin/x86_64-conda-linux-gnu-g++")
        tc_path="$TC/bin:"
    fi
    # ltx-kernels for SM120 with the venv's pip nvcc 13.2 (CUDA_HOME unset on purpose, as on the Spark). A build
    # directory from an earlier attempt is moved aside, not deleted.
    build="$SOURCES/LTX-2/packages/ltx-kernels/build"
    if [[ -d "$build" ]]; then
        run mkdir -p "$SOLH3_HOME/cache"
        run mv "$build" "$SOLH3_HOME/cache/ltx-kernels-build.$(date +%Y%m%d-%H%M%S)"
    fi
    run env -u CUDA_HOME -u CUDA_PATH PATH="$V/bin:$tc_path$PATH" TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=6 "${build_env[@]}" \
        "${P[@]}" install --no-deps --no-build-isolation -v -e "$SOURCES/LTX-2/packages/ltx-kernels"
    run "${P[@]}" check || true
    run mkdir -p "$LOGS"
    if [[ "$DRY_RUN" != 1 ]]; then "${P[@]}" freeze > "$LOGS/stage2-freeze.txt"; fi
    run touch "$DONE"
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
if [[ "$SOLH3_BACKEND" == rtx5090 ]]; then
    say "not needed on the RTX 5090: the prompt cache comes from a DGX Spark (80-prompt-cache.sh)"
elif [[ -d "$DEPS/offline-context/comfy_kitchen" && "$FORCE" != 1 ]]; then
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
[[ "$ADOPT" == 1 ]] && run touch "$DONE"
if [[ "$SOLH3_BACKEND" == rtx5090 ]]; then
    say "done. Next: install/80-prompt-cache.sh (no Qwen image on the RTX 5090; there 80 comes before 70)"
else
    say "done. Next: install/60-qwen-image.sh"
fi
