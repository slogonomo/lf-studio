#!/usr/bin/env bash
# 00-check.sh: check the prerequisites before installing. Changes nothing.
# DGX Spark (SOLH3_BACKEND=spark, the default): aarch64, NVIDIA GB10, driver, CUDA 13 at /usr/local/cuda, python3.12
# with venv, git, ffmpeg/ffprobe, curl, docker access, free disk (~300 GB; ~60 GB if SOLH3_CHECKPOINTS already holds
# the weights), Hugging Face login (hf auth login or HF_TOKEN) and access to the gated Lightricks/LTX-2.5 repo.
# RTX 5090 (SOLH3_BACKEND=rtx5090): x86_64, one RTX 5090, driver, git, curl, tar, sha256sum, ffmpeg/ffprobe, the
# host C++ compiler (a GCC newer than 15 gets a GCC 14 toolchain in 50-stage2.sh), free disk, Hugging Face as above.
# No CUDA toolkit, system Python or Docker is needed there.
# It also says which backend this machine looks like (SOLH3_BACKEND, see bin/env.example).
# --dry-run skips the docker and Hugging Face checks (this script changes nothing either way).
# Validated on DGX OS (Ubuntu 24.04 aarch64), driver 580.178.04, CUDA 13.0; and on Arch Linux x86_64 with one
# RTX 5090.
# shellcheck disable=SC2015  # "test && ok || bad" is safe: ok() never fails
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"

FAILS=0
ok() { echo "  ok    $*"; }
bad() { echo "  FAIL  $*"; FAILS=$((FAILS + 1)); }
note() { echo "  note  $*"; }

step "prerequisites (SOLH3_HOME=$SOLH3_HOME)"
detected=$(detect_backend)
if [[ -n "$detected" && "$detected" != "$SOLH3_BACKEND" ]]; then
    bad "this machine looks like SOLH3_BACKEND=$detected, but the install uses $SOLH3_BACKEND$([[ "$SOLH3_BACKEND_SET" == 1 ]] || echo ' (the default)'):
        export SOLH3_BACKEND=$detected, then run the install scripts again (90-env.sh writes it into the settings file)"
elif [[ -n "$detected" ]]; then
    ok "SOLH3_BACKEND=$SOLH3_BACKEND"
fi
if [[ "$SOLH3_BACKEND" == rtx5090 ]]; then
    [[ "$(uname -m)" == x86_64 ]] && ok "x86_64" || bad "this is $(uname -m); Sol-H3-RTX5090 targets x86_64"
    if command -v nvidia-smi >/dev/null; then
        mapfile -t gpus < <(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)
        gpu=${gpus[0]:-}
        [[ "$gpu" == *"RTX 5090"* ]] && ok "GPU: $gpu" || bad "GPU is '$gpu'; Sol-H3-RTX5090 needs one GeForce RTX 5090"
        (( ${#gpus[@]} <= 1 )) || note "${#gpus[@]} GPUs: the pipeline uses one; the studio sees the first (CUDA_VISIBLE_DEVICES picks another)"
        driver=${gpu##*, }
        [[ "${driver%%.*}" -ge 580 ]] 2>/dev/null && ok "driver $driver (CUDA 13 needs 580 or newer; validated with 615.78.08)" ||
            bad "driver $driver; CUDA 13 (torch cu130 and cu132) needs 580 or newer"
    else
        bad "nvidia-smi not found"
    fi
    for c in git curl tar sha256sum ffmpeg ffprobe; do
        command -v "$c" >/dev/null && ok "$c" || bad "$c not found (install it with your distribution's package manager)"
    done
    # ltx-kernels (50-stage2.sh) is compiled with nvcc 13.2, which accepts host compilers up to GCC 15
    gcc_major=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
    if [[ -z "$gcc_major" ]]; then
        note "no gcc: 50-stage2.sh installs a GCC 14 toolchain for the ltx-kernels build (micromamba, ~1.2 GB)"
    elif (( gcc_major > 15 )); then
        note "gcc $gcc_major is newer than nvcc 13.2 accepts: 50-stage2.sh installs a GCC 14 toolchain for the ltx-kernels build (micromamba, ~1.2 GB)"
    else
        ok "gcc $gcc_major (nvcc 13.2 accepts up to 15)"
    fi
    note "no Docker, CUDA toolkit or system Python needed: Qwen runs natively, nvcc comes from pip, Python from uv (05-python.sh)"
else
    [[ "$(uname -m)" == aarch64 ]] && ok "aarch64" || bad "this is $(uname -m); Sol-H3-Spark targets aarch64 (DGX Spark)"

    if command -v nvidia-smi >/dev/null; then
        gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
        [[ "$gpu" == *GB10* ]] && ok "GPU: $gpu" || bad "GPU is '$gpu'; Sol-H3-Spark needs one NVIDIA GB10 (DGX Spark)"
        driver=${gpu##*, }
        [[ "${driver%%.*}" -ge 580 ]] 2>/dev/null && ok "driver $driver (validated with 580.178.04)" ||
            note "driver $driver; this repo was validated with 580.178.04"
    else
        bad "nvidia-smi not found"
    fi

    if [[ -x /usr/local/cuda/bin/nvcc ]]; then
        cuda=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        [[ "${cuda%%.*}" == 13 ]] && ok "CUDA $cuda at /usr/local/cuda" || bad "CUDA $cuda at /usr/local/cuda; need 13.x (fastvideo-kernel build)"
    else
        bad "no CUDA toolkit at /usr/local/cuda (DGX OS ships CUDA 13)"
    fi

    if command -v python3.12 >/dev/null; then
        python3.12 -c 'import venv, ensurepip' 2>/dev/null && ok "python3.12 with venv" ||
            bad "python3.12 lacks venv/ensurepip (sudo apt install python3.12-venv)"
    else
        bad "python3.12 not found"
    fi
    for c in git curl ffmpeg ffprobe; do
        command -v "$c" >/dev/null && ok "$c" || bad "$c not found (sudo apt install $([[ $c == ff* ]] && echo ffmpeg || echo "$c"))"
    done

    if [[ "$DRY_RUN" == 1 ]]; then
        echo "+ docker info   (dry run: not run)"
    elif command -v docker >/dev/null; then
        if docker info >/dev/null 2>&1; then
            ok "docker works as $(id -un)$([[ -w /var/run/docker.sock ]] || echo ' (through sg docker: log in again to refresh your groups)')"
        else
            bad "docker does not work as $(id -un): sudo usermod -aG docker \$USER, then log in again"
        fi
        # the NVIDIA runtime, or the CDI devices of the NVIDIA Container Toolkit (not the "-nvidia" kernel name)
        docker info 2>/dev/null | grep -qiE 'Runtimes:.*nvidia|nvidia\.com/gpu' && ok "docker knows the NVIDIA runtime" ||
            note "could not confirm the NVIDIA container runtime in 'docker info' (DGX OS has it)"
    else
        bad "docker not found"
    fi
fi

# disk: weights ~224 GB (all three tasks plus the offline weights); envs and sources ~14 GB, the Qwen image ~21 GB
# (Docker's own disk) plus outputs and caches: ~60 GB. If SOLH3_CHECKPOINTS already holds weights (for example from
# NVIDIA's own install), only the ~60 GB are needed.
# RTX 5090: weights ~183 GB (no offline weights: the prompt cache comes from a DGX Spark), envs ~14 GB, the GCC
# toolchain ~1.2 GB, the pip cache ~6 GB, plus outputs and caches: ~40 GB (measured on a workstation with one RTX 5090).
rest_gb=60
[[ "$SOLH3_BACKEND" == rtx5090 ]] && rest_gb=40
fs_probe() { local p=$1; while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done; printf '%s' "$p"; }
free_gb_at() { df -B1G --output=avail "$1" | tail -1 | tr -d ' '; }
home_probe=$(fs_probe "$SOLH3_HOME")
ckpt_probe=$(fs_probe "$SOLH3_CHECKPOINTS")
weights_gb=240
[[ "$SOLH3_BACKEND" == rtx5090 ]] && weights_gb=190
if [[ -d "$SOLH3_CHECKPOINTS" ]] &&
    [[ -n "$(find "$SOLH3_CHECKPOINTS" -name '*.safetensors' -print -quit 2>/dev/null)" ]]; then
    weights_gb=0
    note "weights found in $SOLH3_CHECKPOINTS; 20-weights.sh verifies them and downloads only what is missing"
fi
if [[ "$(stat -c %d "$home_probe")" == "$(stat -c %d "$ckpt_probe")" ]]; then
    need=$((weights_gb + rest_gb)); free_gb=$(free_gb_at "$home_probe")
    (( free_gb >= need )) && ok "${free_gb} GB free at $home_probe (need ~${need} GB)" ||
        bad "only ${free_gb} GB free at $home_probe; need ~${need} GB"
else
    free_gb=$(free_gb_at "$home_probe")
    (( free_gb >= rest_gb )) && ok "${free_gb} GB free at $home_probe (need ~${rest_gb} GB)" ||
        bad "only ${free_gb} GB free at $home_probe; need ~${rest_gb} GB"
    if (( weights_gb > 0 )); then
        free_gb=$(free_gb_at "$ckpt_probe")
        (( free_gb >= weights_gb )) &&
            ok "${free_gb} GB free for the weights at $ckpt_probe (need ~${weights_gb} GB)" ||
            bad "only ${free_gb} GB free for the weights at $ckpt_probe; need ~${weights_gb} GB"
    fi
fi

# Hugging Face: logged in, and the gated LTX-2.5 terms accepted for that account
token_file=${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}
token=${HF_TOKEN:-}
[[ -z "$token" && -r "$token_file" ]] && token=$(tr -d '[:space:]' < "$token_file")
if [[ "$DRY_RUN" == 1 ]]; then
    echo "+ curl -I <one small file of Lightricks/LTX-2.5 with your HF token>   (dry run: not run)"
elif [[ -z "$token" ]]; then
    bad "no Hugging Face token. Log in with the download tool (the same venv 20-weights.sh uses$([[ "$SOLH3_BACKEND" == rtx5090 ]] && echo "; after install/05-python.sh")):
          $HOST_PY -m venv $ENVS/download && $ENVS/download/bin/pip install huggingface-hub==1.29.0
          $ENVS/download/bin/hf auth login
        or export HF_TOKEN=<a read token> for this shell."
elif command -v curl >/dev/null; then
    # HEAD of one small pinned LTX-2.5 file: 302/200 = access, 401/403 = terms not accepted or bad token
    url="https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/vae/ltx-2.5-audio-vae-bf16.safetensors"
    code=$(curl -s -o /dev/null -I -m 20 -w '%{http_code}' -H @<(printf 'Authorization: Bearer %s\n' "$token") "$url")
    case "$code" in
        200|302|307) ok "Hugging Face token can read the gated Lightricks/LTX-2.5 repo" ;;
        401|403) bad "no access to Lightricks/LTX-2.5 (HTTP $code): accept its terms at https://huggingface.co/Lightricks/LTX-2.5 with the account of your token" ;;
        *) note "could not check LTX-2.5 access (HTTP ${code:-none}); check your network" ;;
    esac
fi
unset token

echo
if (( FAILS > 0 )); then
    echo "$FAILS check(s) failed. Fix them, then run this again."
    exit 1
fi
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Checks done; the docker and Hugging Face checks were skipped in the dry run. Run it without --dry-run."
    exit 0
fi
next=install/10-sana.sh
[[ "$SOLH3_BACKEND" == rtx5090 ]] && next=install/05-python.sh   # uv and its Pythons first
echo "All checks passed. Next: $next"
