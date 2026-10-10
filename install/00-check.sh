#!/usr/bin/env bash
# 00-check.sh: check the prerequisites before installing. Changes nothing.
# Checks: aarch64, NVIDIA GB10, driver, CUDA 13 at /usr/local/cuda, python3.12 with venv, git, ffmpeg/ffprobe,
# curl, docker access, free disk (~300 GB; ~60 GB if SOLH3_CHECKPOINTS already holds the weights), Hugging Face
# login (hf auth login or HF_TOKEN) and access to the gated Lightricks/LTX-2.5 repo.
# --dry-run skips the docker and Hugging Face checks (this script changes nothing either way).
# Validated on DGX OS (Ubuntu 24.04 aarch64), driver 580.178.04, CUDA 13.0.
# shellcheck disable=SC2015  # "test && ok || bad" is safe: ok() never fails
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"

FAILS=0
ok() { echo "  ok    $*"; }
bad() { echo "  FAIL  $*"; FAILS=$((FAILS + 1)); }
note() { echo "  note  $*"; }

step "prerequisites (SOLH3_HOME=$SOLH3_HOME)"
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

# disk: weights ~224 GB (all three tasks plus the offline weights); envs and sources ~14 GB, the Qwen image ~21 GB
# (Docker's own disk) plus outputs and caches: ~60 GB. If SOLH3_CHECKPOINTS already holds weights (for example from
# NVIDIA's own install), only the ~60 GB are needed.
fs_probe() { local p=$1; while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done; printf '%s' "$p"; }
free_gb_at() { df -B1G --output=avail "$1" | tail -1 | tr -d ' '; }
home_probe=$(fs_probe "$SOLH3_HOME")
ckpt_probe=$(fs_probe "$SOLH3_CHECKPOINTS")
weights_gb=240
if [[ -d "$SOLH3_CHECKPOINTS" ]] &&
    [[ -n "$(find "$SOLH3_CHECKPOINTS" -name '*.safetensors' -print -quit 2>/dev/null)" ]]; then
    weights_gb=0
    note "weights found in $SOLH3_CHECKPOINTS; 20-weights.sh verifies them and downloads only what is missing"
fi
if [[ "$(stat -c %d "$home_probe")" == "$(stat -c %d "$ckpt_probe")" ]]; then
    need=$((weights_gb + 60)); free_gb=$(free_gb_at "$home_probe")
    (( free_gb >= need )) && ok "${free_gb} GB free at $home_probe (need ~${need} GB)" ||
        bad "only ${free_gb} GB free at $home_probe; need ~${need} GB"
else
    free_gb=$(free_gb_at "$home_probe")
    (( free_gb >= 60 )) && ok "${free_gb} GB free at $home_probe (need ~60 GB)" ||
        bad "only ${free_gb} GB free at $home_probe; need ~60 GB"
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
    bad "no Hugging Face token. Log in with the download tool (the same venv 20-weights.sh uses):
          python3.12 -m venv $ENVS/download && $ENVS/download/bin/pip install huggingface-hub==1.29.0
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
echo "All checks passed. Next: install/10-sana.sh"
