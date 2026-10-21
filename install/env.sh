# shellcheck shell=bash disable=SC2034  # variables here are used by the scripts that source this file
# Settings for install/*.sh and bench/*.sh. Not run directly.
# Reads ${SOLH3_ENV:-~/.config/sol-h3-studio/env} with the same rules as bin/sol-h3-studio (one KEY=VALUE per line,
# shell quoting, $VAR and ~ expand; variables set before the file is read win; within the file a later line for a
# key overrides an earlier one) and fills in the defaults.
SOLH3_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

load_env_file() {   # the same function as in bin/sol-h3-studio
    local file=$1 line key value n=0
    [[ -r "$file" ]] || return 0
    local -A preset=()
    while IFS= read -r key; do preset[$key]=1; done < <(compgen -v)
    while IFS= read -r line || [[ -n "$line" ]]; do
        n=$((n + 1))
        line=${line%$'\r'}
        [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        key=${BASH_REMATCH[2]}
        value=${BASH_REMATCH[3]}
        [[ -n "${preset[$key]:-}" ]] && continue
        # try it in a subshell first: under set -u an unset $VAR would otherwise end the script without this message
        if ! (set -u; eval "export ${key}=${value}") || ! eval "export ${key}=${value}"; then
            echo "sol-h3-studio: cannot parse $file line $n: $line" >&2
            exit 2
        fi
    done < "$file"
}
ENV_FILE=${SOLH3_ENV:-$HOME/.config/sol-h3-studio/env}
load_env_file "$ENV_FILE"

export SOLH3_HOME=${SOLH3_HOME:-$HOME/sol-h3}
export SOLH3_SANA=${SOLH3_SANA:-$SOLH3_HOME/Sana}
export SOLH3_CHECKPOINTS=${SOLH3_CHECKPOINTS:-$SOLH3_HOME/checkpoints}
export SOLH3_PYTHON=${SOLH3_PYTHON:-$SOLH3_HOME/envs/stage1/bin/python}
[[ -n "${SOLH3_PATHS:-}" ]] || SOLH3_PATHS="$SOLH3_HOME/paths-{task}.json"
export SOLH3_PATHS
export SOL_H3_SPARK_RUNTIME_ROOT=${SOL_H3_SPARK_RUNTIME_ROOT:-$SOLH3_HOME/runtime}
export SOL_H3_SPARK_QWEN_IMAGE=${SOL_H3_SPARK_QWEN_IMAGE:-sol-h3-spark-qwen}
export SOL_H3_SPARK_QWEN_WEIGHTS_ROOT=${SOL_H3_SPARK_QWEN_WEIGHTS_ROOT:-$SOLH3_CHECKPOINTS/Comfy-MiniMax-H3}
export SOL_H3_SPARK_COMFY_ROOT=${SOL_H3_SPARK_COMFY_ROOT:-$SOLH3_HOME/sources/ComfyUI}
SOLH3_TASKS=${SOLH3_TASKS:-t2va fl2va ref2va}   # which tasks to download and prepare
# The machine (docs/RTX5090.md): spark (default) = NVIDIA's Sol-H3-Spark package on a DGX Spark, rtx5090 =
# Sol-H3-RTX5090 on a workstation with one RTX 5090. It picks the package, the Sana commit, the patch and the
# install variants. install/00-check.sh and 90-env.sh detect the GPU and suggest or write it.
SOLH3_BACKEND_SET=0
[[ -n "${SOLH3_BACKEND:-}" ]] && SOLH3_BACKEND_SET=1   # set by the user (00-check.sh and 90-env.sh say so)
export SOLH3_BACKEND=${SOLH3_BACKEND:-spark}
case "$SOLH3_BACKEND" in
    spark)
        PKG_NAME=Sol-H3-Spark
        PATCH_NAME=sol-h3-studio.patch
        BASE_COMMIT_FILE=BASE_COMMIT
        OTHER_PKG_NAME=Sol-H3-RTX5090
        OTHER_BACKEND=rtx5090
        ;;
    rtx5090)
        PKG_NAME=Sol-H3-RTX5090
        PATCH_NAME=sol-h3-studio-rtx5090.patch
        BASE_COMMIT_FILE=BASE_COMMIT_RTX5090
        OTHER_PKG_NAME=Sol-H3-Spark
        OTHER_BACKEND=spark
        ;;
    *)
        echo "ERROR: SOLH3_BACKEND=$SOLH3_BACKEND: must be spark or rtx5090 (see bin/env.example)" >&2
        exit 2
        ;;
esac
PKG="$SOLH3_SANA/models/minimax_h3/$PKG_NAME"
SOURCES="$SOLH3_HOME/sources"
ENVS="$SOLH3_HOME/envs"
DEPS="$SOLH3_HOME/dependencies"
PROMPT_CACHE="$SOLH3_HOME/fixed-prompt.pt"
LOGS="$SOLH3_HOME/logs"
BASE_COMMIT=$(tr -d '[:space:]' < "$SOLH3_REPO/patches/$BASE_COMMIT_FILE")
PATCH="$SOLH3_REPO/patches/$PATCH_NAME"
SANA_URL=${SOLH3_SANA_URL:-https://github.com/NVlabs/Sana}
# The Python that makes the venvs and runs NVIDIA's tools (prepare.py, download_checkpoints.py): the system's
# python3.12 on a DGX Spark; on the RTX 5090, CPython 3.12.15 and 3.13.16 installed by uv (install/05-python.sh)
# under $SOLH3_HOME/python, so the system's Python does not matter. There pip's and uv's caches go under
# $SOLH3_HOME/cache too (pip's: ~6.3 GB of wheels measured after steps 40 and 50), so a rerun or FORCE=1 reuses
# them and the install writes nothing outside $SOLH3_HOME. The Spark keeps pip's default cache (~/.cache/pip), as
# before.
if [[ "$SOLH3_BACKEND" == rtx5090 ]]; then
    UV="$SOLH3_HOME/bin/uv"
    export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$SOLH3_HOME/python}
    export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SOLH3_HOME/cache/pip}
    export UV_CACHE_DIR=${UV_CACHE_DIR:-$SOLH3_HOME/cache/uv}
    PY312_VERSION=3.12.15
    PY313_VERSION=3.13.16
    HOST_PY="$UV_PYTHON_INSTALL_DIR/cpython-$PY312_VERSION-linux-x86_64-gnu/bin/python3.12"
    HOST_PY313="$UV_PYTHON_INSTALL_DIR/cpython-$PY313_VERSION-linux-x86_64-gnu/bin/python3.13"   # Stage 2
    # the RTX 5090 patch's memory features keep their files next to the checkpoints (bin/sol-h3-studio, env.example)
    export SOL_H3_FILEBACKED_DIR=${SOL_H3_FILEBACKED_DIR:-$(dirname -- "$SOLH3_CHECKPOINTS")/scratch}
    export SOL_H3_LTX_BLOCKS=${SOL_H3_LTX_BLOCKS:-$(dirname -- "$SOLH3_CHECKPOINTS")/derived/ltx-fused-blocks}
    export SOL_H3_STAGE1_SNAPSHOTS=${SOL_H3_STAGE1_SNAPSHOTS:-$(dirname -- "$SOLH3_CHECKPOINTS")/derived/stage1-snapshots}
else
    HOST_PY=python3.12
fi

manifest_for() { printf '%s' "${SOLH3_PATHS//\{task\}/$1}"; }
mem_avail_gib() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }

# need_host_py [HINT]: the Python above exists (a dry run only says when it does not). On a Spark it is
# need_cmd python3.12 [HINT], with the same message.
need_host_py() {
    if [[ "$SOLH3_BACKEND" == rtx5090 ]]; then
        [[ -x "$HOST_PY" ]] && return 0
        [[ "${DRY_RUN:-0}" == 1 ]] && { echo "(dry run: $HOST_PY does not exist yet; install/05-python.sh installs it)"; return 0; }
        echo "ERROR: no Python at $HOST_PY; run install/05-python.sh first" >&2
        exit 1
    fi
    command -v python3.12 >/dev/null && return 0
    echo "ERROR: python3.12 not found${1:+ ($1)}" >&2
    exit 1
}

# detect_backend: the backend this machine looks like, from the CPU and the GPU: spark (aarch64 and an NVIDIA GB10),
# rtx5090 (x86_64 and an RTX 5090), or nothing
detect_backend() {
    local gpu
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$(uname -m):$gpu" in
        aarch64:*GB10*) echo spark ;;
        x86_64:*"RTX 5090"*) echo rtx5090 ;;
    esac
}
