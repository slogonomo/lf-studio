# shellcheck shell=bash disable=SC2034  # variables here are used by the scripts that source this file
# Settings for install/*.sh and bench/*.sh. Not run directly.
# Reads ${LFSTUDIO_ENV:-~/.config/lf-studio/env} with the same rules as bin/lf-studio (one KEY=VALUE per line,
# shell quoting, $VAR and ~ expand; variables set before the file is read win; within the file a later line for a
# key overrides an earlier one) and fills in the defaults.
LFSTUDIO_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# The same settings code as in bin/lf-studio.
# Until 0.6 a setting may also have its name up to 0.4, SOLH3_X for LFSTUDIO_X (OLD_SETTINGS; not the test hooks),
# in the environment and in the file. The new name wins over the old one; one warning lists the old names used.
OLD_SETTINGS="ALLOWED_HOSTS BACKEND BENCH_LOGS CHECKPOINTS DRY_RUN ENV GUARD_GIB HOME HOST LOG MAX_QUEUED MCP_MAX_WAIT_S
    MCP_OUT_ANYWHERE MCP_WAIT_S MIN_FREE_GIB MIN_FREE_VRAM_GIB OUT_DIR PATHS PIDFILE PORT PRELOAD PROMPT_CACHE_FROM
    PUBLISH_DIR PYTHON RESHAPE SANA SANA_COMMIT SANA_URL START_TIMEOUT_S STATE_DIR STOP_WAIT_S TASKS TRASH_DAYS URL"
declare -A OLD_USED=()    # SOLH3_X -> "" (used) or " (LFSTUDIO_X wins)"
old_env_settings() {      # until 0.6: the environment's SOLH3_X as LFSTUDIO_X, before the file is read
    local x old new
    for x in $OLD_SETTINGS; do
        old=SOLH3_$x new=LFSTUDIO_$x
        [[ -n "${!old+set}" ]] || continue
        if [[ -z "${!new+set}" ]]; then export "$new=${!old}"; OLD_USED[$old]=; else OLD_USED[$old]=" ($new wins)"; fi
        unset "$old"
    done
}
old_settings_warning() {  # until 0.6
    local k list=
    (( ${#OLD_USED[@]} > 0 )) || return 0
    for k in $(printf '%s\n' "${!OLD_USED[@]}" | sort); do list+="${list:+, }$k${OLD_USED[$k]}"; done
    echo "lf-studio: settings under their 0.4 names, which work until 0.6: $list. Rename them LFSTUDIO_*" >&2
}
# old_path VAR OLD: until 0.6, a default place that does not exist gives way to OLD, its place up to 0.4, if that exists
old_path() {
    [[ ! -e "${!1}" && -e "$2" ]] || return 0
    echo "lf-studio: using $2: ${!1} does not exist (the old place works until 0.6; docs/INSTALL.md)" >&2
    printf -v "$1" '%s' "$2"
}
load_env_file() {
    local file=$1 line key value n=0 x
    [[ -r "$file" ]] || return 0
    local -A preset=() old=() new_line=()
    while IFS= read -r key; do preset[$key]=1; done < <(compgen -v)
    for x in $OLD_SETTINGS; do old[$x]=1; local -n "SOLH3_$x=LFSTUDIO_$x"; done   # until 0.6: $SOLH3_X in a value
    while IFS= read -r line || [[ -n "$line" ]]; do
        n=$((n + 1))
        line=${line%$'\r'}
        [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        key=${BASH_REMATCH[2]}
        value=${BASH_REMATCH[3]}
        x=${key#*_}
        if [[ "$key" == SOLH3_* && -n "${old[$x]:-}" ]]; then                     # until 0.6: an old name
            [[ -n "${preset[LFSTUDIO_$x]:-}" ]] && continue
            if [[ -n "${new_line[$x]:-}" ]]; then OLD_USED[$key]=" (LFSTUDIO_$x wins)"; continue; fi
            OLD_USED[$key]=
            key=LFSTUDIO_$x
        elif [[ "$key" == LFSTUDIO_* && -z "${preset[$key]:-}" ]]; then          # wins over an old-name line
            new_line[$x]=1
            [[ -n "${OLD_USED[SOLH3_$x]+set}" ]] && OLD_USED[SOLH3_$x]=" ($key wins)"
        fi
        [[ -n "${preset[$key]:-}" ]] && continue
        # try it in a subshell first: under set -u an unset $VAR would otherwise end the script without this message
        if ! (set -u; eval "export ${key}=${value}") || ! eval "export ${key}=${value}"; then
            echo "lf-studio: cannot parse $file line $n: $line" >&2
            exit 2
        fi
    done < "$file"
}
old_env_settings
ENV_FILE=${LFSTUDIO_ENV:-$HOME/.config/lf-studio/env}
[[ -n "${LFSTUDIO_ENV:-}" ]] || old_path ENV_FILE "$HOME/.config/sol-h3-studio/env"
load_env_file "$ENV_FILE"
old_settings_warning

export LFSTUDIO_HOME=${LFSTUDIO_HOME:-$HOME/sol-h3}
export LFSTUDIO_SANA=${LFSTUDIO_SANA:-$LFSTUDIO_HOME/Sana}
export LFSTUDIO_CHECKPOINTS=${LFSTUDIO_CHECKPOINTS:-$LFSTUDIO_HOME/checkpoints}
export LFSTUDIO_PYTHON=${LFSTUDIO_PYTHON:-$LFSTUDIO_HOME/envs/stage1/bin/python}
[[ -n "${LFSTUDIO_PATHS:-}" ]] || LFSTUDIO_PATHS="$LFSTUDIO_HOME/paths-{task}.json"
export LFSTUDIO_PATHS
export SOL_H3_SPARK_RUNTIME_ROOT=${SOL_H3_SPARK_RUNTIME_ROOT:-$LFSTUDIO_HOME/runtime}
export SOL_H3_SPARK_QWEN_IMAGE=${SOL_H3_SPARK_QWEN_IMAGE:-sol-h3-spark-qwen}
export SOL_H3_SPARK_QWEN_WEIGHTS_ROOT=${SOL_H3_SPARK_QWEN_WEIGHTS_ROOT:-$LFSTUDIO_CHECKPOINTS/Comfy-MiniMax-H3}
export SOL_H3_SPARK_COMFY_ROOT=${SOL_H3_SPARK_COMFY_ROOT:-$LFSTUDIO_HOME/sources/ComfyUI}
LFSTUDIO_TASKS=${LFSTUDIO_TASKS:-t2va fl2va ref2va}   # which tasks to download and prepare
# The machine (docs/RTX5090.md): spark (default) = NVIDIA's Sol-H3-Spark package on a DGX Spark, rtx5090 =
# Sol-H3-RTX5090 on a workstation with one RTX 5090. It picks the package, the Sana commit, the patch and the
# install variants. install/00-check.sh and 90-env.sh detect the GPU and suggest or write it.
LFSTUDIO_BACKEND_SET=0
[[ -n "${LFSTUDIO_BACKEND:-}" ]] && LFSTUDIO_BACKEND_SET=1   # set by the user (00-check.sh and 90-env.sh say so)
export LFSTUDIO_BACKEND=${LFSTUDIO_BACKEND:-spark}
case "$LFSTUDIO_BACKEND" in
    spark)
        PKG_NAME=Sol-H3-Spark
        PATCH_NAME=lf-studio.patch
        BASE_COMMIT_FILE=BASE_COMMIT
        ;;
    rtx5090)
        PKG_NAME=Sol-H3-RTX5090
        PATCH_NAME=lf-studio-rtx5090.patch
        BASE_COMMIT_FILE=BASE_COMMIT_RTX5090
        ;;
    *)
        echo "ERROR: LFSTUDIO_BACKEND=$LFSTUDIO_BACKEND: must be spark or rtx5090 (see bin/env.example)" >&2
        exit 2
        ;;
esac
PKG="$LFSTUDIO_SANA/models/minimax_h3/$PKG_NAME"
SOURCES="$LFSTUDIO_HOME/sources"
ENVS="$LFSTUDIO_HOME/envs"
DEPS="$LFSTUDIO_HOME/dependencies"
PROMPT_CACHE="$LFSTUDIO_HOME/fixed-prompt.pt"
LOGS="$LFSTUDIO_HOME/logs"
BASE_COMMIT=$(tr -d '[:space:]' < "$LFSTUDIO_REPO/patches/$BASE_COMMIT_FILE")
PATCH="$LFSTUDIO_REPO/patches/$PATCH_NAME"
SANA_URL=${LFSTUDIO_SANA_URL:-https://github.com/NVlabs/Sana}
# The Python that makes the venvs and runs NVIDIA's tools (prepare.py, download_checkpoints.py): the system's
# python3.12 on a DGX Spark; on the RTX 5090, CPython 3.12.15 and 3.13.16 installed by uv (install/05-python.sh)
# under $LFSTUDIO_HOME/python, so the system's Python does not matter. There pip's and uv's caches go under
# $LFSTUDIO_HOME/cache too (pip's: ~6.3 GB of wheels measured after steps 40 and 50), so a rerun or FORCE=1 reuses
# them and the install writes nothing outside $LFSTUDIO_HOME. The Spark keeps pip's default cache (~/.cache/pip), as
# before.
if [[ "$LFSTUDIO_BACKEND" == rtx5090 ]]; then
    UV="$LFSTUDIO_HOME/bin/uv"
    export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$LFSTUDIO_HOME/python}
    export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$LFSTUDIO_HOME/cache/pip}
    export UV_CACHE_DIR=${UV_CACHE_DIR:-$LFSTUDIO_HOME/cache/uv}
    PY312_VERSION=3.12.15
    PY313_VERSION=3.13.16
    HOST_PY="$UV_PYTHON_INSTALL_DIR/cpython-$PY312_VERSION-linux-x86_64-gnu/bin/python3.12"
    HOST_PY313="$UV_PYTHON_INSTALL_DIR/cpython-$PY313_VERSION-linux-x86_64-gnu/bin/python3.13"   # Stage 2
    # the RTX 5090 patch's memory features keep their files next to the checkpoints (bin/lf-studio, env.example)
    export SOL_H3_FILEBACKED_DIR=${SOL_H3_FILEBACKED_DIR:-$(dirname -- "$LFSTUDIO_CHECKPOINTS")/scratch}
    export SOL_H3_LTX_BLOCKS=${SOL_H3_LTX_BLOCKS:-$(dirname -- "$LFSTUDIO_CHECKPOINTS")/derived/ltx-fused-blocks}
    export SOL_H3_STAGE1_SNAPSHOTS=${SOL_H3_STAGE1_SNAPSHOTS:-$(dirname -- "$LFSTUDIO_CHECKPOINTS")/derived/stage1-snapshots}
else
    HOST_PY=python3.12
fi

manifest_for() { printf '%s' "${LFSTUDIO_PATHS//\{task\}/$1}"; }
mem_avail_gib() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }

# need_host_py [HINT]: the Python above exists (a dry run only says when it does not). On a Spark it is
# need_cmd python3.12 [HINT], with the same message.
need_host_py() {
    if [[ "$LFSTUDIO_BACKEND" == rtx5090 ]]; then
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
