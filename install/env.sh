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
PKG="$SOLH3_SANA/models/minimax_h3/Sol-H3-Spark"
SOURCES="$SOLH3_HOME/sources"
ENVS="$SOLH3_HOME/envs"
DEPS="$SOLH3_HOME/dependencies"
PROMPT_CACHE="$SOLH3_HOME/fixed-prompt.pt"
LOGS="$SOLH3_HOME/logs"
BASE_COMMIT=$(tr -d '[:space:]' < "$SOLH3_REPO/patches/BASE_COMMIT")
PATCH="$SOLH3_REPO/patches/sol-h3-studio.patch"
SANA_URL=${SOLH3_SANA_URL:-https://github.com/NVlabs/Sana}

manifest_for() { printf '%s' "${SOLH3_PATHS//\{task\}/$1}"; }
mem_avail_gib() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }
