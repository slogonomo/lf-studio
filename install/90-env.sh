#!/usr/bin/env bash
# 90-env.sh: write the settings file that bin/sol-h3-studio and bench/*.sh read: ${SOLH3_ENV:-~/.config/sol-h3-studio/env},
# from the values the install used (your environment, an existing file, or the defaults under $SOLH3_HOME).
# An existing file is kept; FORCE=1 rewrites it (old copy env.bak). Then checks the result with
# bin/sol-h3-studio --check (memory check off, since the box may be busy right now).
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
export SOLH3_STATE_DIR=${SOLH3_STATE_DIR:-$HOME/.local/state/sol-h3-studio}
KEYS=(SOLH3_HOME SOLH3_SANA SOLH3_CHECKPOINTS SOLH3_PYTHON SOLH3_PATHS SOL_H3_SPARK_RUNTIME_ROOT
      SOL_H3_SPARK_QWEN_IMAGE SOL_H3_SPARK_QWEN_WEIGHTS_ROOT SOL_H3_SPARK_COMFY_ROOT SOLH3_STATE_DIR SOLH3_TASKS)
DEFAULTS=(SOLH3_HOST=127.0.0.1 SOLH3_PORT=8190 SOLH3_PUBLISH_DIR= SOLH3_PRELOAD=1 SOLH3_MIN_FREE_GIB=110
          SOLH3_GUARD_GIB=4 SOLH3_PIDFILE= SOL_H3_COMPILE_CACHE=)

step "settings file $ENV_FILE"
write_env() {
    echo "# Sol-H3 Studio settings, written by install/90-env.sh on $(date +%F). See bin/env.example for every key."
    local k kv
    for k in "${KEYS[@]}"; do printf '%s=%q\n' "$k" "${!k}"; done
    for kv in "${DEFAULTS[@]}"; do
        k=${kv%%=*}
        printf '%s=%q\n' "$k" "${!k-${kv#*=}}"
    done
}
if [[ -f "$ENV_FILE" && "$FORCE" != 1 ]]; then
    say "exists; kept (FORCE=1 rewrites it). It would now contain:"
    write_env | sed 's/^/    /'
else
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "+ write $ENV_FILE:"
        write_env | sed 's/^/    /'
    else
        mkdir -p "$(dirname "$ENV_FILE")"
        [[ -f "$ENV_FILE" ]] && cp -p "$ENV_FILE" "$ENV_FILE.bak"
        write_env > "$ENV_FILE"
        say "wrote $ENV_FILE"
    fi
fi
run mkdir -p "$SOL_H3_SPARK_RUNTIME_ROOT" "$SOLH3_STATE_DIR"

step "launcher check"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "+ SOLH3_MIN_FREE_GIB=0 $REPO/bin/sol-h3-studio --check"
else
    SOLH3_ENV="$ENV_FILE" SOLH3_MIN_FREE_GIB=0 bash "$REPO/bin/sol-h3-studio" --check || exit 1
fi
say "done. Start the studio:  $REPO/bin/sol-h3-studio   then open http://localhost:${SOLH3_PORT:-8190}"
