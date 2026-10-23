#!/usr/bin/env bash
# 90-env.sh: write the settings file that bin/lf-studio and bench/*.sh read: ${LFSTUDIO_ENV:-~/.config/lf-studio/env},
# from the values the install used (your environment, an existing file, or the defaults under $LFSTUDIO_HOME).
# An existing file is kept; FORCE=1 rewrites it (old copy env.bak). Then checks the result with
# bin/lf-studio --check (memory checks off, since the box may be busy right now).
# LFSTUDIO_BACKEND goes into the file too: the value the install used (set in the environment or an existing file),
# else spark. It refuses when LFSTUDIO_BACKEND was never set but this machine looks like an RTX 5090 (detect_backend:
# an NVIDIA GB10 = spark, an RTX 5090 = rtx5090), rather than write a value the install did not use. On the RTX 5090
# the launcher's memory floors are left to its defaults (docs/RTX5090.md), and the Qwen image keys are left out.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
if [[ -z "${LFSTUDIO_STATE_DIR:-}" ]]; then
    LFSTUDIO_STATE_DIR=$HOME/.local/state/lf-studio
    old_path LFSTUDIO_STATE_DIR "$HOME/.local/state/sol-h3-studio" jobs.db   # until 0.6: 0.4's, if it has the history
fi
export LFSTUDIO_STATE_DIR
detected=$(detect_backend)
if [[ "$LFSTUDIO_BACKEND_SET" != 1 && -n "$detected" && "$detected" != "$LFSTUDIO_BACKEND" ]]; then
    die "this machine looks like LFSTUDIO_BACKEND=$detected, but LFSTUDIO_BACKEND is not set (so the install used $LFSTUDIO_BACKEND).
Set LFSTUDIO_BACKEND=$detected in the environment if the install steps ran with it, then run this again."
fi
KEYS=(LFSTUDIO_BACKEND LFSTUDIO_HOME LFSTUDIO_SANA LFSTUDIO_CHECKPOINTS LFSTUDIO_PYTHON LFSTUDIO_PATHS
      SOL_H3_SPARK_RUNTIME_ROOT SOL_H3_SPARK_QWEN_IMAGE SOL_H3_SPARK_QWEN_WEIGHTS_ROOT SOL_H3_SPARK_COMFY_ROOT
      LFSTUDIO_STATE_DIR LFSTUDIO_TASKS)
DEFAULTS=(LFSTUDIO_HOST=127.0.0.1 LFSTUDIO_PORT=8190 LFSTUDIO_PUBLISH_DIR= LFSTUDIO_PRELOAD=1 LFSTUDIO_MIN_FREE_GIB=110
          LFSTUDIO_GUARD_GIB=4 LFSTUDIO_PIDFILE= SOL_H3_COMPILE_CACHE=)
if [[ "$LFSTUDIO_BACKEND" == rtx5090 ]]; then
    KEYS=(LFSTUDIO_BACKEND LFSTUDIO_HOME LFSTUDIO_SANA LFSTUDIO_CHECKPOINTS LFSTUDIO_PYTHON LFSTUDIO_PATHS
          SOL_H3_SPARK_RUNTIME_ROOT SOL_H3_SPARK_COMFY_ROOT LFSTUDIO_STATE_DIR LFSTUDIO_TASKS SOL_H3_FILEBACKED_DIR
          SOL_H3_LTX_BLOCKS
          SOL_H3_STAGE1_SNAPSHOTS)
    DEFAULTS=(LFSTUDIO_HOST=127.0.0.1 LFSTUDIO_PORT=8190 LFSTUDIO_PUBLISH_DIR= LFSTUDIO_PRELOAD=1 LFSTUDIO_GUARD_GIB=4
              LFSTUDIO_PIDFILE= SOL_H3_COMPILE_CACHE=)
fi

step "settings file $ENV_FILE"
write_env() {
    echo "# LF Studio settings, written by install/90-env.sh on $(date +%F). See bin/env.example for every key."
    local k kv
    for k in "${KEYS[@]}"; do printf '%s=%q\n' "$k" "${!k}"; done
    for kv in "${DEFAULTS[@]}"; do
        k=${kv%%=*}
        printf '%s=%q\n' "$k" "${!k-${kv#*=}}"
    done
    if [[ "$LFSTUDIO_BACKEND" == rtx5090 ]]; then
        echo "# LFSTUDIO_MIN_FREE_GIB, LFSTUDIO_MIN_FREE_VRAM_GIB: the launcher's defaults for the RTX 5090" \
            "(docs/RTX5090.md)"
    fi
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
run mkdir -p "$SOL_H3_SPARK_RUNTIME_ROOT" "$LFSTUDIO_STATE_DIR"

step "launcher check"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "+ LFSTUDIO_MIN_FREE_GIB=0 LFSTUDIO_MIN_FREE_VRAM_GIB=0 $REPO/bin/lf-studio --check"
else
    LFSTUDIO_ENV="$ENV_FILE" LFSTUDIO_MIN_FREE_GIB=0 LFSTUDIO_MIN_FREE_VRAM_GIB=0 bash "$REPO/bin/lf-studio" --check || exit 1
fi
say "done. Start the studio:  $REPO/bin/lf-studio   then open http://localhost:${LFSTUDIO_PORT:-8190}"
