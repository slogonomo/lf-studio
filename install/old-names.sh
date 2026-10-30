# shellcheck shell=bash
# The stop for the names up to 0.4, when LF Studio was called Sol-H3 Studio. 0.6 no longer reads them
# (docs/INSTALL.md, "From 0.5 to 0.6"). Sourced by bin/lf-studio and install/env.sh before they read the settings file;
# old_names_stop, called right after that, stops before anything starts, with one message that names each old name
# still in use and its new name. bin/lf-cli has its own (old_names). This file goes in a later release.

# The settings LFSTUDIO_X whose name up to 0.4 was SOLH3_X. Other SOLH3_ names (NVIDIA's own, for example) don't count.
OLD_SETTINGS="ALLOWED_HOSTS BACKEND BENCH_LOGS CHECKPOINTS DRY_RUN ENV GUARD_GIB HOME HOST LOG MAX_QUEUED MCP_MAX_WAIT_S
    MCP_OUT_ANYWHERE MCP_WAIT_S MIN_FREE_GIB MIN_FREE_VRAM_GIB OUT_DIR PATHS PIDFILE PORT PRELOAD PROMPT_CACHE_FROM
    PUBLISH_DIR PYTHON RESHAPE SANA SANA_COMMIT SANA_URL START_TIMEOUT_S STATE_DIR STOP_WAIT_S TASKS TRASH_DAYS URL"
OLD_IN_ENV=()   # the ones set in the environment, before the settings file adds any
for _x in $OLD_SETTINGS; do _v=SOLH3_$_x; [[ -z "${!_v+set}" ]] || OLD_IN_ENV+=("$_x"); done
unset _x _v

# old_names_stop FILE [install]: FILE is the settings file just read. With "install" (install/env.sh: the install steps
# and bench/*.sh) also the venvs' done markers and the checkout's copy of the applied patch. The old places count where
# 0.5 still used them: while the new place is missing and no setting (new or old name) names another.
old_names_stop() {
    local f line n x v dir home sana found=() files=("$1")
    for x in "${OLD_IN_ENV[@]}"; do found+=("SOLH3_$x (environment) -> LFSTUDIO_$x"); done
    [[ "${0##*/}" == sol-h3-studio ]] && found+=("bin/sol-h3-studio -> bin/lf-studio (the launcher)")
    if [[ -z "${LFSTUDIO_ENV:-}${SOLH3_ENV:-}" && -e "$HOME/.config/sol-h3-studio" && ! -e "$HOME/.config/lf-studio/env" ]]
    then
        found+=("$HOME/.config/sol-h3-studio -> $HOME/.config/lf-studio (the settings folder)")
        files+=("$HOME/.config/sol-h3-studio/env")
    fi
    for f in "${files[@]}"; do
        [[ -r "$f" && -f "$f" ]] || continue
        n=0
        while IFS= read -r line || [[ -n "$line" ]]; do
            n=$((n + 1))
            [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?SOLH3_([A-Z0-9_]+)= ]] || continue
            x=${BASH_REMATCH[2]}
            [[ " ${OLD_SETTINGS//[[:space:]]/ } " == *" $x "* ]] && found+=("SOLH3_$x ($f line $n) -> LFSTUDIO_$x")
        done < "$f"
    done
    if [[ -z "${LFSTUDIO_STATE_DIR:-}${SOLH3_STATE_DIR:-}" && -e "$HOME/.local/state/sol-h3-studio/jobs.db" &&
            ! -e "$HOME/.local/state/lf-studio/jobs.db" ]]; then
        found+=("$HOME/.local/state/sol-h3-studio -> $HOME/.local/state/lf-studio (the state folder: the clip history)")
    fi
    if [[ "${2:-}" == install ]]; then
        home=${LFSTUDIO_HOME:-${SOLH3_HOME:-$HOME/sol-h3}}
        for v in "$home/envs/stage1" "$home/envs/stage2"; do
            [[ -e "$v/.sol-h3-studio-done" && ! -e "$v/.lf-studio-done" ]] &&
                found+=("$v/.sol-h3-studio-done -> $v/.lf-studio-done (the venv's done marker)")
        done
        sana=${LFSTUDIO_SANA:-${SOLH3_SANA:-$home/Sana}}
        if dir=$(git -C "$sana" rev-parse --absolute-git-dir 2>/dev/null) &&
                [[ -e "$dir/sol-h3-studio-applied.patch" && ! -e "$dir/lf-studio-applied.patch" ]]; then
            found+=("$dir/sol-h3-studio-applied.patch -> $dir/lf-studio-applied.patch (and its .base: the patch applied)")
        fi
    fi
    (( ${#found[@]} > 0 )) || return 0
    {
        echo '>>> STATUS: REFUSED - LF Studio 0.6 no longer reads the names it had up to 0.4. Rename each, then try' \
            'again (docs/INSTALL.md, "From 0.5 to 0.6"):'
        printf '    %s\n' "${found[@]}"
    } >&2
    exit 2
}
