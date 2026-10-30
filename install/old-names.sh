# shellcheck shell=bash
# The stop for the names up to 0.4, when LF Studio was called Sol-H3 Studio. 0.6 no longer reads them
# (docs/INSTALL.md, "From 0.5 to 0.6"). Sourced by bin/lf-studio and install/env.sh, which call old_names_scan before
# they read the settings file and old_names_stop right after. The stop exits before anything starts, with one message
# that names each old name still in use and its new name. bin/lf-cli has its own (old_names). This file goes in a later
# release.

# The settings LFSTUDIO_X whose name up to 0.4 was SOLH3_X. Other SOLH3_ names (NVIDIA's own, for example) don't count.
OLD_SETTINGS="ALLOWED_HOSTS BACKEND BENCH_LOGS CHECKPOINTS DRY_RUN ENV GUARD_GIB HOME HOST LOG MAX_QUEUED MCP_MAX_WAIT_S
    MCP_OUT_ANYWHERE MCP_WAIT_S MIN_FREE_GIB MIN_FREE_VRAM_GIB OUT_DIR PATHS PIDFILE PORT PRELOAD PROMPT_CACHE_FROM
    PUBLISH_DIR PYTHON RESHAPE SANA SANA_COMMIT SANA_URL START_TIMEOUT_S STATE_DIR STOP_WAIT_S TASKS TRASH_DAYS URL"
OLD_FOUND=()    # the message's lines
OLD_IN_ENV=()   # the ones set in the environment, before the settings file adds any
for _x in $OLD_SETTINGS; do _v=SOLH3_$_x; [[ -z "${!_v+set}" ]] || OLD_IN_ENV+=("$_x"); done
unset _x _v

old_setting() { [[ " ${OLD_SETTINGS//[[:space:]]/ } " == *" $1 "* ]]; }

# old_names_scan FILE: FILE is the settings file about to be read. Lists the environment's old names, the launcher's,
# and each old name in the settings files, as a key (SOLH3_X=) and in a value ($SOLH3_X, ${SOLH3_X...}). The files:
# FILE, the one the old SOLH3_ENV names, and the old settings folder's while that folder counts (as in 0.5: while
# ~/.config/lf-studio/env is missing and no setting, new or old name, names a settings file). A setting that names a
# file in the old folder that is no longer there (the folder was moved) is listed with its new place, whose file is
# scanned. An old name in one of FILE's values that nothing has set by then would end the reading of FILE with a parse
# error, or quietly give its default: it is set empty, so FILE is read and old_names_stop gives the message.
old_names_scan() {
    local f line n x y rest named=${LFSTUDIO_ENV:-${SOLH3_ENV:-}} old=$HOME/.config/sol-h3-studio
    local new=$HOME/.config/lf-studio ref='[$][{]?SOLH3_([A-Z0-9_]+)(.*)$' files=("$1")
    local -A seen=() set_before=()
    for x in "${OLD_IN_ENV[@]}"; do OLD_FOUND+=("SOLH3_$x (environment) -> LFSTUDIO_$x"); set_before[$x]=1; done
    [[ "${0##*/}" == sol-h3-studio ]] && OLD_FOUND+=("bin/sol-h3-studio -> bin/lf-studio (the launcher)")
    [[ -z "${LFSTUDIO_ENV:-}" && -n "${SOLH3_ENV:-}" ]] && files+=("$SOLH3_ENV")
    if [[ "$named" == "$old/"* && ! -e "$named" ]]; then
        OLD_FOUND+=("$named -> $new/${named#"$old/"} (the settings file: its folder was moved)")
        files+=("$new/${named#"$old/"}")
    elif [[ -z "$named" && -e "$old" && ! -e "$new/env" ]]; then
        OLD_FOUND+=("$old -> $new (the settings folder)")
        files+=("$old/env")
    fi
    for f in "${files[@]}"; do
        [[ -r "$f" && -f "$f" && -z "${seen[$f]:-}" ]] || continue
        seen[$f]=1
        n=0
        while IFS= read -r line || [[ -n "$line" ]]; do
            n=$((n + 1))
            [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
            x=${BASH_REMATCH[2]#SOLH3_} rest=${BASH_REMATCH[3]}
            [[ "$x" != "${BASH_REMATCH[2]}" ]] && old_setting "$x" || x=
            [[ -n "$x" ]] && OLD_FOUND+=("SOLH3_$x ($f line $n) -> LFSTUDIO_$x")
            while [[ "$rest" =~ $ref ]]; do
                y=${BASH_REMATCH[1]} rest=${BASH_REMATCH[2]}
                old_setting "$y" || continue
                OLD_FOUND+=("\$SOLH3_$y ($f line $n) -> \$LFSTUDIO_$y")
                if [[ "$f" == "$1" && -z "${set_before[$y]:-}" ]]; then declare -g "SOLH3_$y="; set_before[$y]=1; fi
            done
            [[ -n "$x" && "$f" == "$1" ]] && set_before[$x]=1
        done < "$f"
    done
}

# old_names_stop [install]: right after the settings file is read. Adds the old state folder while it counts (while it
# has the clip history and the new one has none, and no setting names a state folder), with "install" (install/env.sh:
# the install steps and bench/*.sh) also the venvs' done markers and the checkout's copy of the applied patch, then
# stops if anything was found.
old_names_stop() {
    local v dir home sana
    if [[ -z "${LFSTUDIO_STATE_DIR:-}${SOLH3_STATE_DIR:-}" && -e "$HOME/.local/state/sol-h3-studio/jobs.db" &&
            ! -e "$HOME/.local/state/lf-studio/jobs.db" ]]; then
        OLD_FOUND+=("$HOME/.local/state/sol-h3-studio -> $HOME/.local/state/lf-studio (the state folder: the clip history)")
    fi
    if [[ "${1:-}" == install ]]; then
        home=${LFSTUDIO_HOME:-${SOLH3_HOME:-$HOME/sol-h3}}
        for v in "$home/envs/stage1" "$home/envs/stage2"; do
            [[ -e "$v/.sol-h3-studio-done" && ! -e "$v/.lf-studio-done" ]] &&
                OLD_FOUND+=("$v/.sol-h3-studio-done -> $v/.lf-studio-done (the venv's done marker)")
        done
        sana=${LFSTUDIO_SANA:-${SOLH3_SANA:-$home/Sana}}
        if dir=$(git -C "$sana" rev-parse --absolute-git-dir 2>/dev/null) &&
                [[ -e "$dir/sol-h3-studio-applied.patch" && ! -e "$dir/lf-studio-applied.patch" ]]; then
            OLD_FOUND+=("$dir/sol-h3-studio-applied.patch -> $dir/lf-studio-applied.patch (and its .base: the patch applied)")
        fi
    fi
    (( ${#OLD_FOUND[@]} > 0 )) || return 0
    {
        echo '>>> STATUS: REFUSED - LF Studio 0.6 no longer reads the names it had up to 0.4. Rename each, then try' \
            'again (docs/INSTALL.md, "From 0.5 to 0.6"):'
        printf '    %s\n' "${OLD_FOUND[@]}"
    } >&2
    exit 2
}
