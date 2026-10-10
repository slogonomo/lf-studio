#!/usr/bin/env bash
# 10-sana.sh: clone NVlabs/Sana (branch sol-engine) at patches/BASE_COMMIT into $SOLH3_SANA, apply
# patches/sol-h3-studio.patch, and run the package's CPU tests with the Stage1 python if it exists.
# Idempotent: an existing checkout is reused; an already applied patch is detected and left alone.
# Updates: a copy of the applied patch is kept as $SOLH3_HOME/applied.patch. After a `git pull` of this repo, run
# this script again. If the checkout carries the earlier patch and nothing else, it reverses the earlier patch
# with that copy, checks out the new patches/BASE_COMMIT if it changed, and applies the new patch. Any other local
# change in the checkout stops it with a message, before anything is changed.
# SOLH3_SANA_COMMIT overrides the commit (the patch also applies to sol-engine 670482d, checked 2026-10-09).
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
need_cmd git
COMMIT=${SOLH3_SANA_COMMIT:-$BASE_COMMIT}
APPLIED="$SOLH3_HOME/applied.patch"     # the patch as it was applied last time, for the next update

sana() { git -C "$SOLH3_SANA" "$@"; }

# fetch_and_checkout COMMIT: check out COMMIT, fetching it first if the clone does not have it
fetch_and_checkout() {
    if [[ "$DRY_RUN" == 1 ]] || ! sana cat-file -e "$1^{commit}" 2>/dev/null; then
        run git -C "$SOLH3_SANA" fetch --filter=blob:none origin "$1"
    fi
    run git -C "$SOLH3_SANA" checkout --detach "$1"
}

# keep_copy: remember the patch that is now applied
keep_copy() {
    cmp -s "$PATCH" "$APPLIED" 2>/dev/null && return 0
    run mkdir -p "$SOLH3_HOME"
    run cp "$PATCH" "$APPLIED"
}

# only_changes_of PATCH: true if the checkout differs from its HEAD by exactly PATCH. Lists other changes on stderr.
only_changes_of() {
    local patch tmp f rc=0
    local -a touched changed existing=()
    patch=$(realpath -- "$1") || return 1
    mapfile -t touched < <(sana apply --numstat "$patch" | cut -f3-)
    mapfile -t changed < <({ sana diff --name-only HEAD; sana ls-files --others --exclude-standard; } | sort -u)
    for f in "${changed[@]}"; do
        printf '%s\n' "${touched[@]}" | grep -qxF -- "$f" || { echo "  local change not from the patch: $f" >&2; rc=1; }
    done
    (( rc == 0 )) || return 1
    # the files the patch touches must be exactly HEAD + PATCH: rebuild them in a scratch repo and compare
    tmp=$(mktemp -d) || return 1
    git init -q "$tmp" || rc=1
    for f in "${touched[@]}"; do sana cat-file -e "HEAD:$f" 2>/dev/null && existing+=("$f"); done
    if (( rc == 0 && ${#existing[@]} > 0 )); then
        sana archive HEAD -- "${existing[@]}" | tar -x -C "$tmp" || rc=1
    fi
    (( rc == 0 )) && { git -C "$tmp" apply "$patch" || rc=1; }
    if (( rc == 0 )); then
        for f in "${touched[@]}"; do
            if [[ -e "$tmp/$f" ]]; then
                cmp -s "$tmp/$f" "$SOLH3_SANA/$f" || { echo "  changed beyond the patch: $f" >&2; rc=1; }
            elif [[ -e "$SOLH3_SANA/$f" ]]; then
                echo "  present although the patch deletes it: $f" >&2; rc=1
            fi
        done
    fi
    rm -rf "$tmp"
    return "$rc"
}

step "NVlabs/Sana sol-engine @ $COMMIT -> $SOLH3_SANA"
if [[ ! -e "$SOLH3_SANA/.git" ]]; then   # a file in a git worktree
    [[ -e "$SOLH3_SANA" ]] && die "$SOLH3_SANA exists but is not a git checkout; move it away or set SOLH3_SANA"
    run mkdir -p "$(dirname "$SOLH3_SANA")"
    # blob-less clone of the branch: full history (BASE_COMMIT is behind the branch head), files fetched on checkout
    run git clone --filter=blob:none --no-checkout --single-branch -b sol-engine "$SANA_URL" "$SOLH3_SANA"
    if [[ "$DRY_RUN" != 1 ]] && ! sana cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
        run git -C "$SOLH3_SANA" fetch --filter=blob:none origin "$COMMIT"
    fi
    run git -C "$SOLH3_SANA" checkout --detach "$COMMIT"
else
    say "existing checkout at $(sana rev-parse HEAD)"
fi

step "apply patches/sol-h3-studio.patch"
if [[ "$DRY_RUN" == 1 && ! -e "$SOLH3_SANA/.git" ]]; then
    run git -C "$SOLH3_SANA" apply --check "$PATCH"
    run git -C "$SOLH3_SANA" apply "$PATCH"
    keep_copy
else
    head=$(sana rev-parse HEAD)
    current=       # the patch the checkout carries now: this one, the earlier copy, or none
    if sana apply --reverse --check "$PATCH" 2>/dev/null; then
        current=$PATCH
    elif [[ -f "$APPLIED" ]] && sana apply --reverse --check "$APPLIED" 2>/dev/null; then
        current=$APPLIED
    fi
    # the earlier patch was made for the commit named in its first line. If the checkout is still there and
    # patches/BASE_COMMIT (or SOLH3_SANA_COMMIT) now names another commit, move to it.
    old_base=
    [[ -f "$APPLIED" ]] && old_base=$(sed -n '1s/.*base commit \([0-9a-f]\{40\}\).*/\1/p' "$APPLIED")
    move=0
    [[ -n "$current" && "$head" != "$COMMIT" && "$head" == "$old_base" ]] && move=1
    if [[ "$current" == "$PATCH" && "$move" == 0 ]]; then
        say "already applied"
        [[ "$head" == "$COMMIT" ]] || warn "the checkout is at $head, not $COMMIT (the patch is applied there)"
        keep_copy
    else
        replaced=0
        if [[ -n "$current" ]]; then
            if [[ "$current" == "$PATCH" ]]; then
                say "patches/BASE_COMMIT changed ($head -> $COMMIT); moving the checkout and applying the patch there"
            else
                say "the checkout carries an earlier version of the patch ($APPLIED); replacing it"
            fi
            only_changes_of "$current" ||
                die "$SOLH3_SANA has local changes besides the patch (listed above); nothing was changed.
Move them away (or commit them elsewhere), or start from a fresh clone: move $SOLH3_SANA away and run this again."
            run git -C "$SOLH3_SANA" apply -R "$current"
            replaced=1
            if [[ "$move" == 1 ]]; then
                say "moving the checkout from $head to $COMMIT"
                fetch_and_checkout "$COMMIT"
                head=$COMMIT
            fi
        fi
        [[ "$head" == "$COMMIT" ]] || warn "the checkout is at $head, not $COMMIT"
        if [[ "$DRY_RUN" == 1 && "$replaced" == 1 ]]; then
            run git -C "$SOLH3_SANA" apply "$PATCH"      # cannot be checked before the earlier patch is reversed
            keep_copy
        elif sana apply --check "$PATCH" 2>/dev/null; then
            run git -C "$SOLH3_SANA" apply "$PATCH"
            keep_copy
        else
            sana status --short | head -20
            die "the patch does not apply to $SOLH3_SANA (local changes, or a different commit). git apply --check says:
$(sana apply --check "$PATCH" 2>&1 | head -5)
If an earlier version of this patch is applied and $APPLIED does not hold it, reverse it with that earlier patch
file (git -C $SOLH3_SANA apply -R <earlier patch>), or move $SOLH3_SANA away and run this again."
        fi
    fi
fi

step "CPU tests (no GPU, no weights)"
if [[ -x "$ENVS/stage1/bin/python" ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "+ (cd $PKG && CUDA_VISIBLE_DEVICES= $ENVS/stage1/bin/python -B -m unittest discover -s tests)"
    else
        (cd "$PKG" && CUDA_VISIBLE_DEVICES='' "$ENVS/stage1/bin/python" -B -m unittest discover -s tests) ||
            die "CPU tests failed"
    fi
else
    say "no Stage1 env yet ($ENVS/stage1); run this script again after 40-stage1.sh to run the 69 CPU tests"
fi
say "done. Next: install/20-weights.sh"
