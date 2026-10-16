#!/usr/bin/env bash
# 10-sana.sh: clone NVlabs/Sana (branch sol-engine) at patches/BASE_COMMIT into $SOLH3_SANA, apply
# patches/sol-h3-studio.patch, and run the package's CPU tests with the Stage1 python if it exists.
# Idempotent: an existing checkout is reused; an already applied patch is detected and left alone.
# Updates: a copy of the applied patch is kept with the checkout (sol-h3-studio-applied.patch in its git directory,
# so each clone and each git worktree has its own). After a `git pull` of this repo, run this script again. If the
# checkout carries an earlier patch and nothing else, it reverses that patch, checks out the new
# patches/BASE_COMMIT if it changed, and applies the new patch. The earlier patch is found as that copy, else as
# $SOLH3_HOME/applied.patch (where versions up to 0.2 kept it), else among the earlier versions of
# patches/sol-h3-studio.patch in this repo's history. Any other local change in the checkout stops it with a
# message, before anything is changed.
# SOLH3_SANA_COMMIT overrides the commit (the patch also applies to sol-engine 670482d, checked 2026-10-09).
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
need_cmd git
COMMIT=${SOLH3_SANA_COMMIT:-$BASE_COMMIT}
LEGACY_APPLIED="$SOLH3_HOME/applied.patch"   # up to 0.2: one copy per SOLH3_HOME, shared by its checkouts
HISTORY=$(mktemp -d) || die "cannot create a temporary directory (TMPDIR=${TMPDIR:-/tmp})"   # earlier patch versions
trap 'rm -rf "$HISTORY"' EXIT

sana() { git -C "$SOLH3_SANA" "$@"; }

# have_commit COMMIT: make sure the clone has COMMIT, fetching it if needed (before anything in the checkout changes)
have_commit() {
    sana cat-file -e "$1^{commit}" 2>/dev/null && return 0
    run git -C "$SOLH3_SANA" fetch --filter=blob:none origin "$1" || return 1
    [[ "$DRY_RUN" == 1 ]] || sana cat-file -e "$1^{commit}" 2>/dev/null
}

# applied_copy: where this checkout keeps the patch as it was applied last time, for the next update
applied_copy() {
    local dir
    dir=$(git -C "$SOLH3_SANA" rev-parse --absolute-git-dir 2>/dev/null) || dir="$SOLH3_SANA/.git"   # (dry run)
    echo "$dir/sol-h3-studio-applied.patch"
}

# keep_copy: remember the patch that is now applied, and the commit it is applied at (<copy>.base)
keep_copy() {
    local copy
    copy=$(applied_copy)
    cmp -s "$PATCH" "$copy" 2>/dev/null || run cp "$PATCH" "$copy"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "+ git -C $SOLH3_SANA rev-parse HEAD > $copy.base"
    else
        sana rev-parse HEAD > "$copy.base"
    fi
}

# applied_base [PATCH]: the commit the earlier patch was applied at: the record next to the copy, else the base named
# in the first line of the copy, of the 0.2 copy, or of PATCH (the earlier patch found)
applied_base() {
    local f copy
    copy=$(applied_copy)
    if [[ -s "$copy.base" ]]; then
        tr -d '[:space:]' < "$copy.base"
        return 0
    fi
    for f in "$copy" "$LEGACY_APPLIED" "${1:-}"; do
        [[ -n "$f" && -f "$f" ]] || continue
        sed -n '1s/.*base commit \([0-9a-f]\{40\}\).*/\1/p' "$f"
        return 0
    done
}

# earlier_patch: the earlier patch this checkout carries: its copy, the 0.2 location, or a version from this repo's
# history. The first candidate that the checkout differs from HEAD by exactly wins; else the first that only
# reverse-applies (the caller then lists the other local changes). Prints its file name.
earlier_patch() {
    local f sha n=0 first=
    local -a candidates=("$(applied_copy)" "$LEGACY_APPLIED")
    if git -C "$SOLH3_REPO" rev-parse --git-dir >/dev/null 2>&1; then
        while read -r sha; do
            n=$((n + 1))
            f="$HISTORY/patch-$n-${sha:0:7}.patch"
            git -C "$SOLH3_REPO" show "$sha:patches/sol-h3-studio.patch" > "$f" 2>/dev/null && candidates+=("$f")
        done < <(git -C "$SOLH3_REPO" log --format=%H -- patches/sol-h3-studio.patch)
    fi
    for f in "${candidates[@]}"; do
        [[ -f "$f" ]] && sana apply --reverse --check "$f" 2>/dev/null || continue
        only_changes_of "$f" 2>/dev/null && { echo "$f"; return 0; }
        [[ -n "$first" ]] || first=$f
    done
    [[ -n "$first" ]] && { echo "$first"; return 0; }
    return 1
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
    current=       # the patch the checkout carries now: this one, an earlier one, or none
    if sana apply --reverse --check "$PATCH" 2>/dev/null; then
        current=$PATCH
    else
        current=$(earlier_patch) || current=
    fi
    # the earlier patch was applied at a known commit (applied_base). If the checkout is still there and
    # patches/BASE_COMMIT (or SOLH3_SANA_COMMIT) now names another commit, move to it.
    old_base=
    if [[ -n "$current" ]]; then
        if [[ "$current" == "$PATCH" ]]; then old_base=$(applied_base); else old_base=$(applied_base "$current"); fi
    fi
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
                say "the checkout carries an earlier version of the patch ($current); replacing it"
            fi
            only_changes_of "$current" ||
                die "$SOLH3_SANA has local changes besides the patch (listed above); nothing was changed.
Move them away (or commit them elsewhere), or start from a fresh clone: move $SOLH3_SANA away and run this again."
            if [[ "$move" == 1 ]]; then        # fetch before anything changes, so a failed fetch changes nothing
                have_commit "$COMMIT" || die "cannot get $COMMIT from origin ($(sana remote get-url origin 2>/dev/null)); nothing was changed"
            fi
            run git -C "$SOLH3_SANA" apply -R "$current"
            replaced=1
            if [[ "$move" == 1 ]]; then
                say "moving the checkout from $head to $COMMIT"
                run git -C "$SOLH3_SANA" checkout --detach "$COMMIT" ||
                    die "checkout of $COMMIT failed; the earlier patch is reversed. Put it back with: git -C $SOLH3_SANA apply $current"
                [[ "$DRY_RUN" == 1 ]] && head=$COMMIT || head=$(sana rev-parse HEAD)
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
If the checkout carries a patch that is neither this one nor an earlier version from this repo, reverse it with
that patch file (git -C $SOLH3_SANA apply -R <that patch>), or move $SOLH3_SANA away and run this again."
        fi
    fi
fi

step "CPU tests (no GPU, no weights)"
# The tests check the patch at its defaults, so they run without the switches of the settings file (SOL_H3_*,
# SPARK_*): a user's own SOL_H3_ORIENTATION or SOL_H3_RESIDENT_QWEN_EARLY=0 would change what they expect, and the
# patch would seem to fail. The SOL_H3_SPARK_* paths stay (the prompt-token tests read SOL_H3_SPARK_COMFY_ROOT).
unset_switches=()
for name in $(compgen -e); do
    case "$name" in
        SOL_H3_SPARK_*) ;;
        SOL_H3_*|SPARK_*) unset_switches+=(-u "$name") ;;
    esac
done
if [[ -x "$ENVS/stage1/bin/python" ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "+ (cd $PKG && env ${unset_switches[*]} CUDA_VISIBLE_DEVICES= $ENVS/stage1/bin/python -B" \
            "-m unittest discover -s tests)"
    else
        (cd "$PKG" && env "${unset_switches[@]}" CUDA_VISIBLE_DEVICES='' "$ENVS/stage1/bin/python" -B \
            -m unittest discover -s tests) || die "CPU tests failed"
    fi
else
    say "no Stage1 env yet ($ENVS/stage1); run this script again after 40-stage1.sh to run the 135 CPU tests"
fi
say "done. Next: install/20-weights.sh"
