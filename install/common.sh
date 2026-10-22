# shellcheck shell=bash disable=SC2034  # variables here are used by the scripts that source this file
# Shared by install/NN-*.sh. Not run directly.
# Loads the settings (install/env.sh: ${LFSTUDIO_ENV:-~/.config/lf-studio/env}, defaults under $LFSTUDIO_HOME,
# default ~/sol-h3) and provides run/step helpers.
# Every script accepts --dry-run (or LFSTUDIO_DRY_RUN=1): it prints the commands that change something instead of
# running them. Read-only checks still run. FORCE=1 redoes steps that are already done.
set -uo pipefail

INSTALL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$INSTALL_DIR/.." && pwd)
if [[ -n "${LFSTUDIO_DRY_RUN+set}" ]]; then   # set, even empty, the new name wins (as in env.sh)
    DRY_RUN=${LFSTUDIO_DRY_RUN:-0}
else
    DRY_RUN=${SOLH3_DRY_RUN:-0}                 # SOLH3_DRY_RUN: its name up to 0.4, until 0.6
fi
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) awk 'NR > 1 && /^# shellcheck/ { next } NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg (only --dry-run)" >&2; exit 2 ;;
    esac
done
FORCE=${FORCE:-0}

# shellcheck source=./env.sh
source "$INSTALL_DIR/env.sh"

# A backend that doesn't match this machine (a GB10 set to rtx5090, an RTX 5090 left at spark) stops every step but
# 00-check.sh, which reports it as a failed check; a dry run only warns. Nothing detected (another GPU): no check.
if [[ "$(basename -- "$0")" != 00-check.sh ]]; then
    machine=$(detect_backend)
    if [[ -n "$machine" && "$machine" != "$LFSTUDIO_BACKEND" ]]; then
        mismatch="this machine looks like LFSTUDIO_BACKEND=$machine, but the install uses $LFSTUDIO_BACKEND$([[ "$LFSTUDIO_BACKEND_SET" == 1 ]] || echo ' (the default)'): export LFSTUDIO_BACKEND=$machine, then run the install scripts again"
        if [[ "$DRY_RUN" == 1 ]]; then
            echo "WARNING: $mismatch (dry run: going on)" >&2
        else
            echo "ERROR: $mismatch" >&2
            exit 1
        fi
    fi
fi

say() { echo ">>> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }
step() { echo; echo "=== $(basename "$0"): $*"; }

# run CMD...: print it, then run it (or only print it with --dry-run)
run() {
    printf '+'; printf ' %q' "$@"; printf '\n'
    [[ "$DRY_RUN" == 1 ]] && return 0
    "$@"
}

# run_in DIR CMD...: the same, with DIR as the working directory (in a subshell)
run_in() {
    local dir=$1
    shift
    printf '+ (cd %q &&' "$dir"; printf ' %q' "$@"; printf ')\n'
    [[ "$DRY_RUN" == 1 ]] && return 0
    (cd "$dir" && "$@")
}

need_cmd() { command -v "$1" >/dev/null || die "$1 not found${2:+ ($2)}"; }

# fetch_checked URL FILE SHA256: download URL to FILE and check its SHA-256 (a FILE that already has it is kept).
# The download goes to FILE.part first; a mismatch stops with FILE.part left for inspection.
fetch_checked() {
    local url=$1 file=$2 sha=$3
    if [[ -f "$file" ]] && echo "$sha  $file" | sha256sum -c --status - 2>/dev/null; then
        say "have $file (SHA-256 checked)"
        return 0
    fi
    run mkdir -p "$(dirname "$file")"
    run curl -fL --retry 3 -o "$file.part" "$url" || die "download failed: $url"
    echo "+ check the SHA-256 of $file.part: $sha"
    if [[ "$DRY_RUN" != 1 ]]; then
        echo "$sha  $file.part" | sha256sum -c --status - ||
            die "$url does not have the pinned SHA-256 $sha (the download is kept as $file.part)"
    fi
    run mv -f "$file.part" "$file"
}

# freeze_matches PYTHON REFERENCE: PYTHON's `pip freeze` is the reference freeze, comment lines left out (read-only,
# so it also runs in a dry run). On the RTX 5090, 40 and 50 adopt a venv built by hand to the same recipe this way
# instead of reinstalling into it; a venv that differs in any line is rebuilt as usual.
freeze_matches() {
    local py=$1 ref=$2
    [[ -x "$py" && -f "$ref" ]] || return 1
    diff -q <(grep -v '^#' "$ref") <("$py" -m pip freeze 2>/dev/null) >/dev/null
}

# Pillow from 13.0.0 on (#69): the image-decoder advisories of 2026-10-22 (GHSA-v9pv-5gc8-w2m6 = CVE-2026-96639 and
# others) affect every earlier Pillow, and the runtime decodes the images clients upload (keyframes, references) in
# Stage 1 and, on a DGX Spark, in the Qwen image (60-qwen-image.sh).
PILLOW=13.0.0

# pillow_update VENV [PACKAGE...]: uninstall the PACKAGEs VENV has (ones that cap Pillow below 13), then install
# Pillow $PILLOW if VENV's is older than 13. 40 and 50 run it every time, so it also updates a venv built before.
pillow_update() {
    local v=$1 p have=()
    shift
    step "Pillow >= 13 in $v (the image-decoder advisories of 2026-10-22, #69)"
    if [[ "$DRY_RUN" == 1 ]]; then
        if (( $# )); then run "$v/bin/python" -m pip uninstall -y "$@"; fi
        run "$v/bin/python" -m pip install --only-binary pillow "pillow==$PILLOW"
        return 0
    fi
    for p in "$@"; do
        "$v/bin/python" -m pip show -q "$p" >/dev/null 2>&1 && have+=("$p")
    done
    if (( ${#have[@]} )); then
        run "$v/bin/python" -m pip uninstall -y "${have[@]}" || die "could not uninstall ${have[*]} from $v"
    fi
    if "$v/bin/python" -c 'import sys, PIL; sys.exit(int(PIL.__version__.split(".")[0]) < 13)' 2>/dev/null; then
        say "Pillow $("$v/bin/python" -c 'import PIL; print(PIL.__version__)') already"
    else
        run "$v/bin/python" -m pip install --only-binary pillow "pillow==$PILLOW" ||
            die "Pillow $PILLOW did not install into $v"
        "$v/bin/python" -c 'import sys, PIL; sys.exit(int(PIL.__version__.split(".")[0]) < 13)' ||
            die "$v still has a Pillow older than 13"
    fi
}

# NVIDIA Sync logins can predate docker group membership; fall back to `sg docker`.
docker() {
    if [[ -w /var/run/docker.sock ]]; then
        command docker "$@"
    else
        sg docker -c "$(printf '%q ' docker "$@")"
    fi
}

if [[ "$DRY_RUN" == 1 ]]; then
    echo "(dry run: commands that change something are printed, not run)"
fi
