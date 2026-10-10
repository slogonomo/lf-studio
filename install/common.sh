# shellcheck shell=bash disable=SC2034  # variables here are used by the scripts that source this file
# Shared by install/NN-*.sh. Not run directly.
# Loads the settings (install/env.sh: ${SOLH3_ENV:-~/.config/sol-h3-studio/env}, defaults under $SOLH3_HOME,
# default ~/sol-h3) and provides run/step helpers.
# Every script accepts --dry-run (or SOLH3_DRY_RUN=1): it prints the commands that change something instead of
# running them. Read-only checks still run. FORCE=1 redoes steps that are already done.
set -uo pipefail

INSTALL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$INSTALL_DIR/.." && pwd)
DRY_RUN=${SOLH3_DRY_RUN:-0}
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
