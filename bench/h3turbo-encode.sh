#!/usr/bin/env bash
# h3turbo-encode.sh --case DIR --prompt-file FILE|--prompt-from-case DIR --orientation O [--first IMG] [--last IMG]
#                   [--resolution R] [--canvas draft|canvas] [--seed N] [--case-id ID] [--no-text-only] [--check]
# The experimental LightX2V Turbo's prompt encode by hand: a clip's prompt conditioning from a transient Qwen worker,
# as the studio's server makes it for jobs/h3turbo_job.py (bench/h3turbo_encode.py's docstring says what and where).
# GPU: ~15 GiB while the worker lives; stop the studio and other GPU apps first (--check starts nothing). Settings: the
# same env file as bin/lf-studio; the Stage-1 Python ($LFSTUDIO_PYTHON), working in the patched package as the
# launcher runs the server.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../install/env.sh
source "$HERE/../install/env.sh"
[[ $# -ge 1 ]] || { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
unset SOL_H3_SECONDS SOL_H3_ORIENTATION      # as the web server does: the clip's own length and format
export LFSTUDIO_PATHS LFSTUDIO_HOME
# as bin/lf-studio: a login whose group list predates the docker group can still run docker through sg
docker() {
    if [[ -w /var/run/docker.sock ]]; then
        command docker "$@"
    else
        sg docker -c "$(printf '%q ' docker "$@")"
    fi
}
export -f docker
[[ -d "$PKG/runtime" ]] || { echo "no patched package at $PKG (install/10-sana.sh)" >&2; exit 2; }
exec "$LFSTUDIO_PYTHON" -I -B "$HERE/h3turbo_encode.py" --package "$PKG" "$@" </dev/null
