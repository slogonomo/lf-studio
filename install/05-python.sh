#!/usr/bin/env bash
# 05-python.sh: RTX 5090 only (LFSTUDIO_BACKEND=rtx5090). Installs uv 0.13.1, the standalone binary from its GitHub
# release checked against a pinned SHA-256, into $LFSTUDIO_HOME/bin, and with it CPython 3.12.15 (Stage 1, the Qwen
# prompt encoder and NVIDIA's tools) and 3.13.16 (Stage 2) into $LFSTUDIO_HOME/python (UV_PYTHON_INSTALL_DIR). The
# system's Python is not used, and nothing is written outside $LFSTUDIO_HOME (--no-bin; uv's cache is
# $LFSTUDIO_HOME/cache/uv, install/env.sh). Under 1 min measured on a workstation with one RTX 5090; downloads ~17 MB
# for uv and ~60 MB for the two Pythons, ~270 MB installed (uv and uvx ~38 MB, python/ ~230 MB).
# On a DGX Spark it does nothing: the install uses the system's python3.12 there.
# Idempotent: uv of this version and installed Pythons are kept (FORCE=1 installs uv again).
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
if [[ "$LFSTUDIO_BACKEND" != rtx5090 ]]; then
    say "LFSTUDIO_BACKEND=$LFSTUDIO_BACKEND: nothing to do (the install uses the system's python3.12). Next: install/10-sana.sh"
    exit 0
fi
need_cmd curl
need_cmd tar
need_cmd sha256sum
UV_VERSION=0.13.1
UV_ASSET=uv-x86_64-unknown-linux-gnu.tar.gz
UV_SHA256=eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220   # the release's own .sha256, 2026-10-16
DOWNLOADS="$LFSTUDIO_HOME/cache/downloads"

step "uv $UV_VERSION -> $UV"
if [[ -x "$UV" && "$("$UV" --version 2>/dev/null)" == "uv $UV_VERSION"* && "$FORCE" != 1 ]]; then
    say "already there"
else
    fetch_checked "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$UV_ASSET" \
        "$DOWNLOADS/uv-$UV_VERSION/$UV_ASSET" "$UV_SHA256"
    run tar -xzf "$DOWNLOADS/uv-$UV_VERSION/$UV_ASSET" -C "$DOWNLOADS/uv-$UV_VERSION"
    run mkdir -p "$(dirname "$UV")"
    run install -m 755 "$DOWNLOADS/uv-$UV_VERSION/${UV_ASSET%.tar.gz}/uv" "$DOWNLOADS/uv-$UV_VERSION/${UV_ASSET%.tar.gz}/uvx" \
        "$(dirname "$UV")/"
fi

step "CPython $PY312_VERSION and $PY313_VERSION -> $UV_PYTHON_INSTALL_DIR"
run env UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" "$UV" python install --no-bin "$PY312_VERSION" "$PY313_VERSION" ||
    die "uv python install failed"
if [[ "$DRY_RUN" != 1 ]]; then
    for py in "$HOST_PY" "$HOST_PY313"; do
        "$py" -c 'import venv, ensurepip, sys; print(sys.executable, sys.version.split()[0])' ||
            die "$py does not run (uv $UV_VERSION names its Pythons cpython-<version>-linux-x86_64-gnu)"
    done
fi
say "done. Next: install/10-sana.sh"
