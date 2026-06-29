#!/usr/bin/env bash
# Set up a virtualenv and install UniServe. Equivalent to the steps in README.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
VENV="${VENV:-$REPO_ROOT/.venv}"

if ! command -v cargo >/dev/null 2>&1; then
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
  . "$HOME/.cargo/env"
fi

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
if ! "$VENV/bin/python" -c 'import torch' 2>/dev/null; then
  echo "error: this venv cannot import torch; install PyTorch first" >&2
  exit 1
fi

"$VENV/bin/python" -m pip install -U pip >/dev/null
"$VENV/bin/python" -m pip install -e .
