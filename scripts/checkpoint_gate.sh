#!/usr/bin/env bash
# Checkpoint performance-protection gate for the decode-runtime construction protocol.
#
# Runs the fixed triplet (Qwen3 ShareGPT r16, SenseNova T2I c32, SenseNova I2T c32) on the
# frozen candidate tree and classifies every required metric against the immutable anchor
# (and, when provided, the previous accepted checkpoint) via scripts/checkpoint_controller.py.
# With --major-boundary it additionally requires the SenseNova UEval interleave c4 point.
#
# Usage: checkpoint_gate.sh <checkpoint-id> [--previous <dir>] [--major-boundary]
# Example: checkpoint_gate.sh cp1
set -euo pipefail
cd /home/hal-ysun/uniserve-dev

CP="${1:?checkpoint id, e.g. cp1}"; shift || true
PREVIOUS=""
MAJOR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --previous) PREVIOUS="$2"; shift 2;;
    --major-boundary) MAJOR="--major-boundary"; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done

export UNISERVE_QWEN3_MODEL=/home/hal-ysun/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
export UNISERVE_SENSENOVA_MODEL=/home/hal-ysun/models/SenseNova-U1-8B-MoT-Interleaved-local

ANCHOR=artifacts/qualification/decode_runtime/anchor
ROOT=artifacts/qualification/decode_runtime/${CP}/candidate
ACCEPT=artifacts/qualification/decode_runtime/${CP}/acceptance

# The candidate gate tree must be clean: any executable change after the gate invalidates it.
if [ -n "$(git status --porcelain | grep -vE 'artifacts/|specs/tasks.md')" ]; then
  echo "GATE ABORT: candidate tree has uncommitted executable changes; freeze the tree first." >&2
  git status --short | grep -vE 'artifacts/|specs/tasks.md' >&2
  exit 3
fi
echo "candidate tree: $(git rev-parse --short HEAD) (clean)"

# Provision the declared GPU provider pack and rebuild the pyo3 worker extension from the candidate
# source. The benchmark requires the visible-end FA4 provider and the candidate IPC wire layout.
uv pip install --python .venv/bin/python -e uniserve_kernel >/dev/null
uv pip install --python .venv/bin/python -e . --no-deps >/dev/null
.venv/bin/python - <<'PY'
from uniserve_kernel import mm_attn_varlen

mm_attn_varlen.require_available()
PY

BENCHES="qwen-uniserve-sharegpt-r16,sensenova-uniserve-t2i-c32,sensenova-uniserve-i2t-c32"
# NB: not GROUPS -- that is a read-only bash special array (the caller's unix groups);
# assigning to it is silently ignored and "$GROUPS" expands to the primary gid.
SERVER_GROUPS="qwen-uniserve,sensenova-uniserve"
if [ -n "$MAJOR" ]; then BENCHES="${BENCHES},sensenova-uniserve-interleave-c4"; fi

rm -rf "$ROOT"
.venv/bin/python scripts/run_benchmarks.py --benchmark main --output-root "$ROOT" \
  --only "$SERVER_GROUPS" --only-bench "$BENCHES" --require-clean-gpu

ARGS=(--checkpoint "$CP" --candidate-root "$ROOT" --anchor-root "$ANCHOR" --out "$ACCEPT" $MAJOR)
if [ -n "$PREVIOUS" ]; then ARGS+=(--previous-root "$PREVIOUS"); fi
.venv/bin/python scripts/checkpoint_controller.py "${ARGS[@]}"
echo "acceptance artifact: $ACCEPT/acceptance.md"
