#!/usr/bin/env bash
# Launch vLLM-Omni (from refs/vllm-omni) serving SenseNova-U1 for benchmarking.
#
#   MODEL=...                 checkpoint dir (same one UniServe serves)
#   PORT=8091                 HTTP port
#   OMNI_VENV=...             venv with vllm 0.24 (refs' target: version stamp
#                             0.24.0rc2) + refs' requirements/common.txt, and
#                             refs/vllm-omni installed into it non-editably:
#                               VLLM_OMNI_TARGET_DEVICE=cuda uv pip install \
#                                 --no-deps refs/vllm-omni
#                             (default: /home/hal-ysun/omni-bench-venv)
#   TEACACHE=0                1 -> --cache-backend tea_cache (thresh 0.2 default)
#
# Installing the refs checkout (non-editable; refs stays pristine) registers
# the omni CLI plugin so the standard `vllm serve --omni` path works, and the
# served code is exactly the refs revision.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${MODEL:-/home/hal-ysun/models/SenseNova-U1-8B-MoT-Interleaved-local}"
PORT="${PORT:-8091}"
OMNI_VENV="${OMNI_VENV:-/home/hal-ysun/omni-bench-venv}"
TEACACHE="${TEACACHE:-0}"
RUN_DIR="${RUN_DIR:-$ROOT/benchmarks/serving/sensenova/run}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CACHE_ARGS=()
if [[ "$TEACACHE" == "1" ]]; then
  CACHE_ARGS=(--cache-backend tea_cache)
fi

nohup "$OMNI_VENV/bin/vllm" serve "$MODEL" --omni \
  --host 127.0.0.1 --port "$PORT" \
  "${CACHE_ARGS[@]}" \
  > "$RUN_DIR/vllm_omni.log" 2>&1 &
echo $! > "$RUN_DIR/vllm_omni.pid"
echo "vllm-omni pid $(cat "$RUN_DIR/vllm_omni.pid"), log $RUN_DIR/vllm_omni.log"

for _ in $(seq 1 900); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "ready: 127.0.0.1:$PORT"
    exit 0
  fi
  sleep 1
done
echo "server did not become ready" >&2
exit 1
