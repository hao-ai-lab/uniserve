#!/usr/bin/env bash
# Launch vLLM-Omni (from refs/vllm-omni) serving SenseNova-U1 for benchmarking.
#
#   MODEL=...                 checkpoint dir (same one UniServe serves)
#   PORT=8091                 HTTP port
#   OMNI_VENV=...             venv with vllm + vllm-omni deps installed
#                             (default: /home/hal-ysun/vllm-omni/.venv)
#
# The refs/ checkout is placed first on PYTHONPATH so the benchmarked code is
# exactly the revision under refs/vllm-omni, while the compiled deps (vllm,
# torch) come from the installed venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${MODEL:-/home/hal-ysun/models/SenseNova-U1-8B-MoT-Interleaved-local}"
PORT="${PORT:-8091}"
OMNI_VENV="${OMNI_VENV:-/home/hal-ysun/vllm-omni/.venv}"
RUN_DIR="${RUN_DIR:-$ROOT/benchmarks/serving/sensenova/run}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT/refs/vllm-omni${PYTHONPATH:+:$PYTHONPATH}"

nohup "$OMNI_VENV/bin/vllm" serve "$MODEL" --omni \
  --host 127.0.0.1 --port "$PORT" \
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
