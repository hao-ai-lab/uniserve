#!/usr/bin/env bash
# Launch UniServe serving SenseNova-U1 for benchmarking.
#
#   MODEL=...                 checkpoint dir  (default: SenseNova-U1-8B-MoT-Interleaved-local)
#   PORT=18082                HTTP port
#   TP=1                      tensor-parallel ranks (1 or 4)
#   TEACACHE=0                1 -> enable the denoise residual cache (threshold 0.2)
#   TEACACHE_THRESHOLD=0.2
#   DECODE_BURST=8            worker-side decode burst length (1 disables; greedy
#                             text decode only — semantics-identical, ~+20% tok/s)
#
# Writes the server pid to $RUN_DIR/uniserve.pid and logs to $RUN_DIR/uniserve.log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${MODEL:-/home/hal-ysun/models/SenseNova-U1-8B-MoT-Interleaved-local}"
PORT="${PORT:-18082}"
TP="${TP:-1}"
TEACACHE="${TEACACHE:-0}"
TEACACHE_THRESHOLD="${TEACACHE_THRESHOLD:-0.2}"
export UNISERVE_DECODE_TOKEN_BURST="${DECODE_BURST:-8}"
RUN_DIR="${RUN_DIR:-$ROOT/benchmarks/serving/sensenova/run}"
mkdir -p "$RUN_DIR"

if [[ "$TP" == "4" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  RANK_ARGS=(--worker-ranks 4)
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  RANK_ARGS=()
fi

if [[ "$TEACACHE" == "1" ]]; then
  export UNISERVE_DENOISE_RESIDUAL_CACHE=1
  export UNISERVE_DENOISE_RESIDUAL_CACHE_THRESHOLD="$TEACACHE_THRESHOLD"
fi

nohup "$ROOT/target/debug/uniserve" serve "$MODEL" \
  --served-model-name SenseNova-U1 \
  --host 127.0.0.1 --port "$PORT" \
  --worker-python "$ROOT/.venv/bin/python" \
  --kv-token-capacity 65536 --max-num-seqs 1 --max-batch 4 \
  --max-num-batched-tokens 4096 \
  "${RANK_ARGS[@]}" \
  > "$RUN_DIR/uniserve.log" 2>&1 &
echo $! > "$RUN_DIR/uniserve.pid"
echo "uniserve pid $(cat "$RUN_DIR/uniserve.pid"), log $RUN_DIR/uniserve.log"

for _ in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "ready: 127.0.0.1:$PORT"
    exit 0
  fi
  sleep 1
done
echo "server did not become ready" >&2
exit 1
