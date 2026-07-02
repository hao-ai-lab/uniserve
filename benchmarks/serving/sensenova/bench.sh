#!/usr/bin/env bash
# Run one benchmark point against an already-running server.
#
#   ./bench.sh t2i <base-url> <output-dir>            # 2048x1152, 50 steps, seed 42
#   ./bench.sh i2t-native <base-url> <output-dir>     # UniServe /generate understand
#   ./bench.sh i2t-chat <base-url> <output-dir>       # OpenAI chat + image_url (vLLM-Omni)
#
#   NUM_PROMPTS=8  MAX_TOKENS=256  STEPS=50  WIDTH=2048  HEIGHT=1152  SEED=42
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KIND="$1"; BASE_URL="$2"; OUT="$3"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
STEPS="${STEPS:-50}"
WIDTH="${WIDTH:-2048}"
HEIGHT="${HEIGHT:-1152}"
SEED="${SEED:-42}"
PY="$ROOT/.venv/bin/python"

case "$KIND" in
  t2i)
    exec "$PY" -m benchmarks.serving.uniserve_bench.cli \
      --base-url "$BASE_URL" --task t2i --model SenseNova-U1 \
      --output-dir "$OUT" \
      --dataset trace --dataset-path "$ROOT/benchmarks/serving/sensenova/t2i_prompts.jsonl" \
      --num-prompts "$NUM_PROMPTS" --max-concurrency 1 --warmup-requests 1 \
      --width "$WIDTH" --height "$HEIGHT" --steps "$STEPS" --seed "$SEED"
    ;;
  i2t-native|i2t-chat)
    WIRE=native
    [[ "$KIND" == "i2t-chat" ]] && WIRE=openai_chat
    exec "$PY" -m benchmarks.serving.uniserve_bench.cli \
      --base-url "$BASE_URL" --task i2t --model SenseNova-U1 \
      --output-dir "$OUT" \
      --dataset synthetic-images \
      --num-prompts "$NUM_PROMPTS" --max-concurrency 1 --warmup-requests 1 \
      --max-tokens "$MAX_TOKENS" --temperature 0 --seed "$SEED" \
      --i2t-wire "$WIRE"
    ;;
  *)
    echo "unknown benchmark kind: $KIND" >&2
    exit 2
    ;;
esac
