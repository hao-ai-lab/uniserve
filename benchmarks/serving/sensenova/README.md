# SenseNova-U1: UniServe vs vLLM-Omni serving benchmarks

Reproducible single-user (closed-loop, concurrency 1) comparison of UniServe
and vLLM-Omni on equivalent SenseNova-U1 workloads:

- **t2i** — text-to-image through each server's `/v1/images/generations`
  (2048×1152, 50 steps, cfg 4.0, timestep-shift 3.0, non-think, seed 42, bf16).
  Both servers resolve those generation defaults from the same checkpoint
  config, and the request carries the step count under both accepted
  spellings; the committed `t2i_prompts.jsonl` supplies the prompts, so runs
  are hermetic (no dataset downloads).
- **i2t** — image understanding over each system's public API: UniServe native
  `/generate` `mode:"understand"` SSE vs vLLM-Omni OpenAI chat completions
  with an `image_url` data URI and `modalities:["text"]`. Deterministic
  synthetic images (seeded PIL scenes, 1024×768), fixed question, greedy
  decode, `max_tokens 256`.

Metrics come from one shared pipeline (`uniserve_bench`): TTFT / E2E / output
tokens / output tok/s for i2t, per-image latency and images/min for t2i, plus
harness-side peak GPU memory sampled identically for both backends via
`nvidia-smi` (`summary.json` → `gpu_memory`). Warmup: 1 request (excluded);
cold-start numbers are the warmup request visible in each server log.

## Environment assumptions

- UniServe built at this checkout (`cargo build`) with its `.venv`.
- vLLM-Omni deps installed in `OMNI_VENV` (default `/home/hal-ysun/vllm-omni/.venv`,
  vllm 0.22.0 + torch 2.11 cu130); the *code under test* is `refs/vllm-omni`,
  forced first on `PYTHONPATH` by `serve_vllm_omni.sh`.
- Model checkpoint at `/home/hal-ysun/models/SenseNova-U1-8B-MoT-Interleaved-local`
  (override with `MODEL=`).
- 1× GPU per server by default (`CUDA_VISIBLE_DEVICES=0`); `TP=4` launches
  UniServe tensor-parallel over 4 GPUs.

## Running

```bash
# UniServe (baseline, then the opt-in denoise residual cache):
benchmarks/serving/sensenova/serve_uniserve.sh                 # port 18082
benchmarks/serving/sensenova/bench.sh t2i        http://127.0.0.1:18082 results/uniserve/t2i
benchmarks/serving/sensenova/bench.sh i2t-native http://127.0.0.1:18082 results/uniserve/i2t
kill "$(cat benchmarks/serving/sensenova/run/uniserve.pid)"

TEACACHE=1 benchmarks/serving/sensenova/serve_uniserve.sh
benchmarks/serving/sensenova/bench.sh t2i http://127.0.0.1:18082 results/uniserve-teacache/t2i
kill "$(cat benchmarks/serving/sensenova/run/uniserve.pid)"

# vLLM-Omni (from refs/vllm-omni):
benchmarks/serving/sensenova/serve_vllm_omni.sh                # port 8091
benchmarks/serving/sensenova/bench.sh t2i      http://127.0.0.1:8091 results/vllm-omni/t2i
benchmarks/serving/sensenova/bench.sh i2t-chat http://127.0.0.1:8091 results/vllm-omni/i2t
kill "$(cat benchmarks/serving/sensenova/run/vllm_omni.pid)"
```

Knobs (env): `NUM_PROMPTS` (8), `MAX_TOKENS` (256), `STEPS` (50),
`WIDTH×HEIGHT` (2048×1152 — a SenseNova resolution bucket; UniServe pins
generation to the model's blessed buckets by policy), `SEED` (42),
`TEACACHE_THRESHOLD` (0.2).

Results land under each `--output-dir` as `summary.json` / `summary.md` /
`requests.jsonl`. Only image *speed* is measured here; image quality gates
live in `scripts/e2e.py` workloads (see docs/e2e-verification.md).
