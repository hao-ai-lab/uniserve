# UniServe

UniServe is an OpenAI-compatible inference engine for omni models, built around a strict **control-plane / data-plane split**:

- **Control plane (Rust).** The OpenAI/gRPC API, tokenization, request scheduling, continuous batching, and KV-cache bookkeeping run in a single Rust process. The control plane owns every decision about *what* runs and *when*.
- **Data plane (Python + GPU).** Model weights, KV-cache pages, activations, and image latents live entirely inside a thin, forward-only Python worker on the GPU. They are never serialized out.

The two halves communicate over a zero-copy shared-memory transport (iceoryx2) that carries **only small control-plane descriptors and scalar results** — block tables, token ids, sampled outputs. Tensors and KV pages stay GPU-resident and never cross the boundary.

## Requirements


| Component | Requirement                                                                   |
| --------- | ----------------------------------------------------------------------------- |
| OS / arch | Linux, x86-64 or aarch64                                                      |
| GPU       | NVIDIA GPU + recent driver (CUDA). `--device cpu` is available for CPU-compatible paths |
| Python    | 3.11+ with **PyTorch 2.8+** built for your CUDA/arch                          |
| Rust      | stable toolchain (`rustup`), edition 2024 — needed at install time            |


The tested baseline is the **NVIDIA NGC PyTorch container** (`nvcr.io/nvidia/pytorch`), which ships a CUDA/arch-matched PyTorch plus `flash-attn` and `triton`. A bare machine works too, as long as a working PyTorch is installed first.

## Install

```bash
# 1. Rust toolchain (skip if `cargo` is already on PATH).
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable && . "$HOME/.cargo/env"

# 2. A virtualenv that inherits the system PyTorch.
uv venv --system-site-packages .venv && source .venv/bin/activate

# 3. Install (builds the `uniserve` binary and the IPC extension via setuptools-rust).
uv pip install -e .
```

## Quick start

With the venv active, serving is a single command:

```bash
uniserve serve --model-path Qwen/Qwen3-32B          # downloads from the Hub on first run
uniserve serve --model-path /path/to/Qwen3-32B      # …or serve a local model directory
```

Then:

```bash
# List models
curl -s http://127.0.0.1:8000/v1/models

# Chat completion (Qwen3 is a reasoning model; disable thinking for a direct answer)
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-32B",
    "messages": [{"role": "user", "content": "Give me one concise fact about matrix multiplication."}],
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }'

# Streaming
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-32B","messages":[{"role":"user","content":"Count to five."}],"stream":true}'
```

Any OpenAI client works — point its `base_url` at `http://<host>:<port>/v1`.

## Configuration

Run `uniserve serve --help` for the full list. The most useful flags:


| Flag                       | Default                           | Description                                                                               |
| -------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| `--model-path`             | —                                 | Local model directory or Hugging Face repo id                                             |
| `--host` / `--port`        | `127.0.0.1` / `8000`              | HTTP bind address                                                                         |
| `--uds <path>`             | —                                 | Bind a Unix domain socket instead of host/port                                            |
| `--device`                 | `cuda`                            | `cuda`, `cpu`                                                                             |
| `--tp-size`                | `1`                               | Tensor-parallel worker processes (set >1 for models that exceed one GPU)                  |
| `--max-model-len`          | model's `max_position_embeddings` | Context-length cap                                                                        |
| `--max-running-requests`   | engine default                    | Scheduler active-request limit                                                            |
| `--max-concurrent-requests` | —                                | Front-door HTTP admission limit for in-flight inference requests                          |
| `--max-num-batched-tokens` | engine default                    | Per-step token budget (chunked prefill)                                                   |
| `--max-total-tokens`       | auto-fit                          | KV token capacity override                                                                |
| `--attention-backend`      | `auto`                            | `auto` picks flashinfer/flash-attn/sgl-kernel if present, else a correct PyTorch fallback |
| `--api-key`                | —                                 | Bearer token for public API routes                                                        |
| `--admin-api-key`          | —                                 | Bearer token for sensitive management routes                                              |
| `--request-timeout`        | —                                 | Per-request wall-clock timeout in seconds                                                 |
| `--log-level` / `--log-level-http` | `INFO` / inherited        | Default and HTTP-target log levels                                                        |
| `--log-stats`              | enabled                           | Set `false` to disable periodic engine statistics logging                                 |
| `--enable-lora`            | off                               | Mount runtime LoRA management routes                                                      |
| `--lora-allowed-path-prefixes` | —                              | Comma-separated absolute prefixes for local runtime LoRA adapter paths                    |
| `SGLANG_GRPC_PORT`         | —                                 | gRPC Generate service port when `SGLANG_ENABLE_GRPC` is enabled                           |
| `--served-model-name`      | `--model-path`                    | Public model id(s) returned by the API                                                    |


KV-cache size is auto-fitted to free GPU memory; override with `--max-total-tokens`. Context length, when not pinned with `--max-model-len`, is read from the model config (e.g. 40960 for Qwen3-32B).

### Multiple GPUs

For a model that does not fit on one GPU, run it tensor-parallel with one worker rank per GPU:

```bash
uniserve serve --model-path /path/to/big-model --tp-size 4
```

## Dependencies

- **Runtime** (`uv pip install -e .`): PyTorch, transformers, safetensors, einops, accelerate, numpy, pillow, sentencepiece, huggingface_hub.
- **gpu** (`uv pip install -e ".[gpu]"`): optional accelerator kernels (`flashinfer-python`); worker imports are guarded and fall back to a pure-PyTorch attention path when absent.
- **dev** / **test** / **bench**: linting, type-checking, testing, and the serving benchmark harness.

## Development

The `justfile` wraps the common local checks (install [just](https://github.com/casey/just) if not already available):

```bash
just fmt                       # cargo fmt --check
just clippy                    # cargo clippy -D warnings
just test-rust                 # cargo test --workspace
just lint                      # fmt + clippy + ruff + mypy
just test-python-fast          # unit / contract / architecture tests
just test-python-integration   # fake/simulated-backend tests
just test-python-e2e           # black-box server tests
just test-all                  # all of the above in one shot
```

GPU/model end-to-end validation is opt-in because it loads real checkpoints:

```bash
UNISERVE_RUN_GPU_E2E=1 just test-python-gpu
```

## Project layout

```
crates/                        Rust workspace (see Cargo.toml for the full crate list)
  foundation/                  core types, config, observability
  protocol/                    wire formats, gRPC/OpenAI types, worker IPC (incl. the PyO3 extension)
  engine/                      scheduler, KV, executor, worker-IPC host, process supervisor
  frontend/                    tokenizer, chat templates, OpenAI/native APIs, parsers
  server/                      HTTP + gRPC server apps
  bin/                         the `uniserve` CLI binary
uniserve_worker/               forward-only Python worker: models, layers, model loaders, runtime
uniserve_eval/                  serving evaluation driver + measurement harness (`uniserve-eval`)
justfile                       development & release recipes (replaces Makefile)
```

## License

Apache-2.0.
