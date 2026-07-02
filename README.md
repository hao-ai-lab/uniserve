# UniServe

UniServe is an OpenAI-compatible inference engine for omni models, built around a strict **control-plane / data-plane split**:

- **Control plane (Rust).** The OpenAI/gRPC API, tokenization, request scheduling, continuous batching, and KV-cache bookkeeping run in a single Rust process. The control plane owns every decision about *what* runs and *when*.
- **Data plane (Python + GPU).** Model weights, KV-cache pages, activations, and image latents live entirely inside a thin, forward-only Python worker on the GPU. They are never serialized out.

The two halves communicate over a zero-copy shared-memory transport (iceoryx2) that carries **only small control-plane descriptors and scalar results** — block tables, token ids, sampled outputs. Tensors and KV pages stay GPU-resident and never cross the boundary.

## Requirements


| Component | Requirement                                                                   |
| --------- | ----------------------------------------------------------------------------- |
| OS / arch | Linux, x86-64 or aarch64                                                      |
| GPU       | NVIDIA GPU + recent driver (CUDA). `--device cpu` / `--sim` for GPU-free runs |
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
uniserve serve Qwen/Qwen3-32B          # downloads from the Hub on first run
uniserve serve /path/to/Qwen3-32B      # …or serve a local model directory
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
    "messages": [{"role": "user", "content": "Give me one fun fact about octopuses."}],
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }'

# Streaming
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-32B","messages":[{"role":"user","content":"Count to five."}],"stream":true}'
```

Any OpenAI client works — point its `base_url` at `http://<host>:<port>/v1`.

### Without a GPU

`--sim` runs the built-in CPU simulation engine (no Python worker, no GPU, no weights) , which is useful for exercising the API surface and scheduler:

```bash
uniserve serve sim-model --sim --port 8000
```

## Configuration

Run `uniserve serve --help` for the full list. The most useful flags:


| Flag                       | Default                           | Description                                                                               |
| -------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| `<MODEL>`                  | —                                 | Local model directory or Hugging Face repo id                                             |
| `--host` / `--port`        | `127.0.0.1` / `8000`              | HTTP bind address                                                                         |
| `--uds <path>`             | —                                 | Bind a Unix domain socket instead of host/port                                            |
| `--device`                 | `cuda`                            | `cuda`, `cpu`                                                                             |
| `--worker-python`          | auto (venv beside the binary)     | Interpreter for the worker; resolved automatically, override only if you must             |
| `--worker-ranks`           | `1`                               | Tensor-parallel worker processes (set >1 for models that exceed one GPU)                  |
| `--max-model-len`          | model's `max_position_embeddings` | Context-length cap                                                                        |
| `--max-num-seqs`           | engine default                    | Max concurrent running requests                                                           |
| `--max-num-batched-tokens` | engine default                    | Per-step token budget (chunked prefill)                                                   |
| `--attention-backend`      | `auto`                            | `auto` picks flashinfer/flash-attn/sgl-kernel if present, else a correct PyTorch fallback |
| `--grpc-port`              | —                                 | Also start the gRPC Generate service                                                      |
| `--sim`                    | off                               | GPU-free CPU simulation engine                                                            |
| `--served-model-name`      | `--model`                         | Public model id(s) returned by the API                                                    |


KV-cache size is auto-fitted to free GPU memory; override with `--kv-token-capacity`. Context length, when not pinned with `--max-model-len`, is read from the model config (e.g. 40960 for Qwen3-32B).

### Multiple GPUs

For a model that does not fit on one GPU, run it tensor-parallel with one worker rank per GPU:

```bash
uniserve serve /path/to/big-model --worker-ranks 4
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
just test-python-e2e           # black-box server tests (uses --sim)
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
uniserve_e2e/                  e2e verification driver + serving measurement harness (`uniserve-e2e`)
justfile                       development & release recipes (replaces Makefile)
```

## License

Apache-2.0.