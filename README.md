# UniServe

UniServe is an OpenAI-compatible inference server for configured text and omni models. Rust owns HTTP admission, tokenization, scheduling, generation state, cache accounting, and response assembly; Python workers own model forward execution and device tensors.

The configured model descriptions are `qwen3`, `sensenova`, and `bagel`. A server process loads exactly one description and exposes one served-model identity.

## Requirements

| Component | Requirement |
| --- | --- |
| OS / architecture | Linux on x86-64 or aarch64 |
| GPU | NVIDIA GPU with a CUDA-compatible driver for production model execution |
| Python | Python 3.11+ with a compatible PyTorch installation |
| Rust | Stable Rust toolchain with edition 2024 support |

The NVIDIA NGC PyTorch container is the recommended environment because it supplies an accelerator-matched PyTorch and CUDA toolchain.

## Installation

```bash
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
. "$HOME/.cargo/env"
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -e .
```

The installation builds the `uniserve` binary and the native worker IPC extension.

## Start a server

Every server invocation supplies the model path or Hugging Face repository and its closed description:

```bash
uniserve serve Qwen/Qwen3-32B \
  --model-description qwen3 \
  --served-model-name Qwen3-32B
```

Local model directories use the same command shape:

```bash
uniserve serve /models/SenseNova-U1 \
  --model-description sensenova \
  --served-model-name SenseNova-U1
```

Run `uniserve serve --help` for the complete option set.

## Public HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Process and route readiness |
| `GET /metrics` | Runtime metrics |
| `GET /version` | Build and protocol provenance |
| `GET /v1/models` | Configured served-model identity and capabilities |
| `POST /v1/chat/completions` | Streaming and non-streaming text, image-input, image-output, and interleaved generation |
| `POST /v1/images/generations` | Single-image generation adapter for configured omni descriptions |

List the configured model:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Submit a chat completion:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-32B",
    "messages": [{"role": "user", "content": "Give one concise fact about matrix multiplication."}],
    "max_completion_tokens": 128
  }'
```

Stream a chat completion:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-32B",
    "messages": [{"role": "user", "content": "Count to five."}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Generate one image from a SenseNova server:

```bash
curl -s http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "SenseNova-U1",
    "prompt": "A red bicycle against a brick wall in golden-hour light",
    "n": 1,
    "size": "2048x1152",
    "steps": 50,
    "seed": 42
  }'
```

## Serving configuration

| Option | Default | Purpose |
| --- | --- | --- |
| Positional `MODEL` | Required | Local model directory or Hugging Face repository |
| `--model-description` | Required | `qwen3`, `sensenova`, or `bagel` preprocessing and output contract |
| `--served-model-name` | Resolved model ID | Single public model ID |
| `--host`, `--port` | `127.0.0.1`, `8000` | TCP listener |
| `--uds` | Unset | Unix-domain listener instead of TCP |
| `--device` | `cuda` | Worker device |
| `--tp-size` | `1` | Tensor-parallel worker ranks |
| `--max-model-len` | Model configuration | Context-length ceiling |
| `--max-total-tokens` | Runtime sizing | KV token-capacity override |
| `--max-running-requests` | `128` | Scheduler active-request bound |
| `--max-num-batched-tokens` | `8192` | Per-step scheduling token budget |
| `--chunked-prefill-size` | `8192` | Per-request prefill bound |
| `--attention-backend` | `auto` | Worker attention provider selection |
| `--api-key` | Unset | Bearer token for public routes |
| `--request-timeout` | Unset | Request wall-clock timeout in seconds |
| `--max-concurrent-requests` | Unset | HTTP in-flight admission bound |
| `--shutdown-timeout` | `30` | Graceful drain bound in seconds |

For tensor-parallel execution, select one rank per participating GPU:

```bash
uniserve serve /models/Qwen3-32B \
  --model-description qwen3 \
  --served-model-name Qwen3-32B \
  --tp-size 4
```

## Development and verification

The `justfile` exposes the canonical repository checks:

```bash
just lint
just test-rust
just test-python-fast
just test-python-integration
just test-python-e2e
```

Real-device end-to-end validation uses the configured model environment variables:

```bash
UNISERVE_QWEN3_MODEL=/models/Qwen3-32B \
UNISERVE_SENSENOVA_MODEL=/models/SenseNova-U1 \
UNISERVE_BAGEL_MODEL=/models/BAGEL-7B-MoT \
UNISERVE_RUN_GPU_E2E=1 \
just test-python-gpu
```

The serving benchmark protocol is documented in [`docs/benchmark-protocol.md`](docs/benchmark-protocol.md), and the executable profile matrix is defined in [`uniserve_eval/profiles.json`](uniserve_eval/profiles.json).

## Repository layout

```text
crates/foundation/       Shared runtime types, configuration, and observability
crates/protocol/         Engine and worker wire contracts
crates/engine/           Scheduler, executor, worker transport, and engine process
crates/frontend/         Model profiles, serving request funnel, and OpenAI adapters
crates/server/           Configured HTTP application
crates/bin/uniserve/     `uniserve` CLI
uniserve_worker/         Forward-only Python model workers
uniserve_eval/           Evaluation driver and benchmark harness
specs/                   Builder-facing runtime and serving contracts
docs/                    User-facing protocols and evaluation documentation
```

## License

Apache-2.0.
