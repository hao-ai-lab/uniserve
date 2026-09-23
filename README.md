# UniServe

UniServe provides a Python computation library and an OpenAI-compatible inference server for text and omni models. `uniserve` supplies numerical layers, loading and resource binding; `uniserve_models` composes the concrete models; `uniserve_worker` executes serving requests with those same numerical implementations. Rust owns HTTP admission, tokenization, scheduling, generation state, cache accounting and response assembly.

The configured model descriptions are `qwen3`, `sensenova`, `bagel`, and `minimax-h3`. A server process loads exactly one description and exposes one served-model identity.

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

The three Python packages (`uniserve`, `uniserve_models` and `uniserve_worker`) are installed together with their `uniserve-kernels` dependency, which holds UniServe's own device kernels.

The `gpu` extra adds FlashAttention-4 and the native `uniserve-kernels` extensions for sparse video attention, CUDA IPC and peer-storage mappings. Install the source workspace with `uv sync --extra gpu`; building these extensions requires a CUDA toolkit compatible with PyTorch, a C++ compiler, and Ninja.

## Start a server

Every server invocation supplies the model path or Hugging Face repository. The checkpoint's own configuration selects the serving profile:

```bash
uniserve serve Qwen/Qwen3-32B \
  --served-model-name Qwen3-32B
```

Local model directories use the same command shape:

```bash
uniserve serve /models/SenseNova-U1 \
  --served-model-name SenseNova-U1
```

Run `uniserve serve --help` for the complete option set.

## Public HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Process and route readiness |
| `GET /metrics` | Runtime metrics |
| `GET /version` | Build information |
| `GET /v1/models` | Configured served model |
| `POST /v1/chat/completions` | Streaming and non-streaming text, image-input, image-output, and interleaved generation |
| `POST /v1/images/generations` | Single-image generation adapter for configured omni descriptions |

List the configured model:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Model discovery returns exactly one entry with the standard `id`, `object`, `created`, and `owned_by` fields. `id` is the configured served-model name.

The metrics endpoint publishes serving lifecycle state as `uniserve:serving_requests`, labeled by served-model name, profile, description, and state. `active` is the instantaneous in-flight count; `accepted`, `scheduled`, `finished`, `rejected`, `cancelled`, `aborted`, and `failed` are cumulative for the running serving runtime. Scheduler, worker, cache, request-latency, and HTTP metrics share the same OpenMetrics response.

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

Each image response entry carries `b64_json`, pixel `height` and `width`, and the encoded PNG byte count `bytes`; `revised_prompt` is included when available. Request timing starts before preprocessing and includes queueing and generation.


## Serving configuration

| Option | Default | Purpose |
| --- | --- | --- |
| Positional `MODEL` | Required | Local model directory or Hugging Face repository |
| `--served-model-name` | Resolved model ID | Single public model ID |
| `--host`, `--port` | `127.0.0.1`, `8000` | TCP listener |
| `--uds` | Unset | Unix-domain listener instead of TCP |
| `--device` | `cuda` | Worker device |
| `--worker-ranks` | `1` | Ranks in the default Worker instance when `--workers` is omitted |
| `--workers` | One `model` entry over every rank | Path to a JSON deployment configuration: Worker instances, node/device ranks, and the components placed on them |
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
  --served-model-name Qwen3-32B \
  --worker-ranks 4
```

For text-to-video-and-audio generation with the FastH3 checkpoints, including the packed NVFP4 releases, use the [FastH3 cheat sheet](docs/fast_h3/fast_h3.md).

## Development and verification

The `justfile` exposes the canonical repository checks:

```bash
just lint
just test-rust
just test-python-fast
just test-python-integration
just test-python-cuda
just test-python-e2e
```

The fast and integration suites run without a CUDA device; tests that need one carry the `gpu` marker, which `test-python-cuda` selects at the unit and integration layers.

Real-device end-to-end validation uses the configured model environment variables:

```bash
UNISERVE_QWEN3_MODEL=/models/Qwen3-32B \
UNISERVE_SENSENOVA_MODEL=/models/SenseNova-U1 \
UNISERVE_BAGEL_MODEL=/models/BAGEL-7B-MoT \
UNISERVE_RUN_GPU_E2E=1 \
just test-python-gpu
```

The serving evaluator runs HTTP workloads against serving models, measures performance, validates response correctness, and writes reproducible result bundles. Benchmark points are defined in [`uniserve_eval/profiles.toml`](uniserve_eval/profiles.toml).

## Repository layout

```text
crates/foundation/core/                  Shared values and the engine-process codec
crates/foundation/observability/         Runtime metrics and process registry
crates/foundation/observability-derive/  Metrics proc-macro
crates/worker-ipc/                       Worker messages, serialization, and iceoryx endpoints
crates/worker-ipc-py/                    Python worker IPC extension
crates/engine/                           Scheduler, KV pool, executors, and engine process
crates/server/                           Model profiles, serving funnel, OpenAI API, HTTP, and engine clients
crates/bin/uniserve/                     `serve` and `engine` CLI entrypoints
uniserve/                                 Numerical layers, loading, and resource binding
uniserve_models/                          Concrete models, typed configs, and checkpoint catalog
uniserve_worker/                          Worker lifecycle and rank-local execution
  protocol/                              Validated batches, calls, WorkerInfo, and IPC envelopes
  execution/                             Submission, request progress, publication, and retirement
  model_executor/                        Capability runners, numerical inputs, and CUDA graphs
  sampling/                              Sampling metadata, execution, and numerical results
  storage/                               KV, latent, tensor, request-slot, and output backing
  transport/                             Local, SHM, CUDA VMM, and channel transfers
uniserve_eval/                            Serving evaluator
specs/                                    Builder-facing implementation notes
docs/fast_h3/                             FastH3 deployment cheat sheet and container files
```

## License

Apache-2.0.
