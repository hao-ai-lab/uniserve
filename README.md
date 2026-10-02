# UniServe

UniServe provides a Python computation library and an OpenAI-compatible inference server for text and omni models. `uniserve` supplies numerical layers, loading and resource binding; `uniserve_models` composes the concrete models; `uniserve_worker` executes serving requests with those same numerical implementations. Rust owns HTTP admission, tokenization, scheduling, generation state, cache accounting and response assembly.

The configured model descriptions are `qwen3`, `sensenova`, `bagel`, `minimax-h3`, and `diffusion-gemma`. A server process loads exactly one description and exposes one served-model identity.

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
| `POST /v1/systemone` | TypeSafe System One decision readout (DiffusionGemma) |

DiffusionGemma readout settings are fixed when the server starts. `--readout-canvas full` uses the checkpoint's full canvas; `compact` rounds each answer scaffold up to a multiple of 16. A numeric value, such as `--readout-canvas 64`, fixes every canvas to that length and splits larger question sets across complete canvases. Numeric lengths must be positive multiples of 16 no greater than the checkpoint's canvas length. `--readout-candidates variants` sums the supported token spellings of each answer; `primary` reads only the space-prefixed spelling (` A`, ` B`, ` yes`, ` no`, and so on). Defaults are `full` and `variants`. Canvas length and candidate selection change the returned distribution and must match between systems in a numerical or performance comparison.

See the [DiffusionGemma serving guide](docs/diffusion_gemma/serving.md) for checkpoint setup, four-GPU serving, decision and chat examples, precision choices, and measurement contracts.

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

Models with image input (SenseNova, Bagel, and DiffusionGemma) accept `image_url` content parts whose URL is either a `data:image/<subtype>;base64,...` URL or an `http(s)` URL. The server fetches `http(s)` URLs itself before tokenization: it follows at most three redirects, requires an `image/*` `Content-Type` or a recognizable PNG, JPEG, GIF, WebP, or BMP file, and enforces `--image-fetch-timeout` and `--image-fetch-max-bytes`. Every hop must resolve to a public address unless `--allow-private-image-urls` is set, and environment proxy settings are not used for these fetches. An image that cannot be fetched or decoded fails the request with 400.

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

A DiffusionGemma server generates chat replies by block diffusion: the reply is denoised in canvases of the checkpoint's canvas length (256 tokens), each committed to the context before the next begins, and a streamed reply publishes one chunk per committed canvas. Sampling follows the checkpoint's `generation_config.json` (48 steps, entropy bound 0.1, temperature falling from 0.8 to 0.4, confidence 0.005, stability 1), which `--diffusion-generation-config` overrides for the whole server; `seed` selects each request's random stream. `max_completion_tokens` truncates the reply inside the canvas that reaches it, and `stop` strings and the checkpoint's end-of-sequence tokens end it. `chat_template_kwargs` passes template variables such as `enable_thinking`, and Gemma-4 thought channels and tool calls are returned as `reasoning_content` and `tool_calls`. Token-sampling controls have no meaning for a denoised canvas, so `temperature`, `top_p`, `top_k`, `min_p`, the penalties, `logit_bias`, `allowed_token_ids`, `bad_words`, `logprobs`, `prompt_logprobs`, `min_tokens`, and `ignore_eos` are refused with 400, as are grammar constraints such as `response_format`, which the chat API does not accept for any model.

A DiffusionGemma server answers System One 0.2.0 requests: a `state` and named `noul`, `choice`, or `score` questions. It renders the questions into readout prompts, denoises each answer canvas once over its prompt, and reads every answer from the full-vocabulary probabilities of its answer tokens; nothing is generated, so `usage.output_tokens` is 0. Every answer also carries `x_candidate_mass`, the unnormalized probability of all its answer tokens; a value near 0 means the model did not answer within them. The optional `x_images` field attaches 1 to 8 images, as `data:image/...;base64` or `http(s)` URLs fetched under the same rules as chat images, which the prompt names Image 1 to Image n. Validation failures are 422 with FastAPI's `{"detail": [...]}` body, another model name is 404, and inference failures are 500 with `{"detail": ...}`.

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffusion-gemma",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
      "spam": {"type": "noul", "instructions": "Is this message spam?"},
      "team": {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"}}
    }
  }'
```


## Serving configuration

| Option | Default | Purpose |
| --- | --- | --- |
| Positional `MODEL` | Required | Local model directory or Hugging Face repository |
| `--served-model-name` | Resolved model ID | Single public model ID |
| `--host`, `--port` | `127.0.0.1`, `8000` | TCP listener |
| `--uds` | Unset | Unix-domain listener instead of TCP; a stale socket file is replaced and the socket file is removed at shutdown |
| `--device` | `cuda` | Worker device |
| `--worker-ranks` | `1` | Ranks, forming one tensor-parallel group, in each replica's default Worker instance when `--workers` is omitted |
| `--data-parallel-size` | `1` | Independent model replicas, each with its own scheduler, KV cache and ranks; every request goes to the replica with the fewest requests in flight. With `--workers`, the file lists the replicas as equal consecutive blocks of Worker instances |
| `--expert-parallel` | Off | Shard the routed experts of a mixture-of-experts model across the data-parallel replicas, one rank each: every replica keeps its share of each expert layer and exchanges tokens with the others over NVLink at every such layer, while attention and every other layer stay data-parallel |
| `--expert-exchange` | `alltoall` | How expert-parallel replicas exchange tokens at every expert layer: `alltoall` runs FlashInfer's NVLink all-to-all around each GPU's grouped expert kernel; `megamoe` runs the fused MegaMoE kernel, which dispatches, computes and combines in one launch per layer over NVSHMEM and serves NVFP4 experts |
| `--workers` | One `model` entry over every rank of each replica | Path to a JSON deployment configuration: Worker instances, node/device ranks, and the components placed on them |
| `--max-model-len` | Model configuration | Context-length ceiling |
| `--max-total-tokens` | Runtime sizing | KV token-capacity override |
| `--page-size` | Chosen by the worker | Tokens per KV page of the cache group with the widest rows; unset, the worker takes the largest power of two up to 64 whose pages its attention kernels read in every cache group (32 for DiffusionGemma, whose full-attention pages then hold 64 tokens) |
| `--max-running-requests` | `128` | Scheduler active-request bound |
| `--max-num-batched-tokens` | `8192` | Per-step scheduling token budget |
| `--chunked-prefill-size` | `8192` | Per-request prefill bound |
| `--attention-backend` | `auto` | Worker attention provider: `auto` selects a native kernel for each attention call on the GPU and rejects, with an error naming its shape, dtype and mask, any call no native kernel serves; a provider name (`trtllm`, `flash_attn_4`, `flashinfer`, `torch`, ...) serves every call with that provider |
| `--api-key` | Unset | Bearer token for public routes |
| `--request-timeout` | Unset | Seconds until a response head is sent; streamed bodies (chat SSE and video downloads) are not bounded |
| `--max-concurrent-requests` | Unset | In-flight bound for chat completion, image generation, and System One requests (503 above it); video requests share the video job slots instead |
| `--shutdown-timeout` | `30` | Graceful drain bound in seconds |
| `--image-fetch-timeout` | `20` | Seconds allowed to fetch one `http(s)` image URL, including redirects and the complete body |
| `--image-fetch-max-bytes` | `20000000` | Largest accepted input image in bytes, for fetched URLs and `data:` URLs alike |
| `--allow-private-image-urls` | Off | Allow image URLs that resolve to loopback, private, link-local, unique-local, or cloud metadata addresses |
| `--readout-layout` | `joint` | System One question grouping: `joint` packs questions in request order into shared canvases; `independent` gives each question its own prompt and canvas |
| `--readout-canvas` | `full` | System One canvas length: `full` is the checkpoint's canvas length; `compact` is the smallest multiple of 16 holding the scaffold; a positive multiple of 16 fixes the length, bounded by the checkpoint's canvas |
| `--readout-candidates` | `variants` | Sum supported answer-token spellings, or use `primary` for only the space-prefixed spelling |
| `--diffusion-generation-config` | Checkpoint `generation_config.json` | DiffusionGemma block-diffusion sampling for every reply, as a JSON object that replaces any of `max_denoising_steps`, `entropy_bound`, `t_min`, `t_max`, `confidence_threshold`, and `stability_threshold` |

On a GPU, startup captures CUDA graphs for decode steps and for prefill steps before the server reports ready, and serving replays them. Graphs cover every step the scheduler forms. Prefill graphs hold up to `--max-num-batched-tokens` prompt tokens (plus one image's feature tokens for models that read images) and up to 31 prompts; decode graphs hold up to 128 requests. Both hold at most `--max-running-requests` and at most as many requests as the KV pool holds a page of every cache group for. The worker reports these bounds, and the scheduler never places more requests in one prefill or decode step. A prefill or decode step that no captured graph holds fails instead of running eagerly. Larger token budgets capture more graphs, so startup takes longer. Startup also runs every image encoder and decoder once, so the `uniserve-kernel-table` line it logs names the kernel of every call the server makes. `--graph-policy off` serves every call without graphs.

For tensor-parallel execution, select one rank per participating GPU:

```bash
uniserve serve /models/Qwen3-32B \
  --served-model-name Qwen3-32B \
  --worker-ranks 4
```

For data-parallel serving, run one full replica per GPU; a model that tensor parallelism cannot split, such as DiffusionGemma, serves four GPUs this way:

```bash
uniserve serve /models/diffusiongemma-26B-A4B-it \
  --served-model-name diffusiongemma \
  --data-parallel-size 4
```

Adding `--expert-parallel` keeps the four replicas' attention data-parallel and shards each expert layer across them, so every GPU holds a quarter of the experts and the replicas exchange tokens at each expert layer. Every expert layer then runs in steps the replicas take together: a replica without work joins each step another replica starts, and all replicas pad a step to the largest one's captured graph. With an NVFP4 checkpoint, `--expert-exchange megamoe` fuses each expert layer's exchange and expert computation into one kernel.

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

For decision-session replay, use the `systemone` dataset with a non-empty `session_id` on every JSONL row. The evaluator preserves file order within each session and sends the next decision only after that session's previous response. Set `load.request_rate = inf`, `load.warmup_requests = 0`, and `load.max_concurrency` to the number of sessions. Each initial cold request is included in the measured trace. Session identifiers are benchmark metadata and are not sent to the model. Rows without session identifiers retain the seeded shuffled workload behavior.

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
