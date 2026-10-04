# UniServe

UniServe serves FastH3 8-Step text-to-video-with-audio generation on NVIDIA Blackwell and Hopper GPUs. Each request returns a finished MP4: 1344×768 H.264 video at 24 fps with stereo 32-kHz AAC audio. The [UniServe FastH3 post](https://hao-ai-lab.github.io/blogs/uniserve-fasth3/) describes the design and its measurements, and the [FastH3 guide](docs/fast_h3/fast_h3.md) covers every deployment, precision and option. UniServe also serves the base MiniMax-H3 checkpoint, with text-to-video, keyframe and reference requests, and FastH3 OmniRef reference requests; the [MiniMax-H3 guide](docs/minimax_h3/minimax_h3.md) covers them.

UniServe is a Python computation library and a Rust server. `uniserve` supplies numerical layers, loading and resource binding; `uniserve_models` composes the models; `uniserve_worker` executes serving requests with those same numerical implementations. Rust owns HTTP admission, scheduling, request state and response assembly.

The configured model descriptions are `qwen3`, `sensenova`, `bagel`, `minimax-h3`, and `diffusion-gemma`. A server process loads exactly one description and exposes one served-model identity.

## Requirements

| Component | Requirement |
| --- | --- |
| OS / architecture | Linux on x86-64 or aarch64 |
| GPU | NVIDIA Hopper or Blackwell GPUs; H200, GB200 and RTX PRO 6000 Blackwell Server Edition have end-to-end validation |
| CUDA | CUDA 13 toolkit and a matching driver |
| Python | Python 3.12 |
| Rust | Stable Rust toolchain with edition 2024 support |

## Installation

Run from the repository root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

uv sync --locked --python /usr/bin/python3.12 --extra gpu
source .venv/bin/activate
```

The sync builds the `uniserve` server and `uniserve-host` launcher binaries and the native worker IPC extension, and installs the `gpu` extra's providers: FlashInfer, FlashAttention-4, and UniServe's sparse-attention and peer-storage kernels. The first `uniserve serve` compiles UniServe's native kernels, which needs a CUDA toolkit compatible with PyTorch, a C++ compiler, and Ninja; the [FastH3 guide](docs/fast_h3/fast_h3.md#install) lists the system packages and a container build.

## Quickstart

Download the checkpoint and serve it on four GPUs:

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
hf download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir "$H3_MODEL"

uniserve serve "$H3_MODEL" \
  --workers configs/fast_h3/ulysses4.json \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 2
```

Startup prepares and captures every admitted request shape before `/health` reports ready; on four GB200 GPUs this takes about 11 minutes. The first startup on a machine also compiles and caches the kernels FastH3 runs and takes about 19 minutes. Until preparation finishes, the log reports `worker still busy during Worker startup` with the elapsed seconds. Then generate a video:

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","prompt":"A clear stream flows through a green forest while birds sing.","task":"t2va","target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5},"seed":1000}' \
  --output forest.mp4
```

## Deployments

A deployment file passed to `--workers` places FastH3's components on devices and hosts. `configs/fast_h3/` holds the deployments with published measurements:

| File | GPUs | Layout |
| --- | --- | --- |
| `ulysses4.json` | Four on one host | Ulysses4 denoiser, TP4 text encoder |
| `ulysses8.json` | Eight on one host | Ulysses8 denoiser, TP8 text encoder |
| `ulysses8-two-node.json` | Four on each of two hosts | Ulysses8 denoiser, TP8 text encoder |
| `ulysses4x2.json` | Eight on one host | Two Ulysses4 replicas |
| `ulysses4x2-two-node.json` | Four on each of two hosts | One Ulysses4 replica per host |
| `dp8-text-tp8.json` | Eight on one host | Eight one-GPU replicas, shared TP8 text encoder |
| `gather8.json` | Eight on one host | All-gather sequence-parallel denoiser, TP8 text encoder |

One replica spanning every GPU gives the lowest latency; replicas serve more requests at once. The [FastH3 guide](docs/fast_h3/fast_h3.md#deployments) lists the recommended deployment and options for each GPU type and goal, and describes two-host startup.

## HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Process and route readiness |
| `GET /metrics` | Runtime metrics |
| `GET /version` | Build information |
| `GET /v1/models` | Configured served model |
| `GET /v1/capabilities` | Served tasks, canvases, duration limits, schedule, accepted request fields and job retention |
| `POST /v1/videos/sync` | Generate one video and return the MP4 |
| `POST /v1/videos` | Create an asynchronous video job |
| `GET /v1/videos`, `GET /v1/videos/{id}` | List jobs, or read one job's state and progress |
| `GET /v1/videos/{id}/content` | Download a completed job's MP4 |
| `DELETE /v1/videos/{id}` | Cancel or delete a job |
| `POST /v1/chat/completions` | Streaming and non-streaming text, image-input, image-output, and interleaved generation |
| `POST /v1/images/generations` | Single-image generation adapter for configured omni descriptions |
| `POST /v1/systemone` | TypeSafe System One decision readout (DiffusionGemma) |

DiffusionGemma readout settings are fixed when the server starts. `--readout-canvas full` uses the checkpoint's full canvas; `compact` rounds each answer scaffold up to a multiple of 16. A numeric value, such as `--readout-canvas 64`, fixes every canvas to that length and splits larger question sets across complete canvases. Numeric lengths must be positive multiples of 16 no greater than the checkpoint's canvas length. `--readout-candidates variants` sums the supported token spellings of each answer; `primary` reads only the space-prefixed spelling (` A`, ` B`, ` yes`, ` no`, and so on). Defaults are `full` and `variants`. Canvas length and candidate selection change the returned distribution and must match between systems in a numerical or performance comparison.

See the [DiffusionGemma serving guide](docs/diffusion_gemma/serving.md) for checkpoint setup, four-GPU serving, decision and chat examples, precision choices, and measurement contracts.

A video request is the MiniMax-H3 request body: `model`, `prompt`, `task` (`t2va`), `target` with `short_edge` 768, `aspect_ratio` `16:9` and `duration_seconds` (4 to 15), and an optional `seed` (default 42). The [FastH3 guide](docs/fast_h3/fast_h3.md#generate-a-video) lists every field. Model discovery returns exactly one entry with the standard `id`, `object`, `created`, and `owned_by` fields; `id` is the configured served-model name.

The metrics endpoint publishes serving lifecycle state as `uniserve:serving_requests`, labeled by served-model name, profile, description, and state. `active` is the instantaneous in-flight count; `accepted`, `scheduled`, `finished`, `rejected`, `cancelled`, `aborted`, and `failed` are cumulative for the running serving runtime. Scheduler, worker, request-latency, and HTTP metrics share the same OpenMetrics response.

## Serving options

| Option | Default | Purpose |
| --- | --- | --- |
| Positional `MODEL` | Required | Local model directory or Hugging Face repository |
| `--workers` | One model worker over every rank | Deployment file: worker instances, node/device ranks, and the components placed on them |
| `--served-model-name` | Resolved model ID | Single public model ID |
| `--host`, `--port` | `127.0.0.1`, `8000` | TCP listener |
| `--uds` | Unset | Unix-domain listener instead of TCP; a stale socket file is replaced and the socket file is removed at shutdown |
| `--host-identity` | `localhost` | This host's name in a multi-host deployment file |
| `--device` | `cuda` | Worker device |
| `--worker-ranks` | `1` | Tensor-parallel ranks in each replica when `--workers` is omitted |
| `--data-parallel-size` | `1` | Independent replicas, each with its own scheduler, KV cache and ranks |
| `--expert-parallel` | Off | Shard routed experts across one-rank data-parallel replicas |
| `--expert-exchange` | `alltoall` | FlashInfer NVLink token exchange, `megamoe` fused NVFP4 dispatch/compute/combine, or `dwdp` asynchronous expert-weight prefetch |
| `--max-video-seconds` | `15` | Longest admitted clip, from 4 to 15 seconds |
| `--max-model-len` | Model configuration | Context-length ceiling |
| `--video-text-capacities` | `1024`, then steps of 2048 | Prompt-token capacities prepared at startup |
| `--max-running-requests` | `128`, clamped to worker capacity | Scheduler active-request bound |
| `--max-total-tokens` | Runtime sizing | KV token-capacity override |
| `--page-size` | Chosen by the worker | Base KV page size supported by every cache group's attention readers. See [paged KV ownership](docs/cache.md). |
| `--max-num-batched-tokens` | `8192` | Per-step scheduling token budget |
| `--chunked-prefill-size` | `8192` | Per-request prefill bound |
| `--mem-fraction-static` | `0.70` | Each rank's share of device storage |
| `--graph-policy` | `auto` | CUDA graph policy: `auto`, `full`, or `off` |
| `--quantization-config` | `{}`, the `quality` preset | Precision preset or per-component overrides |
| `--attention-backend` | `auto` | Attention provider selection; explicit provider names select that implementation |
| `--api-key` | Unset | Bearer token for public routes |
| `--request-timeout` | Unset | Seconds until response headers; streamed bodies are not bounded |
| `--max-concurrent-requests` | Unset | In-flight bound for chat, image generation and System One requests; video requests use video job slots |
| `--shutdown-timeout` | `30` | Graceful drain bound in seconds |
| `--image-fetch-timeout` | `20` | Seconds allowed to fetch one `http(s)` image URL, including redirects and the complete body |
| `--image-fetch-max-bytes` | `20000000` | Largest accepted input image in bytes, for fetched URLs and `data:` URLs alike |
| `--allow-private-image-urls` | Off | Allow image URLs that resolve to loopback, private, link-local, unique-local, or cloud metadata addresses |
| `--readout-layout` | `joint` | System One question grouping: `joint` packs questions in request order into shared canvases; `independent` gives each question its own prompt and canvas |
| `--readout-canvas` | `full` | System One canvas length: `full` is the checkpoint's canvas length; `compact` is the smallest multiple of 16 holding the scaffold; a positive multiple of 16 fixes the length, bounded by the checkpoint's canvas |
| `--readout-candidates` | `variants` | Sum supported answer-token spellings, or use `primary` for only the space-prefixed spelling |
| `--diffusion-generation-config` | Checkpoint `generation_config.json` | DiffusionGemma block-diffusion sampling for every reply, as a JSON object that replaces any of `max_denoising_steps`, `entropy_bound`, `t_min`, `t_max`, `confidence_threshold`, and `stability_threshold` |

On a GPU, startup captures CUDA graphs for decode steps and for prefill steps before the server reports ready, and serving replays them. Graphs cover every step the scheduler forms. Prefill graphs hold up to `--max-num-batched-tokens` prompt tokens (plus one image's feature tokens for models that read images) and up to 31 prompts; decode graphs hold up to 128 requests. Both hold at most `--max-running-requests` and at most as many requests as the KV pool holds a page of every cache group for. The worker reports these bounds, and the scheduler never places more requests in one prefill or decode step. A prefill or decode step that no captured graph holds fails instead of running eagerly. Larger token budgets capture more graphs, so startup takes longer. Startup also runs every image encoder and decoder once, so the `uniserve-kernel-table` line it logs names the kernel of every call the server makes. `--graph-policy off` serves every call without graphs.

Run `uniserve serve --help` for the complete option set.

## Reproduce the measurements

The serving evaluator runs HTTP workloads against a server, validates every response, and writes reproducible result bundles. `uniserve_eval/fast_h3.toml` holds the deployments and workload of the UniServe FastH3 post, and `uniserve_eval/fast_h3_h200.toml` the H200 measurements in the FastH3 guide. The `bench` extra installs the evaluator; `uv sync` keeps exactly the extras it names, so name `gpu` beside it:

```bash
uv sync --locked --python /usr/bin/python3.12 --extra gpu --extra bench
export UNISERVE_FAST_H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml plan gb200-4-bf16
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml run gb200-4-bf16
```

The [FastH3 guide](docs/fast_h3/fast_h3.md#reproduce-the-measurements) lists every suite and the two-host procedure.

## Text and expert parallelism

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

The fast and integration suites run without a CUDA device; tests that need one carry the `gpu` marker, which `test-python-cuda` selects at the unit and integration layers. The FastH3 GPU tests load a complete checkpoint on four GPUs:

```bash
UNISERVE_H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2 \
  .venv/bin/python -m pytest \
  tests/python/integration/model_loading/test_h3_parallel_latents.py \
  tests/python/e2e/test_h3_parallel_http.py
```

The serving evaluator runs HTTP workloads against serving models, measures performance, validates response correctness, and writes reproducible result bundles. Benchmark points are defined in [`uniserve_eval/profiles.toml`](uniserve_eval/profiles.toml).

## Repository layout

```text
crates/foundation/core/                  Shared values and the engine-process codec
crates/foundation/observability/         Runtime metrics and process registry
crates/foundation/observability-derive/  Metrics proc-macro
crates/worker-ipc/                       Worker messages, serialization, and iceoryx endpoints
crates/worker-ipc-py/                    Python worker IPC extension
crates/engine/                           Scheduler, executors, and engine process
crates/server/                           Model profiles, serving funnel, HTTP API, and engine clients
crates/bin/uniserve/                     `serve` and `engine` CLI entrypoints
crates/bin/uniserve-host/                Per-host launcher for multi-host deployments
crates/bin/dynamo-worker/                NVIDIA Dynamo worker backend
uniserve/                                 Numerical layers, loading, and resource binding
uniserve_models/                          Concrete models, typed configs, and checkpoint catalog
uniserve_worker/                          Worker lifecycle and rank-local execution
  protocol/                              Validated batches, calls, WorkerInfo, and IPC envelopes
  execution/                             Submission, request progress, publication, and retirement
  model_executor/                        Capability runners, numerical inputs, and CUDA graphs
  sampling/                              Sampling metadata, execution, and numerical results
  storage/                               KV, latent, tensor, request-slot, and output backing
  transport/                             Local, SHM, CUDA VMM, and channel transfers
uniserve_eval/                            Serving evaluator, profiles, and request workloads
configs/fast_h3/                          FastH3 deployment files
docs/fast_h3/                             FastH3 guide, Dynamo guide, and container files
```

## License

Apache-2.0.
