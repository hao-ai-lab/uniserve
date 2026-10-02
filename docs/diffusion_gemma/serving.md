# DiffusionGemma serving

UniServe serves DiffusionGemma through decision readout at `/v1/systemone` and block-diffusion generation at `/v1/chat/completions`. Both use the same loaded model and prefix cache. The supported checkpoints are [Google's BF16 model](https://huggingface.co/google/diffusiongemma-26B-A4B-it) and [NVIDIA's NVFP4 model](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4). The NVFP4 kernels require a supported Blackwell GPU.

See the [Python readout and DJev adapter examples](../../examples/diffusion_gemma/README.md) for library and HTTP integration, and the [four-GB200 performance comparison](performance.md) for measured decision throughput and its numerical scope.

## Install and load

From the source checkout, install the locked GPU environment. A compatible CUDA toolkit, C++ compiler, and Rust toolchain are required to build the server and native extensions.

```bash
uv sync --locked --python /usr/bin/python3.12 --extra dev --extra test --extra bench --extra gpu
.venv/bin/hf download google/diffusiongemma-26B-A4B-it --local-dir /models/diffusiongemma-26B-A4B-it
```

Start one independent model replica on each of four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/uniserve serve /models/diffusiongemma-26B-A4B-it \
  --served-model-name diffusiongemma \
  --data-parallel-size 4 \
  --max-model-len 4096
```

For one GPU, omit `--data-parallel-size` and expose that GPU alone. For NVFP4, download and supply the NVFP4 checkpoint directory; loading interprets its quantization metadata. BF16 and NVFP4 are distinct precision configurations and can return different distributions and generated text.

Each independent replica owns its scheduler and KV cache. The frontend sends a request to the replica with the fewest requests in flight. `--expert-parallel` partitions the routed experts across the replicas; attention and dense layers remain replicated. Its default `alltoall` transport exchanges tokens at every expert layer over NVLink. This changes the communication and numerical batching, so compare it with independent replicas on the intended workload. `--expert-exchange megamoe` selects the fused NVFP4 expert exchange. Tensor parallelism is not supported for this checkpoint's attention partition.

`--expert-parallel --expert-exchange dwdp` selects distributed weight data parallelism. Each replica retains its expert shard and asynchronously reads missing weights over NVLink into two reusable buffers. Replicas advance independently; an idle replica keeps its immutable weights available without joining inference collectives. The runtime maps resident and prefetched pages into contiguous weight views consumed by the same CuTeDSL BF16 or NVFP4 kernels, preserving the checkpoint's precision and activation calibration. This mode requires one execution lane and peer-accessible CUDA devices on one host. Its weight transfers trade memory capacity for bandwidth; measure it against full-weight replicas on the intended workload.

Startup loads weights, prepares native kernels, and captures execution graphs. Wait for serving readiness before sending measured traffic. Use `GET /health`, `GET /v1/models`, and `GET /metrics` for readiness, model identity, and serving counters. Keep the default native attention selection and graph policy for normal serving.

## Decision readout

A decision request supplies one state and named questions. The model reads a scaffold containing masked answer slots in one denoising pass; it does not generate a text completion.

```bash
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffusiongemma",
    "state": "The north exit is blocked. The east exit is clear.",
    "questions": {
      "north_clear": {"type": "noul", "instructions": "Is the north exit clear?"},
      "action": {
        "type": "choice",
        "instructions": "Choose a clear exit.",
        "criteria": {"north": "Take the north exit", "east": "Take the east exit"}
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is it to change direction when heading north?",
        "criteria": ["No change needed", "Change direction now"]
      }
    }
  }'
```

The response contains `model`, `answers`, and `usage`. `usage.output_tokens` is zero. Each answer includes `x_candidate_mass`, the unnormalized vocabulary probability assigned to its answer candidates. A low mass means that most vocabulary probability falls outside the requested answer set; normalized probabilities alone do not show this.

| Question type | Returned value |
| --- | --- |
| `noul` | `noul`, the probability of yes |
| `choice` | `choice`, `probabilities`, and `confidence` |
| `score` | `score`, the probability-weighted zero-based level; `legend`, `probabilities`, and `confidence` |

Choice supports 1–255 named options. Score supports 1–10 ordered levels. Choices with more than 52 options use two-token labels and conditional readout passes. Question order and candidate order are significant. With the default `joint` layout, questions share a prompt and canvas until the scaffold reaches capacity; larger question sets split across complete canvases. `--readout-layout independent` gives each question its own prompt and canvas, changing both cost and conditioning.

Readout settings are fixed for the deployment:

| Option | Default | Meaning |
| --- | --- | --- |
| `--readout-canvas` | `full` | Use the checkpoint's 256-token canvas; `compact` rounds each scaffold to a multiple of 16; a positive multiple of 16 such as `64` fixes the canvas length, bounded by 256 |
| `--readout-candidates` | `variants` | Sum supported token spellings; `primary` reads only the primary space-prefixed spelling |
| `--readout-layout` | `joint` | Pack complete questions together; `independent` conditions each question separately |

For applications whose readout contract specifies 64-token canvases and primary candidates, append `--readout-canvas 64 --readout-candidates primary` to the serving command. Short canvases use graphs sized for their numerical length on independent replicas. These options change the model's input or answer projection, so qualify their accuracy and calibration on the application's data before choosing them. They do not change the canvas length used for chat generation.

Attach images with `x_images`, an ordered list of 1–8 data URLs or HTTP(S) URLs. The prompt identifies them as Image 1, Image 2, and so on. Remote images follow the server's image-fetch limits and public-address policy, documented in the [main README](../../README.md#public-http-api).

Malformed System One requests return 422 with a `detail` list; an unknown served-model name returns 404; admission above a configured concurrency limit returns 503. Questions reject unknown fields. State and rubric fields accept strings and structured JSON values as described by the System One schema.

## Chat generation

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffusiongemma",
    "messages": [{"role": "user", "content": "Explain why the sky looks blue."}],
    "max_completion_tokens": 256,
    "seed": 11,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```

Generation denoises a 256-token block, commits it to the prefix cache, and starts the next block when needed. `max_completion_tokens` bounds the returned tokens, including truncation inside the final block. `stop` strings and checkpoint end-of-sequence tokens can stop the reply earlier. Streaming emits a chunk when a block is committed; it does not emit one event per denoising step. Thinking and tool-call channels are parsed into the chat response's corresponding fields.

The checkpoint's generation configuration controls denoising steps, entropy acceptance, temperature schedule, confidence, and stability. Change these for the deployment with `--diffusion-generation-config`, for example `'{"max_denoising_steps":48}'`. Autoregressive controls such as request-level `temperature`, `top_p`, token penalties, and log-probability requests are rejected because they do not define this block-diffusion sampler.

`seed` selects the request's random stream. Independent replicas return the same response for the same isolated seeded request. Different batch compositions, including expert-parallel steps whose capacities depend on other ranks, can change floating-point rounding and the resulting response.

## Reproducible measurement

Use `uniserve-eval` to resolve a workload before running it:

```bash
.venv/bin/uniserve-eval --config profiles.toml plan decision-workload
.venv/bin/uniserve-eval --config profiles.toml run decision-workload
```

For a closed-loop decision workload, place a nonempty `session_id` on every System One JSONL row, set `request_rate = inf`, `warmup_requests = 0`, and `max_concurrency` to the number of sessions. Each session sends its next request only after the previous response. Use distinct evolving states while retaining the prefix reuse the real application permits. Fixed-count traces include each session's initial cold request and final completion.

Record checkpoint revision, precision, topology, canvas length, candidate spellings, question layout, prefix-cache behavior, inputs, concurrency, and metric definitions. Run measurement points serially. A performance comparison with MCJev fast also needs to disclose its query-relative sliding canvas attention: UniServe follows HF DynamicCache semantics, with every canvas query attending to the same retained prefix and the complete canvas. Matching token IDs alone does not establish that these attention masks are equivalent. Compare task quality separately from latency and throughput.
