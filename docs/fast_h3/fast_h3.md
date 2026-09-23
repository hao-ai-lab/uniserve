# FastH3 cheat sheet

UniServe serves the FastH3 text-to-video-and-audio checkpoints. The output is an H.264/AAC MP4 at 1344×768, 24 fps, with stereo 32-kHz audio.

## Requirements

- Linux, CUDA 13, Python 3.12, a stable Rust toolchain, and NVIDIA SM100 GPUs such as B200 or GB200. GB200 has end-to-end validation; other hardware requires its own validation before performance claims.
- One of the complete FastH3 VSA checkpoints listed below. Base partitions and adapter-only checkpoints are unsupported.
- Enough GPU storage for weights, two resident requests, and CUDA graphs. Four local GPUs are the standard setup.
- Shared storage and pinned-storage access for worker IPC. The container command below provisions 4 GiB of shared storage and unlimited locked storage.

## Install

Run from the UniServe repository root:

```bash
apt-get update
apt-get install -y build-essential python3.12-dev pkg-config cmake git curl ca-certificates

curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

export CUDA_HOME=/usr/local/cuda
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$CUDA_HOME/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/opt/uniserve-venv
export MAX_JOBS=2

uv sync --locked --python /usr/bin/python3.12 --extra gpu
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

The `gpu` extra installs the locked GPU providers FastH3 serves through: FlashInfer, FlashAttention-4, the peer-storage and sparse-attention kernels, and the CuTe and CUTLASS DSLs. These are shared runtime capabilities rather than model-specific packages, so there is no FastH3-specific dependency group. The sync also builds the `_uniserve_ipc` extension and the `uniserve` binary from this checkout. FastVideo itself is not a runtime dependency.

## Supported checkpoints

| Repository | Weights | Denoiser forwards | VSA sparsity | Download size |
| --- | --- | --- | --- | --- |
| [`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) | BF16 | 4 | 0.9 | 148 GB |
| [`FastVideo/FastVideo-FastH3-8-Step-V2`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) | BF16 | 8 | 0.8 | 148 GB |
| [`skx618/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4`](https://huggingface.co/skx618/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4) | ModelOpt NVFP4 | 4 | 0.9 | 123 GB |
| [`skx618/FastVideo-FastH3-8-Step-V2-NVFP4`](https://huggingface.co/skx618/FastVideo-FastH3-8-Step-V2-NVFP4) | ModelOpt NVFP4 | 8 | 0.8 | 123 GB |

The two `skx618` repositories are ModelOpt PTQ checkpoints in ModelOpt's unified Hugging Face layout, which UniServe loads with no precision flags: see [Precision and graphs](#precision-and-graphs).

Each checkpoint's `fastvideo_inference.json` supplies its sampling schedule: the trained DMD rungs in `dmd_denoising_steps`, one transformer forward per rung, and the VSA sparsity. The video and audio shifts come from `scheduler/` and `audio_scheduler/`, and a manifest that restates them must agree. The loader refuses a manifest that is not a `fasth3-inference-contract-v1` text-to-video-and-audio contract without guidance. The table lists the checkpoints UniServe has validated end to end.

`uniserve serve` takes either a repository id, which loads the latest published revision, or a local model directory.

## Download the model

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2-NVFP4

hf download skx618/FastVideo-FastH3-8-Step-V2-NVFP4 --local-dir "$H3_MODEL"
```

Pass the model root containing `modular_model_index.json`, `fastvideo_inference.json`, `transformer/`, `text_encoder/`, `vae/`, `audio_vae/`, `scheduler/`, `audio_scheduler/`, and `tokenizer/`. In an NVFP4 root in ModelOpt's unified layout, `transformer/config.json` and `vae/config.json` also declare a `quantization_config` with `quant_method: modelopt`. The server identifies the model from the pipeline class that `modular_model_index.json` declares and reads the tokenizer from the component folder the index names; the loader then validates the inference contract in `fastvideo_inference.json`.

## Start the server

```bash
uniserve serve "$H3_MODEL" \
  --workers config/minimax-h3-four-devices.json \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 2
```

`config/minimax-h3-four-devices.json` is a deployment file: it lists the participating devices and, under `components`, the placement and parallel configuration of each of FastH3's five components. It places the numerical components on four devices of one host, as the `model` worker: four-way Ulysses denoising, TP4 text encoding, one video and one audio media unit per rank. A second worker, `host`, has five ranks on the host with `"device": "cpu"` and holds the two host components every video deployment needs: `video_encoder` on ranks 0 to 3, one media unit of each decode round per rank, and `muxer` on rank 4, which encodes the audio track and assembles the MP4. Each host rank is one codec slot: it runs one encode or assembly step at a time in its own process. The muxer must be on the head's host, and every worker that decodes video needs an encoder that keeps each media unit on the host that decoded it; the server refuses a placement that violates either at startup and routes each request to such an encoder. It does not shard denoiser weights. Serving a different width, or sharding the denoiser by tensor or pipeline, is a different deployment file; the schema is in [parallel execution](../parallel-execution.md).

`config/minimax-h3-eight-devices.json` is the same placement over eight devices on two hosts, named `rank-0` and `rank-1` in the file: eight-way Ulysses denoising, TP8 text encoding, one video and one audio media unit per rank, and a host worker with four encoder ranks on each host, which encode that host's four media units per round, and a muxer rank on `rank-0`. The head runs on the host named `rank-0`, which holds ranks 0 to 3 of the model worker, and a launcher on the other host runs ranks 4 to 7 of the model worker and encoder ranks 4 to 7 of the host worker. Start the head first:

```bash
uniserve serve "$H3_MODEL" \
  --workers config/minimax-h3-eight-devices.json \
  --host-identity rank-0 \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 2
```

The head logs `awaiting a launcher for each host this instance does not run on address=...`; on the other host, start the launcher with that address and the host identity the file names:

```bash
uniserve-host --head <address the head logs> --host-identity rank-1
```

One launcher serves every worker of the deployment that places a rank on its host, here ranks 4 to 7 of the model worker and ranks 4 to 7 of the host worker. The launcher starts each rank with the Python interpreter and launch descriptor the head resolved, so the other host needs the same Python environment and the checkpoint at the same path.

`config/minimax-h3-eight-devices-single-node.json` places the same components on eight devices of one host, so it needs no launcher and starts exactly like the four-device file: eight-way Ulysses denoising, TP8 text encoding, one video and one audio media unit per rank, and a host worker with eight encoder ranks, one per media unit of a round, and a muxer rank.

## Data parallel serving

The single-node DP8 deployment uses eight independent one-GPU flow workers instead of one eight-rank Ulysses worker. Each flow worker owns a complete denoiser, video decoder, and audio decoder, so eight requests can denoise and decode concurrently without a collective between GPUs. The scheduler selects one flow replica when it admits a request, keeps that request on the same replica through latent preparation, denoising, and device decoding, and accounts its request rows, product buffers, queue, and execution lane in that worker's address space. Replica capacities add; a shared component remains an independently bounded stage of the route.

The NVFP4 checkpoints are the intended DP8 weights. Dense BF16 denoiser replication is not expected to fit on 96 GB devices. Start with the shared-TP8 conditioning layout:

```bash
uniserve serve "$H3_MODEL" \
  --workers config/minimax-h3-dp8-text-tp8.json \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 8 \
  --video-graph-shapes 5x1000
```

The two supplied placements isolate the main topology choices:

| Deployment | Text encoder | Denoiser and device decode | CPU post-processing | Intended use |
| --- | --- | --- | --- | --- |
| `minimax-h3-dp8-text-tp8.json` | One TP8 worker shared by all requests | Eight complete one-GPU replicas | Eight one-rank encoder workers and one muxer worker | Lower text-weight storage per GPU and the conservative starting point; conditioning is one shared stage and its TP collective spans all devices. |
| `minimax-h3-dp8-text-tp4x2.json` | Two TP4 replicas, one on devices 0–3 and one on devices 4–7 | Eight complete one-GPU replicas | The same encoder and muxer workers | Two conditioning requests can run concurrently and each collective stays within a four-GPU island, at the cost of a larger text shard on every GPU. |

`config/minimax-h3-dp8-text-tp8-two-node.json` extends the shared-TP8 layout across two four-GPU hosts. The text collective spans both hosts, each host owns four one-GPU flow replicas, and each host holds the four one-rank encoder workers that serve its flow replicas, with the muxer worker on `rank-0`. A request is routed to an encoder on its flow replica's host, so every decoded unit is borrowed in place from shared storage; only the encoded unit rows, a fraction of a raw unit, cross to the muxer. Start it with the same head-plus-`uniserve-host` procedure as the two-host Ulysses placement above.

Each GPU decoder has one native media unit per rank, so a one-GPU flow replica reconstructs an eight-unit request in eight bounded decode rounds, and its units reach the encoder one at a time. Each request is therefore bound to one single-rank encoder worker on its replica's host, chosen at admission, and the eight encoder workers encode eight requests' units concurrently. Audio encoding and MP4 assembly run on the one muxer worker, serialized per request because they are short relative to denoising and need one ordered artifact owner.

The queue depths are intentional capacity values. A resident media request slot occupies three positions of its worker's batch queue, one reserved pipeline position and two unresolved outputs, and every flow worker must expose at least two resident slots, so each one-GPU flow uses depth 6. The shared TP8 text worker uses depth 24 for eight slots; each of the two TP4 text workers uses depth 12 for four slots. The muxer worker also uses depth 24 so its shared request-row bank exposes eight slots, and each encoder worker uses depth 6 for two. The narrowest aggregate component capacity is therefore eight complete routes in either deployment.

The `memory_fraction` on each GPU worker is its per-process static storage ceiling. A text worker and a flow worker intentionally share every GPU, so neither inherits the global `--mem-fraction-static` default. Both supplied layouts grant 0.18 to each text rank and 0.81 to each flow replica; the physical free-storage check still caps their combined allocations. On 96 GB RTX PRO 6000 devices, the TP8 layout retained 5.5 GiB or more at the measured concurrency-eight peak. TP4×2 reached 97,244 MiB on the two GPUs holding text rank 0, leaving only 100 MiB; it is a measured maximum-throughput option, not the production default. Treat the supplied split as part of the five-second, 1000-token deployment contract and revalidate it when checkpoint precision, maximum duration, prompt bound, graph shapes, or hardware change.

DP improves throughput only when the offered concurrency keeps multiple replicas occupied. At concurrency one, Ulysses can retain lower latency because all GPUs cooperate on one denoising call; at concurrency eight, DP removes that per-step collective and keeps queueing behind one request from dominating service time. Compare the layouts with the same checkpoint, prompts, duration, graph warmup, and concurrency rather than comparing an uncaptured first request with steady state.

The `fast_h3_dp8` evaluation suite fixes that comparison protocol for the packed four-step checkpoint. It first runs the existing eight-way Ulysses placement with its two latency-oriented resident slots, then the shared-TP8 and replicated-TP4 DP8 placements. All three points use the same checkpoint and eight GPUs, 16 measured requests at concurrency eight after eight warmup requests, five-second outputs, 1000-token prompts, and the same seeds. The Ulysses worker keeps a 0.93 static ceiling; the DP workers use the explicit per-process ceilings validated above because independent text and flow processes share each device.

The reference RTX PRO 6000 Blackwell Server Edition run completed all 16 measured requests in every point and validated every output as 1344×768 H.264 video with stereo 32-kHz AAC audio. The shared-TP8 DP8 layout is the recommended deployment because it more than doubles throughput over Ulysses while retaining useful GPU memory headroom. TP4×2 is 17.1% faster than TP8 in this workload, but its 100 MiB minimum headroom is too small for a general production recommendation.

| Deployment | Videos/s | Video latency p50 | Video latency p95 | Peak single-GPU memory | Change from Ulysses |
| --- | ---: | ---: | ---: | ---: | --- |
| Ulysses8 | 0.1235 | 64.221 s | 64.661 s | 45,799 MiB | Control |
| DP8 + text TP8 | 0.2602 | 28.236 s | 31.552 s | 91,726 MiB | 2.108× throughput; 56.0% lower p50 |
| DP8 + text TP4×2 | 0.3048 | 26.007 s | 26.346 s | 97,244 MiB | 2.469× throughput; 59.5% lower p50 |

Resolve the commands and paths before starting the serial artifact-producing run:

```bash
.venv/bin/uniserve-eval --config uniserve_eval/profiles.toml plan fast_h3_dp8
.venv/bin/uniserve-eval --config uniserve_eval/profiles.toml run fast_h3_dp8
```

The `fast_h3_pareto_ulysses4`, `fast_h3_pareto_dp4`, `fast_h3_pareto_ulysses8`, and `fast_h3_pareto_dp8` suites sweep concurrency 1/2/4 and, for eight GPUs, 8. They use two excluded warmups and 16 measured five-second/1000-token requests per point. Run the points serially against their matching placement; the suite fixes request data and metrics, while the deployment command fixes whether the point is Ulysses or data parallel.

The eight-device Ulysses file also does not shard denoiser weights, so every rank holds the whole denoiser, and a rank's residency can exceed the default `--mem-fraction-static` on a 96 GB device at the default `--max-video-seconds 15` and `--max-model-len 16384`. Raise the fraction, or shard the denoiser with `"tensor_parallel_size"`, if that deployment refuses to start with a static storage grant error. For DP8, adjust the text and flow workers' explicit `memory_fraction` values together instead of raising the global default.

Cross-rank products move over the mechanism named for that edge. CUDA VMM reads inspect the visible CUDA peer topology: a consumer with direct access maps the allocation on its destination GPU, while a consumer outside the producer's peer set maps it on the source GPU and uses CUDA's host-staged cross-device copy. The choice depends on the runtime topology rather than the accelerator model. `--transfer model->model=shm` remains available when an operator wants to force host staging for every model-worker product.

A FastH3 deployment defaults to `--max-video-seconds 15` and `--max-model-len 16384`; set them only to change those limits. `--max-running-requests` caps concurrently resident requests, and the engine clamps that cap to the worker's advertised request-slot capacity; lowering it trades throughput for per-request latency and storage headroom.

`--video-graph-shapes 5x1000,15x10000` declares the duration and prompt length of the requests a deployment serves, and warmup captures the denoising ladder of each declared shape before the server reports ready. A ladder serves every request slot and every prompt length in the same 64-token text tile at the same duration, so `5x1000` covers five-second requests with 961 to 1024 prompt tokens, each with exactly the values an uncaptured evaluation produces. The server never captures a graph while serving: a request that no declared shape covers still serves, but runs its denoising steps without graphs.

Check the live limits and served model name after startup:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/capabilities | python -m json.tool
```

`/v1/capabilities` reports the accepted request fields, the frame geometry, the duration limits, and the job retention bounds described below.

## Build and run with Docker

Use the repository root as the build context. Docker automatically uses the adjacent `Dockerfile.dockerignore` file.

```bash
docker build -f docs/fast_h3/Dockerfile -t uniserve-h3 .

docker run --rm \
  --gpus all \
  --shm-size=4g \
  --ulimit memlock=-1 \
  -p 8000:8000 \
  -v "$H3_MODEL:/models/fast_h3:ro" \
  uniserve-h3 serve /models/fast_h3 \
    --workers config/minimax-h3-four-devices.json \
    --served-model-name FastH3 \
    --host 0.0.0.0 \
    --max-running-requests 2
```

The first startup compiles the native GPU providers and captures shapes on first use. Those artifacts land in `~/.cache/torch_extensions`; mount that path, or point `TORCH_EXTENSIONS_DIR` at a mounted directory, if container restarts must reuse them.

## Generate a video

Only `model`, `prompt`, `seconds`, and `seed` are accepted, as JSON or as `multipart/form-data`. `seconds` defaults to 5 and `seed` defaults to 0.

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","prompt":"A clear stream flows through a green forest while birds sing.","seconds":5,"seed":1000}' \
  --output forest.mp4
```

The synchronous endpoint returns MP4 bytes. H3 rounds the requested duration to whole frames and then up to its temporal geometry of `17n + 5` frames: a 5-second request produces 124 frames (about 5.17 seconds), and a 15-second request produces 362 frames (about 15.08 seconds). The shortest geometry is 22 frames, so any accepted request produces at least about 0.92 seconds; a duration that rounds to five frames or fewer, below about 0.23 seconds, is rejected.

## Use asynchronous jobs

Create a job:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","prompt":"A clear stream flows through a green forest while birds sing.","seconds":5,"seed":1000}'
```

Use the returned `video_...` ID:

```bash
export VIDEO_ID=video_example

curl --fail-with-body "http://127.0.0.1:8000/v1/videos/$VIDEO_ID"
curl --fail-with-body "http://127.0.0.1:8000/v1/videos/$VIDEO_ID/content" --output forest.mp4
curl --fail-with-body http://127.0.0.1:8000/v1/videos
curl --fail-with-body -X DELETE "http://127.0.0.1:8000/v1/videos/$VIDEO_ID"
```

Job states are `queued`, `in_progress`, `completed`, and `failed`. A running job also reports `phase` (`encoding`, `preparing`, `denoising`, `decoding`, then `finalizing`), `completed_steps` against `total_steps`, and `actual_seconds` for the aligned frame count. Jobs and retained MP4s live in the server process, expire after one hour, and disappear on restart. The server retains at most 128 jobs and 1 GiB of artifacts. Deleting a queued or running job cancels it.

Creating a job while 128 are retained returns HTTP 429 with code `video_job_capacity_exceeded`. A failed job reports `error.code`: `invalid_request` when the deployment cannot serve the request as specified, `server_overloaded` when the engine's waiting queue was full and the same request may be resubmitted, and `generation_failed` when execution failed. The synchronous endpoint returns the same conditions as HTTP 400, 503, and 500.

## Precision and graphs

A dense BF16 checkpoint selects its precision at startup with `--quantization-config`:

| Mode | Denoiser attention | Denoiser MLP | Text encoder | Video VAE |
| --- | --- | --- | --- | --- |
| `quality` | BF16 | BF16 | BF16 | FP16 |
| `balanced` (default) | BF16 | BF16 | BF16 | NVFP4 |
| `performance` | BF16 | FP8 | BF16 | NVFP4 |
| `maximum` | BF16 | NVFP4 | FP8 | NVFP4 |

Omitting `--quantization-config` selects `balanced`.

```bash
--quantization-config '{"mode":"maximum"}'
```

`components` overrides individual parts of the selected mode by the names `attention`, `mlp`, `text_encoder`, and `video_vae`:

```bash
--quantization-config '{"mode":"performance","components":{"video_vae":"bf16"}}'
```

CUDA graphs are always on. The denoising step and the media-decoding calls are captured on first use and replayed afterwards, so the first request after startup is slower than steady state; a capture failure is raised rather than silently degrading to eager execution. Precision, placement, duration capacity, and prompt capacity are startup settings; restart the server after changing them.

The four tiers above are runtime dynamic-quantization presets for dense checkpoints. A ModelOpt PTQ checkpoint loads without `--quantization-config`. Each quantized Linear stores its packed NVFP4 values under `.weight`, its K16 block scales in `.weight_scale`, its FP32 weight scale in `.weight_scale_2`, and the static activation scale its calibration cohort recorded in `.input_scale`; the runtime computes only the per-input K16 NVFP4 block encoding against that input scale and does not search a new global scale. Because every row interval uses that same checkpoint-owned scale, singleton DP replicas encode and project bounded 64 MiB intervals instead of materializing a full-token MLP intermediate. Packed checkpoints reject runtime precision presets and component overrides because their weights and scales form one immutable numerical contract.

Both published packed checkpoints quantize their calibrated denoiser MLP projections and Video VAE Transformer projections to NVFP4 and retain the BF16 text encoder. The stored tensors are the contract: exactly the Linears whose weights are packed run in NVFP4, every other module runs in BF16 with FP32 decoders and latent heads, and loading refuses a ModelOpt recipe other than static NVFP4 weights and activations with 16-element blocks.

On four GB200 devices, the packed 8-Step V2 checkpoint serves every measured duration and prompt length faster than any runtime tier applied to the dense 8-Step V2 checkpoint, and at lower peak device memory. Measurements outside that hardware and workload require their own run.

```bash
uniserve serve skx618/FastVideo-FastH3-8-Step-V2-NVFP4 \
  --workers config/minimax-h3-four-devices.json \
  --served-model-name FastH3
```

## Troubleshooting

| Error | Action |
| --- | --- |
| `_uniserve_ipc` import or protocol error | Run `uv sync --locked --python /usr/bin/python3.12 --extra gpu` again and use the resulting `uniserve` executable. |
| CUDA or sparse-attention compile error | Check CUDA 13 `nvcc`, `CUDA_HOME`, SM100 hardware, C++ build tools, and a writable `TORCH_EXTENSIONS_DIR`. |
| Missing audio VAE or codec | Restore the locked environment with `uv sync`; do not mix in older Diffusers or PyAV packages. |
| `unsupported FastH3 checkpoint` | Use a complete checkpoint from the table above; the message names the model ID and revision it expects. |
| `checkpoint format 'modelopt_nvfp4' owns its numerical configuration` | Drop `--quantization-config`: a packed NVFP4 checkpoint carries its own precision contract. |
| GPU out of memory | Reduce resident capacity, or write a deployment configuration that shards the denoiser; sequence parallelism alone replicates denoiser weights. |
| MP4 contains an error body | Use `--fail-with-body`, inspect the HTTP status, and confirm the model name and `/v1/capabilities` limits. |
