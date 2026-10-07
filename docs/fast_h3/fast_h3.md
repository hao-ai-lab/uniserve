# FastH3 cheat sheet

UniServe serves the FastH3 text-to-video-and-audio checkpoints. The output is an H.264/AAC MP4 at one of the checkpoint's training buckets, 1344×768 and 768×1344 in the default deployment (see [Generate a video](#generate-a-video)), 24 fps, with stereo 32-kHz audio.

## Requirements

- Linux, CUDA 13, Python 3.12, a stable Rust toolchain, and NVIDIA Hopper or Blackwell GPUs. H200, GB200, and RTX PRO 6000 Blackwell Server Edition have end-to-end validation.
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
export MAX_JOBS=2

uv sync --locked --python /usr/bin/python3.12 --extra gpu
source .venv/bin/activate
```

The `gpu` extra installs the locked GPU providers FastH3 serves through: FlashInfer, FlashAttention-4, and the CuTe and CUTLASS DSLs. These are shared runtime capabilities rather than model-specific packages, so there is no FastH3-specific dependency group. The sync also builds the `_uniserve_ipc` extension, the `uniserve` and `uniserve-host` binaries, and UniServe's native kernels, including the peer-storage and sparse-attention kernels, from this checkout. The kernels compile for the visible GPUs, or for the architectures `TORCH_CUDA_ARCH_LIST` names, and `MAX_JOBS` bounds the compilation's parallelism. A later sync recompiles them when their sources, `UNISERVE_KERNELS_DEVICE` or `TORCH_CUDA_ARCH_LIST` change. FastVideo itself is not a runtime dependency. The environment is the repository's `.venv`, which the evaluation profiles and the [Dynamo guide](dynamo.md) run from.

## Supported checkpoints

| Repository | Weights | Denoiser forwards | VSA sparsity | Download size |
| --- | --- | --- | --- | --- |
| [`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) | BF16 | 4 | 0.9 | 148 GB |
| [`FastVideo/FastVideo-FastH3-8-Step-V2`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) | BF16 | 8 | 0.8 | 148 GB |
| [`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4) | ModelOpt NVFP4 | 4 | 0.9 | 123 GB |
| [`FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4) | ModelOpt NVFP4 | 8 | 0.8 | 123 GB |

The two NVFP4 repositories are ModelOpt PTQ checkpoints in ModelOpt's unified Hugging Face layout, which UniServe loads with no precision flags and which require Blackwell; the BF16 checkpoints run on Hopper and Blackwell. See [Precision and graphs](#precision-and-graphs).

Each checkpoint's `fastvideo_inference.json` supplies its sampling schedule: the trained DMD rungs in `dmd_denoising_steps`, one transformer forward per rung, and the VSA sparsity. The video and audio shifts come from `scheduler/` and `audio_scheduler/`, and a manifest that restates them must agree. The loader refuses a manifest that is not a `fasth3-inference-contract-v1` text-to-video-and-audio contract without guidance. The table lists the checkpoints UniServe has validated end to end.

`uniserve serve` takes either a repository id, which loads the latest published revision, or a local model directory.

## Download the model

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2

hf download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir "$H3_MODEL"
```

Pass the model root containing `modular_model_index.json`, `fastvideo_inference.json`, `transformer/`, `text_encoder/`, `vae/`, `audio_vae/`, `scheduler/`, `audio_scheduler/`, and `tokenizer/`. In an NVFP4 root in ModelOpt's unified layout, `transformer/config.json` and `vae/config.json` also declare a `quantization_config` with `quant_method: modelopt`. The server identifies the model from the pipeline class that `modular_model_index.json` declares and reads the tokenizer from the component folder the index names; the loader then validates the inference contract in `fastvideo_inference.json`.

## Start the server

```bash
uniserve serve "$H3_MODEL" \
  --workers configs/fast_h3/ulysses4.json \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 2
```

`configs/fast_h3/ulysses4.json` is a deployment file: it lists the participating devices and, under `components`, the placement and parallel configuration of each of FastH3's five components. It places the numerical components on four devices of one host, as the `model` worker: four-way Ulysses denoising, TP4 text encoding, one video and one audio media unit per rank. A second worker, `host`, has five ranks on the host with `"device": "cpu"` and holds the two host components every video deployment needs: `video_codec` on ranks 0 to 3, one media unit of each decode round per rank, and `muxer` on rank 4, which encodes the audio track and assembles the MP4. Each host rank is one codec slot: it runs one encode or assembly step at a time in its own process. The muxer must be on the head's host, and every worker that decodes video needs a `video_codec` rank that keeps each media unit on the host that decoded it; the server refuses a placement that violates either at startup and routes each request to such a codec. Under Ulysses each rank keeps only its own heads' share of the denoiser's merged query, key, value and gate projections, a quarter of those weights on four devices; the other denoiser weights are replicated on every rank. Serving a different width, or sharding the denoiser by tensor or pipeline, is a different deployment file.

### Deployments

`configs/fast_h3/` holds the deployments UniServe publishes measurements for:

| File | GPUs | Denoiser | Text encoder |
| --- | --- | --- | --- |
| `ulysses4.json` | Four on one host | Ulysses4 | TP4 |
| `ulysses8.json` | Eight on one host | Ulysses8 | TP8 |
| `ulysses8-two-node.json` | Four on each of two hosts | Ulysses8 | TP8 |
| `ulysses4x2.json` | Eight on one host | Two Ulysses4 replicas | One TP4 replica per denoiser replica |
| `ulysses4x2-two-node.json` | Four on each of two hosts | One Ulysses4 replica per host | One TP4 replica per host |
| `dp8-text-tp8.json` | Eight on one host | Eight one-GPU replicas | One TP8 worker shared by all replicas |
| `gather8.json` | Eight on one host | Eight-way all-gather sequence parallelism | TP8 |

A single request is fastest when every GPU works on it, so the one-replica files serve latency; replicas serve more requests at once. These commands serve the 8-Step V2 checkpoints at the default 15-second and 16384-token capacities unless the row states a limit:

| Hardware | Goal | Deployment | Additional options |
| --- | --- | --- | --- |
| 4 x GB200 | Latency and throughput | `ulysses4.json` | `--max-running-requests 2`, `NCCL_NVLS_ENABLE=0` |
| 8 x GB200, two hosts | Latency | `ulysses8-two-node.json` | `--max-running-requests 2 --graph-policy off`, `NCCL_NVLS_ENABLE=0` |
| 8 x GB200, two hosts | Throughput | `ulysses4x2-two-node.json` | `--max-running-requests 4`, `NCCL_NVLS_ENABLE=0` |
| 8 x RTX PRO 6000 | Latency | `ulysses8.json` | `--max-running-requests 2 --graph-policy off --mem-fraction-static 0.92 --video-text-capacities 1024,10240,16384` |
| 8 x RTX PRO 6000 | Throughput | `ulysses4x2.json` | `--max-running-requests 4 --graph-policy off --mem-fraction-static 0.92 --video-text-capacities 1024,10240,16384` |
| 4 x H200 | Latency and throughput | `ulysses4.json` | `--max-running-requests 2 --mem-fraction-static 0.92 --video-text-capacities 1024,16384` |
| 8 x H200 | Latency, up to 10240 prompt tokens | `ulysses8.json` | `--max-running-requests 2 --mem-fraction-static 0.92 --max-model-len 10240 --video-text-capacities 1024,10240` |
| 8 x H200 | Throughput, up to 5 s and 1024 prompt tokens | `dp8-text-tp8.json` | `--max-running-requests 8 --max-video-seconds 5 --max-model-len 1024` |

Every GB200 measurement ran with `NCCL_NVLS_ENABLE=0`, which disables NCCL's NVLink SHARP multicast; a four-GPU GB200 host also serves without it. The [UniServe FastH3 post](https://hao-ai-lab.github.io/blogs/uniserve-fasth3/) assembles the same commands from a hardware, goal and checkpoint selection. `gather8.json` attends each rank's own query rows to the complete gathered keys and values and holds the complete denoiser weights on every rank; it is an alternative to Ulysses where head sharding does not apply.

### Two hosts

The two-host files name their hosts `rank-0` and `rank-1`. The head runs on the host named `rank-0`, and a launcher on the other host runs the ranks the file places there. Start the head first:

```bash
NCCL_NVLS_ENABLE=0 uniserve serve "$H3_MODEL" \
  --workers configs/fast_h3/ulysses8-two-node.json \
  --host-identity rank-0 \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 2 \
  --graph-policy off
```

The head logs `awaiting a launcher for each host this instance does not run on address="0.0.0.0:<port>"`: it listens on every interface at that port. On the other host, start the launcher with an address of `rank-0` it can reach, that port, and the host identity the file names:

```bash
NCCL_NVLS_ENABLE=0 uniserve-host --head <rank-0 address>:<port> --host-identity rank-1
```

One launcher serves every worker of the deployment that places a rank on its host: in `ulysses8-two-node.json`, ranks 4 to 7 of the model worker and encoder ranks 4 to 7 of the host worker, which encode that host's four media units per round, while the muxer rank stays on `rank-0`. The launcher starts each rank with the Python interpreter and launch descriptor the head resolved, so the other host needs the same Python environment and the checkpoint at the same path. The launcher and the head speak one launch protocol, so run a `uniserve-host` built from the same source as the head's `uniserve`.

### Replicas

A replica is a complete denoiser, video decoder and audio decoder that serves one request at a time. `ulysses4x2.json` and `ulysses4x2-two-node.json` run two four-GPU Ulysses replicas, each with its own TP4 text encoder; `dp8-text-tp8.json` runs eight one-GPU replicas that share one TP8 text encoder. The scheduler selects one flow replica when it admits a request, keeps that request on the same replica through latent preparation, denoising, and device decoding, and accounts its request rows, product buffers, queue, and execution lane in that worker's address space. Replica capacities add; a shared component remains an independently bounded stage of the route.

Each GPU decoder has one native media unit per rank, so a one-GPU flow replica reconstructs an eight-unit request in eight bounded decode rounds, and its units reach the encoder one at a time. Each request is therefore bound to one single-rank encoder worker, chosen at admission, and the eight encoder workers encode eight requests' units concurrently. Audio encoding and MP4 assembly run on the one muxer worker, serialized per request because they are short relative to denoising and need one ordered artifact owner.

The queue depths are capacity values. A resident media request slot occupies three positions of its worker's batch queue, one reserved pipeline position and two unresolved outputs, and every flow worker must expose at least two resident slots, so each flow worker uses depth 6. In `dp8-text-tp8.json` the shared TP8 text worker and the muxer worker use depth 24 for eight slots, and each encoder worker uses depth 6 for two, so the narrowest aggregate component capacity is eight complete routes.

The `memory_fraction` on each GPU worker is its per-process static storage ceiling. It bounds everything the process holds on the device as NVML reports it per process: the caching allocator's segments and graph pools, product arenas, communicator buffers, captured graph executables, loaded kernels and the CUDA context. NVML must therefore report that usage under the process ID the worker sees. A container whose NVML reports host process IDs fails startup with `NVML reports no device storage for process`; run such a container in the host's process ID namespace (`--pid=host`). Text and flow workers share every GPU, so neither inherits the global `--mem-fraction-static` default: the replica files grant 0.26 to each text rank and 0.73 to each flow rank, and `dp8-text-tp8.json` grants 0.18 and 0.81. The physical free-storage check still caps their combined allocations. The DP8 split holds a five-second duration and 1024-token prompt capacity; revalidate it when checkpoint precision, maximum duration, prompt bound, graph shapes, or hardware change, and adjust the text and flow fractions together.

DP improves throughput only when the offered concurrency keeps multiple replicas occupied. At concurrency one, Ulysses retains lower latency because all GPUs cooperate on one denoising call; at higher concurrency, replicas remove that per-step collective and keep queueing behind one request from dominating service time.

Ulysses shards the merged query, key, value and gate weights by attention head while retaining the other denoiser weights on every rank. A rank's residency can exceed the default `--mem-fraction-static` of 0.70 on a 96 GB or 141 GB device at the default `--max-video-seconds 15` and `--max-model-len 16384`. Raise the fraction, or shard the denoiser further with `"tensor_parallel_size"`, if a deployment refuses to start with a static storage grant error.

Cross-rank products move over the mechanism named for that edge. CUDA VMM reads inspect the visible CUDA peer topology: a consumer with direct access maps the allocation on its destination GPU, while a consumer outside the producer's peer set maps it on the source GPU and uses CUDA's host-staged cross-device copy. The choice depends on the runtime topology rather than the accelerator model. `--transfer model->model=shm` remains available when an operator wants to force host staging for every model-worker product.

A FastH3 deployment defaults to `--max-video-seconds 15` and `--max-model-len 16384`; set them only to change those limits. `--max-running-requests` caps concurrently resident requests, and the engine clamps that cap to the worker's advertised request-slot capacity; lowering it trades throughput for per-request latency and storage headroom.

Startup prepares every computation an admitted request reaches. The denoiser is prepared for each of the 16 output frame counts up to `--max-video-seconds`, at every text capacity and every served canvas (see [Generate a video](#generate-a-video)): `--video-text-capacities` lists those capacities in prompt tokens (default: 1024, then steps of 2048 up to `--max-model-len`), and a request evaluates in the smallest capacity that holds its prompt. The text encoder and the text refiner are prepared at every text capacity, and the audio decoder at every admitted duration. The video decoder reconstructs each media unit from a fixed window of latent frames, so it is prepared once per served canvas and every duration shares it; a request's unit is unpacked from its complete latent before the window is decoded. Each served canvas therefore adds a copy of the denoiser layouts and their startup captures. Request storage and the denoiser's shared workspace are sized for the canvas that needs the most: 768p 21:9, 16:9 and 9:16 pack the same number of latent tokens and every other bucket fewer, but 21:9 pads to about 8% more sparse-attention tiles, so adding 768p 21:9 grows the denoiser's storage while the other canvases add startup time but no per-request storage. With graphs enabled (`--graph-policy auto`, the default, or `full`), startup captures one denoising graph per layout and every text-encoding and decoding call before the server reports ready, and accepted requests replay them: serving never captures and never falls back to eager execution. A layout's graph evaluates any of the eight solver steps, because the step index, the timesteps and the schedule are inputs each replay copies in. The video post-processor, which converts decoded frames for encoding, runs eagerly by design. The denoiser's layouts share one workspace and one graph pool, so its resident memory follows the largest layout rather than their count, while startup time grows with the count; fewer capacities start faster, and finer ones pad shorter prompts less. A decoder's graphs replay one at a time and each replay's output is copied out before the next, so they share one input and one output backing sized for the largest call: the video decoder captures one graph per served canvas and the audio decoder one per admitted duration, and their graph storage stays close to one call's working set. Each captured graph also holds device storage outside the graph pools, about 16 to 20 MiB per denoising graph on GB200, so the graph count, the product of frame counts, canvases and text capacities, bounds how many canvases fit; serve only the canvases a deployment needs, and fewer text capacities when serving many. On eight RTX PRO 6000 devices (`ulysses8.json`, `--graph-policy off`, `--video-text-capacities 1024,10240,16384`), the default two canvases start in about 16 minutes with about 63 GiB per model worker process; serving all 12 buckets (`--video-resolutions 768p,480p --video-aspect-ratios 21:9,16:9,4:3,1:1,3:4,9:16`) prepares 576 denoiser layouts, starts in about 65 minutes from warm kernel caches (76 minutes the first time, which compiles the sparse-attention kernels for each new layout), and holds about 72 GiB per process. Serving more canvases does not change a request's latency: 5-second 1344×768 requests took 24.5 seconds in both deployments. With the default settings on four GB200 devices (`configs/fast_h3/ulysses4.json`), the default two canvases capture 288 denoising graphs, one per layout, and the server becomes ready in about 17 minutes from warm kernel caches. The first startup on a machine also compiles the kernels those steps run and takes longer; later startups load them from the caches described in [Build and run with Docker](#build-and-run-with-docker). Until preparation finishes, the log reports `worker still busy during Worker startup` with the elapsed seconds. Each model worker process then holds about 104 to 105 GiB on its device as NVML reports it, the figure its storage grant charges. All 12 buckets at the default nine text capacities capture 1728 denoising graphs, which exceed the `0.70` grant, while all 12 buckets at `--video-text-capacities 1024,10240,16384` fit at about 119 to 121 GiB per process. A deployment whose startup storage does not fit its devices fails before it reports ready; the startup log lists the bytes each process holds on each device, the caching allocator's share of them, and the graph storage of each component. `--graph-policy off` serves the same layouts without graphs.

Check the live limits and served model name after startup:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/capabilities | python -m json.tool
```

`/v1/capabilities` reports the served tasks and their condition rules, the canvas rule and the canvases the checkpoint serves, the duration limits, the checkpoint's schedule, the prompt capacity, the accepted media sources and request fields, and the job retention bounds described below.

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
    --workers configs/fast_h3/ulysses4.json \
    --served-model-name FastH3 \
    --host 0.0.0.0 \
    --max-running-requests 2
```

The image build compiles UniServe's native kernels for the architectures the Dockerfile's `TORCH_CUDA_ARCH_LIST` names: Hopper (9.0), data-center Blackwell (10.0) and RTX PRO 6000 Blackwell (12.0). The first startup compiles the Triton and FlashInfer kernels FastH3 runs; later startups load the cached builds without compiling. They land in `~/.triton/cache` and `~/.cache/flashinfer`. If container restarts must reuse them, mount `/root/.cache` and `/root/.triton`, or point `TRITON_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE` at mounted directories.

## Generate a video

A request is the MiniMax-H3 request body, sent as JSON or as `multipart/form-data` with the same field names and `conditions` and `target` as JSON text. FastH3 generates text-to-video-and-audio (`t2va`) at the checkpoint's training buckets: `target.short_edge` selects the resolution and `target.aspect_ratio` the aspect ratio.

| `target.short_edge` | `21:9` | `16:9` | `4:3` | `1:1` | `3:4` | `9:16` |
| --- | --- | --- | --- | --- | --- | --- |
| 768 | 1536×672 | 1344×768 | 1024×768 | 768×768 | 768×1024 | 768×1344 |
| 480 | 992×416 | 832×480 | 640×480 | 480×480 | 480×640 | 480×832 |

A deployment serves the canvases it prepares at startup: every resolution in `--video-resolutions` (`768p` or `480p`; default `768p`) crossed with every aspect ratio in `--video-aspect-ratios` (default `16:9,9:16`), so the default deployment serves 1344×768 and 768×1344. A request for any other canvas is rejected. `GET /v1/capabilities` lists the served canvases under `canvas.canvases`, the served short edges and aspect ratios under `canvas.short_edges` and `canvas.aspect_ratios`, and the canvas of each served pair under `canvas.sizes`.

| Field | Rule |
| --- | --- |
| `model` | Required; the served model name |
| `prompt` | Required; not blank |
| `task` | Required; `t2va` |
| `conditions` | Optional; empty for `t2va` |
| `target` | Required: `short_edge` 768 or 480, `aspect_ratio` a served ratio or `auto` (16:9 for `t2va`), and `duration_seconds`, a finite number of seconds from 4 to 15 inclusive, fractional values included |
| `seed` | Optional unsigned integer; defaults to 42 |
| `num_inference_steps`, `flow_shift`, `audio_flow_shift` | Optional; when present, each must equal the checkpoint's schedule that `/v1/capabilities` reports under `schedule`. `num_inference_steps` counts sigma points including the clean endpoint, one more than the denoiser forwards: 9 for the 8-Step checkpoints and 5 for the 4-step ones |
| `n`, `num_outputs_per_prompt` | Optional; only 1 |
| `quality` | Optional; only `lossless` |
| `seconds`, `size`, `width`, `height` | Optional; accepted only when they agree with the duration and canvas the target resolves to |

Any other field is rejected, and so is a request without `task` or `target`; the response names the field. `--max-video-seconds` sets the deployment's duration capacity within the API range (default 15); a request longer than the capacity is rejected, and `GET /v1/capabilities` reports the capacity as `max_seconds` next to the API range `min_seconds` and `model_max_seconds`.

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","prompt":"A clear stream flows through a green forest while birds sing.","task":"t2va","target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5},"seed":1000}' \
  --output forest.mp4
```

The synchronous endpoint returns MP4 bytes at 24 frames per second. H3 converts the requested `duration_seconds` to `duration_seconds * 24` frames, rounded half to even, and extends that count up to its next complete temporal window of `17n + 5` frames, so the video can last slightly longer than requested. A 4-second request produces 107 frames (about 4.46 seconds), a 5-second request 124 frames (about 5.17 seconds), and a 15-second request 362 frames (about 15.08 seconds). A fixed output resolution therefore has exactly 16 frame counts: 107, 124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345, and 362.

## Use asynchronous jobs

Create a job:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","prompt":"A clear stream flows through a green forest while birds sing.","task":"t2va","target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5},"seed":1000}'
```

Use the returned `video_...` ID:

```bash
export VIDEO_ID=video_example

curl --fail-with-body "http://127.0.0.1:8000/v1/videos/$VIDEO_ID"
curl --fail-with-body "http://127.0.0.1:8000/v1/videos/$VIDEO_ID/content" --output forest.mp4
curl --fail-with-body http://127.0.0.1:8000/v1/videos
curl --fail-with-body -X DELETE "http://127.0.0.1:8000/v1/videos/$VIDEO_ID"
```

Job states are `queued`, `in_progress`, `completed`, and `failed`. A running job also reports `phase` (`encoding`, `preparing`, `denoising`, `decoding`, then `finalizing`), `completed_steps` against `total_steps`, the requested duration `seconds`, the generated `num_frames` with their duration `actual_seconds`, and the generated canvas `size` (`1344x768`). Jobs and retained MP4s live in the server process, expire after one hour, and disappear on restart. The server retains at most 1 GiB of artifacts, and retained jobs and in-flight synchronous requests share 128 job slots. Deleting a queued or running job cancels it.

A request to either endpoint while all 128 job slots are taken returns HTTP 429 with code `video_job_capacity_exceeded`. A failed job reports `error.code`: `invalid_request_error` when the deployment cannot serve the request as specified, `server_overloaded` when the engine's waiting queue was full and the same request may be resubmitted, and `generation_failed` when execution failed. The synchronous endpoint returns the same conditions as HTTP 400, 503, and 500.

## Precision and graphs

A dense BF16 checkpoint selects its precision at startup with `--quantization-config`:

| Mode | Denoiser attention | Denoiser MLP | Text encoder | Video VAE |
| --- | --- | --- | --- | --- |
| `quality` (default) | BF16 | BF16 | BF16 | FP16 |
| `balanced` | BF16 | BF16 | BF16 | NVFP4 |
| `performance` | BF16 | FP8 | BF16 | NVFP4 |
| `maximum` | BF16 | NVFP4 | FP8 | NVFP4 |

Omitting `--quantization-config` selects `quality`, the checkpoint's own BF16 and FP16 representations. The other tiers use NVFP4, which needs Blackwell; see [Hopper](#hopper).

```bash
--quantization-config '{"mode":"maximum"}'
```

`components` overrides individual parts of the selected mode by the names `attention`, `mlp`, `text_encoder`, and `video_vae`:

```bash
--quantization-config '{"mode":"performance","components":{"video_vae":"bf16"}}'
```

CUDA graphs are captured during startup, as described above, so requests never capture; a capture failure stops startup rather than silently degrading to eager execution. Precision, placement, duration capacity, and prompt capacity are startup settings; restart the server after changing them.

The four tiers above are runtime dynamic-quantization presets for dense checkpoints. A ModelOpt PTQ checkpoint loads without `--quantization-config`. Each quantized Linear stores its packed NVFP4 values under `.weight`, its K16 block scales in `.weight_scale`, its FP32 weight scale in `.weight_scale_2`, and the static activation scale its calibration cohort recorded in `.input_scale`; the runtime computes only the per-input K16 NVFP4 block encoding against that input scale and does not search a new global scale. Because every row interval uses that same checkpoint-owned scale, singleton DP replicas encode and project bounded 64 MiB intervals instead of materializing a full-token MLP intermediate. Packed checkpoints reject runtime precision presets and component overrides because their weights and scales form one immutable numerical contract.

Both published packed checkpoints quantize their calibrated denoiser MLP projections and Video VAE Transformer projections to NVFP4 and retain the BF16 text encoder. The stored tensors are the contract: exactly the Linears whose weights are packed run in NVFP4, every other module runs in BF16 with FP32 decoders and latent heads, and loading refuses a ModelOpt recipe other than static NVFP4 weights and activations with 16-element blocks.

```bash
uniserve serve FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4 \
  --workers configs/fast_h3/ulysses4.json \
  --served-model-name FastH3
```

## Hopper

The BF16 checkpoints run on Hopper in the default `quality` precision. Hopper computes BF16, FP16, and FP8; NVFP4 and MXFP8 need Blackwell tensor cores, so the `balanced`, `performance`, and `maximum` tiers and the NVFP4 checkpoints do not run there. FP8 denoiser MLPs are available as a component override:

```bash
--quantization-config '{"components":{"mlp":"fp8"}}'
```

H200 deployments at 15-second capacity use `--mem-fraction-static 0.92`, the grant their published measurements use.

## Reproduce the measurements

The evaluator runs the published workloads against the deployments above and validates every MP4. Run points serially, one server at a time, from the repository root. Each run writes its resolved commands, request data, per-request latencies, media validation and GPU memory observations under `artifacts/fast_h3/`.

The `bench` extra installs the evaluator. `uv sync` keeps exactly the extras it names, so name `gpu` beside it:

```bash
uv sync --locked --python /usr/bin/python3.12 --extra gpu --extra bench
```

The profiles start `.venv/bin/uniserve` with `.venv/bin/python` as the worker interpreter, so run them from the checkout whose `.venv` holds this environment.

### UniServe FastH3 post

`uniserve_eval/fast_h3.toml` holds the post's deployments and workload: 72 latency requests one at a time, 12 for each combination of a 5, 10 or 15 second clip and a 1000- or 10000-token prompt, and 32 throughput requests at each concurrency after a priming phase, all preceded by six unmeasured warmup requests. The request manifests are in `uniserve_eval/workloads/fast_h3/`. One suite covers each hardware configuration and checkpoint: `gb200-4-bf16`, `gb200-4-nvfp4`, `gb200-8-bf16`, `gb200-8-nvfp4`, `rtx-pro-6000-8-bf16`, and `rtx-pro-6000-8-nvfp4`.

```bash
export UNISERVE_FAST_H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
export UNISERVE_FAST_H3_NVFP4_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2-NVFP4
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml plan gb200-4-bf16
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml run gb200-4-bf16
```

The `gb200-8` suites start the head on `rank-0`; start `uniserve-host` on `rank-1` with `rank-0`'s address and the port the head logs, as in [Two hosts](#two-hosts), once for each server the suite starts.

### H200

`uniserve_eval/fast_h3_h200.toml` holds the H200 measurements below. They use `FastVideo/FastVideo-FastH3-8-Step-V2` at revision `3da2ddfe1954d9cda4c05b643dc0f26007a655c5`, the evaluator's synthesized `minimax-h3` prompts with seed 1000, full graphs, `--mem-fraction-static 0.92`, and `NCCL_CUMEM_HOST_ENABLE=1`. Latency is the mean ± SD of three complete MP4 requests at concurrency one after one warmup. The DP8 points use eight warmups and 16 measured requests at concurrency eight.

Eight-H200 Ulysses in `quality` precision:

| Video duration | 1000-token prompt | 10000-token prompt |
| --- | ---: | ---: |
| 5 s | 8.194 ± 0.818 s | 14.218 ± 0.880 s |
| 10 s | 18.539 ± 1.071 s | 26.578 ± 0.600 s |
| 15 s | 31.069 ± 0.045 s | 44.228 ± 0.878 s |

| Deployment | Precision | Workload | Result |
| --- | --- | --- | --- |
| Four-H200 Ulysses | `quality` | 5 s, 1000 tokens | 13.864 ± 0.013 s |
| Gather8 | `quality` | 5 s, 10000 tokens | 20.170 ± 0.698 s |
| Eight-H200 Ulysses | FP8 MLPs | 5 s, 1000 tokens | 7.448 ± 0.734 s |
| Eight-H200 Ulysses | FP8 MLPs | 15 s, 10000 tokens | 42.380 ± 0.807 s |
| DP8 + text TP8 | `quality` | 5 s, 1000 tokens, concurrency 8 | 0.1519 videos/s, 52.231 ± 0.358 s, 90,329 MiB peak per GPU |
| DP8 + text TP8 | FP8 MLPs | 5 s, 1000 tokens, concurrency 8 | 0.1676 videos/s, 47.314 ± 0.386 s, 81,531 MiB peak per GPU |

The `h200-ulysses8` suite reproduces the first table; the `h200-ulysses4` and `h200-fp8` suites and the `h200-gather8-5s-10k` and `h200-dp8-5s-1k` points reproduce the second:

```bash
export UNISERVE_FAST_H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3_h200.toml plan h200-ulysses8
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3_h200.toml run h200-ulysses8 --reuse-deployment
```

## Troubleshooting

| Error | Action |
| --- | --- |
| `_uniserve_ipc` import or protocol error | Run `uv sync --locked --python /usr/bin/python3.12 --extra gpu` again and use the resulting `uniserve` executable. |
| CUDA or sparse-attention compile error during `uv sync` | Check CUDA 13 `nvcc`, `CUDA_HOME`, C++ build tools, and a `TORCH_CUDA_ARCH_LIST` naming the target GPUs when none is visible to the build. |
| Missing audio VAE or codec | Restore the locked environment with `uv sync`; do not mix in older Diffusers or PyAV packages. |
| `unsupported FastH3 checkpoint` | Use a complete checkpoint from the table above; the message names the model ID and revision it expects. |
| `checkpoint format 'modelopt_nvfp4' owns its numerical configuration` | Drop `--quantization-config`: a packed NVFP4 checkpoint carries its own precision contract. |
| `nvfp4 conversion requires an SM100-class CUDA device` | NVFP4 and MXFP8 need Blackwell. On Hopper, keep the default `quality` precision or use FP8 component overrides. |
| `NVML reports no device storage for process` | NVML attributes no device usage to the worker's process ID, as when it reports host process IDs inside a container; run the container in the host's process ID namespace (`--pid=host`). |
| GPU out of memory | Reduce resident capacity, or shard the denoiser with tensor or pipeline parallelism; Ulysses retains most denoiser weights on each rank. |
| MP4 contains an error body | Use `--fail-with-body`, inspect the HTTP status, and confirm the model name and `/v1/capabilities` limits. |
