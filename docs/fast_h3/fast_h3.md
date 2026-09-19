# FastH3 cheat sheet

UniServe serves the FastH3 text-to-video-and-audio checkpoints. The output is an H.264/AAC MP4 at 1344×768, 24 fps, with stereo 32-kHz audio.

## Requirements

- Linux, CUDA 13, Python 3.12, a stable Rust toolchain, and NVIDIA SM100 GPUs such as B200 or GB200. GB200 has end-to-end validation; other hardware requires its own validation before performance claims.
- One of the complete FastH3 VSA checkpoints listed below. Base partitions and adapter-only checkpoints are unsupported.
- Enough GPU memory for weights, two resident requests, and CUDA graphs. Four local GPUs are the standard setup.
- Shared memory and pinned-memory access for worker IPC. The container command below provisions 4 GiB of shared memory and unlimited locked memory.

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

The `gpu` extra installs the locked GPU providers FastH3 serves through: FlashInfer, FlashAttention-4, the peer-memory and sparse-attention kernels, and the CuTe and CUTLASS DSLs. These are shared runtime capabilities rather than model-specific packages, so there is no FastH3-specific dependency group. The sync also builds the `_uniserve_ipc` extension and the `uniserve` binary from this checkout. FastVideo itself is not a runtime dependency.

## Supported checkpoints

| Repository | Weights | Denoiser forwards | VSA sparsity | Download size |
| --- | --- | --- | --- | --- |
| [`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) | BF16 | 4 | 0.9 | 148 GB |
| [`FastVideo/FastVideo-FastH3-8-Step-V2`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) | BF16 | 8 | 0.8 | 148 GB |
| [`skx618/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4`](https://huggingface.co/skx618/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4) | Packed NVFP4 | 4 | 0.9 | 123 GB |
| [`skx618/FastVideo-FastH3-8-Step-V2-NVFP4`](https://huggingface.co/skx618/FastVideo-FastH3-8-Step-V2-NVFP4) | Packed NVFP4 | 8 | 0.8 | 123 GB |

The two `skx618` repositories are self-describing ModelOpt PTQ checkpoints. UniServe loads them natively, with no conversion step and no precision flags: see [Precision and graphs](#precision-and-graphs).

`uniserve serve` takes either a repository id, which loads the latest published revision, or a local model directory.

## Download the model

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2-NVFP4

hf download skx618/FastVideo-FastH3-8-Step-V2-NVFP4 --local-dir "$H3_MODEL"
```

Pass the model root containing `modular_model_index.json`, `fastvideo_inference.json`, `transformer/`, `text_encoder/`, `vae/`, `audio_vae/`, `scheduler/`, `audio_scheduler/`, and `tokenizer/`. A packed NVFP4 root also contains `modelopt_manifest.json`. The server recognizes a FastH3 deployment by the presence of `fastvideo_inference.json`, and that manifest must match one of the checkpoints above.

## Start the server

```bash
uniserve serve "$H3_MODEL" \
  --workers config/minimax-h3-four-devices.json \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 2
```

`config/minimax-h3-four-devices.json` is a deployment file: it lists the participating devices and, under `components`, the placement and parallel configuration of each of FastH3's five components. It places them on four devices of one host: four-way Ulysses denoising, TP4 text encoding, one video and one audio media unit per rank, and rank 0 for MP4 assembly. It does not shard denoiser weights. Serving a different width, or sharding the denoiser by tensor or pipeline, is a different deployment file; the schema is in [parallel execution](../parallel-execution.md).

`config/minimax-h3-eight-devices.json` is the same placement over eight devices on two hosts, named `rank-0` and `rank-1` in the file: eight-way Ulysses denoising, TP8 text encoding, one video and one audio media unit per rank, and the muxer on rank 0. The head runs on the host named `rank-0`, which holds ranks 0 to 3, and a launcher on the other host runs ranks 4 to 7. Start the head first:

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

The launcher starts each rank with the Python interpreter and launch descriptor the head resolved, so the other host needs the same Python environment and the checkpoint at the same path.

A FastH3 deployment defaults to `--max-video-seconds 15` and `--max-model-len 16384`; set them only to change those limits. `--max-running-requests` caps concurrently resident requests, and the engine clamps that cap to the worker's advertised request-slot capacity; lowering it trades throughput for per-request latency and memory headroom.

`--video-graph-shapes 5x1000,15x10000` declares the duration and prompt length of the requests a deployment serves, and warmup captures each declared shape's denoising ladder on every request slot before the server reports ready. Without it the first request of each shape on each slot captures one graph per denoising step on its own path, which costs several seconds. A request whose duration or prompt length is not declared still serves.

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

The four tiers above are runtime dynamic-quantization presets for dense checkpoints. A packed ModelOpt PTQ checkpoint is self-describing and loads without `--quantization-config`. Its static activation tensor scales come from the recorded calibration cohort; the runtime computes only the per-input K16 NVFP4 block encoding and does not search a new global scale. Packed checkpoints reject runtime precision presets and component overrides because their weights and scales form one immutable numerical contract.

Both published packed checkpoints quantize their calibrated denoiser MLP projections and Video VAE Transformer projections to NVFP4 and retain the BF16 text encoder. Each repository's `modelopt_manifest.json` is the authoritative component and scale contract, and loading fails if it does not describe exactly those modules.

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
