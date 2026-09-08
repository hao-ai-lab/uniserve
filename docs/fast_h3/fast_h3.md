# FastH3 cheat sheet

UniServe serves `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` as text-to-video-and-audio. The output is an H.264/AAC MP4 at 1344×768, 24 fps, with stereo 32-kHz audio.

## Requirements

- Linux, CUDA 13, Python 3.12, a stable Rust toolchain, and NVIDIA SM100 GPUs such as B200 or GB200.
- The full FastH3 VSA checkpoint. Base partitions and adapter-only checkpoints are unsupported.
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

uv sync --locked --python /usr/bin/python3.12 --extra h3
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

The `h3` extra installs the locked H3 runtime, FlashInfer, peer-memory and sparse-attention kernels, Diffusers audio decoding, CuTe DSL, CUTLASS DSL, and PyAV. FastVideo itself is not a runtime dependency.

## Download the model

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree

hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
  --revision 5ea076f35b84da4c3c82217112fa733d8eea2ae1 \
  --local-dir "$H3_MODEL"
```

Pass the model root containing `modular_model_index.json`, `fastvideo_inference.json`, `transformer/`, `text_encoder/`, `vae/`, `audio_vae/`, `scheduler/`, `audio_scheduler/`, and `tokenizer/`.

## Start the server

```bash
uniserve doctor --model "$H3_MODEL" --worker-ranks 4

uniserve serve "$H3_MODEL" \
  --worker-ranks 4 \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-model-len 16384 \
  --max-video-seconds 15 \
  --max-running-requests 2 \
  --quantization-config '{"mode":"balanced"}' \
  --graph-policy auto
```

`--worker-ranks 4` uses four-way Ulysses denoising, TP4 text encoding, temporal video decoding on all four ranks, and rank 0 for audio decoding and MP4 assembly. It does not shard denoiser weights. Use `--workers` for explicit component placement or denoiser tensor/pipeline parallelism.

Check the live limits and served model name after startup:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/capabilities | python -m json.tool
```

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
    --worker-ranks 4 \
    --served-model-name FastH3 \
    --host 0.0.0.0 \
    --max-model-len 16384 \
    --max-video-seconds 15 \
    --max-running-requests 2 \
    --quantization-config '{"mode":"balanced"}' \
    --graph-policy auto
```

The first startup compiles native GPU providers and captures shapes on first use. Mount a writable compiler cache if container restarts must reuse those artifacts.

## Generate a video

Only `model`, `prompt`, `seconds`, and `seed` are accepted. `seconds` defaults to 5 and `seed` defaults to 0.

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","prompt":"A clear stream flows through a green forest while birds sing.","seconds":5,"seed":1000}' \
  --output forest.mp4
```

The synchronous endpoint returns MP4 bytes. H3 rounds duration to its temporal geometry: a 5-second request produces 124 frames (about 5.17 seconds), and a 15-second request produces 362 frames (about 15.08 seconds).

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

Job states are `queued`, `in_progress`, `completed`, and `failed`. Jobs and retained MP4s live in the server process, expire after one hour, and disappear on restart. The server retains at most 128 jobs and 1 GiB of artifacts. Deleting a queued or running job cancels it.

## Precision and graphs

Precision is selected at startup with `--quantization-config`:

| Mode | Denoiser attention | Denoiser MLP | Text encoder | Video VAE |
| --- | --- | --- | --- | --- |
| `quality` | BF16 | BF16 | BF16 | FP16 |
| `balanced` (default) | BF16 | BF16 | BF16 | NVFP4 |
| `performance` | BF16 | FP8 | BF16 | NVFP4 |
| `maximum` | NVFP4 | MXFP8 | FP8 | NVFP4 |

Example:

```bash
--quantization-config '{"mode":"maximum"}'
```

`--graph-policy auto` captures supported entries, `full` requires every declared numerical entry to capture, and `off` disables CUDA graphs. Precision, placement, duration capacity, prompt capacity, and graph policy are startup settings; restart the server after changing them.

## Fixed model contract

- Four denoiser forwards with inference grid `[1, 0.75, 0.5, 0.25, 0]` and video/audio sigma shifts `12/3`.
- VSA sparse attention with tile size 64, sparsity 0.9, and the SM100a kernel.
- Fixed 1344×768 output, 24 fps, and stereo 32-kHz audio.
- Text-only conditioning. Image/video references, LoRA, variable resolution, guidance changes, and step-count changes are rejected.

## Quick fixes

| Error | Action |
| --- | --- |
| `_uniserve_ipc` import or protocol error | Run `uv sync --locked --python /usr/bin/python3.12 --extra h3` again and use the resulting `uniserve` executable. |
| CUDA or sparse-attention compile error | Check CUDA 13 `nvcc`, `CUDA_HOME`, SM100 hardware, C++ build tools, and writable compiler caches. |
| Missing audio VAE or codec | Restore the locked environment with `uv sync`; do not mix in older Diffusers or PyAV packages. |
| Checkpoint validation error | Use the complete pinned model root and revision shown above. |
| GPU out of memory | Reduce resident capacity or choose a sharded `--workers` layout; sequence parallelism alone replicates denoiser weights. |
| MP4 contains an error body | Use `--fail-with-body`, inspect the HTTP status, and confirm the model name and `/v1/capabilities` limits. |
