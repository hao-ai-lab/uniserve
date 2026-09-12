# FastH3 cheat sheet

UniServe serves full FastVideo-exported FastH3 checkpoints as text-to-video-and-audio, including the pinned four-step VSA release and `FastVideo/FastVideo-FastH3-8-step-Preview-v1-VSA80-DataFree-Shift10`. The output is an H.264/AAC MP4 at 1344×768, 24 fps, with stereo 32-kHz audio.

## Requirements

- Linux, CUDA 13, Python 3.12, a stable Rust toolchain, and NVIDIA SM100 GPUs such as B200 or GB200.
- A full FastH3 VSA checkpoint or the pinned top-level MiniMax-H3 base root described below. Nested task partitions and adapter-only checkpoints are unsupported.
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

The `h3` extra installs the locked H3 runtime, FlashInfer, FlashAttention-4, peer-memory and sparse-attention kernels, Diffusers audio decoding, CuTe DSL, CUTLASS DSL, and PyAV. FastVideo itself is not a runtime dependency.

## Download the model

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree

hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
  --revision 5ea076f35b84da4c3c82217112fa733d8eea2ae1 \
  --local-dir "$H3_MODEL"
```

Pass the model root containing `modular_model_index.json`, `fastvideo_inference.json`, `transformer/`, `text_encoder/`, `vae/`, `audio_vae/`, `scheduler/`, `audio_scheduler/`, and `tokenizer/`.

## Checkpoint contracts

`H3_MODEL` selects the full checkpoint root. UniServe reads `fastvideo_inference.json` before constructing components. Published four-step and eight-step identities retain strict hash and recipe checks; another export must use its own identity. Hash fields identify exporter receipts, not a claim that UniServe rehashes every weight shard. Scheduler component configurations must agree with the declared shifts.

For the eight-step release:

```bash
export H3_MODEL=/mnt/lustre/vlm-wlsaidhi/fastvideo/exports/FastVideo-FastH3-8-step-Preview-v1-VSA80-DataFree-Shift10
unset UNISERVE_H3_CONTRACT H3_CONTRACT
python -m uniserve_worker.bootstrap.inspect_model --model "$H3_MODEL"
```

For a manifest-less local export, write an external JSON sidecar and select it explicitly. Do not copy a published manifest or invent its hashes. Obtain the export's content/metadata receipts and source commit, and choose its inference ladder explicitly; a directory name or training grid does not select an inference recipe. The checkpoint remains immutable.

| Field | Contract |
|---|---|
| `schema_version` | `fasth3-inference-contract-v1` |
| `checkpoint_root` | Resolved absolute checkpoint root; required for an external sidecar without an embedded manifest |
| `model_id` | Nonempty export identity, distinct from published releases |
| `checkpoint_content_sha256`, `checkpoint_metadata_sha256` | Actual exporter receipt identities, lowercase 64-digit SHA-256 |
| `fastvideo_commit` | Exact lowercase 40-digit FastVideo source commit |
| `task` | `t2av` |
| `transformer_forwards` | Positive integer equal to ladder length |
| `num_inference_steps` | Forward count plus one terminal grid point |
| `dmd_denoising_steps` | Strictly descending integer list in `(0, 1000]`; divided by 1000 before shifting, with terminal zero appended |
| `video_scheduler_shift`, `audio_scheduler_shift` | Positive finite shifts; video precedes audio |
| `guidance_scale` | `1.0` (CFG is not implemented) |
| `attention_backend` | `VIDEO_SPARSE_ATTN_H3` or `VIDEO_SPARSE_ATTN` |
| `vsa_tile_size`, `vsa_sparsity` | Tile size `64`; finite sparsity in `[0, 1)` |
| `sequence_parallel_size` | Positive integer recording recipe SP size; physical placement remains controlled by workers |

```bash
export H3_MODEL=/mnt/lustre/vlm-wlsaidhi/fastvideo/exports/minimax_h3_pdd_v23_step1600_fp32_grid32/inference/checkpoint-1600
export UNISERVE_H3_CONTRACT=/absolute/path/to/operator-contract.json
python -m uniserve_worker.bootstrap.inspect_model --model "$H3_MODEL"
```

An explicit sidecar must equal an embedded manifest when both exist; it cannot override a release. For `uniserve-deploy/serve-fasth3.sbatch`, export `H3_MODEL` and optionally `H3_CONTRACT` (the script exports it as `UNISERVE_H3_CONTRACT`). Record both the checkpoint and sidecar in the serving run before submission. Use `UNISERVE_ROOT` to select the installed UniServe checkout. H3 VSA uses segment-pure prefix tiles, dense prefix queries, always-visible prefix keys, video-only top-k selection, and the trained gated pooled-attention branch. Padding tiles are excluded from logical attention and compression.

### Dense base checkpoint

The top-level `MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da` root resolves a dense T2VA recipe with 50 FP32 scheduler grid points, 49 transformer forwards, and video/audio shifts 12/3. It requires revision receipts for the consumed files (or the corresponding pinned Hugging Face snapshot layout), complete indexed shards, and no distilled manifest. The base `quality` preset uses FP32 video VAE execution and is the default for this root. An explicitly selected missing sidecar is an error; it never falls through to base resolution. Base contract and schedule integration have CPU coverage, not end-to-end GPU validation in this worktree. This root currently loads `transformer`, not `transformer_ref`.

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

The request accepts `model`, `prompt`, `seconds`, `seed`, optional `steps`, and `references`. `seconds` defaults to 5 and `seed` defaults to 0. `steps` counts scheduler grid points, including the terminal point, and must match the checkpoint's configured count; omission uses that count. `references` defaults to an empty array. Nonempty references are not executable and are rejected as unsupported conditioning; they are never silently treated as T2VA.

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

- The checkpoint contract selects denoiser forward count, schedule and attention settings. The pinned four-step release preserves `[1, 0.75, 0.5, 0.25, 0]` and video/audio sigma shifts `12/3`. The eight-step release uses integer ladder `[999, 874, 749, 624, 500, 375, 250, 125] / 1000`, terminal zero, shifts `10/3`, guidance 1, and eight forwards.
- VSA sparse attention uses tile size 64 and manifest sparsity (0.9 for the pinned four-step release, 0.8 for the eight-step H3 VSA release). The shared provider selects the installed implementation from the execution device; SM100 uses the native SM100a kernel and supports incremental row production. Dense attention selects a compatible provider from each request’s dtype, head layout and mask. GB200 has end-to-end validation; other hardware requires its own validation before performance claims.
- Fixed 1344×768 output, 24 fps, and stereo 32-kHz audio.
- Text-only conditioning. Nonempty image/video/audio reference bundles, LoRA, variable resolution, guidance changes, and changes to the checkpoint's configured grid-point count are rejected. Parsing an ordered reference descriptor does not enable Ref2VA serving.

## Quick fixes

| Error | Action |
| --- | --- |
| `_uniserve_ipc` import or protocol error | Run `uv sync --locked --python /usr/bin/python3.12 --extra h3` again and use the resulting `uniserve` executable. |
| CUDA or sparse-attention compile error | Check CUDA 13 `nvcc`, `CUDA_HOME`, SM100 hardware, C++ build tools, and writable compiler caches. |
| Missing audio VAE or codec | Restore the locked environment with `uv sync`; do not mix in older Diffusers or PyAV packages. |
| Checkpoint validation error | Use the complete pinned model root and revision shown above. |
| GPU out of memory | Reduce resident capacity or choose a sharded `--workers` layout; sequence parallelism alone replicates denoiser weights. |
| MP4 contains an error body | Use `--fail-with-body`, inspect the HTTP status, and confirm the model name and `/v1/capabilities` limits. |
