# Serving guide

UniServe serves FastH3 text-to-video and keyframe requests (`t2va`, `fl2va`), FastH3 OmniRef reference requests (`ref2va`), and the MiniMax-H3 checkpoint they are distilled from. Every response is an H.264 MP4 at 24 fps with stereo 32-kHz AAC audio.

## Install

```bash
apt-get install -y build-essential python3.12-dev pkg-config cmake git curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
export CUDA_HOME=/usr/local/cuda MAX_JOBS=2
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$CUDA_HOME/bin:$PATH"
uv sync --locked --python 3.12 --extra gpu
source .venv/bin/activate
```

The sync builds the `uniserve` and `uniserve-host` binaries and UniServe's kernels for the visible GPUs, or for the architectures `TORCH_CUDA_ARCH_LIST` names. Keyframe and reference requests also need FFmpeg: `ffprobe` and `ffmpeg` from `PATH`, or the paths `--ffprobe` and `--ffmpeg` name.

## Checkpoints

| Checkpoint | Tasks | Resolutions | Forwards |
| --- | --- | --- | --- |
| [`FastVideo/FastVideo-FastH3-8-Step-V2`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2), [`-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4) | `t2va`, `fl2va` | 768p, 480p | 8 |
| [`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree), [`-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4) | `t2va`, `fl2va` | 768p, 480p | 4 |
| [`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1) (FastH3 OmniRef) | `ref2va` | 768p, 480p | 8 |
| [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) | `t2va`, `fl2va`; `ref2va` with its `transformer_ref` | 768p | 49 |

The NVFP4 checkpoints need Blackwell. FastH3 OmniRef holds only its denoiser; UniServe reads the text encoder, tokenizer and VAEs from the Hugging Face cache at the MiniMax-H3 revision its `fastvideo_inference.json` pins, and downloads what the cache lacks. A server places one denoiser, so it serves either `t2va` and `fl2va` or `ref2va`.

## Deployments

`--workers` takes a deployment file from `configs/minimax_h3/`, which places the components on GPUs and hosts:

| File | GPUs | Layout |
| --- | --- | --- |
| `ulysses4.json`, `ulysses8.json` | 4 or 8 on one host | One Ulysses replica, TP text encoder |
| `ulysses8-two-node.json` | 4 on each of two hosts | One Ulysses8 replica |
| `ulysses4x2.json` | 8 on one host | Two Ulysses4 replicas |
| `ulysses4x2-two-node.json` | 4 on each of two hosts | One Ulysses4 replica per host |
| `dp8-text-tp8.json`, `dp8-text-tp4x2.json` | 8 on one host | Eight one-GPU replicas, shared text encoder |
| `dp8-text-tp8-host-weights.json` | 8 on one host | As `dp8-text-tp8.json`, with the denoiser's projection weights in host memory (needs `--graph-policy off`) |
| `gather8.json` | 8 on one host | All-gather sequence parallel |
| `ulysses1-reference.json`, `ulysses2-reference.json`, `ulysses4-reference.json`, `ulysses8-reference.json` | 1 to 8 on one host | One reference replica |
| `ulysses8-reference-two-node.json` | 4 on each of two hosts | One Ulysses8 reference replica |

The `-reference` files serve `ref2va`; the others serve `t2va` and `fl2va`. One replica across every GPU gives the lowest latency; more replicas serve more requests at once.

Validated settings for FastH3 (`t2va`, `fl2va`):

| Hardware | Goal | Deployment | Options |
| --- | --- | --- | --- |
| 4 x GB200 | Latency and throughput | `ulysses4.json` | `--max-running-requests 2` |
| 8 x GB200, two hosts | Latency | `ulysses8-two-node.json` | `--max-running-requests 2 --graph-policy off` |
| 8 x GB200, two hosts | Throughput | `ulysses4x2-two-node.json` | `--max-running-requests 4` |
| 4 x H200 | Latency and throughput | `ulysses4.json` | `--max-running-requests 2 --mem-fraction-static 0.92 --video-text-capacities 1024` |
| 8 x H200 | Latency | `ulysses8.json` | `--max-running-requests 2 --mem-fraction-static 0.92 --max-model-len 10240 --video-text-capacities 1024` |
| 8 x H200 | Throughput, up to 5 s | `dp8-text-tp8.json` | `--max-running-requests 8 --max-video-seconds 5 --max-model-len 1024` |
| 8 x RTX PRO 6000 | Latency | `ulysses8.json` | `--max-running-requests 2 --graph-policy off --mem-fraction-static 0.92 --video-text-capacities 1024,10240` |
| 8 x RTX PRO 6000 | Throughput | `ulysses4x2.json` | `--max-running-requests 4 --graph-policy off --mem-fraction-static 0.92 --video-text-capacities 1024,10240` |

Validated settings for FastH3 OmniRef (`ref2va`, up to 15 s), each with `--max-running-requests 2 --max-model-len 24576 --video-text-capacities 4096,7168,8192,9216,10240,12288,15360 --max-condition-rows 56192` unless stated:

| Hardware | Deployment | Options |
| --- | --- | --- |
| 1 x GB200 | `ulysses1-reference.json` | `--video-resolutions 768p,480p --mem-fraction-static 0.96 --max-model-len 15360 --max-condition-rows 45056`; add `--graph-policy off` for 768p beyond 10 s or with a 10-second reference video |
| 2 x GB200 | `ulysses2-reference.json` | `--video-resolutions 768p,480p --mem-fraction-static 0.92` |
| 4 x GB200 | `ulysses4-reference.json` | `--video-resolutions 768p,480p` |
| 8 x GB200, two hosts | `ulysses8-reference-two-node.json` | `--video-resolutions 768p,480p --host-identity rank-0` |
| 8 x H200 | `ulysses8-reference.json` | `--mem-fraction-static 0.92` (768p) |
| 8 x RTX PRO 6000 | `ulysses8-reference.json` | `--graph-policy off --mem-fraction-static 0.92` (768p) |

Run GB200 deployments with `NCCL_NVLS_ENABLE=0`, as validated.

### Two hosts

The two-host files name their hosts `rank-0` and `rank-1`. Start the server on `rank-0` with `--host-identity rank-0`; it logs `awaiting a launcher ... address="0.0.0.0:<port>"`. Then, on `rank-1`, with the same environment, checkpoint path and build:

```bash
NCCL_NVLS_ENABLE=0 uniserve-host --head <rank-0 address>:<port> --host-identity rank-1
```

## Requests

`POST /v1/videos/sync` returns the MP4. The body is JSON, or `multipart/form-data` with `conditions` and `target` as JSON text:

```json
{
  "model": "FastH3-OmniRef",
  "task": "ref2va",
  "prompt": "<Subject 1> is the man in <Picture 1>. <Audio 1> is his voice. ...",
  "conditions": [
    {"type": "image", "uri": "file:///srv/media/subject.png", "role": "reference"},
    {"type": "audio", "uri": "file:///srv/media/voice.mp3", "role": "reference"}
  ],
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
  "seed": 0
}
```

| Field | Rule |
| --- | --- |
| `model`, `prompt` | The served model name, and plain text. A reference prompt names its references in request order as `<Picture n>`, `<Video n>` and `<Audio n>`. |
| `task` | `t2va`, `fl2va` or `ref2va`, one the served denoiser serves. |
| `conditions` | `t2va` takes none; `fl2va` one or two image keyframes (`frame_index` 0 or -1); `ref2va` up to 9 images, 3 videos and 3 audio tracks, 12 in all. Each has `type` (`image`, `video`, `video_audio` or `audio`), `uri` (`data:`, `http(s)://` or `file://`), `role` (`keyframe` or `reference`), and optionally `frame_index` or `start_time_seconds`. |
| `target` | `short_edge` (768, or 480 for FastH3 checkpoints), `aspect_ratio` (below, or `auto`: the keyframe's aspect for `fl2va`, 16:9 otherwise) and `duration_seconds` (4 to the served maximum). |
| `seed` | Default 42. |

| `short_edge` | `21:9` | `16:9` | `4:3` | `1:1` | `3:4` | `9:16` |
| --- | --- | --- | --- | --- | --- | --- |
| 768 | 1536×672 | 1344×768 | 1024×768 | 768×768 | 768×1024 | 768×1344 |
| 480 | 992×416 | 832×480 | 640×480 | 480×480 | 480×640 | 480×832 |

A video has `duration_seconds × 24` frames, extended to the next count of the form 17n + 5: 5 seconds give 124 frames and 15 give 362. Any other field is refused. `POST /v1/videos` creates an asynchronous job instead; `GET /v1/videos/{id}` reports its state, `GET /v1/videos/{id}/content` returns the MP4 and `DELETE /v1/videos/{id}` cancels it. `GET /v1/capabilities` reports the served tasks, canvases, durations and capacities.

## Options

| Option | Meaning |
| --- | --- |
| `--max-video-seconds` | Longest admitted duration, 4 to 15; 15 by default. |
| `--video-resolutions`, `--video-aspect-ratios` | Canvases startup prepares; `768p` and `16:9,9:16` by default. Other canvases are refused. |
| `--max-model-len` | Longest prompt in tokens, vision tokens included; 16384 by default. |
| `--video-text-capacities` | Prompt capacities startup prepares; each prompt runs in the smallest that holds it. |
| `--max-condition-rows` | Denoiser rows one request's conditions may occupy. A request beyond it is refused with `invalid_request`, naming the rows it needs. |
| `--max-running-requests` | Concurrently resident requests. |
| `--mm-processor-kwargs` | Reference sizing, as JSON: `reference_image_short_edge` (at most 2048, the default) and `enlarge_references` (`false` by default: a reference is only ever scaled down). |
| `--media-directory`, `--remote-media` | Where `file://` references resolve (refused without it), and whether `http(s)://` references are fetched (on by default). |
| `--mem-fraction-static` | Device memory each worker may hold; 0.70 by default. |
| `--graph-policy` | `auto` (default) captures CUDA graphs at startup; `off` runs eagerly and needs less memory. |
| `--quantization-config` | Precision of a BF16 checkpoint, below. |

Startup prepares every admitted request shape and logs `worker still busy during Worker startup` until it is ready; fewer canvases, capacities and seconds start faster and use less memory. A deployment whose prepared shapes do not fit fails at startup and logs what each component holds; a large reference request can still run out of device memory at run time, as long 768p requests do on one GB200 with graphs on.

A reference request packs its prompt, its condition rows and its generated rows into one sequence:

- A 16:9 768p target generates 37,710 rows at 5 seconds and 109,062 at 15.
- An image reference is (height / 32) × (width / 32) prompt tokens and as many condition rows, at its own size unless its short edge exceeds 2048 pixels.
- FastH3 OmniRef packs each condition into whole 128-row tiles.
- `--max-model-len` and `--max-condition-rows` bound what a deployment accepts. On eight GB200 the OmniRef settings above serve nine square images with three 2-second clips at 15 seconds.

## Precision

| `--quantization-config` mode | Denoiser MLP | Text encoder | Video VAE |
| --- | --- | --- | --- |
| `quality` (default) | BF16 | BF16 | FP16 |
| `balanced` | BF16 | BF16 | NVFP4 |
| `performance` | FP8 | BF16 | NVFP4 |
| `maximum` | NVFP4 | FP8 | NVFP4 |

Attention stays BF16, and NVFP4 needs Blackwell. `components` overrides parts of a mode, as in `{"mode":"performance","components":{"video_vae":"bf16"}}`. The NVFP4 checkpoints carry their own precision and take no `--quantization-config`. Modes other than `quality` change the generated media.

## Docker

```bash
docker build -f docs/minimax_h3/Dockerfile -t uniserve .
docker run --rm --gpus all --pid=host --shm-size=4g --ulimit memlock=-1 -p 8000:8000 \
  -v "$PWD/models/FastH3:/models/FastH3:ro" \
  uniserve serve /models/FastH3 --workers configs/minimax_h3/ulysses4.json \
    --served-model-name FastH3 --host 0.0.0.0
```

Mount `/root/.cache` and `/root/.triton` to keep the kernels the first start compiles.

## Troubleshooting

| Symptom | Remedy |
| --- | --- |
| `_uniserve_ipc` import or protocol error | Run `uv sync` again and use the resulting `uniserve`. |
| Compile error during `uv sync` | Check CUDA 13 `nvcc`, `CUDA_HOME`, the C++ tools, and `TORCH_CUDA_ARCH_LIST` when no GPU is visible. |
| `unsupported FastH3 checkpoint` | Use a checkpoint from the table above; the message names the field it cannot serve. |
| `nvfp4 conversion requires an SM100-class CUDA device` | Use `quality`, or FP8 overrides, on Hopper. |
| `NVML reports no device storage for process` | Run the container with `--pid=host`. |
| Startup refuses a memory grant, or out of memory | Raise `--mem-fraction-static`, serve fewer canvases, capacities or seconds, use `--graph-policy off`, or spread the denoiser over more GPUs. |
| `invalid_request` naming rows or the prompt length | Serve with a larger `--max-condition-rows` or `--max-model-len`, or send fewer or smaller references. |
| A refused `file://` reference | Set `--media-directory` to a directory that holds it. |
| `server_error` naming `cannot run ffprobe` | Put FFmpeg on `PATH` or name it with `--ffprobe`. |
| FastH3 OmniRef cannot find a base file | Let the server download the pinned MiniMax-H3 revision, or download it into the Hugging Face cache. |
