# FastH3 and MiniMax-H3 guide

UniServe serves the MiniMax-H3 video-and-audio models: the FastH3 DMD students for text-to-video (`t2va`), FastH3 OmniRef students for reference-conditioned video (`ref2va`), and the base MiniMax-H3 checkpoint for `t2va`, keyframe-conditioned video (`fl2va`) and `ref2va`. Every response is an H.264/AAC MP4 at 24 fps with stereo 32-kHz audio.

## Requirements

- Linux, CUDA 13, Python 3.12, a stable Rust toolchain, and NVIDIA Hopper or Blackwell GPUs. Four local GPUs are the standard setup. FastH3 is validated on H200, GB200 and RTX PRO 6000 Blackwell Server Edition; the base checkpoint and OmniRef on four GB200. OmniRef needs a data-center Blackwell (SM100) GPU, and the server refuses it elsewhere at startup.
- FFmpeg for keyframe and reference requests: the server probes condition media with `ffprobe` and the worker decodes reference videos with `ffmpeg`, both from `PATH` unless `--ffprobe` and `--ffmpeg` name them.
- Shared storage and pinned-storage access for worker IPC, as the container command below provisions.

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

The sync installs the locked GPU providers (FlashInfer, FlashAttention-4, the CuTe and CUTLASS DSLs) and builds the `uniserve` and `uniserve-host` binaries, the `_uniserve_ipc` extension and UniServe's native kernels. The kernels compile for the visible GPUs, or for the architectures `TORCH_CUDA_ARCH_LIST` names, with `MAX_JOBS` parallel jobs; a later sync recompiles them when their sources change.

## Checkpoints

| Checkpoint | Tasks | Resolutions | Denoiser forwards |
| --- | --- | --- | --- |
| [`FastVideo/FastVideo-FastH3-8-Step-V2`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2), [`-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4) | `t2va` | 768p, 480p | 8 |
| [`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree), [`-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4) | `t2va` | 768p, 480p | 4 |
| FastH3 OmniRef component exports | `ref2va` | 768p | The export's schedule |
| [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) `denoiser` | `t2va`, `fl2va` | 768p | 49 |
| [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) `reference_denoiser` | `ref2va` | 768p | 49 |

The NVFP4 repositories are ModelOpt PTQ checkpoints and need Blackwell. A checkpoint's schedule comes from its `fastvideo_inference.json` and schedulers, or for the base checkpoint 50 sigma points with video shift 12 and audio shift 3. A server places one denoiser, so a base deployment serves either `t2va` and `fl2va` or `ref2va`.

An OmniRef export holds only its reference denoiser and schedulers. Its `fastvideo_inference.json` pins the base revision (`base_model_revision`) whose text encoder, tokenizer, processor and VAEs it uses; the server reads them from the Hugging Face cache at that revision, or from the local copy `--base-model` names, which must hold every file at the pinned revision.

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
hf download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir "$H3_MODEL"

export H3_ROOT=/workspace/models/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --local-dir "$H3_ROOT"

# The base components an OmniRef export pins.
hf download MiniMaxAI/MiniMax-H3 --revision <base_model_revision> --local-dir "$OMNIREF_BASE" \
  --include 'text_encoder/*' --include 'tokenizer/*' --include 'processor/*' \
  --include 'vae/*' --include 'audio_vae/*'
```

## Start the server

FastH3 text-to-video:

```bash
uniserve serve "$H3_MODEL" \
  --workers configs/minimax_h3/fasth3-ulysses4.json \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 2
```

Base text-to-video and keyframe requests:

```bash
uniserve serve "$H3_ROOT" \
  --workers configs/minimax_h3/ulysses4.json \
  --served-model-name MiniMax-H3 \
  --host 0.0.0.0 \
  --max-video-seconds 8 --max-model-len 2048 --video-text-capacities 2048
```

Reference requests, with the base reference denoiser or an OmniRef export:

```bash
uniserve serve "$H3_ROOT" \
  --workers configs/minimax_h3/ulysses4-reference.json \
  --served-model-name MiniMax-H3 \
  --host 0.0.0.0 \
  --max-video-seconds 5 --max-model-len 8192 --video-text-capacities 8192 \
  --max-condition-rows 40960 --media-directory /srv/media

uniserve serve "$OMNIREF_EXPORT" --base-model "$OMNIREF_BASE" \
  --workers configs/minimax_h3/ulysses4-reference.json \
  --served-model-name MiniMax-H3-OmniRef \
  --host 0.0.0.0 \
  --max-video-seconds 5 --max-model-len 8192 --video-text-capacities 8192 \
  --max-condition-rows 49152 --media-directory /srv/media
```

A deployment file places each component on devices. These four-GPU files run four-way Ulysses denoising, TP4 text encoding and one video and audio decoding unit per rank in a `model` worker, and a CPU `host` worker that holds the H.264 encoders and the muxer, plus the media reader that decodes condition media in the `minimax_h3` files. Check a running server with:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/capabilities | python -m json.tool
```

`/v1/capabilities` reports the served tasks and condition rules, canvases, durations, schedule, prompt and condition capacities, and media policy.

### Deployments

The deployment files are in `configs/minimax_h3/`; the `fasth3-` files place FastH3's text-to-video components, the others add the condition encoders and media reader the base checkpoint and OmniRef use.

| File | GPUs | Denoiser | Text encoder |
| --- | --- | --- | --- |
| `fasth3-ulysses4.json` | Four on one host | Ulysses4 | TP4 |
| `fasth3-ulysses8.json` | Eight on one host | Ulysses8 | TP8 |
| `fasth3-ulysses8-two-node.json`, `ulysses8-two-node.json` | Four on each of two hosts | Ulysses8 | TP8 |
| `fasth3-ulysses4x2.json` | Eight on one host | Two Ulysses4 replicas | TP4 per replica |
| `fasth3-ulysses4x2-two-node.json` | Four on each of two hosts | One Ulysses4 replica per host | TP4 per replica |
| `fasth3-dp8-text-tp8.json` | Eight on one host | Eight one-GPU replicas | One shared TP8 |
| `fasth3-gather8.json` | Eight on one host | Eight-way all-gather | TP8 |

One replica across every GPU gives the lowest latency; replicas serve more requests at once when the offered concurrency keeps them busy. The validated FastH3 settings:

| Hardware | Goal | Deployment | Options |
| --- | --- | --- | --- |
| 4 x GB200 | Latency and throughput | `fasth3-ulysses4.json` | `--max-running-requests 2` |
| 8 x GB200, two hosts | Latency | `fasth3-ulysses8-two-node.json` | `--max-running-requests 2 --graph-policy off` |
| 8 x GB200, two hosts | Throughput | `fasth3-ulysses4x2-two-node.json` | `--max-running-requests 4` |
| 8 x RTX PRO 6000 | Latency | `fasth3-ulysses8.json` | `--max-running-requests 2 --graph-policy off --mem-fraction-static 0.92 --video-text-capacities 1024,10240,16384` |
| 8 x RTX PRO 6000 | Throughput | `fasth3-ulysses4x2.json` | `--max-running-requests 4 --graph-policy off --mem-fraction-static 0.92 --video-text-capacities 1024,10240,16384` |
| 4 x H200 | Latency and throughput | `fasth3-ulysses4.json` | `--max-running-requests 2 --mem-fraction-static 0.92 --video-text-capacities 1024,16384` |
| 8 x H200 | Latency, up to 10240 prompt tokens | `fasth3-ulysses8.json` | `--max-running-requests 2 --mem-fraction-static 0.92 --max-model-len 10240 --video-text-capacities 1024,10240` |
| 8 x H200 | Throughput, up to 5 s and 1024 prompt tokens | `fasth3-dp8-text-tp8.json` | `--max-running-requests 8 --max-video-seconds 5 --max-model-len 1024` |

The GB200 measurements ran with `NCCL_NVLS_ENABLE=0`.

### Two hosts

The two-host files name their hosts `rank-0` and `rank-1`. Start the head on `rank-0` with `--host-identity rank-0`; it logs `awaiting a launcher for each host this instance does not run on address="0.0.0.0:<port>"`. Then start the launcher on the other host with the same environment, checkpoint path and build:

```bash
NCCL_NVLS_ENABLE=0 uniserve-host --head <rank-0 address>:<port> --host-identity rank-1
```

## Startup and capacity

| Option | Meaning |
| --- | --- |
| `--max-video-seconds` | Longest admitted duration, 4 to 15 seconds; 15 by default. |
| `--video-resolutions`, `--video-aspect-ratios` | The canvases startup prepares: each resolution with each aspect ratio, `768p` and `16:9,9:16` by default. FastH3 serves `768p` and `480p`, the others `768p`. A request for any other canvas is refused. |
| `--max-model-len` | Longest prompt presentation in tokens, vision tokens included; 16384 by default for FastH3. The official keyframe request presents 1935 tokens and the official video-plus-audio request 6913. |
| `--video-text-capacities` | Prompt capacities startup prepares; each prompt runs in the smallest that holds it. By default 1024, then steps of 2048 up to `--max-model-len`. |
| `--max-running-requests` | Concurrently resident requests, up to the workers' request slots. |
| `--max-condition-rows` | Denoiser rows one request's conditions may occupy; the default 2048 holds two keyframes. Reference deployments set it, as below. |
| `--media-directory` | Directory under which `file://` condition URIs resolve; `file://` is refused without it. |
| `--remote-media` | Whether `http(s)://` condition URIs are fetched; on by default. |
| `--max-request-bytes` | Largest request body and fetched media in total; 256 MiB by default. |
| `--mem-fraction-static` | Device storage each worker process may hold; 0.70 by default. |
| `--graph-policy` | `auto` (default) captures CUDA graphs at startup; `off` runs eagerly. |

Startup prepares, and with graphs captures, every layout an admitted request reaches: each frame count up to `--max-video-seconds` at each canvas and text capacity. Fewer canvases, capacities and seconds start faster and hold less storage. Until then the log reports `worker still busy during Worker startup`. On four GB200 with `ulysses4.json` and the defaults, FastH3 is ready in about 17 minutes from warm kernel caches and holds about 105 GiB per model worker; the first startup also compiles kernels. Serving all 12 FastH3 canvases needs fewer text capacities, such as `--video-text-capacities 1024,10240,16384`. 768p 21:9 needs more request storage than the other canvases. A deployment that does not fit fails before it reports ready and logs what each component holds.

A base denoiser counts one condition row per token: a five-second reference video with its soundtrack occupies about 38,000 rows, a 16:9 reference image about 7,300. OmniRef packs each condition into whole 128-row tiles: an image or audio track takes `ceil(rows / 128)` tiles, and a video's token grid of latent frames × height/32 × width/32 takes `ceil(frames / 4) × ceil(height / 32 / 4) × ceil(width / 32 / 8)`. The official video-plus-audio request therefore occupies 38,124 rows on the base reference denoiser and 47,104 on OmniRef. OmniRef also bounds a request's packed sequence (prompt, conditions and generated rows) at 131,072 rows. The prompt carries every visual reference's vision tokens: a reference image, resized to a 2048-pixel short edge, is (height / 32) × (width / 32) prompt tokens and as many condition rows, 7,296 each at 16:9 and 4,096 each when square, and a five-second 16:9 reference video is about 6,100 prompt tokens. With the 37,710 rows of a five-second 16:9 target, three 16:9 images, nine near-square images, a reference video with an image and an audio track, or three reference clips of about two seconds fit; nine 16:9 images, three five-second videos, and nine images with three videos do not. One deployment serves all of these with `--max-model-len 38912 --video-text-capacities 8192,16384,24576,38912 --max-condition-rows 54400`: startup sizes the denoiser for the largest text capacity and the condition capacity together, 131,022 rows here.

The latent encoder's ranks (`temporal_units`) encode a request's condition units in rounds. A reference image is split into bands of whole 32-pixel patch rows that fill one round, the round's units divided among the request's reference images, so a lone image's tiles are encoded on every rank; more images than one round holds encode whole, and when only images remain for a final partial round that can share it evenly, they split into the bands that fill it (nine images on four ranks: two rounds of whole images, then the ninth in four bands). The bands assemble the single-rank encoding exactly. Keyframes and reference-video windows remain one unit each. With graphs, startup captures the encoding of one 256-pixel still-frame tile, which keyframe and reference-image tiles replay; reference-video windows encode eagerly.

## Build and run with Docker

```bash
docker build -f docs/minimax_h3/Dockerfile -t uniserve-h3 .

docker run --rm --gpus all --shm-size=4g --ulimit memlock=-1 -p 8000:8000 \
  -v "$H3_MODEL:/models/h3:ro" \
  uniserve-h3 serve /models/h3 \
    --workers configs/minimax_h3/fasth3-ulysses4.json \
    --served-model-name FastH3 \
    --host 0.0.0.0 \
    --max-running-requests 2
```

The image compiles the native kernels for Hopper, data-center Blackwell and RTX PRO 6000 Blackwell. Mount `/root/.cache` and `/root/.triton` to keep the Triton and FlashInfer kernels the first startup compiles.

## Generate a video

`POST /v1/videos/sync` returns the MP4; the body is JSON, or `multipart/form-data` with `conditions` and `target` as JSON text.

```bash
curl --fail-with-body --max-time 600 http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","task":"t2va","prompt":"A clear stream flows through a green forest while birds sing.","target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5},"seed":1000}' \
  --output forest.mp4
```

A keyframe request anchors the first (`frame_index` 0) or last (`-1`) frame to an image; a reference request lists its references in order, and the prompt names them `<Picture 1>`, `<Video 1>` and `<Audio 1>`:

```json
{
  "model": "MiniMax-H3",
  "task": "ref2va",
  "prompt": "subject_definitions:\n<Subject 1> is the man in <Picture 1>.\n<Audio 1> is the voice timbre reference for <Subject 1>'s voice. ...",
  "conditions": [
    {"type": "image", "uri": "file:///srv/media/subject.png", "role": "reference"},
    {"type": "audio", "uri": "file:///srv/media/voice.mp3", "role": "reference"}
  ],
  "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5}
}
```

| `target.short_edge` | `21:9` | `16:9` | `4:3` | `1:1` | `3:4` | `9:16` |
| --- | --- | --- | --- | --- | --- | --- |
| 768 | 1536×672 | 1344×768 | 1024×768 | 768×768 | 768×1024 | 768×1344 |
| 480 (FastH3) | 992×416 | 832×480 | 640×480 | 480×480 | 480×640 | 480×832 |

| Field | Rule |
| --- | --- |
| `model`, `prompt` | Required: the served model name and plain text without the model's vision tokens. |
| `task` | Required: `t2va`, `fl2va` or `ref2va`, one the served denoiser serves. |
| `conditions` | In request order. `t2va` takes none; `fl2va` one or two image keyframes; `ref2va` up to 9 images, 3 videos and 3 audio tracks, 12 in total, plus keyframes on the base reference denoiser. Each has `type` (`image`, `video`, `video_audio` for a video that must have a soundtrack, or `audio`), `uri` (`data:`, `http(s)://` or `file://`), `role` (`keyframe` or `reference`), and optionally `frame_index` or `start_time_seconds`, an offset into a video reference. |
| `target` | Required: `short_edge`, `aspect_ratio` (a served ratio, or `auto`: the keyframe's aspect for `fl2va`, 16:9 otherwise) and `duration_seconds` (4 to the served maximum; a reference request whose only audio-bearing reference sets the duration may omit it). |
| `seed` | Default 42. |
| `num_inference_steps`, `flow_shift`, `audio_flow_shift` | Optional; only the served schedule `/v1/capabilities` reports. |
| `n`, `num_outputs_per_prompt`, `quality` | Optional; only 1, 1 and `lossless`. |
| `seconds`, `size`, `width`, `height` | Optional; only values that agree with the target. |

Any other field is refused, naming it. A video has `duration_seconds × 24` frames, rounded half to even and extended to the next count of the form 17n + 5: 4 seconds give 107 frames, 5 give 124, 8 give 192 and 15 give 362.

`POST /v1/videos` creates an asynchronous job instead; `GET /v1/videos/{id}` reports its state (`queued`, `in_progress`, `completed`, `failed`) and progress, `GET /v1/videos/{id}/content` returns the MP4, `GET /v1/videos` lists jobs and `DELETE` cancels one. Jobs live in the server process for one hour, within 1 GiB of retained MP4s and 128 job slots shared with synchronous requests; a full server answers 429 `video_job_capacity_exceeded`.

## Precision

A dense BF16 checkpoint selects its precision with `--quantization-config`:

| Mode | Denoiser MLP | Text encoder | Video VAE |
| --- | --- | --- | --- |
| `quality` (default) | BF16 | BF16 | FP16 |
| `balanced` | BF16 | BF16 | NVFP4 |
| `performance` | FP8 | BF16 | NVFP4 |
| `maximum` | NVFP4 | FP8 | NVFP4 |

Denoiser attention stays BF16. `components` overrides parts of a mode by the names `attention`, `mlp`, `text_encoder` and `video_vae`, as in `--quantization-config '{"mode":"performance","components":{"video_vae":"bf16"}}'`. NVFP4 needs Blackwell; on Hopper, use `quality`, optionally with `{"components":{"mlp":"fp8"}}`. The packed NVFP4 checkpoints carry their own precision and take no `--quantization-config`. The presets other than `quality` change the generated media; establish quality for the intended workload before choosing one.

## Further guides

- [Python library](library.md): loading and generating with the same numerical modules.
- [Evaluation](evaluation.md): the published measurements and fixed workloads.
- [NVIDIA Dynamo](dynamo.md): FastH3 behind Dynamo's `/v1/videos`.

## Troubleshooting

| Symptom | Remedy |
| --- | --- |
| `_uniserve_ipc` import or protocol error | Run the `uv sync` above again and use the resulting `uniserve`. |
| CUDA or sparse-attention compile error during `uv sync` | Check CUDA 13 `nvcc`, `CUDA_HOME`, the C++ build tools, and a `TORCH_CUDA_ARCH_LIST` naming the target GPUs when none is visible. |
| `unsupported FastH3 checkpoint` | Use a complete checkpoint from the table above; the message names the expected model and revision. |
| `checkpoint format 'modelopt_nvfp4' owns its numerical configuration` | Drop `--quantization-config` for a packed NVFP4 checkpoint. |
| `nvfp4 conversion requires an SM100-class CUDA device` | Use `quality` or FP8 overrides on Hopper. |
| `NVML reports no device storage for process` | Run the container in the host's process ID namespace (`--pid=host`). |
| Startup refuses a static storage grant, or GPU out of memory | Raise `--mem-fraction-static`, serve fewer canvases, capacities or seconds, or shard the denoiser further. |
| `invalid_request` naming the condition rows and `--max-condition-rows` | Serve with a larger `--max-condition-rows`. |
| `invalid_request` naming the prompt length | Raise `--max-model-len` together with `--video-text-capacities`. |
| A refused `file://` condition | Set `--media-directory` to a directory that holds the path. |
| `server_error` naming `cannot run ffprobe` | Put FFmpeg on `PATH` or name it with `--ffprobe`. |
| An OmniRef start refusing a base file by name | Download the base at the revision the export pins. |
