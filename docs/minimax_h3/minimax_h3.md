# Serving guide

Install UniServe as in the [README](../../README.md#install).

## Checkpoints

| Checkpoint | Tasks | Resolutions | Steps |
| --- | --- | --- | --- |
| [`FastVideo/FastVideo-FastH3-8-Step-V2`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2), [`-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4) | `t2va`, `fl2va` | 768p, 480p | 8 |
| [`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree), [`-NVFP4`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4) | `t2va`, `fl2va` | 768p, 480p | 4 |
| [`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1) (FastH3 Omni) | `ref2va` | 768p, 480p | 8 |

`t2va` is text to video, `fl2va` text and first or last keyframe to video, and `ref2va` reference to video; every task returns an H.264 MP4 at 24 fps with stereo 32-kHz AAC audio. The NVFP4 checkpoints need Blackwell. FastH3 Omni ships only its denoiser: UniServe reads the text encoder, tokenizer and VAEs from the Hugging Face cache at the MiniMax-H3 revision its `fastvideo_inference.json` pins, and downloads what the cache lacks. A server loads one checkpoint, so it serves either `t2va` and `fl2va` or `ref2va`.

UniServe also serves the MiniMax-H3 base release, at 768p and 480p with dense attention, on its released uniform grid of 50 sigma points (49 network evaluations). A copy of the release whose DiT was merged with a step-distillation LoRA, such as larryvrh's MiniMax-H3 Turbo, states its shorter grid in a `schedule.json` at the checkpoint root: `{"points": 9}` evaluates the network 8 times. Only the point count changes; the shifts stay the schedulers'. A FastH3 checkpoint states its schedule in `fastvideo_inference.json` and refuses the file. `GET /v1/capabilities` reports the point count as `schedule.num_inference_steps`. The release was published with the 768p canvas rule alone; its 480p canvases are a FastH3 export's buckets (below), so a request gets the same size from either release, and a deployment prepares them with `--video-resolutions 768p,480p`.

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

### Validated settings

These are the settings the [evaluation](evaluation.md) measured. GB200 deployments ran with `NCCL_NVLS_ENABLE=0`, and the FastH3 H200 deployments with `NCCL_CUMEM_HOST_ENABLE=1`. On a PCIe host such as RTX PRO 6000, leave `NCCL_P2P_LEVEL` at its default: `SYS` cuts eight-GPU all-to-all bandwidth from 24.5 to 1.1 GB/s.

FastH3 (`t2va`, `fl2va`, 768p):

| Hardware | Goal | Deployment | Options |
| --- | --- | --- | --- |
| 4 x GB200 | Latency and throughput | `ulysses4.json` | `--max-running-requests 2` |
| 8 x GB200, two hosts | Latency | `ulysses8-two-node.json` | `--max-running-requests 2 --graph-policy off` |
| 8 x GB200, two hosts | Throughput | `ulysses4x2-two-node.json` | `--max-running-requests 4` |
| 4 x H200 | Latency, up to 5 s | `ulysses4.json` | `--max-running-requests 2 --mem-fraction-static 0.92 --graph-policy full --max-video-seconds 5 --max-model-len 1024 --video-text-capacities 1024` |
| 8 x H200 | Latency | `ulysses8.json` | `--max-running-requests 2 --mem-fraction-static 0.92 --graph-policy full --max-model-len 10240 --video-text-capacities 1024` |
| 8 x H200 | Throughput, up to 5 s | `dp8-text-tp8.json` | `--max-running-requests 8 --mem-fraction-static 0.92 --graph-policy full --max-video-seconds 5 --max-model-len 1024 --video-text-capacities 1024` |
| 8 x RTX PRO 6000 | Latency | `ulysses8.json` | `--max-running-requests 2 --mem-fraction-static 0.92 --graph-policy off --video-text-capacities 1024,10240` |
| 8 x RTX PRO 6000 | Throughput | `ulysses4x2.json` | `--max-running-requests 4 --mem-fraction-static 0.92 --graph-policy off --video-text-capacities 1024,10240` |

FastH3 Omni (`ref2va`, up to 15 s). Every row also takes `--max-running-requests 2 --max-model-len 24576 --video-text-capacities 4096,7168,8192,9216,10240,12288,15360 --max-condition-rows 56192`, except where its options restate one:

| Hardware | Resolutions | Deployment | Options |
| --- | --- | --- | --- |
| 1 x GB200 | 768p, 480p | `ulysses1-reference.json` | `--video-resolutions 768p,480p --mem-fraction-static 0.96 --max-model-len 15360 --max-condition-rows 45056`, and `--graph-policy off` for 768p requests of 15 s, or of 10 s with a reference video |
| 2 x GB200 | 768p, 480p | `ulysses2-reference.json` | `--video-resolutions 768p,480p --mem-fraction-static 0.92` |
| 4 x GB200 | 768p, 480p | `ulysses4-reference.json` | `--video-resolutions 768p,480p` |
| 8 x GB200, two hosts | 768p, 480p | `ulysses8-reference-two-node.json` | `--video-resolutions 768p,480p` |
| 8 x H200 | 768p | `ulysses8-reference.json` | `--mem-fraction-static 0.92` |
| 8 x RTX PRO 6000 | 768p | `ulysses8-reference.json` | `--mem-fraction-static 0.92 --graph-policy off` |
| 4 x GB200, enlarged references, up to 5 s | 768p | `ulysses4-reference.json` | `--mm-processor-kwargs '{"enlarge_references":true}' --max-video-seconds 5 --max-model-len 8192 --video-text-capacities 8192 --max-condition-rows 49152` |
| 8 x GB200, two hosts, enlarged references | 768p | `ulysses8-reference-two-node.json` | `--mm-processor-kwargs '{"enlarge_references":true}' --max-model-len 48128 --video-text-capacities 4096,7168,8192,10240,14336,15360,22528,37888 --max-condition-rows 79104` |
| 8 x H200, enlarged references | 768p | `ulysses8-reference.json` | `--mm-processor-kwargs '{"enlarge_references":true}' --mem-fraction-static 0.92 --max-model-len 48128 --video-text-capacities 4096,7168,8192,10240,14336,15360,22528,37888 --max-condition-rows 79104` |

### Two hosts

The two-host files name their hosts `rank-0` and `rank-1`. Start the server on `rank-0` with `--host-identity rank-0`; it logs `awaiting a launcher ... address="0.0.0.0:<port>"`. Then, on `rank-1`, with the same environment, checkpoint path and build:

```bash
NCCL_NVLS_ENABLE=0 uniserve-host --head <rank-0 address>:<port> --host-identity rank-1
```

## Requests

`POST /v1/videos/sync` returns the MP4. The body is JSON, or `multipart/form-data` with `conditions` and `target` as JSON text:

```json
{
  "model": "FastH3-Omni",
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
| `model` | The served model name. |
| `task` | `t2va`, `fl2va` or `ref2va`, one the served checkpoint serves. |
| `prompt` | Plain text. A `ref2va` prompt names its references in request order as `<Picture n>`, `<Video n>` and `<Audio n>`. |
| `conditions` | `t2va` takes none; `fl2va` one or two image keyframes (`frame_index` 0 or -1); `ref2va` up to 9 images, 3 videos and 3 audio tracks, 12 in all, and one continuation clip besides ([below](#continuing-a-video)). Each has `type` (`image`, `video`, `video_audio` or `audio`), `uri` (`data:`, `http(s)://` or `file://`), `role` (`keyframe`, `reference` or `continuation`), and optionally `frame_index` or `start_time_seconds`. |
| `target` | `short_edge` (768 or 480), `aspect_ratio` (below, or `auto`: the keyframe's aspect for `fl2va`, 16:9 otherwise) and `duration_seconds` (4 to the served maximum). |
| `seed` | Default 42. |

| `short_edge` | `21:9` | `16:9` | `4:3` | `1:1` | `3:4` | `9:16` |
| --- | --- | --- | --- | --- | --- | --- |
| 768 | 1536×672 | 1344×768 | 1024×768 | 768×768 | 768×1024 | 768×1344 |
| 480 | 992×416 | 832×480 | 640×480 | 480×480 | 480×640 | 480×832 |

A video has `duration_seconds × 24` frames, rounded up to the next count of the form 17n + 5: 5 seconds give 124 frames and 15 give 362. Any other field is refused.

### Continuing a video

A `ref2va` request can continue an earlier video. Its `conditions` then hold one `{"type": "video_audio", "role": "continuation", "uri": ...}` (or `"type": "video"`): the earlier video's last 5, 22 or 39 frames and their sound, as a clip of its own. The generated video starts by regenerating the clip's frames in place, and the response holds only the frames after them and the sound from the same instant, so the caller appends it to the earlier video. Of 5, 22 and 39 frames, 22 continue an earlier video most smoothly.

```json
{
  "model": "MiniMax-H3",
  "task": "ref2va",
  "prompt": "She keeps walking toward the camera and ...",
  "conditions": [
    {"type": "video_audio", "uri": "file:///srv/media/tail.mp4", "role": "continuation"}
  ],
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10},
  "seed": 0
}
```

| Rule | Detail |
| --- | --- |
| Clip | Exactly 5, 22 or 39 frames once resampled to 24 fps, on the target canvas (1344×768 above). It is not cropped or scaled, and it is not presented to the text encoder. A `video_audio` clip needs sound within its frames; a `video` clip conditions with its sound when it has any, otherwise with its frames alone. A soundtrack is cut to the clip's duration. `GET /v1/capabilities` lists the frame counts as `task_conditions.ref2va.continuation_frames`. |
| Count | At most one clip a request. It takes none of the reference slots and suffices as the request's only visual condition; images, videos and audio may still follow it as references. A continuation takes no keyframes, `frame_index` or `start_time_seconds`. |
| Duration | `target.duration_seconds` is required and counts the new frames only. The video generates the clip's frames and the new ones, rounded up to `17n + 5`: 22 frames and 5 seconds generate 158 and return 136. Both together may not exceed the longest video served (362 frames at 15 seconds), so a 22-frame clip allows 14.17 seconds, a 5-frame one 14.875 and a 39-frame one 13.46; a longer duration is refused, not shortened. |
| Capacity | The clip's rows count toward `--max-condition-rows`: a 22-frame clip with sound adds 7,130 rows at 1344×768 and 2,804 at 832×480. |
| Prompt | Write it from the clip's actual end state. When its opening contradicts the clip, the model tends to cut to a new shot right after it. |

The clip is placed as FastVideo's `continuation_video` places it, the reference context of [ComfyUI-H3-Continuum](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum): after the references, it advances the rotary clock as a video reference with its soundtrack does, and its latent frames and audio latents then take the coordinates of the generated video's first latent frames and audio latents. Its frames are anchored at the visual-condition timestep and its sound at the audio-reference one, as a reference's are, and FastH3 Omni's sparse attention keeps its tiles dense. Every continuation drifts a little toward higher saturation and sharpness, so a chain of fewer, longer segments (two of 15 seconds rather than six of 5) holds up better.

`POST /v1/videos` creates an asynchronous job instead: `GET /v1/videos/{id}` reports its state, `GET /v1/videos/{id}/content` returns the MP4 and `DELETE /v1/videos/{id}` cancels it. `GET /v1/capabilities` reports the served tasks, canvases, durations and capacities.

## Options

| Option | Meaning |
| --- | --- |
| `--max-video-seconds` | Longest admitted duration, 4 to 15; 15 by default. |
| `--video-resolutions`, `--video-aspect-ratios` | Canvases startup prepares; `768p` and `16:9,9:16` by default. Other canvases are refused. |
| `--max-model-len` | Longest prompt in tokens, reference tokens included; 16384 by default. |
| `--video-text-capacities` | Prompt capacities startup prepares; each prompt runs in the smallest that holds it. |
| `--max-condition-rows` | Denoiser rows one request's conditions may occupy. A request beyond it is refused with `invalid_request`, naming the rows it needs. With the validated FastH3 Omni settings, eight GB200 serve nine square images with three 2-second clips at 15 seconds. |
| `--max-running-requests` | Concurrently resident requests. |
| `--mm-processor-kwargs` | Reference sizing, as JSON: `reference_image_short_edge` (at most 2048, the default) and `enlarge_references` (`false` by default: a reference is only ever scaled down). |
| `--media-directory`, `--remote-media` | Where `file://` references resolve (refused without it), and whether `http(s)://` references are fetched (on by default). |
| `--mem-fraction-static` | Device memory each worker may hold; 0.70 by default. |
| `--graph-policy` | `auto` (default) captures CUDA graphs at startup, `full` refuses to start without them, and `off` runs eagerly with less memory. |
| `--quantization-config` | Precision of a BF16 checkpoint, below. |

Startup prepares every admitted request shape, and `/health` reports ready when it is done; fewer canvases, capacities and seconds start faster and use less memory. A deployment whose prepared shapes do not fit fails at startup and logs what each component holds.

## Precision

| `--quantization-config` mode | Denoiser MLP | Text encoder | Video VAE |
| --- | --- | --- | --- |
| `quality` (default) | BF16 | BF16 | FP16 |
| `balanced` | BF16 | BF16 | NVFP4 |
| `performance` | FP8 | BF16 | NVFP4 |
| `maximum` | NVFP4 | FP8 | NVFP4 |

Attention stays BF16, and NVFP4 needs Blackwell. `components` overrides parts of a mode, as in `{"mode":"performance","components":{"video_vae":"bf16"}}`. The NVFP4 checkpoints carry their own precision and take no `--quantization-config`. Modes other than `quality` change the generated media.

## Build and run with Docker

```bash
docker build -f docs/minimax_h3/Dockerfile -t uniserve .
docker run --rm --gpus all --pid=host --shm-size=4g --ulimit memlock=-1 -p 8000:8000 \
  -v "$PWD/models/FastH3:/models/FastH3:ro" \
  uniserve serve /models/FastH3 --workers configs/minimax_h3/ulysses4.json \
    --served-model-name FastH3 --host 0.0.0.0 --max-running-requests 2
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
| `server_error` naming `cannot run ffprobe` | Run `scripts/install-ffmpeg.sh`, or name FFmpeg with `--ffprobe` and `--ffmpeg`. |
| FastH3 Omni cannot find a base file | Let the server download the pinned MiniMax-H3 revision, or download it into the Hugging Face cache. |
