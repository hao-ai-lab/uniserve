# MiniMax-H3 guide

UniServe serves the MiniMax-H3 video-and-audio models: text-to-video (`t2va`), keyframe-conditioned video (`fl2va`) and reference-conditioned video (`ref2va`). Every response is an H.264/AAC MP4 at 24 fps with a stereo 32-kHz track. The FastH3 DMD students have their own [FastH3 cheat sheet](../fast_h3/fast_h3.md); they use the same request body and serve only `t2va`, at their 768p and 480p training buckets.

## Requirements

- Linux, CUDA 13, Python 3.12, a stable Rust toolchain, and NVIDIA Blackwell GPUs. Four GB200 GPUs on one host are the validated setup for every checkpoint in this guide. FastH3 OmniRef's sparse attention uses 128-row tiles, which have a data-center Blackwell (SM100) kernel only; on other devices the server refuses an OmniRef deployment at startup, before loading weights.
- The MiniMax-H3 diffusers root (`MiniMaxAI/MiniMax-H3`), or a FastH3 OmniRef component export together with the base revision it pins.
- FFmpeg for video and audio conditions: the server probes condition media with `ffprobe` and the worker decodes reference videos with `ffmpeg`, the executables on `PATH` unless `--ffprobe` and `--ffmpeg` name others.
- Shared storage and pinned-storage access for worker IPC, as for FastH3.

## Install

Install as the [FastH3 cheat sheet](../fast_h3/fast_h3.md#install) describes: `uv sync --locked --python /usr/bin/python3.12 --extra gpu` from the repository root. The lock includes `torchaudio`, which resamples condition audio; an environment created before it was added needs the same `uv sync` again.

## Checkpoints and denoisers

| Checkpoint | Denoising component | Tasks | Canvases | Schedule |
| --- | --- | --- | --- | --- |
| `MiniMaxAI/MiniMax-H3` (diffusers root) | `denoiser` (`transformer/`) | `t2va`, `fl2va` | The canvas rule's six named canvases | 50 points, flow shift 12, audio shift 3 |
| `MiniMaxAI/MiniMax-H3` (diffusers root) | `reference_denoiser` (`transformer_ref/`) | `ref2va` | The canvas rule's six named canvases | 50 points, flow shift 12, audio shift 3 |
| FastH3 OmniRef component export | `reference_denoiser` (`transformer_ref/`) | `ref2va` | The canvas rule's six named canvases | The export's parallel-decoding schedule |

A deployment places exactly one denoising component, so one server serves one task family. A diffusers root holds both base DiTs; the deployment file chooses which one a server runs. A component export holds only its denoiser, schedulers and inference contract; its `fastvideo_inference.json` pins the base revision (`base_model_revision`) whose text encoder, tokenizer, processor and VAEs it uses. The server reads those components from the Hugging Face cache at that revision, or from the local copy `--base-model` names; a local copy must hold every file at the pinned revision, as the `.cache/huggingface/download` records of `hf download --revision <revision> --local-dir` show, and the server refuses a file recorded at another revision by name.

```bash
export H3_ROOT=/workspace/models/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --local-dir "$H3_ROOT"
```

## Start the server

`$OMNIREF_EXPORT` and `$OMNIREF_BASE` below name an OmniRef export and a local copy of the base revision it pins, downloaded with `hf download MiniMaxAI/MiniMax-H3 --revision <revision> --local-dir "$OMNIREF_BASE"` for each of `text_encoder`, `tokenizer`, `processor`, `vae` and `audio_vae`.

Text-to-video and keyframe requests:

```bash
uniserve serve "$H3_ROOT" \
  --workers configs/minimax_h3/ulysses4.json \
  --served-model-name MiniMax-H3 \
  --host 0.0.0.0 \
  --max-video-seconds 8 \
  --max-model-len 2048 \
  --video-text-capacities 2048
```

Reference requests, with the base reference DiT:

```bash
uniserve serve "$H3_ROOT" \
  --workers configs/minimax_h3/ulysses4-reference.json \
  --served-model-name MiniMax-H3 \
  --host 0.0.0.0 \
  --max-video-seconds 5 \
  --max-model-len 8192 \
  --video-text-capacities 8192 \
  --max-condition-rows 40960 \
  --media-directory /srv/media
```

Both files place the numerical components on four GPUs of one host: four-way Ulysses denoising, TP4 text encoding with its vision tower, and the VAE encoders and decoders distributed one media unit per rank. A host worker holds the H.264 encoders, the muxer and the `media_reader` that decodes condition media. `ulysses4.json` places the root's `denoiser`; `ulysses4-reference.json` places its `reference_denoiser`, and also serves an OmniRef export:

```bash
uniserve serve "$OMNIREF_EXPORT" \
  --base-model "$OMNIREF_BASE" \
  --workers configs/minimax_h3/ulysses4-reference.json \
  --served-model-name MiniMax-H3-OmniRef \
  --max-condition-rows 49152 \
  --media-directory /srv/media
```

### Two hosts

`configs/minimax_h3/ulysses8-two-node.json` places the root's `denoiser` across two four-GPU hosts named `rank-0` and `rank-1`: eight-way Ulysses denoising, TP8 text encoding, and the VAE units distributed over all eight GPUs. Start the head on `rank-0` with `--host-identity rank-0` and the t2va or fl2va options above, then start `uniserve-host` on `rank-1` with `rank-0`'s address and the port the head logs, as the [FastH3 guide](../fast_h3/fast_h3.md#two-hosts) describes. The measured two-host deployment runs eagerly (`--graph-policy off`).

### Capacity options

| Option | Meaning |
| --- | --- |
| `--max-video-seconds` | Longest admitted duration, 4 to 15 seconds. Startup prepares a layout for every admitted frame count at every prepared canvas, so a shorter maximum starts faster and holds less storage. |
| `--video-aspect-ratios` | Named aspect ratios whose canvases startup prepares, `16:9,9:16` by default; each adds its layouts to startup. A request at any other canvas is refused, including an `fl2va` request whose `auto` canvas falls outside them. These denoisers follow the 768-pixel canvas rule, so `--video-resolutions` stays `768p`. |
| `--max-model-len` | Longest prompt presentation in tokens. A presentation includes the vision tokens of every keyframe, reference image and reference video: the official keyframe request presents 1935 tokens and the official video-plus-audio request 6913. |
| `--video-text-capacities` | Prompt capacities startup prepares; each prompt runs in the smallest that holds it. Fewer capacities start faster. |
| `--max-condition-rows` | Denoiser rows the conditions of one request may occupy. The default, 2048, holds two keyframes at any canvas, so a `t2va` or `fl2va` deployment needs no setting. A reference deployment must state its capacity: a five-second reference video with its soundtrack occupies about 38,000 rows, a 16:9 reference image about 7,300. Each request reserves only its own rows, but the capacity sizes per-request storage. A request beyond it is refused with `invalid_request` naming its rows and the option. An OmniRef export counts whole 128-row tiles, so the same request takes more rows than on a base reference DiT (see below). |
| `--media-directory` | Directory under which `file://` condition URIs resolve; `file://` is refused without it. |
| `--remote-media` | Whether `http(s)://` condition URIs are fetched; on by default. |
| `--max-request-bytes` | Largest request body and fetched media in total; 256 MiB by default. |

OmniRef bounds a request's packed sequence (prompt, conditions and generated rows) at 131,072 rows; the base denoisers have no such bound beyond GPU storage.

A base denoiser counts one condition row per token. An OmniRef export packs each condition into whole 128-row tiles: an image or an audio track takes `ceil(rows / 128)` tiles, and a video's token grid of latent frames × height/32 × width/32 takes `ceil(frames / 4) × ceil(height / 32 / 4) × ceil(width / 32 / 8)` tiles. Its rows are 128 times its tiles, and `--max-condition-rows` bounds that count. The official video-plus-audio request occupies 38,124 rows on the base reference DiT and 47,104 on OmniRef, so an OmniRef deployment serving it states at least `--max-condition-rows 49152`. OmniRef takes no keyframes.

## Generate a video

`/v1/videos/sync` returns the MP4; `/v1/videos` creates an asynchronous job, as in the FastH3 guide. A text-to-video request:

```bash
curl -sS http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -o fox.mp4 \
  -d '{
    "model": "MiniMax-H3",
    "task": "t2va",
    "prompt": "A red fox trots across a snowy meadow at sunrise.",
    "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
    "seed": 42
  }'
```

A first-frame request anchors the first generated frame to an image; `frame_index` is `0` for the first frame and `-1` for the last, and a request may anchor both:

```json
{
  "model": "MiniMax-H3",
  "task": "fl2va",
  "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced. ...",
  "conditions": [
    {"type": "image", "uri": "file:///srv/media/keyframe.png", "role": "keyframe", "frame_index": 0}
  ],
  "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 8}
}
```

A reference request names its references in order; the prompt refers to them by their ordinal labels (`<Picture 1>`, `<Video 1>`, `<Audio 1>`):

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

### Request fields

| Field | Rule |
| --- | --- |
| `task` | Required: `t2va`, `fl2va` or `ref2va`, one the served denoiser serves. |
| `prompt` | Required plain text. Condition placeholders are presented by the server; the prompt must not contain the model's vision tokens. |
| `conditions` | The task's conditions in request order. `t2va` takes none. `fl2va` takes one or two image keyframes (`frame_index` 0, -1, or both). `ref2va` takes up to 9 images, 3 videos and 3 audio tracks, 12 references in total; the base reference DiT also takes keyframes, an OmniRef export none. |
| `conditions[].type` | `image`, `video`, `video_audio` (a video that must have a soundtrack) or `audio`. |
| `conditions[].uri` | `data:` always; `http(s)://` unless `--remote-media false`; `file://` under `--media-directory`. |
| `conditions[].role` | `keyframe` or `reference`. |
| `conditions[].start_time_seconds` | Offset into a video reference; its soundtrack starts at the same offset. |
| `target.short_edge` | 768. |
| `target.aspect_ratio` | `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16` or `auto`. For `fl2va`, `auto` takes the keyframe's displayed aspect ratio, within 1:4 to 4:1; for the other tasks it is 16:9. The canvas has a 768-pixel short edge, an area of at most 768×1344, and sides on a 32-pixel grid: 16:9 is 1344×768 and 9:16 is 768×1344. The deployment serves only the canvases it prepared (`--video-aspect-ratios`). |
| `target.duration_seconds` | 4 to the served maximum. The frame count is the duration at 24 fps aligned up to the next count of the form 17n + 5: five seconds are 124 frames, eight are 192. A reference request whose only audio-bearing reference sets the duration may omit it. |
| `seed` | Default 42. |
| `num_inference_steps`, `flow_shift`, `audio_flow_shift` | Optional; they may only restate the served schedule. |
| `n`, `num_outputs_per_prompt` | Optional; only 1. |

Unknown fields are refused, at every level. `GET /v1/capabilities` reports the served tasks, their condition rules, the served canvases with their short edges and aspect ratios, the schedule, the duration and prompt limits, the condition capacity and the media policy.

## Precision

The default `--quantization-config`, `{"mode":"quality"}`, serves the checkpoint's own BF16 transformers and FP16 video decoder, the lossless path. The base checkpoint also accepts the `balanced`, `performance` and `maximum` presets and the per-component overrides the [FastH3 guide](../fast_h3/fast_h3.md#precision-and-graphs) lists; they quantize the video decoder (`balanced`), the denoiser MLPs (`performance`, `maximum`) and the text encoder (`maximum`), and need Blackwell. On W1 with four GB200s they take 36.2 s, 34.1 s and 33.0 s against `quality`'s 36.4 s (diagnostic runs, eight requests each). They are lossy: `balanced` changes only the decoded frames (34.6 dB PSNR against `quality` at the same seed), and the presets that quantize the denoiser change the generated sample itself, so their outputs diverge from `quality`'s; no quality acceptance has been established for them.

## Numerical behavior

The base DiTs evaluate their block epilogues (modulation, gated residuals, rotary Q/K and SwiGLU gating) the way the diffusers reference does, rounding to BF16 after each operation; OmniRef follows FastVideo's eager arithmetic, which rounds the same way. Seeded draws follow each checkpoint's reference: a request draws its condition noise, then its video and audio noise, on the CPU from its seed. The Python library computes what a server computes on the same placement: given the server's capacity layout (`minimax_h3.generation.generate(..., layout=...)`), a single-GPU library generation equals a single-GPU server's.


The [Python library guide](library.md) describes direct loading and generation with the same numerical modules.

## Troubleshooting

| Symptom | Cause and remedy |
| --- | --- |
| `invalid_request` naming `task` | The body is the duration-only form of earlier FastH3 releases. Send `task` and `target`. |
| `invalid_request` naming the condition rows and `--max-condition-rows` | The request's conditions exceed the deployment's condition capacity. Serve with a larger `--max-condition-rows`. |
| `invalid_request` naming the prompt length | The presentation, vision tokens included, exceeds `--max-model-len`. Raise it together with `--video-text-capacities`. |
| A refused `file://` condition | `--media-directory` is unset or the path resolves outside it. |
| `server_error` naming `cannot run ffprobe` | The server probes video and audio conditions with `ffprobe` and cannot run it. Put FFmpeg's `ffprobe` on `PATH` or name it with `--ffprobe`. |
| `ModuleNotFoundError: torchaudio` in the worker log | The environment predates the `torchaudio` dependency. Run `uv sync --locked` again. |
| An OmniRef start refusing a base file by name | The `--base-model` copy holds that file at another revision. Download the base at the revision the export pins. |
