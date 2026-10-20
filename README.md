# UniServe

UniServe serves FastH3, the fast distilled variant of MiniMax-H3, and the MiniMax-H3 base checkpoint. It generates video with synchronized audio from a text prompt (`t2va`), from keyframes (`fl2va`), and from reference images, videos and audio (`ref2va`, FastH3 OmniRef). Every response is an H.264 MP4 at 24 fps with stereo 32-kHz AAC audio.

A Rust server owns HTTP admission, scheduling, request state and response assembly. Python workers run the models with UniServe's numerical layers and kernels.

## Requirements

| Component | Requirement |
| --- | --- |
| OS | Linux on x86-64 or aarch64 |
| GPU | NVIDIA Hopper or Blackwell: H200, GB200 and RTX PRO 6000 Blackwell Server Edition are validated |
| CUDA | CUDA 13 toolkit and a matching driver |
| Python | Python 3.12 |
| Rust | Stable Rust toolchain with edition 2024 support |
| FFmpeg | `ffprobe` and `ffmpeg`, for keyframe and reference requests |

## Install

Run from the repository root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

uv sync --locked --python 3.12 --extra gpu
source .venv/bin/activate
```

The sync builds the `uniserve` server and `uniserve-host` launcher binaries, the worker IPC extension and UniServe's native kernels for the visible GPUs, and installs FlashInfer and FlashAttention-4. The [guide](docs/minimax_h3/minimax_h3.md#install) lists the system packages and a container build.

## Text to video

Download FastH3 and serve it on four GPUs:

```bash
export H3_MODEL=$PWD/models/FastVideo-FastH3-8-Step-V2
hf download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir "$H3_MODEL"

uniserve serve "$H3_MODEL" \
  --workers configs/minimax_h3/ulysses4.json \
  --served-model-name FastH3 \
  --host 0.0.0.0 \
  --max-running-requests 2
```

Startup prepares every admitted request shape before `/health` reports ready, and the first startup on a machine also compiles kernels. Then generate a video:

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","prompt":"A clear stream flows through a green forest while birds sing.","task":"t2va","target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5},"seed":1000}' \
  --output forest.mp4
```

## Reference to video

FastH3 OmniRef generates a video that follows up to twelve references: at most nine images, three videos and three audio tracks. The checkpoint holds its denoiser; the server reads the text encoder, tokenizer and VAEs from the MiniMax-H3 revision the checkpoint pins, and downloads them into the Hugging Face cache on first use.

```bash
export OMNIREF_MODEL=$PWD/models/FastVideo-FastH3-Omni-8-Step-V1
hf download FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --local-dir "$OMNIREF_MODEL"

uniserve serve "$OMNIREF_MODEL" \
  --workers configs/minimax_h3/ulysses4-reference.json \
  --served-model-name FastH3-OmniRef \
  --host 0.0.0.0 \
  --max-video-seconds 5 --max-model-len 8192 --video-text-capacities 8192 \
  --max-condition-rows 49152 \
  --media-directory /srv/media
```

The prompt names the references in request order as `<Picture n>`, `<Video n>` and `<Audio n>`:

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3-OmniRef","task":"ref2va","prompt":"subject_definitions:\n<Subject 1> is the man in <Picture 1>.\n<Audio 1> is the voice of <Subject 1>. ...","conditions":[{"type":"image","uri":"file:///srv/media/subject.png","role":"reference"},{"type":"audio","uri":"file:///srv/media/voice.mp3","role":"reference"}],"target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5}}' \
  --output reference.mp4
```

The [guide](docs/minimax_h3/minimax_h3.md#startup-and-capacity) explains how the prompt, condition and duration capacities bound the references a deployment accepts.

## Deployments

A deployment file passed to `--workers` places the model's components on devices and hosts. The files in `configs/minimax_h3/`:

| File | GPUs | Serves |
| --- | --- | --- |
| `ulysses4.json` | Four on one host | Text and keyframe requests: Ulysses4 denoiser, TP4 text encoder |
| `ulysses8.json` | Eight on one host | Text and keyframe requests: Ulysses8 denoiser, TP8 text encoder |
| `ulysses8-two-node.json` | Four on each of two hosts | Text and keyframe requests: Ulysses8 denoiser, TP8 text encoder |
| `fasth3-ulysses4x2.json` | Eight on one host | Text and keyframe requests: two Ulysses4 replicas |
| `fasth3-ulysses4x2-two-node.json` | Four on each of two hosts | Text and keyframe requests: one Ulysses4 replica per host |
| `fasth3-dp8-text-tp8.json` | Eight on one host | Text and keyframe requests: eight one-GPU replicas, shared TP8 text encoder |
| `single-gpu-reference.json` | One | Reference requests |
| `ulysses2-reference.json` | Two on one host | Reference requests: Ulysses2 denoiser, TP2 text encoder |
| `ulysses4-reference.json` | Four on one host | Reference requests: Ulysses4 denoiser, TP4 text encoder |
| `ulysses8-reference.json` | Eight on one host | Reference requests: Ulysses8 denoiser, TP8 text encoder |
| `ulysses8-reference-two-node.json` | Four on each of two hosts | Reference requests: Ulysses8 denoiser, TP4 text encoder |

One replica across every GPU gives the lowest latency; several replicas serve more requests at once. The [guide](docs/minimax_h3/minimax_h3.md#deployments) lists the validated settings per GPU type and the two-host startup.

## HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Process and route readiness |
| `GET /metrics` | Runtime metrics (OpenMetrics) |
| `GET /version` | Build information |
| `GET /v1/models` | The served model |
| `GET /v1/capabilities` | Served tasks, canvases, durations, schedule, request fields and job retention |
| `POST /v1/videos/sync` | Generate one video and return the MP4 |
| `POST /v1/videos` | Create an asynchronous video job |
| `GET /v1/videos`, `GET /v1/videos/{id}` | List jobs, or read one job's state and progress |
| `GET /v1/videos/{id}/content` | Download a completed job's MP4 |
| `DELETE /v1/videos/{id}` | Cancel or delete a job |

The [guide](docs/minimax_h3/minimax_h3.md#generate-a-video) defines every request field.

## Documentation

- [FastH3 and MiniMax-H3 guide](docs/minimax_h3/minimax_h3.md): checkpoints, deployments, serving options, requests and precision.
- [Python library](docs/minimax_h3/library.md): the same models without the server.
- [Evaluation](docs/minimax_h3/evaluation.md): measured performance and how to reproduce it with `uniserve-eval`.
- [NVIDIA Dynamo](docs/minimax_h3/dynamo.md): FastH3 behind Dynamo's `/v1/videos`.

## Development

```bash
just lint
just test-rust
just test-python-fast
just test-python-integration
```

These run without a GPU. `just test-python-cuda` runs the tests that need one, and `just test-python-gpu` the end-to-end server tests, which read checkpoints from the environment variables they name.

## Repository layout

```text
crates/server/        HTTP API, request planning and model profiles
crates/engine/        Scheduler, executors and engine process
crates/worker-ipc/    Worker messages and transport endpoints
crates/bin/           `uniserve`, `uniserve-host` and the Dynamo worker
uniserve/             Numerical layers, loading and resource binding
uniserve_models/      Model definitions and checkpoint readers
uniserve_worker/      Worker lifecycle and rank-local execution
uniserve_kernels/     Device kernels
uniserve_eval/        Benchmark driver and workloads
configs/minimax_h3/   Deployment files
docs/minimax_h3/      Guides and container build
tests/                Unit, integration and end-to-end tests
```

## License

UniServe is licensed under the [Apache License 2.0](LICENSE). Third-party sources under `uniserve_kernels/src/uniserve_kernels/` keep their own licenses, each next to its sources: DeepEP and DeepGEMM (MIT), CUTLASS (BSD-3-Clause), {fmt} (MIT), MegaMoE (BSD-3-Clause) and the native sparse-attention kernel (Apache-2.0).
