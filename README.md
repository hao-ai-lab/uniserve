# UniServe

UniServe is a serving engine for FastH3 video generation. It serves text-to-video (FastH3) and reference-to-video (FastH3 OmniRef) requests over HTTP on NVIDIA Hopper and Blackwell GPUs and returns each video as an MP4 with synchronized stereo audio.

## Install

Linux, CUDA 13, Python 3.12 and a stable Rust toolchain:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv sync --locked --python 3.12 --extra gpu
source .venv/bin/activate
```

## Text to video

```bash
hf download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir models/FastH3
uniserve serve models/FastH3 --workers configs/minimax_h3/ulysses4.json \
  --served-model-name FastH3 --host 0.0.0.0
```

The server reports ready on `/health` once it has prepared every request shape; the first start on a machine also compiles kernels. Then:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3","task":"t2va","prompt":"A clear stream flows through a green forest while birds sing.","target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5}}' \
  --output forest.mp4
```

## Reference to video

FastH3 OmniRef follows up to twelve references: at most nine images, three videos and three audio tracks.

```bash
hf download FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --local-dir models/FastH3-OmniRef
uniserve serve models/FastH3-OmniRef --workers configs/minimax_h3/ulysses4-reference.json \
  --served-model-name FastH3-OmniRef --host 0.0.0.0 --media-directory /srv/media \
  --max-model-len 24576 --video-text-capacities 4096,7168,8192,9216,10240,12288,15360 \
  --max-condition-rows 56192
```

The prompt names the references in request order as `<Picture n>`, `<Video n>` and `<Audio n>`:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  -d '{"model":"FastH3-OmniRef","task":"ref2va","prompt":"<Subject 1> is the man in <Picture 1>; <Audio 1> is his voice. ...","conditions":[{"type":"image","uri":"file:///srv/media/subject.png","role":"reference"},{"type":"audio","uri":"file:///srv/media/voice.mp3","role":"reference"}],"target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5}}' \
  --output reference.mp4
```

## Documentation

- [Serving guide](docs/minimax_h3/minimax_h3.md): checkpoints, deployments for one to eight GPUs, options and the request format.
- [Evaluation](docs/minimax_h3/evaluation.md): measured latency and throughput, and how to reproduce them.
- [Python library](docs/minimax_h3/library.md) and [NVIDIA Dynamo](docs/minimax_h3/dynamo.md).

## License

[Apache License 2.0](LICENSE). Third-party kernel sources under `uniserve_kernels/src/uniserve_kernels/` keep their own licenses: DeepEP, DeepGEMM and {fmt} (MIT), CUTLASS and MegaMoE (BSD-3-Clause), and the native sparse-attention kernel (Apache-2.0).
