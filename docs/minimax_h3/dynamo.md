# Run FastH3 through NVIDIA Dynamo

Dynamo's frontend serves `/v1/videos` and routes each request to a UniServe worker, which runs the whole pipeline and returns the MP4. The integration is experimental and serves `t2va` only. This guide uses Dynamo 1.5.0 and four GB200 GPUs on one host.

## Install

Install UniServe as in the [README](../../README.md#install), then build the worker and a separate frontend environment, which keeps UniServe's pinned GPU dependencies apart from Dynamo's:

```bash
apt-get install -y build-essential cmake pkg-config protobuf-compiler libclang-dev
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs
cargo build --locked --release -p uniserve-dynamo-worker

uv venv --python 3.12 artifacts/dynamo/frontend-env
uv pip install --python artifacts/dynamo/frontend-env/bin/python ai-dynamo==1.5.0
hf download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir models/FastH3
```

`LIBRARY_PATH` names the CUDA stub libraries of your CUDA installation.

## Start the frontend and worker

In one terminal:

```bash
artifacts/dynamo/frontend-env/bin/python -m dynamo.frontend \
  --http-host 127.0.0.1 --http-port 18090 \
  --namespace uniserve-fasth3 \
  --discovery-backend file
```

In another, from the same repository on the same host:

```bash
DYN_DISCOVERY_BACKEND=file target/release/uniserve-dynamo-worker \
  --namespace uniserve-fasth3 \
  --model-path models/FastH3 --served-model-name FastH3 \
  --worker-python "$PWD/.venv/bin/python" \
  --workers configs/minimax_h3/ulysses4.json \
  --max-model-len 16384 --max-video-seconds 15 \
  --max-running-requests 2 --quantization-config '{"mode":"quality"}'
```

Both processes share the file discovery directory, `/tmp/dynamo_store_kv` by default, and the namespace. The worker registers once its startup has prepared every request shape; it is ready when `curl -fsS http://127.0.0.1:18090/v1/models` lists `FastH3`.

## Generate a video

`POST /v1/videos` answers once generation completes; with `b64_json` the MP4 is in `data[0].b64_json`:

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:18090/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "FastH3",
    "prompt": "A stream flows over smooth stones in a green forest, with the sound of running water.",
    "seconds": 5,
    "size": "1344x768",
    "response_format": "b64_json",
    "output_format": "mp4",
    "nvext": {"seed": 1000}
  }' -o artifacts/dynamo/response.json

.venv/bin/python - <<'PY'
import base64
import json
from pathlib import Path

response = json.loads(Path("artifacts/dynamo/response.json").read_text())
assert response["status"] == "completed", response
Path("artifacts/dynamo/video.mp4").write_bytes(
    base64.b64decode(response["data"][0]["b64_json"], validate=True)
)
PY
```

`size` selects one of the canvases the worker prepares with `--video-resolutions` and `--video-aspect-ratios`, as for `uniserve serve` (1344x768 and 768x1344 by default); without it the worker generates the first. A client may instead send top-level `resolution` (`768p` or `480p`) and `aspect_ratio`; a `size` that disagrees with them is refused. The clip is `seconds` rounded up to the frame layout, as in the [serving guide](minimax_h3.md#requests). The checkpoint fixes the schedule and frame rate, so `nvext.num_inference_steps`, `nvext.fps` and `nvext.num_frames` are refused; image input, streaming and guidance overrides are unsupported.
