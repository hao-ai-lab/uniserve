# Run FastH3 through NVIDIA Dynamo

Serve FastH3 through Dynamo's `/v1/videos` endpoint using UniServe as the execution backend. Dynamo handles the frontend, discovery and routing; UniServe runs the full text-to-video-and-audio pipeline and returns the completed MP4. The integration is experimental.

This quickstart uses Dynamo 1.5.0 and the FastH3 8-Step V2 checkpoint on four local GB200 GPUs.

## Install

First complete the [FastH3 installation](fast_h3.md#install), including GPU providers, shared storage and a complete checkpoint. Run the following commands from the UniServe repository root. The examples use `.venv/bin/python` for UniServe; substitute your installed UniServe interpreter if it is elsewhere.

The Rust worker additionally needs Clang's library and builtin headers, a C++ compiler, CMake, pkg-config and protoc. On Ubuntu:

```bash
apt-get install -y build-essential cmake pkg-config protobuf-compiler libclang-dev
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs
cargo build --locked --release -p uniserve-dynamo-worker

uv venv --python /usr/bin/python3.12 artifacts/dynamo/frontend-env
uv pip install --python artifacts/dynamo/frontend-env/bin/python ai-dynamo==1.5.0
```

The separate frontend environment preserves UniServe's pinned GPU dependencies. Set the CUDA library path to match your CUDA installation.

Download the complete BF16 checkpoint, or use an existing local copy:

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
.venv/bin/hf download FastVideo/FastVideo-FastH3-8-Step-V2 \
  --revision 3da2ddfe1954d9cda4c05b643dc0f26007a655c5 \
  --local-dir "$H3_MODEL"
```

## Start the frontend and worker

In one terminal, start the frontend:

```bash
artifacts/dynamo/frontend-env/bin/python -m dynamo.frontend \
  --http-host 127.0.0.1 --http-port 18090 \
  --namespace uniserve-fasth3 \
  --discovery-backend file
```

In another terminal, from the same repository and on the same host, start the worker:

```bash
export H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
export DYN_DISCOVERY_BACKEND=file

target/release/uniserve-dynamo-worker \
  --namespace uniserve-fasth3 \
  --model-path "$H3_MODEL" --served-model-name FastH3 \
  --worker-python "$PWD/.venv/bin/python" \
  --workers configs/minimax-h3-four-devices.json \
  --max-model-len 16384 --max-video-seconds 15 \
  --max-running-requests 2 --quantization-config '{"mode":"quality"}'
```

The placement uses four-way Ulysses denoising and TP4 text encoding, distributes temporal decoder units across the GPUs, and starts four host video encoders plus a host muxer. Weights remain resident; the quality configuration retains BF16 text encoding and denoising with the configured FP16 VAE projections.

This single-host setup uses file discovery; requests travel over Dynamo's default TCP request plane. Both processes must share the discovery directory, `/tmp/dynamo_store_kv` by default, and the same namespace.

Model loading and CUDA graph preparation run before the worker registers. Check the frontend's model list:

```bash
curl -fsS http://127.0.0.1:18090/v1/models
```

Proceed when the returned model list contains `FastH3`.

## Generate a video

The endpoint is `POST /v1/videos`. It returns a JSON response after generation completes. With `b64_json`, the MP4 is embedded in `data[0].b64_json`:

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
    "nvext": {"seed": 1000, "num_inference_steps": 8}
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

Open `artifacts/dynamo/video.mp4` to play the generated video and soundtrack. FastH3 aligns the requested duration to its temporal layout, so the resulting clip can be slightly longer than requested.

The checkpoint determines the denoising schedule. `nvext.num_inference_steps` is optional; if supplied, it must match the checkpoint's eight steps. This integration supports text-to-video requests with completed responses; image input, streaming and guidance overrides are unsupported. Use `b64_json` to retrieve the MP4 directly, as above.

Stop the worker and frontend with Ctrl-C in their respective terminals.
