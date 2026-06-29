from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from tests.python.e2e.http_helpers import (
    png_size_from_b64,
    post_sse,
    require_uniserve_binary,
    server_process,
    strip_b64,
)
from tests.python.e2e.quality_checks import png_quality, text_quality

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.sensenova,
    pytest.mark.slow,
]

# Resolve the SenseNova checkpoint from the environment so the test is not bound
# to one developer's machine. Falls back to the historical local path.
MODEL = Path(
    os.environ.get(
        "UNISERVE_SENSENOVA_MODEL",
        "/home/hal-ysun/models/SenseNova-U1-8B-MoT-Interleaved-local",
    )
)
PROMPT = "Generate a travel guide covering Sonoma, Sequoia, Tahoe, and the Golden Gate."


def test_sensenova_gpu_native_interleave_quality_and_resolution():
    if os.environ.get("UNISERVE_RUN_GPU_E2E") != "1":
        pytest.skip("set UNISERVE_RUN_GPU_E2E=1 to run the real SenseNova GPU e2e")
    binary = require_uniserve_binary()
    if not MODEL.exists():
        pytest.fail(f"local SenseNova checkpoint is missing: {MODEL}")

    artifact_root = Path(os.environ.get("UNISERVE_GPU_E2E_ARTIFACT_DIR", "e2e-artifacts/sensenova"))
    artifact_dir = artifact_root / time.strftime("%Y%m%d-%H%M%S")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    request_payload = {
        "prompt": PROMPT,
    }
    (artifact_dir / "request.json").write_text(json.dumps(request_payload, indent=2) + "\n")

    args = [
        str(binary),
        "serve",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
        "--worker-python",
        "/home/hal-ysun/UniServe/.venv/bin/python",
        "--device",
        "cuda",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "8",
        "--max-batch",
        "8",
        "--max-num-batched-tokens",
        "4096",
        "--pipeline-depth",
        "1",
    ]
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    base_url = "http://127.0.0.1:18080"
    with server_process(args, base_url, artifact_dir / "uniserve.log", timeout_s=1800, env=env) as _:
        events = post_sse(base_url, "/generate", request_payload, timeout_s=1800)

    (artifact_dir / "events.metadata.json").write_text(json.dumps(strip_b64(events), indent=2) + "\n")
    text = "".join(event.get("text", "") for event in events if event.get("type") == "text")
    (artifact_dir / "response.txt").write_text(text)

    finished = next((event for event in events if event.get("type") == "finished"), None)
    assert finished is not None, strip_b64(events)

    image_begins = [(idx, event) for idx, event in enumerate(events) if event.get("type") == "image_begin"]
    image_dones = [(idx, event) for idx, event in enumerate(events) if event.get("type") == "image_done"]
    assert len(image_begins) == 4, strip_b64(events)
    assert len(image_dones) == 4, strip_b64(events)
    assert all(begin_idx < done_idx for (begin_idx, _), (done_idx, _) in zip(image_begins, image_dones, strict=True))

    metadata_items = []
    quality_items = []
    for idx, ((_, image_begin), (_, image_done)) in enumerate(zip(image_begins, image_dones, strict=True), start=1):
        assert (image_begin["width"], image_begin["height"]) == (2048, 1152)
        assert (image_done["width"], image_done["height"]) == (2048, 1152)

        png_bytes = base64.b64decode(image_done["pixels_png_b64"])
        png_path = artifact_dir / f"response.image{idx}.png"
        png_path.write_bytes(png_bytes)
        metadata = {
            "image_id": image_done["image_id"],
            "width": image_done["width"],
            "height": image_done["height"],
            "bytes": image_done["bytes"],
            "sha256": image_done["sha256"],
            "file_sha256": hashlib.sha256(png_bytes).hexdigest(),
            "path": str(png_path),
        }
        metadata_items.append(metadata)
        assert metadata["sha256"] == metadata["file_sha256"]
        assert metadata["bytes"] == len(png_bytes)
        assert png_size_from_b64(image_done["pixels_png_b64"]) == (2048, 1152)
        image_ok, image_checks = png_quality(image_done["pixels_png_b64"], 2048, 1152)
        image_checks["ok"] = image_ok
        image_checks["image_id"] = image_done["image_id"]
        quality_items.append(image_checks)

    (artifact_dir / "response.metadata.json").write_text(json.dumps(metadata_items, indent=2) + "\n")
    (artifact_dir / "image_quality.json").write_text(json.dumps(quality_items, indent=2) + "\n")
    assert all(item["ok"] for item in quality_items), quality_items

    text_ok, text_checks = text_quality(text, required_locations=True)
    (artifact_dir / "text_quality.json").write_text(json.dumps(text_checks, indent=2) + "\n")
    assert text_ok, text_checks
