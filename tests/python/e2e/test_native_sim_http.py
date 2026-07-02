from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from uniserve_e2e.harness.runner import BenchmarkRunner
from uniserve_e2e.harness.spec import BenchmarkSpec, TaskName
from tests.python.e2e.http_helpers import (
    find_free_port,
    png_size_from_b64,
    post_sse,
    require_uniserve_binary,
    server_process,
    tiny_input_png_b64,
)

pytestmark = [pytest.mark.e2e]

MODEL = Path("/home/hal-ysun/models/SenseNova-U1-8B-MoT-Interleaved-local")


@contextmanager
def sim_server(tmp_path: Path):
    try:
        binary = require_uniserve_binary()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    if not MODEL.exists():
        pytest.skip(f"local SenseNova checkpoint is missing: {MODEL}")
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    args = [
        str(binary),
        "serve",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--sim",
        "--worker-python",
        str(Path.cwd() / ".venv" / "bin" / "python"),
        "--device",
        "cpu",
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
        "--disable-log-stats",
    ]
    with server_process(args, base_url, tmp_path / "uniserve-sim.log", timeout_s=180) as _:
        yield base_url


def test_sim_http_native_contracts_and_benchmark_smoke(tmp_path: Path):
    # CPU sim mode drives the worker with runtime/stub.StubUniModel, which
    # fabricates deterministic outputs (synthetic gradient PNG at the requested
    # geometry, sequential text tokens with a forced EOS). The image dimension
    # and PNG assertions below therefore validate the HTTP/IPC contract and the
    # geometry plumbing end-to-end, NOT any real model inference -- real-output
    # correctness is covered by the GPU e2e tests, which this stub-driven smoke
    # test intentionally substitutes for so the contract path runs without a GPU.
    with sim_server(tmp_path) as base_url:
        # The stub forces an EOS token once a request has emitted >= 8 tokens
        # (runtime/stub.py:_text). Request more than that so the natural
        # EOS-driven finish path is actually exercised, rather than always
        # terminating on the max_tokens length cap.
        text_events = post_sse(
            base_url,
            "/generate",
            {"prompt": "Say hello from the sim backend.", "mode": "text", "max_tokens": 16},
        )
        assert any(event["type"] == "text" for event in text_events)
        assert text_events[-1]["type"] == "finished"

        image_response = httpx.post(
            f"{base_url}/v1/images/generations",
            json={"prompt": "A California travel poster."},
            timeout=300,
        )
        image_response.raise_for_status()
        image_payload = image_response.json()["data"][0]
        assert (image_payload["width"], image_payload["height"]) == (2048, 1152)
        assert png_size_from_b64(image_payload["b64_json"]) == (2048, 1152)
        assert image_payload["bytes"] > 0
        assert len(image_payload["sha256"]) == 64

        interleave_events = post_sse(
            base_url,
            "/generate",
            {
                "prompt": "Generate a travel guide covering Sonoma, Sequoia, Tahoe, and the Golden Gate.",
                "mode": "interleave",
                "max_tokens": 8,
                "image": {"max_images": 1},
            },
            timeout_s=300,
        )
        assert any(event["type"] == "text" for event in interleave_events)
        image_begin = next(event for event in interleave_events if event["type"] == "image_begin")
        image_done = next(event for event in interleave_events if event["type"] == "image_done")
        assert (image_begin["width"], image_begin["height"]) == (2048, 1152)
        assert (image_done["width"], image_done["height"]) == (2048, 1152)
        assert png_size_from_b64(image_done["pixels_png_b64"]) == (2048, 1152)

        i2i_events = post_sse(
            base_url,
            "/generate",
            {
                "prompt": "Use the input image as a color reference for a California travel image.",
                "mode": "image",
                "input_image_b64": tiny_input_png_b64(),
            },
            timeout_s=300,
        )
        assert any(event["type"] == "image_done" for event in i2i_events)

        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            json.dumps(
                {
                    "id": "smoke-1",
                    "task": "interleave",
                    "prompt": "Generate a travel guide covering Sonoma, Sequoia, Tahoe, and the Golden Gate.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        spec = BenchmarkSpec(
            name="sim_interleave_smoke",
            task=TaskName.INTERLEAVE,
            endpoint="/generate",
            model="SenseNova-U1",
            dataset="trace",
            dataset_path=str(trace_path),
            num_prompts=1,
            warmup_requests=0,
            max_tokens=8,
        )
        result = asyncio.run(BenchmarkRunner(base_url, spec, tmp_path / "bench").run())
        assert result.summary["harness_status"] == "completed"
        assert result.summary["failed_count"] == 0
        assert result.summary["ok_count"] == 1
        assert (tmp_path / "bench" / "summary.json").exists()
