from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    png_size_from_b64,
    post_sse,
    require_uniserve_binary,
    server_process,
    tiny_input_png_b64,
)
from uniserve_eval.harness.runner import BenchmarkRunner
from uniserve_eval.harness.spec import BenchmarkSpec, TaskName

pytestmark = [pytest.mark.e2e]

MODEL = Path("/home/hal-ysun/models/SenseNova-U1-8B-MoT-Default-local")


def chat_sse_text(events: list[dict[str, object]]) -> str:
    chunks: list[str] = []
    for event in events:
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                chunks.append(str(delta["content"]))
    return "".join(chunks)


def chat_sse_finish(events: list[dict[str, object]]) -> str | None:
    for event in events:
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
                return str(choice["finish_reason"])
    return None


def chat_sse_images(events: list[dict[str, object]]) -> list[dict[str, object]]:
    images: list[dict[str, object]] = []
    for event in events:
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("images"), list):
                images.extend(image for image in delta["images"] if isinstance(image, dict))
    return images


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
        "--model-path",
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
        "--max-running-requests",
        "8",
        "--max-num-batched-tokens",
        "4096",
        "--pipeline-depth",
        "1",
        "--log-stats",
        "false",
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
            "/v1/chat/completions",
            {
                "model": "SenseNova-U1",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "Say hello from the sim backend."}],
                "modalities": ["text"],
                "max_completion_tokens": 16,
            },
        )
        assert chat_sse_text(text_events)
        assert chat_sse_finish(text_events) == "stop"
        assert text_events[-1]["type"] == "sse_done"

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

        default_events = post_sse(
            base_url,
            "/v1/chat/completions",
            {
                "model": "SenseNova-U1",
                "stream": True,
                "stream_options": {"include_usage": True},
                "modalities": ["text", "image"],
                "messages": [
                    {
                        "role": "user",
                        "content": "Generate a travel guide covering Sonoma, Sequoia, Tahoe, and the Golden Gate.",
                    }
                ],
                "max_completion_tokens": 8,
                "image_config": {"num_images": 1},
            },
            timeout_s=300,
        )
        assert chat_sse_text(default_events)
        images = chat_sse_images(default_events)
        assert images
        image_b64 = str(images[0]["image_url"]["url"]).split(",", 1)[1]
        assert png_size_from_b64(image_b64) == (2048, 1152)
        assert not any(event.get("type") == "image_begin" for event in default_events)

        i2i_response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "SenseNova-U1",
                "modalities": ["image"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Use the input image as a color reference for a California travel image.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{tiny_input_png_b64()}"
                                },
                            },
                        ],
                    }
                ],
            },
            timeout=300,
        )
        i2i_response.raise_for_status()
        assert i2i_response.json()["choices"][0]["message"]["images"]

        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            json.dumps(
                {
                    "id": "smoke-1",
                    "task": "default",
                    "prompt": "Generate a travel guide covering Sonoma, Sequoia, Tahoe, and the Golden Gate.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        spec = BenchmarkSpec(
            name="sim_default_smoke",
            task=TaskName.DEFAULT,
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
