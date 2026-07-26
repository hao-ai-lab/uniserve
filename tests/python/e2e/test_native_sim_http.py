from __future__ import annotations

import asyncio
import json
import os
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

MODEL_ENV = "UNISERVE_SENSENOVA_MODEL"


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
    model_value = os.environ.get(MODEL_ENV)
    if not model_value:
        pytest.skip(f"{MODEL_ENV} is required for the native sim HTTP gate")
    model = Path(model_value)
    if not model.exists():
        pytest.fail(f"configured SenseNova checkpoint is missing: {model}")
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    args = [
        str(binary),
        "serve",
        str(model),
        "--served-model-name",
        "SenseNova-U1",
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
        "8192",
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
    # CPU simulation emits deterministic text and image fixtures through the
    # production HTTP, scheduler, worker IPC, and geometry contracts.
    with sim_server(tmp_path) as base_url:
        # The deterministic model emits EOS after eight tokens; this request
        # exercises the runtime's EOS completion contract.
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
                "logit_bias": {"151670": 100.0},
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
            task=TaskName.INTERLEAVE,
            model="SenseNova-U1",
            dataset="trace",
            dataset_path=str(trace_path),
            num_prompts=1,
            warmup_requests=0,
            max_tokens=8,
            runtime_profile_id="sensenova-u1",
            plan_evidence_policy="runtime_inspection",
        )
        result = asyncio.run(BenchmarkRunner(base_url, spec, tmp_path / "bench").run())
        assert result.summary["harness_status"] == "completed"
        assert result.summary["failed_count"] == 0
        assert result.summary["ok_count"] == 1
        assert result.summary["artifact"]["plan_evidence"]["source"] == "runtime_inspection"
        plan = result.summary["artifact"]["plan_summary"]
        assert plan["dialect_id"] == "sensenova-u1"
        assert plan["profile_id"].startswith("neo_chat:")
        assert plan["generation"]["temperature"] == 0.0
        assert plan["generation"]["top_p"] == 1.0
        assert plan["generation"]["ignore_eos"] is True
        assert plan["generation"]["image"] == {
            "width": 2048,
            "height": 1152,
            "steps": 50,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_renorm_type": "none",
            "cfg_renorm_min": 0.0,
            "cfg_interval": [0.0, 1.0],
            "timestep_shift": 3.0,
            "seed": 42,
            "max_images": 4,
            "image_prompt_count": 0,
            "retain_images": True,
        }
        assert (tmp_path / "bench" / "summary.json").exists()
