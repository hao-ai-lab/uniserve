"""Public generation on ordered tensor, sequence and pipeline assignments."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    png_size_from_b64,
    require_uniserve_binary,
    server_process,
    tiny_input_png_b64,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


@pytest.mark.parametrize(
    ("description", "model_env"),
    [
        pytest.param("qwen3", "UNISERVE_QWEN3_MODEL", marks=pytest.mark.model("qwen3")),
        pytest.param("bagel", "UNISERVE_BAGEL_MODEL", marks=pytest.mark.model("bagel")),
        pytest.param("sensenova", "UNISERVE_SENSENOVA_MODEL", marks=pytest.mark.model("sensenova")),
    ],
)
@pytest.mark.parametrize("parallel_axis", ["tensor", "pipeline", "sequence"])
def test_ordered_bindings_generate_outputs(
    tmp_path: Path, description: str, model_env: str, parallel_axis: str
):
    checkpoint = os.environ.get(model_env, "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail(f"{model_env} must name the model checkpoint directory")
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    parallel = (
        {"sequence_parallel": {"kind": "ulysses", "ulysses_degree": 2}}
        if parallel_axis == "sequence"
        else {f"{parallel_axis}_parallel_size": 2}
    )
    worker_config = {
        "id": "model",
        "ranks": [{"node": "localhost", "device": f"cuda:{device}"} for device in (3, 1)],
        "entries": {"model": {"ranks": [1, 0], "parallel_config": parallel}},
        "queue_depth": 2,
    }
    command = [
        str(require_uniserve_binary()),
        "serve",
        checkpoint,
        "--model-description",
        description,
        "--served-model-name",
        description,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-python",
        sys.executable,
        "--workers",
        json.dumps([worker_config]),
        "--dtype",
        "bfloat16",
        "--max-total-tokens",
        "65536" if description == "sensenova" else "8192",
        "--max-model-len",
        "32768" if description == "sensenova" else "4096",
        "--max-num-batched-tokens",
        "4096" if description == "sensenova" else "1024",
        "--chunked-prefill-size",
        "512",
        "--max-running-requests",
        "1",
    ]
    with server_process(
        command,
        base_url,
        tmp_path / f"{description}-{parallel_axis}.log",
        timeout_s=300,
        env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    ):
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": description,
                "messages": [{"role": "user", "content": "Name a primary color."}],
                "temperature": 0,
                "max_completion_tokens": 8,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        assert result["model"] == description
        assert result["usage"]["prompt_tokens"] > 0
        assert 1 <= result["usage"]["completion_tokens"] <= 8
        choice = result["choices"][0]
        assert choice["finish_reason"] in ("stop", "length")
        message = choice["message"]
        assert message.get("content") or message.get("reasoning_content")

        if description in {"bagel", "sensenova"}:
            image_size = (512, 512) if description == "bagel" else (2048, 1152)
            image_to_text = httpx.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": description,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe the dominant color."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{tiny_input_png_b64()}"
                                    },
                                },
                            ],
                        }
                    ],
                    "temperature": 0,
                    "max_completion_tokens": 8,
                },
                timeout=120,
            )
            image_to_text.raise_for_status()
            image_result = image_to_text.json()
            image_message = image_result["choices"][0]["message"]
            assert image_message.get("content") or image_message.get("reasoning_content")
            assert 1 <= image_result["usage"]["completion_tokens"] <= 8

            generated = httpx.post(
                f"{base_url}/v1/images/generations",
                json={
                    "model": description,
                    "prompt": "A blue square.",
                    "size": f"{image_size[0]}x{image_size[1]}",
                    "steps": 1,
                    "seed": 7,
                },
                timeout=180,
            )
            generated.raise_for_status()
            images = generated.json()["data"]
            assert len(images) == 1
            assert png_size_from_b64(images[0]["b64_json"]) == image_size
