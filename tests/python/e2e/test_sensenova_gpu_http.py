from __future__ import annotations

import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from tokenizers import Tokenizer

from tests.python.e2e.http_helpers import (
    find_free_port,
    png_size_from_b64,
    post_sse,
    require_uniserve_binary,
    server_process,
    tiny_input_png_b64,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.sensenova,
    pytest.mark.model("sensenova"),
]

MODEL_ENV = "UNISERVE_SENSENOVA_MODEL"
SERVED_MODEL = "SenseNova-U1"
IMAGE_SIZE = (2048, 1152)


def active_model() -> Path:
    model_value = os.environ.get(MODEL_ENV)
    if not model_value:
        pytest.fail(f"{MODEL_ENV} is required for the SenseNova GPU gate")
    model = Path(model_value)
    if not model.is_dir():
        pytest.fail(f"configured checkpoint directory is missing: {model}")
    return model


def image_start_token_id(model: Path) -> int:
    tokenizer_path = model / "tokenizer.json"
    if not tokenizer_path.is_file():
        pytest.fail(f"configured checkpoint has no tokenizer.json: {tokenizer_path}")
    token_id = Tokenizer.from_file(str(tokenizer_path)).token_to_id("<img>")
    if token_id is None:
        pytest.fail("configured SenseNova tokenizer has no <img> control")
    return token_id


@contextmanager
def production_server(tmp_path: Path, model: Path) -> Iterator[str]:
    try:
        binary = require_uniserve_binary()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    args = [
        str(binary),
        "serve",
        str(model),
        "--model-description",
        "sensenova",
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-python",
        str(Path.cwd() / ".venv" / "bin" / "python"),
        "--max-total-tokens",
        "65536",
        "--max-running-requests",
        "1",
        "--max-num-batched-tokens",
        "4096",
        "--prefill-cuda-graph",
        "true",
        "--tp-size",
        "2",
        "--log-stats",
        "false",
    ]
    environment = {"CUDA_VISIBLE_DEVICES": "0,1"}
    with server_process(
        args,
        base_url,
        tmp_path / "sensenova-production.log",
        timeout_s=300,
        env=environment,
    ):
        yield base_url


def post_chat(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={"model": SERVED_MODEL, "temperature": 0.0, **payload},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def assert_bounded_text_completion(payload: dict[str, Any]) -> None:
    assert payload["model"] == SERVED_MODEL
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["usage"]["prompt_tokens"] > 0
    assert payload["usage"]["completion_tokens"] == 2
    assert payload["usage"]["image_count"] == 0


def assert_generated_image(image: dict[str, Any]) -> None:
    assert (image["width"], image["height"]) == IMAGE_SIZE
    assert png_size_from_b64(image["b64_json"]) == IMAGE_SIZE
    assert image["bytes"] > 0
    assert len(image["sha256"]) == 64


def visible_stream_delta(event: dict[str, Any]) -> dict[str, Any] | None:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return None
    if any(delta.get(field) for field in ("content", "reasoning_content", "images")):
        return delta
    return None


@pytest.mark.timeout(600)
def test_sensenova_public_production_lineage(tmp_path: Path):
    model = active_model()
    input_image_url = f"data:image/png;base64,{tiny_input_png_b64()}"
    with production_server(tmp_path, model) as base_url:
        text = post_chat(
            base_url,
            {
                "messages": [{"role": "user", "content": "Reply with one word."}],
                "modalities": ["text"],
                "max_completion_tokens": 2,
            },
        )
        assert_bounded_text_completion(text)

        image_to_text = post_chat(
            base_url,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the dominant color."},
                            {"type": "image_url", "image_url": {"url": input_image_url}},
                        ],
                    }
                ],
                "modalities": ["text"],
                "max_completion_tokens": 2,
            },
        )
        assert_bounded_text_completion(image_to_text)

        image_response = httpx.post(
            f"{base_url}/v1/images/generations",
            json={
                "model": SERVED_MODEL,
                "prompt": "A blue square.",
                "n": 1,
                "steps": 1,
                "seed": 7,
            },
            timeout=600,
        )
        image_response.raise_for_status()
        generated_images = image_response.json()["data"]
        assert len(generated_images) == 1
        assert_generated_image(generated_images[0])

        interleaved = post_sse(
            base_url,
            "/v1/chat/completions",
            {
                "model": SERVED_MODEL,
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Use this color in one generated image."},
                            {"type": "image_url", "image_url": {"url": input_image_url}},
                        ],
                    }
                ],
                "modalities": ["text", "image"],
                "max_completion_tokens": 2,
                "temperature": 0.0,
                "logit_bias": {str(image_start_token_id(model)): 100.0},
                "image_config": {"num_images": 1, "steps": 1, "seed": 7},
            },
            timeout_s=600,
        )
        assert interleaved[-1]["type"] == "sse_done"
        finish_reasons = [
            choice["finish_reason"]
            for event in interleaved
            for choice in event.get("choices", [])
            if choice.get("finish_reason") is not None
        ]
        assert finish_reasons == ["length"]
        usage = next(event["usage"] for event in interleaved if event.get("usage"))
        assert usage["completion_tokens"] == 2
        assert usage["image_count"] == 1
        assert usage["image_steps"] == 1
        assert usage["image_steps_per_image"] == [1]
        visible_events = [
            (event, delta)
            for event in interleaved
            if (delta := visible_stream_delta(event)) is not None
        ]
        assert visible_events
        commits = [event["public_commit"] for event, _ in visible_events]
        assert [commit["event_seq"] for commit in commits] == sorted(
            commit["event_seq"] for commit in commits
        )
        for commit in commits:
            assert commit["modality"] in {"text", "image"}
            assert math.isfinite(commit["committed_at"])
            assert commit["committed_at"] >= 0.0
            assert commit["semantic_root"]["producer_op_id"] > 0
            assert len(commit["semantic_root"]["semantic_digest"]) == 64
        images = [
            image
            for _, delta in visible_events
            for image in delta.get("images", [])
        ]
        assert len(images) == 1
        image_url = images[0]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")
        assert png_size_from_b64(image_url.split(",", 1)[1]) == IMAGE_SIZE
