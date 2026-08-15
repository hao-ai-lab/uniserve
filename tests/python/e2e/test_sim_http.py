from __future__ import annotations

import asyncio
import json
import os
import re
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
from uniserve_eval.harness.spec import BenchmarkSpec, MetricDefinition, TaskName

pytestmark = [pytest.mark.e2e]

MODEL_ENV = "UNISERVE_SENSENOVA_MODEL"
QWEN_MODEL_ENV = "UNISERVE_QWEN3_MODEL"
BAGEL_MODEL_ENV = "UNISERVE_BAGEL_MODEL"
CONTROL_TOKENS = ("<img>", "</img>")
SAMPLING_CONTROLS = [
    "greedy",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "allowed_token_ids",
    "bad_words",
    "min_tokens",
    "logprobs",
    "stop_token_ids",
    "eos",
    "stop_strings",
]


def active_model(environment_variable: str) -> Path:
    model_value = os.environ.get(environment_variable)
    if not model_value:
        pytest.fail(f"{environment_variable} is required for the sim HTTP gate")
    model = Path(model_value)
    if not model.exists():
        pytest.fail(f"configured checkpoint is missing: {model}")
    return model


def active_sensenova_model() -> Path:
    return active_model(MODEL_ENV)


def assert_model_discovery(
    base_url: str,
    *,
    model_id: str,
    description_id: str,
    endpoints: list[str],
    input_modalities: list[str],
    output_modalities: list[str],
    features: list[str],
) -> None:
    response = httpx.get(f"{base_url}/v1/models", timeout=30)
    response.raise_for_status()
    payload = response.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    model = payload["data"][0]
    assert model["id"] == model_id
    assert model["object"] == "model"
    assert model["created"] > 0
    assert model["owned_by"] == "uniserve"
    identity = model["identity"]
    assert identity["description_id"] == description_id
    assert identity["profile_id"].startswith(f"{description_id}:")
    assert re.fullmatch(r"[0-9a-f]{64}", identity["config_fingerprint"])
    assert model["capabilities"] == {
        "endpoints": endpoints,
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
        "features": features,
        "sampling_controls": SAMPLING_CONTROLS,
    }


def serving_lifecycle_metrics(text: str) -> dict[str, float]:
    states: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("uniserve:serving_requests{"):
            continue
        match = re.search(r'state="([^"]+)"', line)
        assert match is not None
        states[match.group(1)] = float(line.rsplit(" ", 1)[1])
    return states


def _control_ids_from_tokenizer_json(model: Path) -> dict[str, int]:
    tokenizer_path = model / "tokenizer.json"
    if not tokenizer_path.is_file():
        pytest.fail(f"configured SenseNova checkpoint has no tokenizer.json: {tokenizer_path}")
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    control_ids = {
        str(token["content"]): int(token["id"])
        for token in payload.get("added_tokens", [])
        if isinstance(token, dict) and token.get("content") in CONTROL_TOKENS
    }
    assert set(control_ids) == set(CONTROL_TOKENS), (
        f"frontend tokenizer.json must define both SenseNova image controls; got {control_ids}"
    )
    return control_ids


def _control_ids_from_added_tokens(model: Path) -> dict[str, int]:
    path = model / "added_tokens.json"
    if not path.is_file():
        pytest.fail(f"configured SenseNova checkpoint has no added_tokens.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    control_ids = {token: int(payload[token]) for token in CONTROL_TOKENS if token in payload}
    assert set(control_ids) == set(CONTROL_TOKENS), (
        f"worker added_tokens.json must define both SenseNova image controls; got {control_ids}"
    )
    return control_ids


def _control_ids_from_tokenizer_config(model: Path) -> dict[str, int]:
    path = model / "tokenizer_config.json"
    if not path.is_file():
        pytest.fail(f"configured SenseNova checkpoint has no tokenizer_config.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    decoder = payload.get("added_tokens_decoder", {})
    control_ids = {
        str(spec["content"]): int(token_id)
        for token_id, spec in decoder.items()
        if isinstance(spec, dict) and spec.get("content") in CONTROL_TOKENS
    }
    assert set(control_ids) == set(CONTROL_TOKENS), (
        f"worker tokenizer_config.json must define both SenseNova image controls; got {control_ids}"
    )
    return control_ids


def active_sensenova_control_ids(model: Path) -> dict[str, int]:
    """Assert frontend and worker loaders resolve the active image controls alike."""
    frontend_ids = _control_ids_from_tokenizer_json(model)
    added_token_ids = _control_ids_from_added_tokens(model)
    tokenizer_config_ids = _control_ids_from_tokenizer_config(model)

    from transformers import AutoTokenizer

    worker_tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=False,
    )
    worker_ids = {
        token: int(worker_tokenizer.convert_tokens_to_ids(token)) for token in CONTROL_TOKENS
    }

    assert frontend_ids == added_token_ids == tokenizer_config_ids == worker_ids, (
        "active SenseNova checkpoint has divergent frontend/worker image-control IDs: "
        f"frontend={frontend_ids}, added_tokens={added_token_ids}, "
        f"tokenizer_config={tokenizer_config_ids}, worker={worker_ids}"
    )
    assert frontend_ids["<img>"] != frontend_ids["</img>"], (
        "SenseNova image start and end controls must remain distinct"
    )
    return frontend_ids


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
def configured_sim_server(
    tmp_path: Path,
    *,
    model: Path,
    description: str,
    served_model_name: str,
):
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
        description,
        "--served-model-name",
        served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--sim",
        "--worker-python",
        str(Path.cwd() / ".venv" / "bin" / "python"),
        "--device",
        "cpu",
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


@contextmanager
def sim_server(tmp_path: Path):
    with configured_sim_server(
        tmp_path,
        model=active_sensenova_model(),
        description="sensenova",
        served_model_name="SenseNova-U1",
    ) as base_url:
        yield base_url


def test_active_sensenova_checkpoint_image_controls_match_worker_tokenizer():
    active_sensenova_control_ids(active_sensenova_model())


def test_qwen3_public_chat_funnel(tmp_path: Path):
    with configured_sim_server(
        tmp_path,
        model=active_model(QWEN_MODEL_ENV),
        description="qwen3",
        served_model_name="Qwen3-32B",
    ) as base_url:
        assert_model_discovery(
            base_url,
            model_id="Qwen3-32B",
            description_id="qwen3",
            endpoints=["chat_completions"],
            input_modalities=["text"],
            output_modalities=["text"],
            features=["streaming", "usage", "logprobs", "reasoning", "tool_calling"],
        )
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "Qwen3-32B",
                "messages": [{"role": "user", "content": "Say hello."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a value.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
                "max_completion_tokens": 8,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        assert payload["model"] == "Qwen3-32B"
        assert payload["choices"][0]["message"]["content"]
        assert payload["choices"][0]["finish_reason"] in {"stop", "length"}

        unsupported_image_output = httpx.post(
            f"{base_url}/v1/images/generations",
            json={"model": "Qwen3-32B", "prompt": "Draw a square."},
            timeout=60,
        )
        assert unsupported_image_output.status_code == 400
        assert "image_output" in unsupported_image_output.json()["error"]["message"]

        unsupported_image_input = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "Qwen3-32B",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this."},
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
            timeout=60,
        )
        assert unsupported_image_input.status_code == 400
        assert "image_input" in unsupported_image_input.json()["error"]["message"]


def test_bagel_public_funnels(tmp_path: Path):
    with configured_sim_server(
        tmp_path,
        model=active_model(BAGEL_MODEL_ENV),
        description="bagel",
        served_model_name="BAGEL",
    ) as base_url:
        assert_model_discovery(
            base_url,
            model_id="BAGEL",
            description_id="bagel",
            endpoints=["chat_completions", "image_generations"],
            input_modalities=["text", "image"],
            output_modalities=["text", "image"],
            features=["streaming", "usage", "logprobs"],
        )
        unsupported_tools = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "BAGEL",
                "messages": [{"role": "user", "content": "Use a tool."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            timeout=60,
        )
        assert unsupported_tools.status_code == 400
        assert "tool_calling" in unsupported_tools.json()["error"]["message"]

        unsupported_reasoning = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "BAGEL",
                "messages": [{"role": "user", "content": "Reason about this."}],
                "reasoning_effort": "high",
            },
            timeout=60,
        )
        assert unsupported_reasoning.status_code == 400
        assert "reasoning" in unsupported_reasoning.json()["error"]["message"]

        image_to_text = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "BAGEL",
                "modalities": ["text"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{tiny_input_png_b64()}"
                                },
                            },
                        ],
                    }
                ],
                "max_completion_tokens": 8,
            },
            timeout=60,
        )
        image_to_text.raise_for_status()
        assert image_to_text.json()["choices"][0]["message"]["content"]

        invalid_count = httpx.post(
            f"{base_url}/v1/images/generations",
            json={"model": "BAGEL", "prompt": "A geometric landscape.", "n": 2},
            timeout=60,
        )
        assert invalid_count.status_code == 400
        assert invalid_count.json()["error"]["param"] == "n"

        response = httpx.post(
            f"{base_url}/v1/images/generations",
            json={"model": "BAGEL", "prompt": "A geometric landscape."},
            timeout=300,
        )
        response.raise_for_status()
        images = response.json()["data"]
        assert len(images) == 1
        for image in images:
            dimensions = (image["width"], image["height"])
            assert png_size_from_b64(image["b64_json"]) == dimensions
            assert image["bytes"] > 0
            assert len(image["sha256"]) == 64


def test_sim_http_configured_routes_and_harness_contract(tmp_path: Path):
    # CPU simulation emits deterministic text and image fixtures through the
    # configured HTTP, scheduler, generation-event, and geometry contracts.
    image_start_id = active_sensenova_control_ids(active_sensenova_model())["<img>"]
    with sim_server(tmp_path) as base_url:
        health_response = httpx.get(f"{base_url}/health", timeout=30)
        health_response.raise_for_status()

        metrics_response = httpx.get(f"{base_url}/metrics", timeout=30)
        metrics_response.raise_for_status()
        assert metrics_response.text

        version_response = httpx.get(f"{base_url}/version", timeout=30)
        version_response.raise_for_status()
        assert version_response.json()["version"]

        assert_model_discovery(
            base_url,
            model_id="SenseNova-U1",
            description_id="sensenova",
            endpoints=["chat_completions", "image_generations"],
            input_modalities=["text", "image"],
            output_modalities=["text", "image"],
            features=[
                "streaming",
                "usage",
                "logprobs",
                "reasoning",
                "repeated_interleave",
            ],
        )

        unknown_chat_control = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "SenseNova-U1",
                "messages": [{"role": "user", "content": "hello"}],
                "unsupported_option": True,
            },
            timeout=30,
        )
        assert unknown_chat_control.status_code == 400
        assert unknown_chat_control.json()["error"]["type"] == "invalid_request_error"

        unknown_image_control = httpx.post(
            f"{base_url}/v1/images/generations",
            json={"prompt": "draw", "response_format": "url"},
            timeout=30,
        )
        assert unknown_image_control.status_code == 400
        assert unknown_image_control.json()["error"]["type"] == "invalid_request_error"

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

        cancellation_id = "stream-cancellation"
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers={"X-Request-Id": cancellation_id},
            json={
                "model": "SenseNova-U1",
                "stream": True,
                "messages": [{"role": "user", "content": "Keep generating."}],
                "modalities": ["text"],
                "min_tokens": 512,
                "max_completion_tokens": 512,
            },
            timeout=60,
        ) as response:
            response.raise_for_status()
            assert any(
                line.startswith("data: ") and line != "data: [DONE]"
                for line in response.iter_lines()
            )

        followup_id = "post-cancellation"
        cancellation_followup = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers={"X-Request-Id": followup_id},
            json={
                "model": "SenseNova-U1",
                "messages": [{"role": "user", "content": "Confirm recovery."}],
                "modalities": ["text"],
                "max_completion_tokens": 16,
            },
            timeout=60,
        )
        cancellation_followup.raise_for_status()
        assert cancellation_followup.json()["id"] == f"chatcmpl-{followup_id}"

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
                "logit_bias": {str(image_start_id): 100.0},
                "image_config": {"num_images": 1},
            },
            timeout_s=300,
        )
        assert chat_sse_text(default_events)
        images = chat_sse_images(default_events)
        assert images
        image_b64 = str(images[0]["image_url"]["url"]).split(",", 1)[1]
        assert png_size_from_b64(image_b64) == (2048, 1152)

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

        final_metrics_response = httpx.get(f"{base_url}/metrics", timeout=30)
        final_metrics_response.raise_for_status()
        lifecycle = serving_lifecycle_metrics(final_metrics_response.text)
        assert set(lifecycle) == {
            "active",
            "accepted",
            "scheduled",
            "finished",
            "rejected",
            "cancelled",
            "aborted",
            "failed",
        }
        assert lifecycle["active"] == 0
        assert lifecycle["accepted"] > 0
        assert lifecycle["scheduled"] > 0
        assert lifecycle["finished"] > 0

        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            json.dumps(
                {
                    "id": "sensenova-interleave-contract",
                    "task": "interleave",
                    "prompt": "Generate a travel guide covering Sonoma, Sequoia, Tahoe, and the Golden Gate.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        spec = BenchmarkSpec(
            name="sensenova_interleave_contract",
            task=TaskName.INTERLEAVE,
            model="SenseNova-U1",
            server="test",
            metrics=(MetricDefinition(("mean_ttft_ms",), "lower"),),
            dataset="trace",
            dataset_path=str(trace_path),
            num_prompts=1,
            warmup_requests=0,
            max_tokens=8,
            minimum_average_images=1.0,
            extra_request_body={"logit_bias": {str(image_start_id): 100.0}},
        )
        result = asyncio.run(BenchmarkRunner(base_url, spec, tmp_path / "bench").run())
        assert result.summary["status"] == "completed"
        assert result.summary["failed_count"] == 0
        assert result.summary["ok_count"] == 1
        assert result.summary["validation"]["valid"] is True
        assert (tmp_path / "bench" / "summary.json").exists()
