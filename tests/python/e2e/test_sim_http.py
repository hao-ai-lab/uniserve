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
CONTROL_TOKENS = ("<img>", "</img>")


def active_sensenova_model() -> Path:
    model_value = os.environ.get(MODEL_ENV)
    if not model_value:
        pytest.skip(f"{MODEL_ENV} is required for the sim HTTP gate")
    model = Path(model_value)
    if not model.exists():
        pytest.fail(f"configured SenseNova checkpoint is missing: {model}")
    return model


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
def sim_server(tmp_path: Path):
    try:
        binary = require_uniserve_binary()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    model = active_sensenova_model()
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


def test_active_sensenova_checkpoint_image_controls_match_worker_tokenizer():
    active_sensenova_control_ids(active_sensenova_model())


def test_sim_http_configured_routes_and_benchmark_smoke(tmp_path: Path):
    # CPU simulation emits deterministic text and image fixtures through the
    # production HTTP, scheduler, worker IPC, and geometry contracts.
    image_start_id = active_sensenova_control_ids(active_sensenova_model())["<img>"]
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
                    "task": "interleave",
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
            max_images=1,
            extra_request_body={"logit_bias": {str(image_start_id): 100.0}},
            runtime_profile_id="sensenova-u1",
        )
        result = asyncio.run(BenchmarkRunner(base_url, spec, tmp_path / "bench").run())
        assert result.summary["harness_status"] == "completed"
        assert result.summary["failed_count"] == 0
        assert result.summary["ok_count"] == 1
        assert result.summary["artifact"]["plan_evidence"]["source"] == "declared_contract"
        plan = result.summary["artifact"]["plan_summary"]
        assert plan["runtime_profile_id"] == "sensenova-u1"
        assert plan["generation"]["temperature"] == 0.0
        assert plan["generation"]["top_p"] == 1.0
        assert plan["generation"]["ignore_eos"] is True
        assert plan["image"]["max_images"] == 1
        assert (tmp_path / "bench" / "summary.json").exists()
