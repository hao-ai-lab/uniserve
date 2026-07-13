from __future__ import annotations

import asyncio
import base64
import json
from io import BytesIO

import pytest
from PIL import Image

import uniserve_eval.harness.runner as harness_runner
from uniserve_eval.harness.core.client import _parse_openai, send_request
from uniserve_eval.harness.datasets import BenchmarkInputs
from uniserve_eval.harness.image_outputs import ImageOutputError, inspect_image_bytes
from uniserve_eval.harness.metrics.common import RequestRecord
from uniserve_eval.harness.report import (
    artifact_bundle_matches,
    attach_execution_contract,
    benchmark_contract,
    canonical_artifact_bundle_matches,
    write_summary_artifacts,
)
from uniserve_eval.harness.runner import BenchmarkRunner
from uniserve_eval.harness.spec import BenchmarkSpec, TaskName
from uniserve_eval.harness.tasks.base import TaskRequest

pytestmark = pytest.mark.unit


def _image_bytes(*, width: int = 2, height: int = 3, image_format: str = "PNG") -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), (29, 43, 71)).save(output, format=image_format)
    return output.getvalue()


def _encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class _JsonResponse:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def json(self) -> dict:
        return self.payload


class _JsonClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def post(self, _url: str, *, json: dict) -> _JsonResponse:
        del json
        return _JsonResponse(self.payload)


def test_images_generations_decodes_exact_bytes_and_records_only_metadata() -> None:
    image_bytes = _image_bytes()
    request = TaskRequest(
        endpoint="/v1/images/generations",
        kind="images_generations",
        payload={"prompt": "x", "size": "2x3", "n": 1},
    )

    record = asyncio.run(
        send_request(
            _JsonClient({"data": [{"b64_json": _encoded(image_bytes)}]}),
            "http://server",
            request,
            "image-1",
            task="t2i",
        )
    )

    assert record.success is True
    assert record.images == 1
    assert record.decoded_images[0].data == image_bytes
    assert record.decoded_images[0].mime == "image/png"
    assert (record.decoded_images[0].width, record.decoded_images[0].height) == (2, 3)
    persisted = record.record_dict()
    assert persisted["image_outputs"] == [record.decoded_images[0].metadata_dict()]
    assert _encoded(image_bytes) not in json.dumps(persisted)


def test_chat_json_image_part_must_match_declared_count_and_dimensions() -> None:
    image_bytes = _image_bytes(width=2, height=3)
    request = TaskRequest(
        endpoint="/v1/chat/completions",
        kind="openai_chat_json",
        payload={
            "modalities": ["image"],
            "image_config": {"num_images": 1, "width": 2, "height": 4},
        },
    )
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_encoded(image_bytes)}"},
                        }
                    ]
                },
            }
        ]
    }

    record = asyncio.run(
        send_request(
            _JsonClient(response),
            "http://server",
            request,
            "image-1",
            task="t2i",
        )
    )

    assert record.success is False
    assert record.classifier == "protocol_image_dimensions_mismatch"
    assert record.images == 1


def test_streaming_image_delta_uses_the_same_decoder_and_rejects_mime_mismatch() -> None:
    png = _image_bytes()
    record = RequestRecord(
        request_id="image-1",
        task="default",
        generated_images_expected=True,
        requested_image_count=1,
        requested_image_width=2,
        requested_image_height=3,
        start_time=10.0,
    )
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{_encoded(png)}"},
                            }
                        ]
                    }
                }
            ],
            "_client_t": 11.0,
        },
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"type": "sse_done"},
    ]

    _parse_openai(events, record, output_len_fallback=0, prompt_len=0)

    assert record.success is False
    assert record.classifier == "protocol_image_mime_mismatch"
    assert record.decoded_images == []


def test_decoder_rejects_malformed_base64_and_non_images() -> None:
    request = TaskRequest(
        endpoint="/v1/images/generations",
        kind="images_generations",
        payload={"prompt": "x"},
    )
    malformed = asyncio.run(
        send_request(
            _JsonClient({"data": [{"b64_json": "not-base64"}]}),
            "http://server",
            request,
            "image-1",
            task="t2i",
        )
    )
    assert malformed.success is False
    assert malformed.classifier == "protocol_invalid_image_base64"

    with pytest.raises(ImageOutputError, match="protocol_undecodable_image"):
        inspect_image_bytes(b"not an image")


def test_runner_commits_exact_samples_and_bundle_rejects_file_drift(tmp_path, monkeypatch) -> None:
    stale_path = tmp_path / "samples" / "stale.png"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_bytes(b"stale")
    rows = [{"id": "image-1", "prompt": "x", "width": 2, "height": 3}]
    image_bytes = _image_bytes()
    image = inspect_image_bytes(image_bytes)
    spec = BenchmarkSpec(
        task=TaskName.T2I,
        model="M",
        num_prompts=1,
        warmup_requests=0,
        width=2,
        height=3,
        max_images=1,
        sample_gpu_memory=False,
    )

    monkeypatch.setattr(
        harness_runner,
        "load_benchmark_inputs",
        lambda _spec: BenchmarkInputs(measured=rows, warmup=[], tokenizer=None),
    )

    async def fake_submit(self, _client, row):
        del self, row
        return RequestRecord(
            request_id="image-1",
            task="t2i",
            success=True,
            classifier="ok",
            latency=1.0,
            images=1,
            generated_images_expected=True,
            requested_image_count=1,
            requested_image_width=2,
            requested_image_height=3,
            decoded_images=[image],
        )

    async def fake_run_load(items, **kwargs):
        return [await kwargs["submit"](items[0])], 1.0

    async def fake_plan(self, _client, _rows):
        return {"source": "declared_contract", "plan": harness_runner.plan_summary(self.spec)}

    async def fake_server_info(self, _client):
        del self
        return None

    monkeypatch.setattr(BenchmarkRunner, "_submit", fake_submit)
    monkeypatch.setattr(BenchmarkRunner, "_collect_plan_evidence", fake_plan)
    monkeypatch.setattr(BenchmarkRunner, "_fetch_server_info", fake_server_info)
    monkeypatch.setattr(harness_runner, "run_load", fake_run_load)

    result = asyncio.run(BenchmarkRunner("http://server", spec, tmp_path).run())
    contract = benchmark_contract(spec, rows)
    sample_path = tmp_path / "samples" / image.sample_filename

    assert not stale_path.exists()
    assert sample_path.read_bytes() == image_bytes
    assert artifact_bundle_matches(tmp_path, result.summary, contract)
    request_json = (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    assert _encoded(image_bytes) not in request_json

    attach_execution_contract(
        result.summary,
        "matrix_contract",
        {"schema_version": 2, "fingerprint": "a" * 64},
    )
    write_summary_artifacts(tmp_path, result.summary)
    assert canonical_artifact_bundle_matches(tmp_path, result.summary, contract)

    (tmp_path / "samples" / "unbound.png").write_bytes(image_bytes)
    assert not canonical_artifact_bundle_matches(tmp_path, result.summary, contract)
    (tmp_path / "samples" / "unbound.png").unlink()
    sample_path.write_bytes(b"mutated")
    assert not canonical_artifact_bundle_matches(tmp_path, result.summary, contract)
    sample_path.write_bytes(image_bytes)
    sample_path.unlink()
    assert not canonical_artifact_bundle_matches(tmp_path, result.summary, contract)
