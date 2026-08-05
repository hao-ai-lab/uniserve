from __future__ import annotations

import io

import pytest
from PIL import Image

from uniserve_eval.harness.image_outputs import inspect_image_bytes
from uniserve_eval.harness.metrics.common import RequestRecord
from uniserve_eval.harness.spec import BenchmarkSpec, MetricDefinition, TaskName
from uniserve_eval.harness.tasks.interleave import InterleaveTask
from uniserve_eval.harness.tasks.t2i import T2ITask

pytestmark = pytest.mark.unit


def _image(width: int = 2, height: int = 3):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return inspect_image_bytes(buffer.getvalue())


def _metric(path: str = "images_per_second", direction: str = "higher") -> MetricDefinition:
    return MetricDefinition(tuple(path.split(".")), direction)  # type: ignore[arg-type]


def test_t2i_requests_and_validation_use_exact_image_count() -> None:
    spec = BenchmarkSpec(
        name="t2i",
        task=TaskName.T2I,
        model="model",
        server="server",
        metrics=(_metric(),),
        num_prompts=2,
        image_count=1,
        width=2,
        height=3,
        wire="openai_chat_json",
    )
    task = T2ITask(spec)
    request = task.build_request({"prompt": "draw"})
    assert request.payload["image_config"]["num_images"] == 1

    image = _image()
    records = [
        RequestRecord(
            request_id=str(index),
            task="t2i",
            success=True,
            images=1,
            decoded_images=[image],
        )
        for index in range(2)
    ]
    assert task.validate(records).valid is True
    records[1].images = 0
    records[1].decoded_images = []
    assert task.validate(records).checks["exact_image_count"] is False


def test_interleave_sends_no_count_and_checks_point_average() -> None:
    spec = BenchmarkSpec(
        name="interleave",
        task=TaskName.INTERLEAVE,
        model="model",
        server="server",
        metrics=(_metric("mean_ttft_ms", "lower"),),
        num_prompts=2,
        minimum_average_images=1.0,
        width=2,
        height=3,
    )
    task = InterleaveTask(spec)
    request = task.build_request({"prompt": "travel"})
    assert "num_images" not in request.payload["image_config"]

    image = _image()
    text_only = RequestRecord(
        request_id="text",
        task="interleave",
        success=True,
        generated_text="answer",
        output_modalities=["text"],
        modality_events=[{"modalities": ["text"], "client_time": 1.0}],
        output_len_source="server_usage",
        prompt_len_source="server_usage",
    )
    multimodal = RequestRecord(
        request_id="mixed",
        task="interleave",
        success=True,
        generated_text="answer",
        output_modalities=["text", "image"],
        modality_events=[
            {"modalities": ["text"], "client_time": 1.0},
            {"modalities": ["image"], "client_time": 2.0},
        ],
        output_len_source="server_usage",
        prompt_len_source="server_usage",
        images=2,
        decoded_images=[image, image],
    )
    validation = task.validate([text_only, multimodal])
    assert validation.valid is True
    assert validation.statistics["images_per_request"] == 1.0
    assert validation.statistics["zero_image_requests"] == 1

    multimodal.images = 1
    multimodal.decoded_images = [image]
    assert task.validate([text_only, multimodal]).checks["minimum_average_images"] is False
