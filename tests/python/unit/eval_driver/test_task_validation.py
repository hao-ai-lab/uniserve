from __future__ import annotations

import io

import pytest
from PIL import Image

from uniserve_eval.tasks.interleave import InterleaveTask
from uniserve_eval.tasks.t2i import T2ITask
from uniserve_eval.transport.images import inspect_image_bytes
from uniserve_eval.types import (
    BenchmarkPoint,
    Example,
    ImageConfig,
    LoadConfig,
    MetricDefinition,
    RequestRecord,
    SamplingConfig,
    TaskName,
)

pytestmark = pytest.mark.unit


def _image(width: int = 2, height: int = 3):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return inspect_image_bytes(buffer.getvalue())


def _metric(path: str = "images_per_second", direction: str = "higher") -> MetricDefinition:
    return MetricDefinition(tuple(path.split(".")), direction)  # type: ignore[arg-type]


def test_t2i_requests_and_validation_use_exact_image_count() -> None:
    point = BenchmarkPoint(
        name="t2i",
        server="server",
        task=TaskName.T2I,
        model="model",
        dataset="mjhq",
        metrics=(_metric(),),
        load=LoadConfig(num_prompts=2),
        sampling=SamplingConfig(stream=False),
        image=ImageConfig(image_count=1, width=2, height=3),
    )
    task = T2ITask(point)
    request = task.build_request(Example(id="row", prompt="draw"))
    assert request.stream is False
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
    point = BenchmarkPoint(
        name="interleave",
        server="server",
        task=TaskName.INTERLEAVE,
        model="model",
        dataset="ueval",
        metrics=(_metric("mean_ttft_ms", "lower"),),
        load=LoadConfig(num_prompts=2),
        sampling=SamplingConfig(stream=True),
        image=ImageConfig(width=2, height=3),
    )
    task = InterleaveTask(point)
    request = task.build_request(Example(id="row", prompt="travel"))
    assert request.stream is True
    assert "num_images" not in request.payload["image_config"]

    image = _image()
    text_only = RequestRecord(
        request_id="text",
        task="interleave",
        success=True,
        generated_text="answer",
        output_len_source="server_usage",
        prompt_len_source="server_usage",
    )
    multimodal = RequestRecord(
        request_id="mixed",
        task="interleave",
        success=True,
        generated_text="answer",
        output_len_source="server_usage",
        prompt_len_source="server_usage",
        images=3,
        decoded_images=[image, image, image],
    )
    validation = task.validate([text_only, multimodal])
    assert validation.valid is True
    assert validation.statistics["images_per_request"] == 1.5
    assert validation.statistics["zero_image_requests"] == 1

    multimodal.images = 2
    multimodal.decoded_images = [image, image]
    assert task.validate([text_only, multimodal]).checks["minimum_average_images"] is False
