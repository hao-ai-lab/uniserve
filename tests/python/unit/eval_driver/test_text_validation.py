from __future__ import annotations

import pytest

from uniserve_eval.tasks.i2t import I2TTask
from uniserve_eval.tasks.text import TextTask
from uniserve_eval.types import (
    BenchmarkPoint,
    Example,
    LoadConfig,
    MetricDefinition,
    RequestRecord,
    SamplingConfig,
    TaskName,
)

pytestmark = pytest.mark.unit


def _metric() -> MetricDefinition:
    return MetricDefinition(("output_throughput",), "higher")


def test_text_ignore_eos_requires_fixed_length_and_server_usage() -> None:
    point = BenchmarkPoint(
        name="text",
        server="server",
        task=TaskName.TEXT,
        model="model",
        dataset="sharegpt",
        tokenizer="model",
        metrics=(_metric(),),
        load=LoadConfig(num_prompts=1),
        sampling=SamplingConfig(ignore_eos=True, stream=True),
    )
    task = TextTask(point)
    request = task.build_request(Example(id="row", prompt="hello", output_len=8))
    assert request.stream is True
    assert request.payload["max_completion_tokens"] == 8
    assert request.payload["ignore_eos"] is True

    record = RequestRecord(
        request_id="row",
        task="text",
        success=True,
        requested_output_len=8,
        output_len=8,
        finish_reason="length",
        output_len_source="server_usage",
        prompt_len_source="server_usage",
    )
    assert task.validate([record]).valid is True
    record.output_len = 3
    record.finish_reason = "stop"
    assert task.validate([record]).checks["fixed_output_length"] is False


def test_i2t_natural_eos_requires_server_usage_only() -> None:
    point = BenchmarkPoint(
        name="i2t",
        server="server",
        task=TaskName.I2T,
        model="model",
        dataset="beans",
        metrics=(_metric(),),
        load=LoadConfig(num_prompts=1),
        sampling=SamplingConfig(ignore_eos=False, stream=False, max_tokens=16),
    )
    task = I2TTask(point)
    request = task.build_request(
        Example(
            id="row",
            prompt="Describe this image in detail.",
            input_image_b64="abc",
            input_image_mime="image/jpeg",
        )
    )
    assert request.stream is False
    assert "stream" not in request.payload

    record = RequestRecord(
        request_id="row",
        task="i2t",
        success=True,
        output_len=4,
        requested_output_len=16,
        finish_reason="stop",
        output_len_source="server_usage",
        prompt_len_source="server_usage",
    )
    assert task.validate([record]).valid is True
    assert "fixed_output_length" not in task.validate([record]).checks
