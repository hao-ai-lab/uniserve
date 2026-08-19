from __future__ import annotations

from collections.abc import Sequence

from ..types import Example, RequestRecord, TaskRequest, ValidationResult
from .base import (
    BenchmarkTask,
    apply_text_sampling,
    fixed_output_length_check,
    input_image_data_url,
    server_usage_check,
)


class I2TTask(BenchmarkTask):
    def build_request(self, example: Example) -> TaskRequest:
        sampling = self.point.sampling
        max_tokens = int(
            example.max_tokens
            if example.max_tokens is not None
            else sampling.max_tokens or 512
        )
        payload: dict[str, object] = {
            "model": self.point.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example.prompt},
                        {"type": "image_url", "image_url": {"url": input_image_data_url(example)}},
                    ],
                }
            ],
            "modalities": ["text"],
            "max_completion_tokens": max_tokens,
        }
        apply_text_sampling(payload, sampling)
        if sampling.stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return TaskRequest(self.point.endpoint, payload, stream=sampling.stream)

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        checks = {"server_usage": server_usage_check(records)}
        if self.point.sampling.ignore_eos:
            checks["fixed_output_length"] = fixed_output_length_check(records)
        return ValidationResult(checks=checks)
