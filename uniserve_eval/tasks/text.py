from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ..types import Example, RequestRecord, TaskName, TaskRequest, ValidationResult
from .base import BenchmarkTask, ImageCountRule


class TextTask(BenchmarkTask):
    name: ClassVar[TaskName] = TaskName.TEXT
    default_stream: ClassVar[bool] = True
    accepts_image: ClassVar[bool] = False
    image_count: ClassVar[ImageCountRule] = ImageCountRule.FORBIDDEN

    def build_request(self, example: Example) -> TaskRequest:
        output_len = example.output_len if example.output_len is not None else self.point.sampling.max_tokens
        payload: dict[str, object] = {
            "model": self.point.model,
            "messages": example.messages
            or [{"role": "user", "content": example.prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        self.apply_text_sampling(payload)
        if output_len is not None:
            payload["max_completion_tokens"] = int(output_len)
        return TaskRequest(self.point.endpoint, payload, stream=True)

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        checks = {"server_usage": self.server_usage_ok(records)}
        if self.point.sampling.ignore_eos:
            checks["fixed_output_length"] = self.fixed_output_length_ok(records)
        return ValidationResult(checks=checks)
