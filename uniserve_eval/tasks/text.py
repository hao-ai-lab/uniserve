from __future__ import annotations

from collections.abc import Sequence

from ..types import Example, RequestRecord, TaskRequest, ValidationResult
from .base import (
    BenchmarkTask,
    apply_text_sampling,
    fixed_output_length_check,
    server_usage_check,
)


class TextTask(BenchmarkTask):
    def build_request(self, example: Example) -> TaskRequest:
        output_len = example.output_len if example.output_len is not None else self.point.sampling.max_tokens
        payload: dict[str, object] = {
            "model": self.point.model,
            "messages": example.messages
            or [{"role": "user", "content": example.prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        apply_text_sampling(payload, self.point.sampling)
        if output_len is not None:
            payload["max_completion_tokens"] = int(output_len)
        return TaskRequest(self.point.endpoint, payload, stream=True)

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        checks = {"server_usage": server_usage_check(records)}
        if self.point.sampling.ignore_eos:
            checks["fixed_output_length"] = fixed_output_length_check(records)
        return ValidationResult(checks=checks)
