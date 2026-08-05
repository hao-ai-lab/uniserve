from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..metrics.common import RequestRecord
from ..validation import ValidationResult
from .base import BenchmarkTask, TaskRequest, apply_text_sampling


class TextTask(BenchmarkTask):
    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        output_len = item.get("output_len", self.spec.max_tokens)
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": item.get("messages")
            or [{"role": "user", "content": item["prompt"]}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        apply_text_sampling(payload, self.spec)
        if output_len is not None:
            payload["max_completion_tokens"] = int(output_len)
        payload.update(self.spec.extra_request_body)
        return TaskRequest(self.spec.endpoint, payload, "openai_chat")

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        server_usage = bool(records) and all(
            record.output_len_source == "server_usage" and record.prompt_len_source == "server_usage"
            for record in records
        )
        checks = {"server_usage": server_usage}
        if self.spec.ignore_eos:
            checks["fixed_output_length"] = bool(records) and all(
                record.requested_output_len > 0
                and record.output_len == record.requested_output_len
                and record.finish_reason == "length"
                for record in records
            )
        return ValidationResult(checks=checks)
