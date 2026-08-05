from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..metrics.common import RequestRecord
from ..validation import ValidationResult
from .base import BenchmarkTask, RequestKind, TaskRequest, apply_text_sampling, input_image_data_url


class I2TTask(BenchmarkTask):
    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        max_tokens = int(item.get("max_tokens", self.spec.max_tokens or 512))
        streamed = self.spec.wire == "openai_chat"
        kind: RequestKind = "openai_chat" if streamed else "openai_chat_json"
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item["prompt"]},
                        {"type": "image_url", "image_url": {"url": input_image_data_url(item)}},
                    ],
                }
            ],
            "modalities": ["text"],
            "max_completion_tokens": max_tokens,
        }
        apply_text_sampling(payload, self.spec)
        if streamed:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.spec.extra_request_body)
        return TaskRequest(self.spec.endpoint, payload, kind)

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        checks = {
            "server_usage": bool(records)
            and all(
                record.output_len_source == "server_usage"
                and record.prompt_len_source == "server_usage"
                for record in records
            )
        }
        if self.spec.ignore_eos:
            checks["fixed_output_length"] = bool(records) and all(
                record.output_len == record.requested_output_len and record.finish_reason == "length"
                for record in records
            )
        return ValidationResult(checks=checks)
