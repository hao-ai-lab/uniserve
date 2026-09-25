"""Defines streamed text-completion benchmark behavior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ..types import (
    Example,
    RequestRecord,
    TaskName,
    TaskRequest,
    ValidationResult,
)
from .base import BenchmarkTask, ImageCountRule


class TextTask(BenchmarkTask):
    """Builds and validates deterministic streamed text completions."""

    name: ClassVar[TaskName] = TaskName.TEXT
    default_stream: ClassVar[bool] = True
    accepts_image: ClassVar[bool] = False
    image_count: ClassVar[ImageCountRule] = ImageCountRule.FORBIDDEN

    def build_request(self, example: Example) -> TaskRequest:
        """Build a chat-completions request with usage-bearing streaming.

        The request always streams, regardless of ``sampling.stream``. A
        row's non-empty ``messages`` replace its ``prompt`` as the
        conversation. The output limit is the row's ``output_len``, else the
        point's ``sampling.max_tokens``; with neither,
        ``max_completion_tokens`` is omitted. As in every chat task,
        ``sampling.extra_body`` overrides any of these fields.
        """
        output_len = (
            example.output_len
            if example.output_len is not None
            else self.point.sampling.max_tokens
        )
        payload: dict[str, object] = {
            "model": self.point.model,
            "messages": example.messages
            or [{"role": "user", "content": example.prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if output_len is not None:
            payload["max_completion_tokens"] = int(output_len)
        self.apply_text_sampling(payload)
        return TaskRequest(self.point.endpoint, payload, stream=True)

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Require authoritative usage and configured fixed-length output."""
        checks = {"server_usage": self.server_usage_ok(records)}
        if self.point.sampling.ignore_eos:
            checks["fixed_output_length"] = self.fixed_output_length_ok(records)
        return ValidationResult(checks=checks)
