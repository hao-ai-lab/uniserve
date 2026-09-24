"""Defines image-to-text benchmark request and validation behavior."""

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


class I2TTask(BenchmarkTask):
    """Builds multimodal prompts and validates their text completions."""

    name: ClassVar[TaskName] = TaskName.I2T
    default_stream: ClassVar[bool] = True
    # `accepts_image` gates the point's image-generation settings table, which
    # does not apply to text output. The input image comes from the dataset
    # row instead.
    accepts_image: ClassVar[bool] = False
    accepts_question: ClassVar[bool] = True
    image_count: ClassVar[ImageCountRule] = ImageCountRule.FORBIDDEN

    def build_request(self, example: Example) -> TaskRequest:
        """Build a chat request containing text and an embedded input image.

        The output limit is the row's ``max_tokens``, else the point's
        ``sampling.max_tokens``, else 512. Unlike the text and interleave
        tasks, this task honors ``sampling.stream`` and sends a non-streaming
        request when it is false.

        Raises:
            ValueError: If the row has no base64 image or an invalid MIME
                type.
        """
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
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.input_image_data_url(example)
                            },
                        },
                    ],
                }
            ],
            "modalities": ["text"],
            "max_completion_tokens": max_tokens,
        }
        self.apply_text_sampling(payload)

        # `validate_output` requires server-reported usage, which an
        # OpenAI-compatible stream carries only when `include_usage` is set.
        if sampling.stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return TaskRequest(self.point.endpoint, payload, stream=sampling.stream)

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Require authoritative usage and configured fixed-length output."""
        checks = {"server_usage": self.server_usage_ok(records)}
        if self.point.sampling.ignore_eos:
            checks["fixed_output_length"] = self.fixed_output_length_ok(records)
        return ValidationResult(checks=checks)
