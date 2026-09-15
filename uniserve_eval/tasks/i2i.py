"""Defines image-to-image benchmark request and validation behavior."""

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


class I2ITask(BenchmarkTask):
    """Builds image-editing chat requests and validates decoded images."""

    name: ClassVar[TaskName] = TaskName.I2I
    default_stream: ClassVar[bool] = False
    accepts_image: ClassVar[bool] = True
    image_count: ClassVar[ImageCountRule] = ImageCountRule.OPTIONAL

    def build_request(self, example: Example) -> TaskRequest:
        """Build a non-streaming image request with an embedded source image."""
        payload: dict[str, object] = {
            "model": self.point.model,
            "modalities": ["image"],
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
            "image_config": self.image_fields(
                example,
                include_count=self.point.image.image_count is not None,
            ),
        }
        self.apply_text_sampling(payload)
        return TaskRequest(self.point.endpoint, payload, stream=False)

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Check image presence, integrity, geometry, and configured count."""
        checks = self.image_integrity_checks(records)
        checks["image_output"] = bool(records) and all(
            record.images > 0 for record in records
        )
        if self.point.image.image_count is not None:
            checks["exact_image_count"] = all(
                record.images == self.point.image.image_count
                for record in records
            )
        return ValidationResult(checks=checks)
