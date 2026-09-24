"""Defines text-to-image behavior for supported public endpoint schemas."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ..types import (
    CHAT_COMPLETIONS,
    IMAGES_GENERATIONS,
    Example,
    RequestRecord,
    TaskName,
    TaskRequest,
    ValidationResult,
)
from .base import BenchmarkTask, ImageCountRule


class T2ITask(BenchmarkTask):
    """Builds image-generation requests and validates decoded outputs."""

    name: ClassVar[TaskName] = TaskName.T2I
    allowed_endpoints: ClassVar[tuple[str, ...]] = (
        CHAT_COMPLETIONS,
        IMAGES_GENERATIONS,
    )
    default_endpoint: ClassVar[str] = CHAT_COMPLETIONS
    default_stream: ClassVar[bool] = False
    accepts_image: ClassVar[bool] = True
    image_count: ClassVar[ImageCountRule] = ImageCountRule.REQUIRED

    def build_request(self, example: Example) -> TaskRequest:
        """Build a request for the selected chat or image endpoint."""
        # Profile parsing runs `check_image`, which rejects a T2I point without
        # `image_count`; `or 0` only narrows the optional type.
        count = int(self.point.image.image_count or 0)

        # The images endpoint receives no text-sampling fields: resolved image
        # settings map onto its schema and `extra_body` is merged last.
        if self.point.endpoint == IMAGES_GENERATIONS:
            payload: dict[str, object] = {
                "model": self.point.model,
                "prompt": example.prompt,
                "n": count,
            }
            self.apply_image_generations_fields(payload, example)
            payload.update(self.point.sampling.extra_body)
            return TaskRequest(self.point.endpoint, payload, stream=False)

        payload = {
            "model": self.point.model,
            "modalities": ["image"],
            "messages": [{"role": "user", "content": example.prompt}],
            "image_config": self.image_fields(example, include_count=True),
        }
        self.apply_text_sampling(payload)
        return TaskRequest(self.point.endpoint, payload, stream=False)

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Check image integrity and exact configured output count.

        Decoded sizes are compared with the point's configured width and
        height; per-row size overrides are not considered.
        """
        count = int(self.point.image.image_count or 0)
        checks = self.image_integrity_checks(records)
        checks["exact_image_count"] = bool(records) and all(
            record.images == count for record in records
        )
        total_images = sum(record.images for record in records)
        return ValidationResult(
            checks=checks,
            statistics={
                "completed_images": total_images,
                "images_per_request": total_images / len(records)
                if records
                else 0.0,
            },
        )
