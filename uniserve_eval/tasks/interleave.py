"""Defines interleaved text-and-image benchmark behavior."""

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


class InterleaveTask(BenchmarkTask):
    """Builds streamed multimodal requests and validates mixed output."""

    name: ClassVar[TaskName] = TaskName.INTERLEAVE
    default_stream: ClassVar[bool] = True
    accepts_image: ClassVar[bool] = True
    image_count: ClassVar[ImageCountRule] = ImageCountRule.FORBIDDEN

    def build_request(self, example: Example) -> TaskRequest:
        """Build a chat request that streams both text and images.

        The request always streams, regardless of ``sampling.stream``. The
        output limit is the row's ``max_tokens``, else the point's
        ``sampling.max_tokens``, else 512. ``image_config`` never carries
        ``num_images`` because this task forbids a configured image count.
        """
        max_tokens = int(
            example.max_tokens
            if example.max_tokens is not None
            else self.point.sampling.max_tokens or 512
        )
        payload: dict[str, object] = {
            "model": self.point.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "modalities": ["text", "image"],
            "messages": [{"role": "user", "content": example.prompt}],
            "max_completion_tokens": max_tokens,
            "image_config": self.image_fields(example, include_count=False),
        }
        self.apply_text_sampling(payload)
        return TaskRequest(self.point.endpoint, payload, stream=True)

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Check visible text, image integrity, usage, and that images appear.

        How many images an answer holds is the model's choice for each prompt:
        the request carries no image count and the dataset states none, so
        the run is valid when interleaving was exercised, that is, when some
        request produced an image. Requests without one are reported as the
        `zero_image_requests` statistic.
        """
        checks = self.image_integrity_checks(records)
        checks["visible_text"] = bool(records) and all(
            record.generated_text for record in records
        )
        checks["server_usage"] = self.server_usage_ok(records)

        total_images = sum(record.images for record in records)
        mean_images = total_images / len(records) if records else 0.0
        checks["produced_images"] = total_images > 0
        return ValidationResult(
            checks=checks,
            statistics={
                "completed_images": total_images,
                "images_per_request": mean_images,
                "zero_image_requests": sum(
                    record.images == 0 for record in records
                ),
            },
        )
