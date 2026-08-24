from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ..types import Example, RequestRecord, TaskName, TaskRequest, ValidationResult
from .base import BenchmarkTask, ImageCountRule


class InterleaveTask(BenchmarkTask):
    name: ClassVar[TaskName] = TaskName.INTERLEAVE
    default_stream: ClassVar[bool] = True
    accepts_image: ClassVar[bool] = True
    image_count: ClassVar[ImageCountRule] = ImageCountRule.FORBIDDEN
    minimum_average_images: ClassVar[float] = 1.1

    def build_request(self, example: Example) -> TaskRequest:
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

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        checks = self.image_integrity_checks(records)
        checks["visible_text"] = bool(records) and all(record.generated_text for record in records)
        checks["server_usage"] = self.server_usage_ok(records)
        total_images = sum(record.images for record in records)
        mean_images = total_images / len(records) if records else 0.0
        checks["minimum_average_images"] = mean_images >= self.minimum_average_images
        return ValidationResult(
            checks=checks,
            statistics={
                "completed_images": total_images,
                "images_per_request": mean_images,
                "zero_image_requests": sum(record.images == 0 for record in records),
            },
        )
