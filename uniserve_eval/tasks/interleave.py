from __future__ import annotations

from collections.abc import Sequence

from ..types import Example, RequestRecord, TaskRequest, ValidationResult
from .base import (
    BenchmarkTask,
    apply_text_sampling,
    image_integrity_checks,
    render_image_config,
    server_usage_check,
)

MINIMUM_AVERAGE_IMAGES = 1.1


class InterleaveTask(BenchmarkTask):
    def build_request(self, example: Example) -> TaskRequest:
        image_config = render_image_config(
            self.point.image,
            example,
            include_count=False,
            fallback_seed=self.point.load.seed,
        )
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
            "image_config": image_config,
        }
        apply_text_sampling(payload, self.point.sampling)
        return TaskRequest(self.point.endpoint, payload, stream=True)

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        checks = image_integrity_checks(records, self.point.image)
        checks["visible_text"] = bool(records) and all(record.generated_text for record in records)
        checks["server_usage"] = server_usage_check(records)
        total_images = sum(record.images for record in records)
        mean_images = total_images / len(records) if records else 0.0
        checks["minimum_average_images"] = mean_images >= MINIMUM_AVERAGE_IMAGES
        return ValidationResult(
            checks=checks,
            statistics={
                "completed_images": total_images,
                "images_per_request": mean_images,
                "zero_image_requests": sum(record.images == 0 for record in records),
            },
        )
