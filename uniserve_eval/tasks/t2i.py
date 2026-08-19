from __future__ import annotations

from collections.abc import Sequence

from ..types import (
    IMAGES_GENERATIONS,
    Example,
    RequestRecord,
    TaskRequest,
    ValidationResult,
)
from .base import (
    BenchmarkTask,
    apply_image_generations_fields,
    apply_text_sampling,
    image_integrity_checks,
    render_image_config,
)


class T2ITask(BenchmarkTask):
    def build_request(self, example: Example) -> TaskRequest:
        count = int(self.point.image.image_count or 0)
        if self.point.endpoint == IMAGES_GENERATIONS:
            payload: dict[str, object] = {
                "model": self.point.model,
                "prompt": example.prompt,
                "n": count,
            }
            apply_image_generations_fields(
                payload,
                self.point.image,
                example,
                fallback_seed=self.point.load.seed,
            )
            payload.update(self.point.sampling.extra_body)
            return TaskRequest(self.point.endpoint, payload, stream=False)

        image_config = render_image_config(
            self.point.image,
            example,
            include_count=True,
            fallback_seed=self.point.load.seed,
        )
        payload = {
            "model": self.point.model,
            "modalities": ["image"],
            "messages": [{"role": "user", "content": example.prompt}],
            "image_config": image_config,
        }
        apply_text_sampling(payload, self.point.sampling)
        return TaskRequest(self.point.endpoint, payload, stream=False)

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        count = int(self.point.image.image_count or 0)
        checks = image_integrity_checks(records, self.point.image)
        checks["exact_image_count"] = bool(records) and all(
            record.images == count for record in records
        )
        total_images = sum(record.images for record in records)
        return ValidationResult(
            checks=checks,
            statistics={
                "completed_images": total_images,
                "images_per_request": total_images / len(records) if records else 0.0,
            },
        )
