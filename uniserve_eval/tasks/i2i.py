from __future__ import annotations

from collections.abc import Sequence

from ..types import Example, RequestRecord, TaskRequest, ValidationResult
from .base import (
    BenchmarkTask,
    apply_text_sampling,
    image_integrity_checks,
    input_image_data_url,
    render_image_config,
)


class I2ITask(BenchmarkTask):
    def build_request(self, example: Example) -> TaskRequest:
        image_config = render_image_config(
            self.point.image,
            example,
            include_count=self.point.image.image_count is not None,
            fallback_seed=self.point.load.seed,
        )
        payload: dict[str, object] = {
            "model": self.point.model,
            "modalities": ["image"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example.prompt},
                        {"type": "image_url", "image_url": {"url": input_image_data_url(example)}},
                    ],
                }
            ],
            "image_config": image_config,
        }
        apply_text_sampling(payload, self.point.sampling)
        return TaskRequest(self.point.endpoint, payload, stream=False)

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        checks = image_integrity_checks(records, self.point.image)
        checks["image_output"] = bool(records) and all(record.images > 0 for record in records)
        if self.point.image.image_count is not None:
            checks["exact_image_count"] = all(
                record.images == self.point.image.image_count for record in records
            )
        return ValidationResult(checks=checks)
