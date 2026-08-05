from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..metrics.common import RequestRecord
from ..validation import ValidationResult
from .base import (
    BenchmarkTask,
    TaskRequest,
    apply_chat_image_parameters,
    image_integrity_checks,
    input_image_data_url,
)


class I2ITask(BenchmarkTask):
    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        image_config: dict[str, Any] = {"seed": int(item.get("seed", self.spec.seed))}
        if self.spec.image_count is not None:
            image_config["num_images"] = self.spec.image_count
        if width is not None and height is not None:
            image_config.update(width=int(width), height=int(height))
        if self.spec.steps is not None:
            image_config["steps"] = self.spec.steps
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "modalities": ["image"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item["prompt"]},
                        {"type": "image_url", "image_url": {"url": input_image_data_url(item)}},
                    ],
                }
            ],
            "temperature": self.spec.temperature,
            "top_p": self.spec.top_p,
            "ignore_eos": self.spec.ignore_eos,
        }
        apply_chat_image_parameters(payload, image_config)
        payload.update(self.spec.extra_request_body)
        return TaskRequest(self.spec.endpoint, payload, "openai_chat_json")

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        checks = image_integrity_checks(records, width=self.spec.width, height=self.spec.height)
        checks["image_output"] = bool(records) and all(record.images > 0 for record in records)
        if self.spec.image_count is not None:
            checks["exact_image_count"] = all(
                record.images == self.spec.image_count for record in records
            )
        return ValidationResult(checks=checks)
