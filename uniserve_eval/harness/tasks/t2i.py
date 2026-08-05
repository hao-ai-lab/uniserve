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
)


class T2ITask(BenchmarkTask):
    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        steps = item.get("steps", self.spec.steps)
        seed = item.get("seed", self.spec.seed)
        count = int(self.spec.image_count or 0)

        if self.spec.wire == "openai_chat_json":
            image_config = self._image_config(width, height, steps, seed)
            image_config["num_images"] = count
            payload: dict[str, Any] = {
                "model": self.spec.model,
                "modalities": ["image"],
                "messages": [{"role": "user", "content": item["prompt"]}],
                "temperature": self.spec.temperature,
                "top_p": self.spec.top_p,
                "ignore_eos": self.spec.ignore_eos,
            }
            apply_chat_image_parameters(payload, image_config)
            payload.update(self.spec.extra_request_body)
            return TaskRequest(
                endpoint=self.spec.endpoint,
                payload=payload,
                kind="openai_chat_json",
            )

        payload = {"model": self.spec.model, "prompt": item["prompt"], "n": count}
        if width is not None and height is not None:
            payload["size"] = f"{width}x{height}"
        if steps is not None:
            payload["steps"] = int(steps)
        if seed is not None:
            payload["seed"] = int(seed)
        self._apply_quality(payload)
        payload.update(self.spec.extra_request_body)
        return TaskRequest(
            endpoint=self.spec.endpoint,
            payload=payload,
            kind="images_generations",
        )

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        count = int(self.spec.image_count or 0)
        checks = image_integrity_checks(records, width=self.spec.width, height=self.spec.height)
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

    def _image_config(
        self,
        width: Any,
        height: Any,
        steps: Any,
        seed: Any,
    ) -> dict[str, Any]:
        image: dict[str, Any] = {}
        if width is not None and height is not None:
            image.update(width=int(width), height=int(height))
        if steps is not None:
            image["steps"] = int(steps)
        if seed is not None:
            image["seed"] = int(seed)
        self._apply_quality(image)
        return image

    def _apply_quality(self, image: dict[str, Any]) -> None:
        for key in (
            "guidance_scale",
            "image_guidance_scale",
            "cfg_norm",
            "timestep_shift",
        ):
            value = getattr(self.spec, key)
            if value is not None:
                image[key] = value
        if self.spec.cfg_interval is not None:
            image["cfg_interval"] = list(self.spec.cfg_interval)
