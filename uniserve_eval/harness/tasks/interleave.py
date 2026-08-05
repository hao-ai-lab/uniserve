from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..metrics.common import RequestRecord
from ..validation import ValidationResult
from .base import (
    BenchmarkTask,
    TaskRequest,
    apply_chat_image_parameters,
    apply_text_sampling,
    image_integrity_checks,
)


class InterleaveTask(BenchmarkTask):
    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        width = item.get("width", self.spec.width)
        height = item.get("height", self.spec.height)
        image: dict[str, Any] = {"seed": int(item.get("seed", self.spec.seed))}
        if width is not None and height is not None:
            image.update(width=int(width), height=int(height))
        if self.spec.steps is not None:
            image["steps"] = int(self.spec.steps)
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
        if item.get("aspect_ratio") is not None:
            image["resolution"] = str(item["aspect_ratio"])

        payload: dict[str, Any] = {
            "model": self.spec.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "modalities": ["text", "image"],
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_completion_tokens": int(item.get("max_tokens", self.spec.max_tokens or 512)),
        }
        apply_chat_image_parameters(payload, image)
        extra = dict(self.spec.extra_request_body)
        extra_image = extra.pop("image_config", None)
        if isinstance(extra_image, dict):
            image.update(extra_image)
        payload.update(extra)
        apply_text_sampling(payload, self.spec)
        return TaskRequest(
            endpoint=self.spec.endpoint,
            payload=payload,
            kind="openai_chat",
        )

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        from ..metrics.stream import interleave_transition_summary

        checks = image_integrity_checks(records, width=self.spec.width, height=self.spec.height)
        checks["visible_text"] = bool(records) and all(
            bool(record.generated_text) and "text" in record.output_modalities
            for record in records
        )
        checks["server_usage"] = bool(records) and all(
            record.output_len_source == "server_usage" and record.prompt_len_source == "server_usage"
            for record in records
        )
        total_images = sum(record.images for record in records)
        mean_images = total_images / len(records) if records else 0.0
        minimum = self.spec.minimum_average_images
        checks["minimum_average_images"] = minimum is None or mean_images >= minimum
        checks["client_transition_timestamps"] = interleave_transition_summary(
            list(records)
        )["valid"]
        zero_image_requests = sum(record.images == 0 for record in records)
        return ValidationResult(
            checks=checks,
            statistics={
                "completed_images": total_images,
                "images_per_request": mean_images,
                "zero_image_requests": zero_image_requests,
            },
        )
