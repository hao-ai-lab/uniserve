from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from ..metrics.common import RequestRecord
from ..spec import BenchmarkSpec
from ..validation import ValidationResult

RequestKind = Literal["openai_chat", "openai_chat_json", "images_generations"]


@dataclass(frozen=True)
class TaskRequest:
    endpoint: str
    payload: dict[str, Any]
    kind: RequestKind
    semantic_task: str | None = None


class BenchmarkTask:
    def __init__(self, spec: BenchmarkSpec) -> None:
        self.spec = spec

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        raise NotImplementedError

    def validate(self, records: Sequence[RequestRecord]) -> ValidationResult:
        common = ValidationResult(
            checks={
                "declared_request_count": len(records) == self.spec.num_prompts,
                "all_requests_succeeded": bool(records) and all(record.success for record in records),
            },
            statistics={
                "request_count": len(records),
                "successful_requests": sum(record.success for record in records),
            },
        )
        return common.merged(self.validate_output(records))

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        return ValidationResult(checks={"observable_output": bool(records)})


def apply_text_sampling(payload: dict[str, Any], spec: BenchmarkSpec) -> None:
    payload["temperature"] = spec.temperature
    payload["top_p"] = spec.top_p
    payload["ignore_eos"] = spec.ignore_eos
    for key in (
        "top_k",
        "min_p",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
    ):
        value = getattr(spec, key)
        if value is not None:
            payload[key] = value
    if spec.sampling_seed is not None:
        payload["seed"] = spec.sampling_seed


def apply_chat_image_parameters(
    payload: dict[str, Any],
    image_config: dict[str, Any],
) -> None:
    payload["image_config"] = image_config


def image_integrity_checks(
    records: Sequence[RequestRecord],
    *,
    width: int | None,
    height: int | None,
) -> dict[str, bool]:
    decoded_counts = all(record.images == len(record.decoded_images) for record in records)
    dimensions = all(
        (width is None or image.width == width) and (height is None or image.height == height)
        for record in records
        for image in record.decoded_images
    )
    return {
        "decoded_image_count": decoded_counts,
        "image_dimensions": dimensions,
    }


def input_image_data_url(item: dict[str, Any]) -> str:
    image_b64 = item.get("input_image_b64")
    if not isinstance(image_b64, str) or not image_b64:
        raise ValueError("input image row has no base64 payload")
    mime = item.get("input_image_mime", "image/png")
    if not isinstance(mime, str) or not mime.startswith("image/"):
        raise ValueError("input image row has an invalid MIME type")
    return f"data:{mime};base64,{image_b64}"
