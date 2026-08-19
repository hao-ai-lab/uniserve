"""Shared task request helpers and the validation envelope."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..types import (
    BenchmarkPoint,
    Example,
    ImageConfig,
    RequestRecord,
    SamplingConfig,
    TaskRequest,
    ValidationResult,
)


class BenchmarkTask:
    def __init__(self, point: BenchmarkPoint) -> None:
        self.point = point

    def build_request(self, example: Example) -> TaskRequest:
        raise NotImplementedError

    def validate(self, records: Sequence[RequestRecord]) -> ValidationResult:
        common = ValidationResult(
            checks={
                "declared_request_count": len(records) == self.point.load.num_prompts,
                "all_requests_succeeded": bool(records)
                and all(record.success for record in records),
            },
            statistics={
                "request_count": len(records),
                "successful_requests": sum(record.success for record in records),
            },
        )
        return common.merged(self.validate_output(records))

    def validate_output(self, records: Sequence[RequestRecord]) -> ValidationResult:
        return ValidationResult(checks={"observable_output": bool(records)})


def apply_text_sampling(payload: dict[str, Any], sampling: SamplingConfig) -> None:
    payload["temperature"] = sampling.temperature
    payload["top_p"] = sampling.top_p
    payload["ignore_eos"] = sampling.ignore_eos
    for key in (
        "top_k",
        "min_p",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
    ):
        value = getattr(sampling, key)
        if value is not None:
            payload[key] = value
    if sampling.sampling_seed is not None:
        payload["seed"] = sampling.sampling_seed
    payload.update(sampling.extra_body)


def render_image_config(
    image: ImageConfig,
    example: Example,
    *,
    include_count: bool,
    fallback_seed: int,
) -> dict[str, Any]:
    width = example.width if example.width is not None else image.width
    height = example.height if example.height is not None else image.height
    steps = example.steps if example.steps is not None else image.steps
    seed = example.seed if example.seed is not None else fallback_seed
    payload: dict[str, Any] = {"seed": int(seed)}
    if include_count and image.image_count is not None:
        payload["num_images"] = image.image_count
    if width is not None and height is not None:
        payload.update(width=int(width), height=int(height))
    if steps is not None:
        payload["steps"] = int(steps)
    for key in (
        "guidance_scale",
        "image_guidance_scale",
        "cfg_norm",
        "timestep_shift",
    ):
        value = getattr(image, key)
        if value is not None:
            payload[key] = value
    if image.cfg_interval is not None:
        payload["cfg_interval"] = list(image.cfg_interval)
    if example.aspect_ratio is not None:
        payload["resolution"] = str(example.aspect_ratio)
    return payload


def apply_image_generations_fields(
    payload: dict[str, Any],
    image: ImageConfig,
    example: Example,
    *,
    fallback_seed: int,
) -> None:
    rendered = render_image_config(
        image, example, include_count=False, fallback_seed=fallback_seed
    )
    width = rendered.get("width")
    height = rendered.get("height")
    if width is not None and height is not None:
        payload["size"] = f"{width}x{height}"
    for key in (
        "steps",
        "seed",
        "guidance_scale",
        "image_guidance_scale",
        "cfg_norm",
        "timestep_shift",
        "cfg_interval",
    ):
        if key in rendered:
            payload[key] = rendered[key]


def image_integrity_checks(
    records: Sequence[RequestRecord],
    image: ImageConfig,
) -> dict[str, bool]:
    decoded_counts = all(record.images == len(record.decoded_images) for record in records)
    dimensions = all(
        (image.width is None or decoded.width == image.width)
        and (image.height is None or decoded.height == image.height)
        for record in records
        for decoded in record.decoded_images
    )
    return {
        "decoded_image_count": decoded_counts,
        "image_dimensions": dimensions,
    }


def input_image_data_url(example: Example) -> str:
    image_b64 = example.input_image_b64
    if not isinstance(image_b64, str) or not image_b64:
        raise ValueError("input image row has no base64 payload")
    mime = example.input_image_mime or "image/png"
    if not mime.startswith("image/"):
        raise ValueError("input image row has an invalid MIME type")
    return f"data:{mime};base64,{image_b64}"


def server_usage_check(records: Sequence[RequestRecord]) -> bool:
    return bool(records) and all(
        record.output_len_source == "server_usage" and record.prompt_len_source == "server_usage"
        for record in records
    )


def fixed_output_length_check(records: Sequence[RequestRecord]) -> bool:
    return bool(records) and all(
        record.requested_output_len > 0
        and record.output_len == record.requested_output_len
        and record.finish_reason == "length"
        for record in records
    )
