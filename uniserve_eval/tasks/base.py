"""Defines shared request construction and observable output validation.

A task adapter maps a normalized dataset `Example` to an HTTP request and
judges the resulting `RequestRecord`s. Profile parsing in `config` consults
the adapter's class attributes and `check_*` class methods before a
`BenchmarkPoint` exists; `run_point` then binds an instance to the point to
build requests and validate the measured records.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, ClassVar

from ..types import (
    CHAT_COMPLETIONS,
    BenchmarkPoint,
    Example,
    ImageConfig,
    RequestRecord,
    TaskName,
    TaskRequest,
    ValidationResult,
)


class ImageCountRule(StrEnum):
    """Specifies whether a task accepts an explicit output-image count."""

    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    OPTIONAL = "optional"


class BenchmarkTask:
    """Defines the endpoint, request, and validation contract for a task.

    `config` reads these attributes and runs the `check_*` class methods
    while parsing a profile; request building does not recheck them.

    Attributes:
        name: The task identifier; `TASKS` registers each adapter under
            this identifier's value.
        allowed_endpoints: Endpoints `check_endpoint` accepts.
        default_endpoint: Endpoint used when a point declares none.
        default_stream: `sampling.stream` value used when a point declares
            none.
        accepts_image: Whether a point may declare an `image` settings table.
        accepts_question: Whether a point may declare a dataset `question`.
        image_count: How `check_image` treats `image.image_count`.
    """

    name: ClassVar[TaskName]
    allowed_endpoints: ClassVar[tuple[str, ...]] = (CHAT_COMPLETIONS,)
    default_endpoint: ClassVar[str] = CHAT_COMPLETIONS
    default_stream: ClassVar[bool] = True
    accepts_image: ClassVar[bool] = False
    accepts_question: ClassVar[bool] = False
    image_count: ClassVar[ImageCountRule] = ImageCountRule.FORBIDDEN

    def __init__(self, point: BenchmarkPoint) -> None:
        """Bind the task adapter to a resolved benchmark point."""
        self.point = point

    def build_request(self, example: Example) -> TaskRequest:
        """Construct the endpoint request for one normalized example.

        Subclasses must implement this.
        """
        raise NotImplementedError

    def validate(self, records: Sequence[RequestRecord]) -> ValidationResult:
        """Combine request-level checks with task-specific output checks.

        `declared_request_count` requires exactly `load.num_prompts` measured
        records, and `all_requests_succeeded` fails on an empty record list.

        Raises:
            ValueError: If `validate_output` returns a check named
                `declared_request_count` or `all_requests_succeeded`.
        """
        common = ValidationResult(
            checks={
                "declared_request_count": len(records)
                == self.point.load.num_prompts,
                "all_requests_succeeded": bool(records)
                and all(record.success for record in records),
            },
            statistics={
                "request_count": len(records),
                "successful_requests": sum(
                    record.success for record in records
                ),
            },
        )
        return common.merged(self.validate_output(records))

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Validate output properties specific to the task.

        Subclasses may override this; the default requires only that records
        exist.
        """
        return ValidationResult(checks={"observable_output": bool(records)})

    @classmethod
    def check_endpoint(cls, endpoint: str | None, context: str) -> str:
        """Resolve and validate an endpoint supported by the task."""
        chosen = endpoint if endpoint is not None else cls.default_endpoint
        if not isinstance(chosen, str) or chosen not in cls.allowed_endpoints:
            allowed = ", ".join(cls.allowed_endpoints)
            raise ValueError(f"{context}.endpoint must be one of: {allowed}")
        return chosen

    @classmethod
    def check_question(cls, question: Any, context: str) -> str | None:
        """Validate an optional dataset question against task capabilities."""
        if question is None:
            return None
        if not cls.accepts_question:
            raise ValueError(
                f"{context}.question is not valid for task {cls.name.value}"
            )
        if not isinstance(question, str) or not question:
            raise ValueError(f"{context}.question must be a non-empty string")
        return question

    @classmethod
    def check_image(cls, image: ImageConfig, context: str) -> None:
        """Validate explicit image-count settings against the task contract."""
        if (
            cls.image_count is ImageCountRule.REQUIRED
            and image.image_count is None
        ):
            raise ValueError(f"{context} requires image.image_count")
        if (
            cls.image_count is ImageCountRule.FORBIDDEN
            and image.image_count is not None
        ):
            raise ValueError(
                f"{context} does not declare a per-request image count"
            )

    def apply_text_sampling(self, payload: dict[str, Any]) -> None:
        """Add configured text-sampling fields to an endpoint payload.

        Optional fields are added only when configured. `extra_body` is merged
        last, so its keys override every field set here.
        """
        sampling = self.point.sampling
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

    def image_fields(
        self, example: Example, *, include_count: bool
    ) -> dict[str, Any]:
        """Resolve image settings by applying per-example overrides.

        Example width, height, and steps override the point's image
        settings, and an example seed overrides `load.seed`, the seed that
        also drives arrival sampling. Width and height are emitted only as a
        pair.

        Args:
            example: The example whose overrides apply.
            include_count: Whether to add `num_images` from
                `image.image_count` when it is set.

        Returns:
            Fields in the chat-completions `image_config` layout, which
            `apply_image_generations_fields` maps onto the image-generations
            schema.
        """
        image = self.point.image
        width = example.width if example.width is not None else image.width
        height = example.height if example.height is not None else image.height
        steps = example.steps if example.steps is not None else image.steps
        seed = (
            example.seed if example.seed is not None else self.point.load.seed
        )

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
        self, payload: dict[str, Any], example: Example
    ) -> None:
        """Map resolved image settings onto the image-generations schema.

        Width and height become one `size` string of the form `WxH`; the
        image count and `resolution` fields are not carried over.
        """
        rendered = self.image_fields(example, include_count=False)
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

    def input_image_data_url(self, example: Example) -> str:
        """Build a validated embedded data URL for an example image.

        The MIME type defaults to `image/png` when the row declares none.

        Raises:
            ValueError: If the row has no base64 payload or its MIME type is
                not an image type.
        """
        image_b64 = example.input_image_b64
        if not isinstance(image_b64, str) or not image_b64:
            raise ValueError("input image row has no base64 payload")
        mime = example.input_image_mime or "image/png"
        if not mime.startswith("image/"):
            raise ValueError("input image row has an invalid MIME type")
        return f"data:{mime};base64,{image_b64}"

    def image_integrity_checks(
        self, records: Sequence[RequestRecord]
    ) -> dict[str, bool]:
        """Check decoded image counts and configured output geometry.

        Decoded sizes are compared with the point's configured width and
        height, not per-example overrides; an unset dimension is not checked.
        Both checks pass trivially for records without images.
        """
        image = self.point.image
        decoded_counts = all(
            record.images == len(record.decoded_images) for record in records
        )
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

    def server_usage_ok(self, records: Sequence[RequestRecord]) -> bool:
        """Report whether every request uses server-reported token counts.

        Both prompt and output lengths must come from server usage rather than
        request-side fallbacks. An empty record list fails.
        """
        return bool(records) and all(
            record.output_len_source == "server_usage"
            and record.prompt_len_source == "server_usage"
            for record in records
        )

    def fixed_output_length_ok(self, records: Sequence[RequestRecord]) -> bool:
        """Report whether every completion reaches its requested token limit.

        `requested_output_len` is the output limit `run_point` passed to the
        transport, 0 when none was declared, so a point without an output
        limit always fails. An empty record list fails.
        """
        return bool(records) and all(
            record.requested_output_len > 0
            and record.output_len == record.requested_output_len
            and record.finish_reason == "length"
            for record in records
        )
