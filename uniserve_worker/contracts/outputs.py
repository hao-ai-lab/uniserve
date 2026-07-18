"""Typed forward outputs consumed by the shared runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

__all__ = [
    "ForwardOutputBase",
    "DeferredForwardOutput",
    "FinalizableSeqResult",
    "TextTokenOutput",
    "FlowOutput",
    "CommitOutput",
    "EncodeOutput",
    "SampleOutput",
    "FrameOutput",
    "ForwardOutput",
]


@runtime_checkable
class FinalizableSeqResult(Protocol):
    """Response-time result whose device-backed values can be materialized later."""

    def finalize(self) -> dict[str, Any]: ...


@runtime_checkable
class DeferredForwardOutput(Protocol):
    """Forward output that defers device-backed response materialization."""

    req_id: int

    def to_seq_result(self) -> FinalizableSeqResult: ...


@dataclass(frozen=True)
class ForwardOutputBase:
    """Base value object for one per-sequence forward result."""

    req_id: int

    _required_fields: ClassVar[tuple[str, ...]] = ()
    _optional_fields: ClassVar[tuple[str, ...]] = ()
    _list_tuple_fields: ClassVar[tuple[str, ...]] = ()

    def to_seq_result(self) -> dict[str, Any]:
        out: dict[str, Any] = {"req_id": self.req_id}
        for field in self._required_fields:
            out[field] = self._wire_value(field, getattr(self, field))
        for field in self._optional_fields:
            value = getattr(self, field)
            if value is not None:
                out[field] = self._wire_value(field, value)
        return out

    def _wire_value(self, field: str, value: Any) -> Any:
        if field in self._list_tuple_fields and isinstance(value, tuple):
            return list(value)
        return value


@dataclass(frozen=True)
class TextTokenOutput(ForwardOutputBase):
    """Sampled text token (and optional logprobs) for one request row."""

    sampled_token_id: int
    sampled_logprob: float | None = None
    top_logprobs: list[tuple[int, float, int]] | None = None
    prompt_logprobs: list[list[tuple[int, float, int]]] | None = None
    num_accepted_tokens: int | None = None

    _required_fields: ClassVar[tuple[str, ...]] = ("sampled_token_id",)
    _optional_fields: ClassVar[tuple[str, ...]] = (
        "sampled_logprob",
        "top_logprobs",
        "prompt_logprobs",
        "num_accepted_tokens",
    )


@dataclass(frozen=True)
class FlowOutput(ForwardOutputBase):
    """Flow-step progress for one image-generation request."""

    denoise_done: bool
    num_steps_done: int

    _required_fields: ClassVar[tuple[str, ...]] = ("denoise_done", "num_steps_done")


@dataclass(frozen=True)
class CommitOutput(ForwardOutputBase):
    """Image commit result after VAE decode and optional re-encode."""

    image_png_b64: str | None = None
    image_hw: tuple[int, int] | None = None
    sampled_token_id: int | None = None
    sampled_logprob: float | None = None
    top_logprobs: list[tuple[int, float, int]] | None = None
    num_tokens: int | None = None
    locator: str | None = None

    _optional_fields: ClassVar[tuple[str, ...]] = (
        "image_png_b64",
        "image_hw",
        "sampled_token_id",
        "sampled_logprob",
        "top_logprobs",
        "num_tokens",
        "locator",
    )
    _list_tuple_fields: ClassVar[tuple[str, ...]] = ("image_hw",)


@dataclass(frozen=True)
class EncodeOutput(ForwardOutputBase):
    """Vision or VAE encoder output handle for one request."""

    encoder_handle: int
    num_tokens: int | None = None
    image_hw: tuple[int, int] | None = None

    _required_fields: ClassVar[tuple[str, ...]] = ("encoder_handle",)
    _optional_fields: ClassVar[tuple[str, ...]] = ("num_tokens", "image_hw")
    _list_tuple_fields: ClassVar[tuple[str, ...]] = ("image_hw",)


SampleOutput = TextTokenOutput


@dataclass(frozen=True)
class FrameOutput(ForwardOutputBase):
    """Output of a PostProcess-stage ``encode_frame`` op."""

    num_tokens: int | None = None
    image_png_b64: str | None = None

    _optional_fields: ClassVar[tuple[str, ...]] = ("num_tokens", "image_png_b64")


ForwardOutput = (
    TextTokenOutput | FlowOutput | CommitOutput | EncodeOutput | FrameOutput | DeferredForwardOutput
)
