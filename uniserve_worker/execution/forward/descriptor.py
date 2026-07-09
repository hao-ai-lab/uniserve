"""Model geometry and neural surface descriptor for forward execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from ...foundation.errors import invalid_descriptor

__all__ = [
    "ForwardModelDescriptor",
    "ForwardModelModules",
    "descriptor_from_model",
]


@dataclass(frozen=True)
class ForwardModelModules:
    embed_text: Callable[..., Any] | None = None
    embed_generation: Callable[..., Any] | None = None
    decoder: Callable[..., Any] | None = None
    logits: Callable[..., Any] | None = None
    velocity: Callable[..., Any] | None = None
    encode: Callable[..., Any] | None = None
    commit: Callable[..., Any] | None = None


@dataclass(frozen=True)
class ForwardModelDescriptor:
    device: torch.device
    dtype: torch.dtype
    hidden_size: int
    vocab_size: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    attention_scale: float = 1.0
    kv_page_size: int | None = None
    supports_text: bool = False
    supports_denoise: bool = False
    supports_encode: bool = False
    supports_commit: bool = False
    graph_capture: Mapping[str, Any] = field(default_factory=dict)
    modules: ForwardModelModules = field(default_factory=ForwardModelModules)
    variant: str = "default"

    def validate(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise invalid_descriptor(f"forward model descriptor {name} must be positive")
        if self.supports_text:
            if int(self.vocab_size) <= 0:
                raise invalid_descriptor("text-capable descriptor must declare vocab_size")
            if self.modules.decoder is None and self.modules.logits is None:
                raise invalid_descriptor("text-capable descriptor is missing text neural surfaces")
        if self.supports_denoise and self.modules.velocity is None:
            raise invalid_descriptor("denoise-capable descriptor is missing velocity surface")
        if self.supports_encode and self.modules.encode is None:
            raise invalid_descriptor("encode-capable descriptor is missing encode surface")
        if self.supports_commit and self.modules.commit is None:
            raise invalid_descriptor("commit-capable descriptor is missing commit surface")


def descriptor_from_model(model: Any) -> ForwardModelDescriptor:
    device = torch.device(str(getattr(model, "device", "cpu") or "cpu"))
    dtype = _dtype(getattr(model, "dtype", None))
    modules = ForwardModelModules(
        embed_text=_first_callable(model, ("embed_tokens", "packed_text_embeddings")),
        embed_generation=_first_callable(model, ("embed_generation", "prepare_generation_embeddings")),
        decoder=_first_callable(model, ("forward", "decoder_forward", "packed_decoder_forward")),
        logits=_first_callable(model, ("compute_logits", "logits", "lm_head"))
        or _first_overridden_callable(model, ("run_text_logits_batch", "run_text_logits")),
        velocity=_first_callable(model, ("predict_velocity", "velocity", "project_velocity")),
        encode=_first_callable(model, ("encode_image", "encode_latents")),
        commit=_first_callable(model, ("decode_image", "commit")),
    )
    supports_text = (
        callable(getattr(model, "forward", None))
        or _overrides(model, "run_text_logits_batch")
        or _overrides(model, "run_text_logits")
    )
    raw_vocab_size = int(_attr(model, ("vocab_size",), 0))
    descriptor = ForwardModelDescriptor(
        device=device,
        dtype=dtype,
        hidden_size=max(1, int(_attr(model, ("hidden_size", "d_model"), 1))),
        vocab_size=max(1 if supports_text else 0, raw_vocab_size),
        num_layers=max(1, int(_attr(model, ("num_layers", "n_layers"), 1))),
        num_q_heads=max(1, int(_attr(model, ("num_q_heads", "num_attention_heads"), 1))),
        num_kv_heads=max(1, int(_attr(model, ("num_kv_heads", "num_key_value_heads"), 1))),
        head_dim=max(1, int(_attr(model, ("head_dim",), 1))),
        attention_scale=float(_attr(model, ("attention_scale",), 1.0)),
        kv_page_size=_optional_int(_attr(model, ("block_size", "kv_page_size"), None)),
        supports_text=supports_text,
        supports_denoise=_overrides(model, "predict_velocity"),
        supports_encode=_overrides(model, "encode_image") or _overrides(model, "encode_latents"),
        supports_commit=_overrides(model, "decode_image"),
        graph_capture={
            "decode": callable(getattr(model, "text_decode_graph_query_geometry", None)),
        },
        modules=modules,
        variant=type(model).__name__,
    )
    descriptor.validate()
    return descriptor


def _dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return torch.float32
    parsed = getattr(torch, str(value), None)
    return parsed if isinstance(parsed, torch.dtype) else torch.float32


def _attr(model: Any, names: tuple[str, ...], default: Any) -> Any:
    config = getattr(model, "config", None)
    for name in names:
        if hasattr(model, name):
            return getattr(model, name)
        if config is not None and hasattr(config, name):
            return getattr(config, name)
    return default


def _first_callable(model: Any, names: tuple[str, ...]) -> Callable[..., Any] | None:
    for name in names:
        value = getattr(model, name, None)
        if callable(value):
            return value
    return None


def _first_overridden_callable(model: Any, names: tuple[str, ...]) -> Callable[..., Any] | None:
    for name in names:
        value = getattr(model, name, None)
        if callable(value) and _overrides(model, name):
            return value
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _overrides(model: Any, name: str) -> bool:
    from ...contracts.model_protocols import ModelHooks

    hook = getattr(type(model), name, None)
    default = getattr(ModelHooks, name, None)
    return hook is not None and hook is not default
