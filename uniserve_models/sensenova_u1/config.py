"""Immutable SenseNova U1 architecture and checkpoint metadata normalization."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uniserve import loading
from uniserve.diffusion import NoiseScale
from uniserve.nn.rope import (
    DynamicScaling,
    LinearScaling,
    LlamaScaling,
    LongRoPEScaling,
    ProportionalScaling,
    RoPEScaling,
    YaRNScaling,
)

from . import flow, vision


def _positive(value: object, name: str, *, integer: bool = False) -> None:
    """Require a finite positive number, optionally a strict integer."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or (integer and not isinstance(value, int))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"SenseNova {name} must be a positive {'integer' if integer else 'finite number'}"
        )


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Text/flow decoder dimensions, normalization, and multi-axis rotary math."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_types: tuple[str, ...]
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    max_position_embeddings: int = 32768
    max_position_embeddings_hw: int = 10000
    rope_theta: float = 10000.0
    rope_theta_hw: float = 10000.0
    rope_scaling: RoPEScaling | None = None
    partial_rotary_factor: float = 1.0
    sliding_window: int | None = None
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "max_position_embeddings_hw",
        ):
            _positive(getattr(self, name), name, integer=True)
        for name in ("rms_norm_eps", "rope_theta", "rope_theta_hw", "partial_rotary_factor"):
            _positive(getattr(self, name), name)

        if self.head_dim % 4:
            raise ValueError("SenseNova head_dim must divide into temporal/height/width partitions")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("SenseNova attention heads must be divisible by KV heads")
        if self.partial_rotary_factor > 1:
            raise ValueError("SenseNova partial_rotary_factor must not exceed one")

        if (
            not isinstance(self.layer_types, tuple)
            or len(self.layer_types) != self.num_hidden_layers
        ):
            raise ValueError("SenseNova layer_types must describe every decoder layer")
        if any(value not in {"full_attention", "sliding_attention"} for value in self.layer_types):
            raise ValueError("SenseNova layer_types contains an unsupported attention type")
        if self.sliding_window is not None:
            _positive(self.sliding_window, "sliding_window", integer=True)
        if "sliding_attention" in self.layer_types:
            if self.sliding_window is None:
                raise ValueError("SenseNova sliding_attention requires sliding_window")
            # The multimodal mask implements full attention within image
            # blocks and causal text visibility, without a local-window cutoff.
            raise ValueError(
                "SenseNova layer_types: sliding_attention is not supported by the MoT decoder"
            )

        for name in ("attention_bias", "tie_word_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"SenseNova {name} must be boolean")
        if self.pad_token_id is not None and (
            not isinstance(self.pad_token_id, int)
            or isinstance(self.pad_token_id, bool)
            or not 0 <= self.pad_token_id < self.vocab_size
        ):
            raise ValueError("SenseNova pad_token_id must be inside the vocabulary")
        if self.hidden_act not in {
            "silu",
            "swish",
            "silu_and_mul",
            "swiglu",
            "gelu",
            "gelu_and_mul",
            "geglu",
            "gelu_pytorch_tanh",
            "gelu_tanh",
        }:
            raise ValueError(f"unsupported SenseNova hidden_act {self.hidden_act!r}")


@dataclass(frozen=True, slots=True)
class Config:
    """Compose the language, patch embedding, and image prediction networks."""

    text: TransformerConfig
    vision: vision.Config
    flow: flow.Config
    max_image_seq_len: int = 4096

    def __post_init__(self) -> None:
        _positive(self.max_image_seq_len, "max_image_seq_len", integer=True)
        if self.vision.output_size != self.text.hidden_size:
            raise ValueError("SenseNova vision output must match text hidden_size")
        if self.vision.num_channels != 3:
            raise ValueError("SenseNova image prediction requires RGB vision inputs")


def _stage_scalar(value: Any, name: str) -> Any:
    """Collapse a per-stage list to its first entry; scalars pass through."""

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"SenseNova vision_config.{name} must not be empty")
        return value[0]
    return value


def _alias(
    primary: Mapping[str, Any], name: str, aliases: tuple[Mapping[str, Any], ...], default: Any
) -> Any:
    """Return one value for a field that checkpoints may duplicate across maps."""

    values = [source[name] for source in (primary, *aliases) if source.get(name) is not None]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"SenseNova checkpoint has conflicting aliases for {name}")
    return values[0] if values else default


def _normalize(raw: Mapping[str, Any]) -> Config:
    """Resolve checkpoint aliases once, without allocating numerical resources."""

    from packaging.version import Version

    required = raw.get("uniserve_sensenova_min_version")
    if required and Version("0.1.0") < Version(str(required)):
        raise RuntimeError(f"checkpoint requires UniServe model code >= {required}")

    text, image = raw["llm_config"], raw["vision_config"]
    if not isinstance(text, Mapping) or not isinstance(image, Mapping):
        raise ValueError("SenseNova llm_config and vision_config must be objects")
    if text.get("num_experts", 0) != 0:
        raise ValueError(
            "SenseNova llm_config.num_experts: sparse MoE is not supported by the dual-route MoT decoder"
        )

    # Rotary metadata appears under both rope_scaling and rope_parameters in
    # different checkpoint generations; the two aliases must agree.
    scaling = text.get("rope_scaling") or {}
    parameters = text.get("rope_parameters") or {}
    if not isinstance(scaling, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError("SenseNova rotary metadata must be an object")
    kind = _alias(
        parameters, "rope_type", (scaling,), parameters.get("type", scaling.get("type", "default"))
    )
    if any(source.get("type", kind) != kind for source in (parameters, scaling)):
        raise ValueError("SenseNova checkpoint has conflicting rotary type aliases")

    recipe = None
    if kind != "default":

        def option(name, default=None):
            return _alias(parameters, name, (scaling,), default)

        factor = option("factor")
        original = option(
            "original_max_position_embeddings", text.get("max_position_embeddings", 32768)
        )
        match kind:
            case "linear":
                recipe = LinearScaling(factor)
            case "dynamic":
                recipe = DynamicScaling(factor)
            case "proportional":
                recipe = ProportionalScaling(factor)
            case "yarn":
                recipe = YaRNScaling(
                    factor,
                    original,
                    option("attention_factor"),
                    option("beta_fast", 32.0),
                    option("beta_slow", 1.0),
                    option("mscale"),
                    option("mscale_all_dim"),
                    option("truncate", True),
                )
            case "longrope":
                recipe = LongRoPEScaling(
                    factor,
                    original,
                    option("attention_factor"),
                    tuple(option("short_factor", ())),
                    tuple(option("long_factor", ())),
                )
            case "llama3":
                recipe = LlamaScaling(
                    factor,
                    original,
                    option("low_freq_factor", 1.0),
                    option("high_freq_factor", 4.0),
                )
            case _:
                raise ValueError(f"unsupported SenseNova rotary recipe {kind!r}")

    layers, heads = text["num_hidden_layers"], text["num_attention_heads"]
    _positive(layers, "llm_config.num_hidden_layers", integer=True)
    _positive(heads, "llm_config.num_attention_heads", integer=True)
    hidden = text["hidden_size"]
    _positive(hidden, "llm_config.hidden_size", integer=True)
    if "head_dim" not in text and hidden % heads:
        raise ValueError("SenseNova requires explicit head_dim for this hidden width")

    use_window = text.get("use_sliding_window", False)
    if not isinstance(use_window, bool):
        raise ValueError("SenseNova use_sliding_window must be boolean")
    window = text.get("sliding_window")
    boundary = text.get("max_window_layers", 0)
    if not isinstance(boundary, int) or isinstance(boundary, bool) or boundary < 0:
        raise ValueError("SenseNova llm_config.max_window_layers must be a non-negative integer")
    layer_types = text.get("layer_types")
    if layer_types is None:
        # Older checkpoints derive per-layer attention kinds from the window fields.
        layer_types = tuple(
            "sliding_attention"
            if use_window and window is not None and index >= boundary
            else "full_attention"
            for index in range(layers)
        )
    if not isinstance(layer_types, (list, tuple)):
        raise ValueError("SenseNova layer_types must be a sequence")

    vision_ratio = _stage_scalar(image.get("downsample_ratio", 0.5), "downsample_ratio")
    if raw.get("downsample_ratio", vision_ratio) != vision_ratio:
        raise ValueError("SenseNova root and vision downsample_ratio must agree")
    head_layers = raw.get("fm_head_layers", 2)
    _positive(head_layers, "fm_head_layers", integer=True)

    return Config(
        text=TransformerConfig(
            vocab_size=text["vocab_size"],
            hidden_size=hidden,
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text.get("head_dim", hidden // heads),
            layer_types=tuple(layer_types),
            hidden_act=text.get("hidden_act", "silu"),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            attention_bias=text.get("attention_bias", False),
            max_position_embeddings=text.get("max_position_embeddings", 32768),
            max_position_embeddings_hw=text.get("max_position_embeddings_hw", 10000),
            rope_theta=_alias(text, "rope_theta", (parameters, scaling), 10000.0),
            rope_theta_hw=text.get("rope_theta_hw", 10000.0),
            rope_scaling=recipe,
            partial_rotary_factor=_alias(text, "partial_rotary_factor", (parameters, scaling), 1.0),
            sliding_window=window,
            pad_token_id=text.get("pad_token_id")
            if text.get("pad_token_id") is not None
            else raw.get("pad_token_id"),
            tie_word_embeddings=_alias(text, "tie_word_embeddings", (raw,), False),
        ),
        vision=vision.Config(
            hidden_size=image.get("hidden_size", 1024),
            output_size=_stage_scalar(image.get("llm_hidden_size", 2048), "llm_hidden_size"),
            downsample_ratio=vision_ratio,
            patch_size=image.get("patch_size", 16),
            num_channels=image.get("num_channels", 3),
            rope_theta=image.get("rope_theta_vision", 10000.0),
        ),
        flow=flow.Config(
            head=flow.HeadConfig(
                hidden_size=raw["fm_head_dim"] if head_layers > 2 else 4096,
                num_layers=head_layers,
                mlp_ratio=raw["fm_head_mlp_ratio"] if head_layers > 2 else 1.0,
            ),
            use_pixel_head=raw.get("use_pixel_head", False),
            add_noise_scale_embedding=raw.get("add_noise_scale_embedding", False),
            noise=NoiseScale(
                raw.get("noise_scale", 1.0),
                "resolution"
                if raw.get("noise_scale_mode") == "dynamic"
                else raw.get("noise_scale_mode", "constant"),
                raw.get("noise_scale_base_image_seq_len", 1.0),
                raw.get("noise_scale_max_value", 1.0),
            ),
        ),
        max_image_seq_len=raw.get("max_image_seq_len", 4096),
    )


def read_config(root: Path, io: loading.Config) -> Config:
    """Normalize checkpoint aliases and typed rotary recipes before construction."""
    return _normalize(json.loads((root / "config.json").read_text()))


# Checkpoint headers needed to resolve architecture before module selection.
config_sources = ()
