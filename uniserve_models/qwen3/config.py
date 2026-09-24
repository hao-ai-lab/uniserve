"""Immutable Qwen3 architecture and checkpoint metadata normalization.

``read_config`` turns a checkpoint's ``config.json`` into the frozen
``Config`` that ``Model`` and ``Transformer`` consume. ``Config`` validates
its own fields on construction, so a config built directly, as the MiniMax H3
text encoder does, receives the same checks as one read from a checkpoint.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from uniserve import loading


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable numerical parameters for Qwen decoders and conditioners.

    Field names follow the checkpoint ``config.json`` keys that
    ``read_config`` reads. ``head_dim`` may differ from
    ``hidden_size // num_attention_heads``. A zero ``num_experts`` builds a
    dense ``GatedMLP`` of width ``intermediate_size`` in every layer; a
    positive value builds an ``MoE`` in every layer, routing each token to
    ``num_experts_per_tok`` experts of width ``moe_intermediate_size``.

    Raises:
        ValueError: From ``__post_init__`` when a field has the wrong type or
            an invalid value, including odd ``head_dim``, query heads not
            divisible by KV heads, ``num_experts_per_tok`` above a nonzero
            ``num_experts``, or an unsupported ``hidden_act``.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    attention_bias: bool
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int

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
            "num_experts_per_tok",
            "moe_intermediate_size",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"Qwen3 {name} must be a positive integer")

        if self.head_dim % 2:
            raise ValueError("Qwen3 head_dim must be even for rotary positions")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "Qwen3 attention heads must be divisible by KV heads"
            )

        if (
            not isinstance(self.num_experts, int)
            or isinstance(self.num_experts, bool)
            or self.num_experts < 0
        ):
            raise ValueError("Qwen3 num_experts must be a non-negative integer")
        if self.num_experts and self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                "Qwen3 num_experts_per_tok must not exceed num_experts"
            )

        for name in ("rms_norm_eps", "rope_theta"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Qwen3 {name} must be finite and positive")

        for name in ("attention_bias", "tie_word_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"Qwen3 {name} must be boolean")

        # These are the aliases ``transformer._activation`` maps onto the
        # gated MLP's activation kernels.
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
            raise ValueError(
                f"unsupported Qwen3 hidden_act {self.hidden_act!r}"
            )


def _required_int(config: Mapping[str, object], name: str) -> int:
    """Read a required non-boolean integer from model configuration."""
    raw = config.get(name)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"Qwen3 config requires integer field {name!r}")
    if raw <= 0:
        raise ValueError(f"Qwen3 config field {name!r} must be positive")
    return raw


def _optional_int(
    config: Mapping[str, object],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    """Read and lower-bound an optional integer model setting."""
    raw = config.get(name, default)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < minimum:
        raise ValueError(
            f"Qwen3 config field {name!r} must be an integer >= {minimum}"
        )
    return raw


def _number(config: Mapping[str, object], name: str, default: float) -> float:
    """Read a finite positive number with a default, rejecting booleans."""
    raw = config.get(name, default)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(f"Qwen3 config field {name!r} must be numeric")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"Qwen3 config field {name!r} must be finite and positive"
        )
    return value


def _boolean(config: Mapping[str, object], name: str, default: bool) -> bool:
    """Read a boolean model setting with a default."""
    raw = config.get(name, default)
    if not isinstance(raw, bool):
        raise ValueError(f"Qwen3 config field {name!r} must be boolean")
    return raw


def _string(config: Mapping[str, object], name: str, default: str) -> str:
    """Read a textual model setting with a default."""
    raw = config.get(name, default)
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            f"Qwen3 config field {name!r} must be a non-empty string"
        )
    return raw


def read_config(root: Path, io: loading.Config) -> Config:
    """Normalize checkpoint metadata into immutable decoder configuration.

    Args:
        root: Local checkpoint directory containing ``config.json``.
        io: Loading options of the package ``read_config`` contract; Qwen3
            reads only the local ``config.json`` and does not use them.

    Returns:
        The validated ``Config``. Optional fields absent from the checkpoint
        take this function's defaults.

    Raises:
        ValueError: For invalid metadata, including a missing required
            field, a field of the wrong type or value, a ``rope_parameters``
            recipe other than ``default``, and disagreeing ``rope_theta``
            locations.
    """
    config = json.loads((root / "config.json").read_text())

    rotary = config.get("rope_parameters") or {}
    if not isinstance(rotary, Mapping):
        raise ValueError("Qwen3 rope_parameters must be an object")
    if rotary.get("rope_type", "default") != "default":
        raise ValueError("Qwen3 requires the default rotary embedding recipe")
    if "rope_theta" in rotary:
        if (
            "rope_theta" in config
            and config["rope_theta"] != rotary["rope_theta"]
        ):
            raise ValueError(
                "Qwen3 checkpoint has conflicting rope_theta aliases"
            )
        # Transformers may serialize rope_theta inside rope_parameters. The
        # top-level key becomes the single normalized value Config receives.
        config["rope_theta"] = rotary["rope_theta"]

    # Without an explicit head_dim, hidden_size must split exactly across the
    # query heads; an explicit head_dim may differ from that split.
    hidden_size = _required_int(config, "hidden_size")
    num_attention_heads = _required_int(config, "num_attention_heads")
    if "head_dim" not in config and hidden_size % num_attention_heads:
        raise ValueError(
            "Qwen3 hidden_size must be divisible by num_attention_heads"
        )
    head_dim = _optional_int(
        config,
        "head_dim",
        hidden_size // num_attention_heads,
        minimum=1,
    )
    num_experts = _optional_int(config, "num_experts", 0, minimum=0)
    num_experts_per_tok = _optional_int(
        config, "num_experts_per_tok", 1, minimum=1
    )
    intermediate_size = _required_int(config, "intermediate_size")

    cfg = Config(
        vocab_size=_required_int(config, "vocab_size"),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=_required_int(config, "num_hidden_layers"),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=_required_int(config, "num_key_value_heads"),
        head_dim=head_dim,
        hidden_act=_string(config, "hidden_act", "silu"),
        rms_norm_eps=_number(config, "rms_norm_eps", 1e-6),
        rope_theta=_number(config, "rope_theta", 1_000_000.0),
        max_position_embeddings=_optional_int(
            config, "max_position_embeddings", 4096, minimum=1
        ),
        attention_bias=_boolean(config, "attention_bias", False),
        tie_word_embeddings=_boolean(config, "tie_word_embeddings", False),
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=_optional_int(
            config,
            "moe_intermediate_size",
            intermediate_size,
            minimum=1,
        ),
    )
    return cfg


# Checkpoint sources whose tensor headers read_config needs before module
# selection. Qwen3 derives every dimension from config.json, so it has none.
config_sources = ()
