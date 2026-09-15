"""Qwen3 numerical composition and checkpoint metadata interpretation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from torch import nn

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.model import CausalLM, EntryPoint, TransformerDecoder
from uniserve.nn.attention import (
    Attention as ScaledAttention,
)
from uniserve.nn.attention import (
    AttentionInput,
    RotaryQKVProjection,
)
from uniserve.nn.functional import add_rms_norm
from uniserve.nn.linear import (
    Linear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelHead,
)
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.moe import FusedMoE
from uniserve.nn.norm import RMSNorm
from uniserve.nn.rope import RotaryEmbedding


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable numerical parameters for Qwen decoders and conditioners."""

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
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Qwen3 {name} must be a positive integer")
        if self.head_dim % 2:
            raise ValueError("Qwen3 head_dim must be even for rotary positions")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Qwen3 attention heads must be divisible by KV heads")
        if (
            not isinstance(self.num_experts, int)
            or isinstance(self.num_experts, bool)
            or self.num_experts < 0
        ):
            raise ValueError("Qwen3 num_experts must be a non-negative integer")
        if self.num_experts and self.num_experts_per_tok > self.num_experts:
            raise ValueError("Qwen3 num_experts_per_tok must not exceed num_experts")
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
            raise ValueError(f"unsupported Qwen3 hidden_act {self.hidden_act!r}")


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
        raise ValueError(f"Qwen3 config field {name!r} must be an integer >= {minimum}")
    return raw


def _number(config: Mapping[str, object], name: str, default: float) -> float:
    """Read a numeric model setting while rejecting boolean values."""

    raw = config.get(name, default)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(f"Qwen3 config field {name!r} must be numeric")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Qwen3 config field {name!r} must be finite and positive")
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
        raise ValueError(f"Qwen3 config field {name!r} must be a non-empty string")
    return raw


def read_config(root: Path, io: loading.Config) -> Config:
    """Normalize checkpoint metadata into immutable decoder configuration."""

    config = json.loads((root / "config.json").read_text())
    rotary = config.get("rope_parameters") or {}
    if not isinstance(rotary, Mapping):
        raise ValueError("Qwen3 rope_parameters must be an object")
    if rotary.get("rope_type", "default") != "default":
        raise ValueError("Qwen3 requires the default rotary embedding recipe")
    if "rope_theta" in rotary:
        if "rope_theta" in config and config["rope_theta"] != rotary["rope_theta"]:
            raise ValueError("Qwen3 checkpoint has conflicting rope_theta aliases")
        # Current Transformers serializes this numerical field under
        # rope_parameters. Constructors consume its single normalized value.
        config["rope_theta"] = rotary["rope_theta"]
    hidden_size = _required_int(config, "hidden_size")
    num_attention_heads = _required_int(config, "num_attention_heads")
    if "head_dim" not in config and hidden_size % num_attention_heads:
        raise ValueError("Qwen3 hidden_size must be divisible by num_attention_heads")
    head_dim = _optional_int(
        config,
        "head_dim",
        hidden_size // num_attention_heads,
        minimum=1,
    )
    num_experts = _optional_int(config, "num_experts", 0, minimum=0)
    num_experts_per_tok = _optional_int(config, "num_experts_per_tok", 1, minimum=1)
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
        max_position_embeddings=_optional_int(config, "max_position_embeddings", 4096, minimum=1),
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


def _activation(name: str) -> str:
    if name in {"silu", "swish", "silu_and_mul", "swiglu"}:
        return "silu"
    if name in {"gelu", "gelu_and_mul", "geglu"}:
        return "gelu"
    return "gelu_pytorch_tanh"


class Attention(nn.Module):
    """QK-normalized rotary attention followed by the output contraction."""

    def __init__(self, config: Config, layer: int):
        super().__init__()
        self.qkv = RotaryQKVProjection(
            QKVParallelLinear(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                bias=config.attention_bias,
            ),
            RMSNorm(config.head_dim, config.rms_norm_eps),
            RMSNorm(config.head_dim, config.rms_norm_eps),
        )
        self.attention = ScaledAttention(
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            cache_name=f"backbone.layers.{layer}.attention.attention",
        )
        self.output = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rotary = RotaryEmbedding(config.head_dim, theta=config.rope_theta)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, attention: AttentionInput):
        # This unscaled recipe depends only on positions, so computing factors
        # does not require a host mirror of the attention lengths.
        cos, sin = self.rotary(
            positions.reshape(-1), dtype=torch.float32, sequence_length=positions.numel()
        )
        query, key, value = self.qkv(hidden, (cos,), (sin,))
        attended = self.attention(query, key, value, attention)
        return self.output(attended.flatten(1))


class MoE(nn.Module):
    """Top-k expert selection with a replicated mathematical router."""

    def __init__(self, config: Config):
        super().__init__()
        self.router = Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = FusedMoE(
            [
                GatedMLP(config.hidden_size, config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ],
            top_k=config.num_experts_per_tok,
            norm_topk_prob=True,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.experts(hidden, self.router(hidden))


class TransformerLayer(nn.Module):
    """Qwen pre-normalization equations with a separate residual stream."""

    def __init__(self, config: Config, layer: int):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config, layer)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = (
            MoE(config)
            if config.num_experts
            else GatedMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=_activation(config.hidden_act),
            )
        )

    def forward(self, hidden, residual, positions, attention):
        if residual is None:
            residual = hidden
            hidden = self.input_norm(hidden)
        else:
            hidden, residual = add_rms_norm(
                hidden, residual, self.input_norm.weight, self.input_norm.eps
            )
        hidden = self.attention(hidden, positions, attention)
        hidden, residual = add_rms_norm(
            hidden, residual, self.post_attention_norm.weight, self.post_attention_norm.eps
        )
        return self.mlp(hidden), residual


class Transformer(TransformerDecoder):
    def __init__(self, config: Config):
        super().__init__(
            VocabParallelEmbedding(config.vocab_size, config.hidden_size),
            nn.ModuleDict(
                {
                    str(index): TransformerLayer(config, index)
                    for index in range(config.num_hidden_layers)
                }
            ),
            RMSNorm(config.hidden_size, config.rms_norm_eps),
        )
        self.config = config


class Model(CausalLM):
    """The same loaded Qwen computation for Python and serving callers."""

    def __init__(self, config: Config):
        super().__init__(
            Transformer(config), VocabParallelHead(config.hidden_size, config.vocab_size)
        )
        self.config = config
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight


checkpoint_sources = (checkpoint.Config(name="primary"),)
entry_paths = MappingProxyType({"model": "forward"})
precisions = MappingProxyType(
    {"bf16": weights.Config(), "fp16": weights.Config(dtype=torch.float16)}
)


def entry_points(config: Config) -> Mapping[str, tuple[EntryPoint, ...]]:
    return MappingProxyType(
        {
            "": (
                EntryPoint("forward", groups=("tp", "sp", "pp")),
                EntryPoint("embed_input_ids", stage="first", groups=("tp",)),
                EntryPoint("compute_logits", stage="last", groups=("tp",)),
            )
        }
    )


def _parameter_sources(config: Config) -> Mapping[str, str]:
    names = {
        "backbone.embedding.weight": "model.embed_tokens.weight",
        "backbone.norm.weight": "model.norm.weight",
        "lm_head.weight": "model.embed_tokens.weight"
        if config.tie_word_embeddings
        else "lm_head.weight",
    }
    for index in range(config.num_hidden_layers):
        target, source = f"backbone.layers.{index}", f"model.layers.{index}"
        names[f"{target}.input_norm.weight"] = f"{source}.input_layernorm.weight"
        names[f"{target}.post_attention_norm.weight"] = f"{source}.post_attention_layernorm.weight"
        for field in ("weight", "bias") if config.attention_bias else ("weight",):
            for branch in ("q", "k", "v"):
                names[f"{target}.attention.qkv.projection.projections.{branch}.{field}"] = (
                    f"{source}.self_attn.{branch}_proj.{field}"
                )
            names[f"{target}.attention.output.{field}"] = f"{source}.self_attn.o_proj.{field}"
        for branch in ("q", "k"):
            norm = "query_norm" if branch == "q" else "key_norm"
            names[f"{target}.attention.qkv.{norm}.weight"] = (
                f"{source}.self_attn.{branch}_norm.weight"
            )
        if config.num_experts:
            names[f"{target}.mlp.router.weight"] = f"{source}.mlp.gate.weight"
            mlps = tuple(
                (f"{target}.mlp.experts.experts.{expert}", f"{source}.mlp.experts.{expert}")
                for expert in range(config.num_experts)
            )
        else:
            mlps = ((f"{target}.mlp", f"{source}.mlp"),)
        for target_mlp, source_mlp in mlps:
            for branch in ("gate", "up"):
                names[f"{target_mlp}.gate_up.projections.{branch}.weight"] = (
                    f"{source_mlp}.{branch}_proj.weight"
                )
            names[f"{target_mlp}.down.weight"] = f"{source_mlp}.down_proj.weight"
    return names


def checkpoint_mappings(model: Model) -> tuple[weights.ModuleMapping, ...]:
    names = _parameter_sources(model.config)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    off_stage = frozenset(source for target, source in names.items() if target not in parameters)
    # Tied heads can have a redundant checkpoint copy; the embedding is the
    # unique source for their shared Parameter on either pipeline endpoint.
    if model.config.tie_word_embeddings:
        off_stage |= {"lm_head.weight"}
    used_sources = {names[target] for target in parameters}
    off_stage -= used_sources

    def assign(reader):
        available = frozenset(reader.names())
        result = []
        for target, parameter in parameters.items():
            source = names[target]
            if source not in available and source.startswith("model.layers."):
                source = source.removeprefix("model.")
            if source in available:
                result.append(weights.Assignment(parameter, reader.get(source)))
        return tuple(result)

    return (
        weights.ModuleMapping(
            model, "primary", assign, frozenset(parameters), nonresident=off_stage
        ),
    )
