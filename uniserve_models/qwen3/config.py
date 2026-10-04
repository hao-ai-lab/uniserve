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
from uniserve.loading import checkpoint
from uniserve.nn.rope import RoPEScaling, YaRNScaling

from ..rotary import read_rotary


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable numerical parameters for Qwen decoders and conditioners.

    Field names follow the checkpoint ``config.json`` keys that
    ``read_config`` reads. ``head_dim`` may differ from
    ``hidden_size // num_attention_heads``. ``rope_scaling`` is a typed
    rotary recipe, None for plain rotary positions. ``mrope_sections`` gives
    the interleaved multimodal rotary widths of a Qwen3-VL language model;
    Qwen3 checkpoints rotate one position axis and leave it None.

    A zero ``num_experts`` builds a dense ``GatedMLP`` of width
    ``intermediate_size`` in every layer (Qwen3). A positive value makes
    layer ``i`` sparse when ``i`` is not in ``mlp_only_layers`` and
    ``(i + 1) % decoder_sparse_step == 0`` (Qwen3-MoE): a sparse layer routes
    each token to ``num_experts_per_tok`` experts of width
    ``moe_intermediate_size`` gated by ``hidden_act``, and any other layer
    keeps the dense MLP.

    Raises:
        ValueError: From ``__post_init__`` when a field has the wrong type or
            an invalid value, including odd ``head_dim``, query heads not
            divisible by KV heads, ``num_experts_per_tok`` above a nonzero
            ``num_experts``, an unsupported ``hidden_act``, or an expert
            activation no expert kernel implements.
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
    rope_scaling: RoPEScaling | None
    max_position_embeddings: int
    attention_bias: bool
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    # Whether the selected experts' softmax weights renormalize to one.
    norm_topk_prob: bool
    decoder_sparse_step: int
    mlp_only_layers: tuple[int, ...]
    mrope_sections: tuple[int, ...] | None = None

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
            "decoder_sparse_step",
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

        if not isinstance(self.mlp_only_layers, tuple) or any(
            not isinstance(layer, int)
            or isinstance(layer, bool)
            or not 0 <= layer < self.num_hidden_layers
            for layer in self.mlp_only_layers
        ):
            raise ValueError(
                "Qwen3 mlp_only_layers must be a tuple of layer indices"
            )
        # Only static recipes: positions alone determine their factors, as
        # the decoder's rotary evaluation assumes.
        if self.rope_scaling is not None and not isinstance(
            self.rope_scaling, YaRNScaling
        ):
            raise ValueError(
                "Qwen3 supports the default and YaRN rotary recipes"
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

        for name in ("attention_bias", "tie_word_embeddings", "norm_topk_prob"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"Qwen3 {name} must be boolean")

        sections = self.mrope_sections
        if sections is not None and (
            not isinstance(sections, tuple)
            or any(type(width) is not int or width < 1 for width in sections)
            or sum(sections) != self.head_dim // 2
        ):
            raise ValueError(
                "Qwen3 mrope_sections must be positive widths summing to "
                "half the head dimension"
            )

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
        if self.num_experts and self.hidden_act not in _EXPERT_ACTIVATIONS:
            raise ValueError(
                f"Qwen3-MoE experts do not implement {self.hidden_act!r}; "
                "expert kernels gate with SiLU or tanh-approximated GELU"
            )

    def sparse(self, layer: int) -> bool:
        """Return whether ``layer`` routes tokens to experts."""
        return (
            self.num_experts > 0
            and layer not in self.mlp_only_layers
            and (layer + 1) % self.decoder_sparse_step == 0
        )


# ``hidden_act`` aliases and the expert gating each one names.
_EXPERT_ACTIVATIONS = {
    "silu": "silu",
    "swish": "silu",
    "silu_and_mul": "silu",
    "swiglu": "silu",
    "gelu_pytorch_tanh": "gelu_tanh",
    "gelu_tanh": "gelu_tanh",
}


def expert_activation(hidden_act: str) -> str:
    """Name the ``FusedMoE`` gating that ``hidden_act`` specifies."""
    return _EXPERT_ACTIVATIONS[hidden_act]


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
    return _positive_float(config.get(name, default), name)


def _positive_float(raw: object, name: str) -> float:
    """Validate one finite positive number, rejecting booleans."""
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


def read_config(
    root: Path,
    io: loading.Config,
    *,
    sources: Mapping[str, checkpoint.Source],
) -> Config:
    """Normalize checkpoint metadata into immutable decoder configuration.

    ``architectures`` selects the family: ``Qwen3ForCausalLM`` (dense,
    ``model_type`` qwen3) or ``Qwen3MoeForCausalLM`` (``model_type``
    qwen3_moe). Each family's absent fields take Transformers' defaults for
    that family, so a field means the same here as in the reference.

    Args:
        root: Local checkpoint directory containing ``config.json``.
        io: Loading options of the package ``read_config`` contract; Qwen3
            reads only the local ``config.json`` and does not use them.
        sources: The resolved ``config_sources`` of the package contract;
            Qwen3 declares none.

    Returns:
        The validated ``Config``.

    Raises:
        ValueError: For invalid metadata, including an unknown or
            inconsistent architecture and ``model_type``, a missing required
            field, a field of the wrong type or value, conflicting aliases,
            sliding-window attention, a rotary recipe other than default or
            YaRN, a partial rotary width, and expert fields in a dense
            checkpoint.
    """
    config = json.loads((root / "config.json").read_text())

    # The dense and mixture-of-experts families parse the same keys with
    # different defaults and meanings, so the checkpoint must say which one
    # it is, consistently.
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or architectures[0] not in _FAMILIES
    ):
        raise ValueError(
            "Qwen3 checkpoint must declare exactly one of the architectures "
            f"{sorted(_FAMILIES)}"
        )
    family = _FAMILIES[architectures[0]]
    if config.get("model_type", family.model_type) != family.model_type:
        raise ValueError(
            f"{architectures[0]} checkpoint has model_type "
            f"{config['model_type']!r}, not {family.model_type!r}"
        )

    hidden_size = _required_int(config, "hidden_size")
    num_attention_heads = _required_int(config, "num_attention_heads")
    num_hidden_layers = _required_int(config, "num_hidden_layers")

    # A dense checkpoint without head_dim uses Transformers' fixed 128; the
    # mixture-of-experts family derives it from the attention width.
    if family.default_head_dim is None:
        if "head_dim" not in config and hidden_size % num_attention_heads:
            raise ValueError(
                "Qwen3-MoE hidden_size must be divisible by num_attention_heads"
            )
        default_head_dim = hidden_size // num_attention_heads
    else:
        default_head_dim = family.default_head_dim
    head_dim = _optional_int(config, "head_dim", default_head_dim, minimum=1)

    # A dense checkpoint's explicit null KV head count means one KV head per
    # query head (Qwen3Config.__post_init__). An absent count would take a
    # fixed class default unrelated to the query heads, so it is required.
    if family.experts is None and config.get("num_key_value_heads", 0) is None:
        num_key_value_heads = num_attention_heads
    else:
        num_key_value_heads = _required_int(config, "num_key_value_heads")

    # Sliding-window attention changes the visible keys: dense checkpoints
    # window the layers from max_window_layers (or those layer_types marks),
    # mixture-of-experts checkpoints window every layer. Neither is served.
    if _boolean(config, "use_sliding_window", False):
        raise ValueError(
            f"{architectures[0]} sliding-window attention is not supported"
        )
    layer_types = config.get("layer_types")
    if layer_types is not None and (
        not isinstance(layer_types, list)
        or any(kind != "full_attention" for kind in layer_types)
    ):
        raise ValueError(
            "Qwen3 layer_types other than full_attention are not supported"
        )

    max_position_embeddings = _optional_int(
        config, "max_position_embeddings", 32768, minimum=1
    )
    rotary = read_rotary(
        config,
        owner="Qwen3",
        default_theta=10000.0,
        default_original=max_position_embeddings,
    )
    if rotary.kind not in {"default", "yarn"}:
        raise ValueError(
            f"Qwen3 supports the default and YaRN rotary recipes, not "
            f"{rotary.kind!r}"
        )
    # Transformers' default Qwen3 rotary ignores a partial factor, while
    # its YaRN honors one; only full-width rotation has one meaning here.
    if rotary.partial_rotary_factor != 1.0:
        raise ValueError("Qwen3 rotary positions must span the full head")

    intermediate_size = _required_int(config, "intermediate_size")
    experts = family.experts
    if experts is None:
        # Dense checkpoints carry no expert fields; one here would mean a
        # mislabeled mixture-of-experts checkpoint.
        present = sorted(set(config) & _EXPERT_FIELDS)
        if present:
            raise ValueError(
                f"Qwen3ForCausalLM checkpoint has expert fields {present}"
            )
        num_experts, num_experts_per_tok = 0, 1
        moe_intermediate_size, norm_topk_prob = intermediate_size, False
        decoder_sparse_step, mlp_only_layers = 1, ()
    else:
        # Transformers 5 serializes the expert count as num_local_experts;
        # released checkpoints name it num_experts.
        if "num_local_experts" in config:
            if (
                "num_experts" in config
                and config["num_experts"] != config["num_local_experts"]
            ):
                raise ValueError(
                    "Qwen3-MoE checkpoint has conflicting expert-count aliases"
                )
            config["num_experts"] = config["num_local_experts"]
        num_experts = _optional_int(
            config, "num_experts", experts.num_experts, minimum=0
        )
        num_experts_per_tok = _optional_int(
            config,
            "num_experts_per_tok",
            experts.num_experts_per_tok,
            minimum=1,
        )
        moe_intermediate_size = _optional_int(
            config,
            "moe_intermediate_size",
            experts.moe_intermediate_size,
            minimum=1,
        )
        norm_topk_prob = _boolean(config, "norm_topk_prob", False)
        decoder_sparse_step = _optional_int(
            config, "decoder_sparse_step", 1, minimum=1
        )
        raw_layers = config.get("mlp_only_layers") or []
        if not isinstance(raw_layers, list):
            raise ValueError("Qwen3-MoE mlp_only_layers must be a list")
        mlp_only_layers = tuple(raw_layers)

    return Config(
        vocab_size=_required_int(config, "vocab_size"),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        hidden_act=_string(config, "hidden_act", "silu"),
        rms_norm_eps=_number(config, "rms_norm_eps", 1e-6),
        rope_theta=_positive_float(rotary.theta, "rope_theta"),
        rope_scaling=rotary.scaling,
        max_position_embeddings=max_position_embeddings,
        attention_bias=_boolean(config, "attention_bias", False),
        tie_word_embeddings=_boolean(config, "tie_word_embeddings", False),
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        norm_topk_prob=norm_topk_prob,
        decoder_sparse_step=decoder_sparse_step,
        mlp_only_layers=mlp_only_layers,
    )


@dataclass(frozen=True, slots=True)
class _Experts:
    """Transformers' expert defaults for a checkpoint that omits them."""

    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int


@dataclass(frozen=True, slots=True)
class _Family:
    """How one Qwen3 architecture interprets its ``config.json``.

    ``default_head_dim`` is None when the head width derives from the
    attention width; ``experts`` is None for dense checkpoints.
    """

    model_type: str
    default_head_dim: int | None
    experts: _Experts | None


# Defaults follow Transformers' Qwen3Config and Qwen3MoeConfig.
_FAMILIES = {
    "Qwen3ForCausalLM": _Family("qwen3", 128, None),
    "Qwen3MoeForCausalLM": _Family("qwen3_moe", None, _Experts(128, 8, 768)),
}

_EXPERT_FIELDS = frozenset(
    {
        "num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "norm_topk_prob",
        "decoder_sparse_step",
        "mlp_only_layers",
    }
)


# Checkpoint sources whose tensor headers read_config needs before module
# selection. Qwen3 derives every dimension from config.json, so it has none.
config_sources = ()
