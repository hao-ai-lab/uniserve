"""Immutable DiffusionGemma architecture and checkpoint metadata normalization.

``read_config`` turns a checkpoint's metadata into the frozen ``Config`` the
model modules consume: the Gemma-4 text stack shared by the causal prompt
pass and the canvas denoiser, the vision tower, and the block-diffusion token
constants. Unsupported mathematical options are rejected here, at the loading
boundary, so modules never reparse metadata.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from uniserve import loading
from uniserve.loading import checkpoint
from uniserve.nn.rope import ProportionalScaling
from uniserve_models.rotary import Rotary, read_rotary

__all__ = [
    "Config",
    "DiffusionConfig",
    "LayerAttention",
    "TextConfig",
    "VisionConfig",
    "config_sources",
    "read_config",
]


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"DiffusionGemma {name} must be a positive integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"DiffusionGemma {name} must be finite and positive")
    return float(value)


@dataclass(frozen=True, slots=True)
class LayerAttention:
    """Attention dimensions and history of one text layer.

    ``window`` is the visible history in tokens (``None`` for full layers).
    The checkpoint's ``sliding_window`` counts the query itself, so its 1024
    keys normalize to a history of 1023 tokens here, once. Full layers have
    no value projection: their values are the normalized key projection.
    ``rotary`` is the layer kind's recipe: sliding layers rotate every
    channel pair; full layers rotate proportionally, turning the leading
    ``partial_rotary_factor`` of the pairs at frequencies of the full head
    width and leaving the rest at a zero angle.
    """

    kind: Literal["sliding", "full"]
    head_dim: int
    num_kv_heads: int
    window: int | None
    rotary: Rotary

    def __post_init__(self) -> None:
        if self.kind not in ("sliding", "full"):
            raise ValueError("DiffusionGemma layers are sliding or full")
        _positive_int(self.head_dim, "head_dim")
        _positive_int(self.num_kv_heads, "kv heads")
        if self.head_dim % 2:
            raise ValueError("DiffusionGemma head_dim must be even")
        if (self.kind == "sliding") != (self.window is not None):
            raise ValueError("only sliding layers carry a history window")
        if self.window is not None and (
            type(self.window) is not int or self.window < 0
        ):
            raise ValueError("DiffusionGemma windows are token counts")

    @property
    def value_projection(self) -> bool:
        """Whether the layer projects values separately from keys."""
        return self.kind == "sliding"


@dataclass(frozen=True, slots=True)
class TextConfig:
    """The Gemma-4 mixture-of-experts text stack.

    Every layer runs a dense gated MLP of ``intermediate_size`` in parallel
    with ``num_experts`` routed experts of ``moe_intermediate_size``, of
    which each token selects ``top_k_experts``.
    """

    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    layers: tuple[LayerAttention, ...]
    intermediate_size: int
    num_experts: int
    top_k_experts: int
    moe_intermediate_size: int
    rms_norm_eps: float
    final_logit_softcapping: float
    max_position_embeddings: int

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "num_attention_heads",
            "intermediate_size",
            "num_experts",
            "top_k_experts",
            "moe_intermediate_size",
            "max_position_embeddings",
        ):
            _positive_int(getattr(self, name), name)
        if not self.layers or not all(
            isinstance(layer, LayerAttention) for layer in self.layers
        ):
            raise ValueError("DiffusionGemma requires typed text layers")
        if any(
            self.num_attention_heads % layer.num_kv_heads
            for layer in self.layers
        ):
            raise ValueError(
                "query heads must be a multiple of every layer's KV heads"
            )
        if self.top_k_experts > self.num_experts:
            raise ValueError("expert top-k cannot exceed the expert count")
        _positive_float(self.rms_norm_eps, "rms_norm_eps")
        _positive_float(self.final_logit_softcapping, "logit softcapping")

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layers)

    @property
    def embed_scale(self) -> float:
        """Token embeddings multiply by sqrt(hidden_size)."""
        return self.hidden_size**0.5


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """The Gemma-4 vision tower: patch encoder, pooler and standardization.

    Images are cut into ``patch_size`` squares; ``pooling_kernel_size``
    squares of patches average into one soft token, and an image holds at
    most ``soft_tokens_per_image`` soft tokens.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    intermediate_size: int
    patch_size: int
    pooling_kernel_size: int
    position_embedding_size: int
    rope_theta: float
    rms_norm_eps: float
    standardize: bool
    soft_tokens_per_image: int

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "head_dim",
            "intermediate_size",
            "patch_size",
            "pooling_kernel_size",
            "position_embedding_size",
            "soft_tokens_per_image",
        ):
            _positive_int(getattr(self, name), f"vision {name}")
        _positive_float(self.rope_theta, "vision rope theta")
        _positive_float(self.rms_norm_eps, "vision rms_norm_eps")
        if type(self.standardize) is not bool:
            raise ValueError("vision standardization is a boolean choice")
        if self.head_dim % 4:
            raise ValueError(
                "vision heads rotate two spatial axes of split-half pairs"
            )

    @property
    def max_patches(self) -> int:
        """Patch budget of one image: its soft tokens times the pooling area."""
        return self.soft_tokens_per_image * self.pooling_kernel_size**2


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    """Canvas length of block diffusion and the special tokens it uses.

    ``mask_token_id`` fills undecided canvas slots, ``pad_token_id`` pads
    canvases, ``end_of_turn_id`` closes a turn, and ``eos_token_ids`` stop
    generation. Image soft tokens sit between the begin- and end-of-image
    tokens.
    """

    canvas_length: int
    mask_token_id: int
    pad_token_id: int
    end_of_turn_id: int
    eos_token_ids: tuple[int, ...]
    image_token_id: int
    begin_image_token_id: int
    end_image_token_id: int

    def __post_init__(self) -> None:
        _positive_int(self.canvas_length, "canvas_length")
        tokens = (
            self.mask_token_id,
            self.pad_token_id,
            self.end_of_turn_id,
            self.image_token_id,
            self.begin_image_token_id,
            self.end_image_token_id,
            *self.eos_token_ids,
        )
        if not self.eos_token_ids or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise ValueError("DiffusionGemma special tokens are token ids")


@dataclass(frozen=True, slots=True)
class Config:
    text: TextConfig
    vision: VisionConfig
    diffusion: DiffusionConfig


def _rotary(parameters: object, kind: str, context: int) -> Rotary:
    """Resolve one layer kind's recipe from its ``rope_parameters`` entry.

    Sliding layers require the default recipe over the whole head and full
    layers the proportional one, both without frequency scaling.
    """
    if not isinstance(parameters, Mapping) or "rope_theta" not in parameters:
        raise ValueError(
            f"DiffusionGemma needs rope_parameters with a theta for {kind}"
        )
    rotary = read_rotary(
        {"rope_parameters": parameters},
        owner="DiffusionGemma",
        default_theta=parameters["rope_theta"],
        default_original=context,
    )
    _positive_float(rotary.theta, f"{kind} rope_theta")
    expected = "proportional" if kind == "full_attention" else "default"
    if rotary.kind != expected:
        layers = "full" if kind == "full_attention" else "sliding"
        raise ValueError(
            f"DiffusionGemma {layers} layers require {expected} rope"
        )
    if not 0 < rotary.partial_rotary_factor <= 1 or (
        expected == "default" and rotary.partial_rotary_factor != 1
    ):
        raise ValueError("DiffusionGemma partial rotary factor is unsupported")
    if rotary.scaling is None:
        return rotary

    # Transformers divides proportional frequencies by a factor that
    # defaults to one; no released checkpoint scales them.
    if rotary.scaling.factor not in (None, 1.0):
        raise ValueError("DiffusionGemma rope scaling factors are unsupported")
    return replace(rotary, scaling=ProportionalScaling(1.0))


def _text(config: Mapping) -> TextConfig:
    if config.get("use_bidirectional_attention") != "vision":
        raise ValueError(
            "DiffusionGemma requires bidirectional attention within images only"
        )
    if config.get("attention_bias", False):
        raise ValueError("DiffusionGemma attention has no biases")
    if config.get("hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError("DiffusionGemma requires tanh-approximated GELU")

    rope = config.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("DiffusionGemma needs per-kind rope_parameters")
    # The checkpoint's sliding window counts the query itself; the visible
    # history is one token shorter.
    window = _positive_int(config.get("sliding_window"), "sliding_window") - 1
    rotaries = {
        kind: _rotary(
            rope.get(kind), kind, config.get("max_position_embeddings")
        )
        for kind in ("sliding_attention", "full_attention")
    }
    layers = []
    for kind in config.get("layer_types", ()):
        if kind == "sliding_attention":
            layers.append(
                LayerAttention(
                    "sliding",
                    _positive_int(config.get("head_dim"), "head_dim"),
                    _positive_int(
                        config.get("num_key_value_heads"), "kv heads"
                    ),
                    window,
                    rotaries[kind],
                )
            )
        elif kind == "full_attention":
            layers.append(
                LayerAttention(
                    "full",
                    _positive_int(
                        config.get("global_head_dim"), "global_head_dim"
                    ),
                    _positive_int(
                        config.get("num_global_key_value_heads"),
                        "global kv heads",
                    ),
                    None,
                    rotaries[kind],
                )
            )
        else:
            raise ValueError(f"unsupported DiffusionGemma layer type {kind!r}")
    if len(layers) != config.get("num_hidden_layers"):
        raise ValueError("layer_types must cover every hidden layer")

    return TextConfig(
        vocab_size=config.get("vocab_size"),
        hidden_size=config.get("hidden_size"),
        num_attention_heads=config.get("num_attention_heads"),
        layers=tuple(layers),
        intermediate_size=config.get("intermediate_size"),
        num_experts=config.get("num_experts"),
        top_k_experts=config.get("top_k_experts"),
        moe_intermediate_size=config.get("moe_intermediate_size"),
        rms_norm_eps=_positive_float(
            config.get("rms_norm_eps"), "rms_norm_eps"
        ),
        final_logit_softcapping=_positive_float(
            config.get("final_logit_softcapping"), "final_logit_softcapping"
        ),
        max_position_embeddings=config.get("max_position_embeddings"),
    )


def _vision(config: Mapping, soft_tokens: int) -> VisionConfig:
    if config.get("use_clipped_linears", False):
        raise ValueError(
            "DiffusionGemma clipped vision linears are unsupported"
        )
    if config.get("attention_bias", False):
        raise ValueError("DiffusionGemma vision attention has no biases")
    if config.get("hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError("DiffusionGemma vision requires tanh GELU")
    rope = config.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError("DiffusionGemma vision requires default rope")
    heads = _positive_int(config.get("num_attention_heads"), "vision heads")
    if config.get("num_key_value_heads", heads) != heads:
        raise ValueError("DiffusionGemma vision attention is multi-head")
    return VisionConfig(
        hidden_size=config.get("hidden_size"),
        num_hidden_layers=config.get("num_hidden_layers"),
        num_attention_heads=heads,
        head_dim=config.get("head_dim"),
        intermediate_size=config.get("intermediate_size"),
        patch_size=config.get("patch_size"),
        pooling_kernel_size=config.get("pooling_kernel_size"),
        position_embedding_size=config.get("position_embedding_size"),
        rope_theta=_positive_float(rope.get("rope_theta"), "vision rope_theta"),
        rms_norm_eps=_positive_float(
            config.get("rms_norm_eps"), "vision rms_norm_eps"
        ),
        standardize=config.get("standardize", False),
        soft_tokens_per_image=soft_tokens,
    )


def _special_tokens(root: Path) -> dict[str, int]:
    """Resolve the canvas's special tokens from the tokenizer's declarations.

    ``tokenizer_config.json`` names the mask, pad and end-of-turn tokens;
    ``tokenizer.json`` lists their ids among its added tokens.
    """
    declared = json.loads((root / "tokenizer_config.json").read_text())
    added = json.loads((root / "tokenizer.json").read_text())["added_tokens"]
    vocabulary = {token["content"]: token["id"] for token in added}
    result = {}
    for role in ("mask_token", "pad_token", "eot_token"):
        token = declared.get(role)
        if not isinstance(token, str) or token not in vocabulary:
            raise ValueError(f"DiffusionGemma tokenizer lacks its {role}")
        result[role] = vocabulary[token]
    return result


def read_config(
    root: Path,
    io: loading.Config,
    *,
    sources: Mapping[str, checkpoint.Source],
) -> Config:
    """Normalize the checkpoint's model, generation and tokenizer metadata.

    Reads ``config.json``, the stop tokens of ``generation_config.json``
    when present, and the canvas special tokens of the tokenizer files.
    DiffusionGemma derives no dimension from tensor headers, so ``io`` and
    ``sources`` go unused.

    Raises:
        ValueError: For missing fields, unsupported mathematical options or
            inconsistent special tokens.
    """
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "diffusion_gemma":
        raise ValueError("DiffusionGemma requires model_type diffusion_gemma")
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("DiffusionGemma needs text_config and vision_config")
    if not config.get("tie_word_embeddings", True):
        raise ValueError("DiffusionGemma ties its head to the token embedding")

    generation_path = root / "generation_config.json"
    generation = (
        json.loads(generation_path.read_text())
        if generation_path.is_file()
        else {}
    )
    eos = generation.get("eos_token_id", config.get("eos_token_id"))
    eos = (eos,) if type(eos) is int else tuple(eos or ())

    soft_tokens = _positive_int(
        config.get("vision_soft_tokens_per_image"), "soft tokens per image"
    )
    special = _special_tokens(root)
    return Config(
        text=_text(text),
        vision=_vision(vision, soft_tokens),
        diffusion=DiffusionConfig(
            canvas_length=_positive_int(
                config.get("canvas_length"), "canvas_length"
            ),
            mask_token_id=special["mask_token"],
            pad_token_id=special["pad_token"],
            end_of_turn_id=special["eot_token"],
            eos_token_ids=eos,
            image_token_id=config.get("image_token_id"),
            begin_image_token_id=config.get("boi_token_id"),
            end_image_token_id=config.get("eoi_token_id"),
        ),
    )


# DiffusionGemma derives every dimension from its JSON metadata.
config_sources = ()
