"""Immutable Qwen3-VL vision tower architecture and its normalization.

``read_vision_config`` turns a checkpoint's ``vision_config`` object (in a
``Qwen3VLForConditionalGeneration`` ``config.json``) into the frozen
``VisionConfig`` that ``VisionTower`` consumes. ``VisionConfig`` validates
its fields on construction, so a directly built config receives the same
checks as one read from a checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

# Activations a non-gated vision MLP may name; ``get_act_fn`` builds them.
_ACTIVATIONS = frozenset({"gelu_pytorch_tanh", "gelu", "silu", "relu"})


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """Dimensions of the Qwen3-VL vision encoder and its token mergers.

    Pixels arrive as tubelets of ``in_channels x temporal_patch_size x
    patch_size x patch_size`` values. ``depth`` blocks of width
    ``hidden_size`` with ``num_heads`` heads encode them; the learned
    position table holds ``num_position_embeddings`` entries on a square
    grid. Every ``spatial_merge_size x spatial_merge_size`` patch block
    merges into one ``out_hidden_size`` token, the language model's width.
    ``deepstack_visual_indexes`` names the blocks, in increasing order,
    whose outputs feed the DeepStack mergers; the ``j``-th merger's tokens
    join the language model after its ``j``-th decoder layer.

    Raises:
        ValueError: From ``__post_init__`` for a non-positive or non-integer
            size, heads that do not divide ``hidden_size``, a head width
            that does not split into two even rotary halves, a non-square
            position table, DeepStack blocks outside ``0..depth-1`` or out
            of order, or an unsupported activation.
    """

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    hidden_act: str
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    num_position_embeddings: int
    deepstack_visual_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in {"hidden_act", "deepstack_visual_indexes"}:
                continue
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"Qwen3-VL vision {field.name} must be a positive integer"
                )
        if self.hidden_size % self.num_heads:
            raise ValueError(
                "Qwen3-VL vision heads must divide the hidden size"
            )
        # The 2D rotary embedding rotates half of each head by the patch row
        # and half by its column, each as split-half pairs.
        if (self.hidden_size // self.num_heads) % 4:
            raise ValueError(
                "Qwen3-VL vision head width must split into two even "
                "rotary halves"
            )
        if math.isqrt(self.num_position_embeddings) ** 2 != (
            self.num_position_embeddings
        ):
            raise ValueError(
                "Qwen3-VL vision position table must be a square grid"
            )
        indexes = self.deepstack_visual_indexes
        if (
            not isinstance(indexes, tuple)
            or any(type(index) is not int for index in indexes)
            or any(not 0 <= index < self.depth for index in indexes)
            or list(indexes) != sorted(set(indexes))
        ):
            raise ValueError(
                "Qwen3-VL DeepStack blocks must be increasing block indexes"
            )
        if self.hidden_act not in _ACTIVATIONS:
            raise ValueError(
                f"unsupported Qwen3-VL vision hidden_act {self.hidden_act!r}"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def position_grid(self) -> int:
        """Side of the square learned position table."""
        return math.isqrt(self.num_position_embeddings)

    @property
    def feature_size(self) -> int:
        """Width of one encoded token: its embedding and DeepStack features.

        ``VisionTower`` emits each merged token as the language embedding
        followed by one ``out_hidden_size`` DeepStack feature per DeepStack
        block, concatenated along the feature axis.
        """
        return (1 + len(self.deepstack_visual_indexes)) * self.out_hidden_size


def read_vision_config(metadata: Mapping[str, object]) -> VisionConfig:
    """Normalize a checkpoint ``vision_config`` object.

    Every ``VisionConfig`` field is required; the JSON list of DeepStack
    blocks becomes a tuple. Other keys (``model_type``,
    ``initializer_range``) carry no numerical meaning and are ignored.

    Raises:
        ValueError: A field is missing or invalid.
    """
    missing = sorted(
        field.name
        for field in fields(VisionConfig)
        if field.name not in metadata
    )
    if missing:
        raise ValueError(
            f"Qwen3-VL vision_config is missing fields: {', '.join(missing)}"
        )
    values = {
        field.name: metadata[field.name] for field in fields(VisionConfig)
    }
    indexes = values["deepstack_visual_indexes"]
    if not isinstance(indexes, (list, tuple)):
        raise ValueError(
            "Qwen3-VL vision_config.deepstack_visual_indexes must be a list"
        )
    values["deepstack_visual_indexes"] = tuple(indexes)
    return VisionConfig(**values)
