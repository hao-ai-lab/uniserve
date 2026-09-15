"""SigLIP checkpoint tower assignments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniserve.loading import checkpoint, weights

if TYPE_CHECKING:
    from .encoder import Encoder


def assignments(
    model: Encoder, reader: checkpoint.Reader, *, prefix: str = ""
) -> tuple[weights.Assignment, ...]:
    """Map SigLIP's checkpoint tower, whose patch matrix already uses HWC order."""
    result = []
    available = frozenset(reader.names())
    for name, parameter in model.named_parameters():
        if name.startswith(("patch_embedding.", "position_embedding.")):
            source = "embeddings." + name
        elif name.startswith("encoder.norm."):
            source = "post_layernorm." + name.removeprefix("encoder.norm.")
        else:
            source = name.replace(".input_norm.", ".layer_norm1.")
            source = source.replace(".output_norm.", ".layer_norm2.")
            source = source.replace(".attention.output.", ".self_attn.out_proj.")
            for branch in ("q", "k", "v"):
                source = source.replace(
                    f".attention.qkv.projections.{branch}.", f".self_attn.{branch}_proj."
                )
            source = source.replace(".mlp.0.", ".mlp.fc1.").replace(".mlp.2.", ".mlp.fc2.")
        if prefix + source in available:
            result.append(weights.Assignment(parameter, reader.get(prefix + source)))
    return tuple(result)
