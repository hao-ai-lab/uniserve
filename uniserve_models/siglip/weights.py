"""SigLIP checkpoint tower assignments.

Parameter names of ``Encoder`` translate to the Hugging Face
``SiglipVisionModel`` naming below the vision tower's prefix (``embeddings``,
``encoder.layers.N.self_attn``, ``post_layernorm``). The composing model
supplies that prefix and owns the ``weights.ModuleMapping`` these assignments
feed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniserve.loading import checkpoint, weights

if TYPE_CHECKING:
    from .encoder import Encoder


def assignments(
    model: Encoder, reader: checkpoint.Reader, *, prefix: str = ""
) -> tuple[weights.Assignment, ...]:
    """Map SigLIP's checkpoint tower, whose patch matrix already uses HWC order."""  # noqa: E501
    result = []
    available = frozenset(reader.names())
    for name, parameter in model.named_parameters():
        # The checkpoint's ``embeddings.patch_embedding.weight`` must already
        # be a linear ``[hidden, patch_size**2 * channels]`` matrix whose
        # columns follow ``patchify``'s (pixel row, pixel column, channel)
        # order; this mapping applies no convolution-to-linear reshape.
        if name.startswith(("patch_embedding.", "position_embedding.")):
            source = "embeddings." + name
        elif name.startswith("encoder.norm."):
            source = "post_layernorm." + name.removeprefix("encoder.norm.")
        else:
            # Transformer layers translate through several renames; each
            # replace is a no-op for names that do not contain its pattern.
            source = name.replace(".input_norm.", ".layer_norm1.")
            source = source.replace(".output_norm.", ".layer_norm2.")
            source = source.replace(
                ".attention.output.", ".self_attn.out_proj."
            )
            for branch in ("q", "k", "v"):
                source = source.replace(
                    f".attention.qkv.projections.{branch}.",
                    f".self_attn.{branch}_proj.",
                )
            source = source.replace(".mlp.0.", ".mlp.fc1.").replace(
                ".mlp.2.", ".mlp.fc2."
            )

        # Only names present in the reader become assignments. The caller's
        # ``weights.ModuleMapping`` required set decides whether a parameter
        # with no assignment fails the load.
        if prefix + source in available:
            result.append(
                weights.Assignment(parameter, reader.get(prefix + source))
            )
    return tuple(result)
