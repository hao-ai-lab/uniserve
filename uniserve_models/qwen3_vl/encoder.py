"""Qwen3-VL language layers conditioned on encoded images and video blocks.

``TextEncoder`` composes the vision encoder (``vision``) with the retained
decoder layers of the language model. ``vision.encode`` turns images and
video blocks into merged tokens; ``encode`` splices those tokens into a
prompt at its image and video placeholders: each token's embedding replaces
the placeholder's token embedding, and its ``j``-th DeepStack feature is
added to the residual stream after retained layer ``j``.
"""

from __future__ import annotations

import torch
from torch import nn

from uniserve.model import (
    EmbeddingReplacement,
    PatchEncoder,
    TransformerDecoder,
)
from uniserve.model import TextEncoder as BaseTextEncoder

from .config import VisionConfig
from .vision import VisionTower


class TextEncoder(BaseTextEncoder):
    """Encode prompts whose placeholders carry Qwen3-VL vision features.

    ``network`` is the language decoder, with the M-RoPE sections of the
    checkpoint, and ``retained_layers`` the decoder layers this encoder
    runs (``uniserve.model.TextEncoder``). ``vision`` is a ``PatchEncoder``
    over ``VisionTower``: it encodes ``VisionInput`` samples of packed
    tubelet rows with their ``(time, height, width)`` grids into
    ``[time * height * width / merge**2, vision.feature_size]`` rows of
    ``dtype``, each a token's embedding followed by its DeepStack features.
    Prompts mark those tokens with ``image_token_id`` or ``video_token_id``.
    """

    def __init__(
        self,
        network: TransformerDecoder,
        retained_layers: tuple[int, ...],
        vision: VisionConfig,
        *,
        image_token_id: int,
        video_token_id: int,
        dtype: torch.dtype,
    ):
        super().__init__(network, retained_layers)
        if vision.out_hidden_size != network.hidden_size:
            raise ValueError(
                "Qwen3-VL vision tokens must match the language width"
            )
        if len(vision.deepstack_visual_indexes) > len(retained_layers):
            raise ValueError(
                "every DeepStack feature requires a retained decoder layer"
            )
        if image_token_id == video_token_id:
            raise ValueError("image and video placeholders must differ")
        self.vision = PatchEncoder(
            VisionTower(vision),
            nn.Identity(),
            patch_size=vision.patch_size,
            downsample=vision.spatial_merge_size,
            output_size=vision.feature_size,
            output_dtype=dtype,
        )
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.deepstack_layers = len(vision.deepstack_visual_indexes)

    def encode(
        self,
        tokens: tuple[torch.Tensor, ...],
        *,
        positions: tuple[torch.Tensor, ...] | None = None,
        visual: tuple[torch.Tensor | None, ...] | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """Encode prompts, splicing vision tokens at their placeholders.

        Args:
            tokens: One ``[length]`` token sequence per prompt.
            positions: One ``[3, length]`` M-RoPE coordinate tensor per
                prompt (``rope_index``). Without them every axis takes the
                text coordinates ``0..length-1``.
            visual: Per prompt, the ``vision.encode`` rows of its vision
                blocks concatenated in the order of its placeholder tokens,
                ``[placeholders, vision.feature_size]``, or None for a prompt
                without placeholders.

        Returns:
            Per prompt ``[length, hidden]`` features on the final pipeline
            stage; None on earlier stages.

        Raises:
            ValueError: A prompt's vision rows do not match its placeholder
                count, or the inputs do not align with the prompts.
        """
        if visual is None:
            return super().encode(tokens, positions=positions)
        if len(visual) != len(tokens):
            raise ValueError("vision rows must align with the prompts")

        # [length] placeholder masks; one host read checks every prompt's
        # placeholder count against its vision rows.
        masks = tuple(
            (value == self.image_token_id) | (value == self.video_token_id)
            for value in tokens
        )
        counts = (
            torch.stack(tuple(mask.sum() for mask in masks)).tolist()
            if masks
            else []
        )
        hidden = self.network.hidden_size
        embeddings, deepstack = [], []
        for mask, count, rows in zip(masks, counts, visual, strict=True):
            if rows is None:
                if count:
                    raise ValueError(
                        "a prompt with vision placeholders requires their rows"
                    )
                embeddings.append(None)
                deepstack.append(None)
                continue
            if rows.shape != (count, (1 + self.deepstack_layers) * hidden):
                raise ValueError(
                    "vision rows must cover the prompt's placeholders"
                )

            # [placeholders, 1 + deepstack, hidden]: the embedding, then one
            # feature per DeepStack layer. Each is scattered onto the
            # prompt's rows; rows without a placeholder hold zeros.
            features = rows.unflatten(-1, (1 + self.deepstack_layers, hidden))
            dense = rows.new_zeros(
                (1 + self.deepstack_layers, mask.numel(), hidden)
            )
            dense[:, mask] = features.transpose(0, 1)
            embeddings.append(EmbeddingReplacement(dense[0], mask))
            deepstack.append(tuple(dense[1:]))
        return super().encode(
            tokens,
            positions=positions,
            embeddings=tuple(embeddings),
            deepstack=tuple(deepstack),
        )
