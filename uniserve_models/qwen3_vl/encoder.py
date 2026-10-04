"""Qwen3-VL language layers conditioned on encoded images and video blocks.

``TextEncoder`` composes the vision encoder (``vision``) with the retained
decoder layers of the language model. ``vision.pack_pixels`` turns decoded
frames into the processor's patch rows, ``vision.encode`` turns those rows
into merged tokens, and ``encode`` splices the tokens into a prompt at its
image and video placeholders: each token's embedding replaces the
placeholder's token embedding, and its ``j``-th DeepStack feature is added
to the residual stream after retained layer ``j``. ``positions`` gives the
prompt's M-RoPE coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from uniserve.model import (
    EmbeddingReplacement,
    PatchEncoder,
    TransformerDecoder,
)
from uniserve.model import TextEncoder as BaseTextEncoder
from uniserve.tensors import OutputLayout

from .config import PixelConfig, VisionConfig
from .positions import rope_index
from .vision import VisionTower


class VisionEncoder(PatchEncoder):
    """Qwen3-VL's vision tower with its processor's pixel packing.

    ``encode`` takes ``[time * height * width, tubelet]`` patch rows with
    their ``(time, height, width)`` grid and returns one row per merged
    token (``VisionTower``). ``pack_pixels`` produces those rows from
    decoded frames exactly as the checkpoint's Qwen3-VL image and video
    processors do on the host.
    """

    def __init__(
        self, config: VisionConfig, pixels: PixelConfig, *, dtype: torch.dtype
    ):
        super().__init__(
            VisionTower(config),
            nn.Identity(),
            patch_size=config.patch_size,
            downsample=config.spatial_merge_size,
            output_size=config.feature_size,
            output_dtype=dtype,
        )
        self.temporal_patch_size = config.temporal_patch_size
        self.pixels = pixels

    def pack_pixels(
        self, frames: torch.Tensor, grid: tuple[int, int, int]
    ) -> torch.Tensor:
        """Resize, normalize and patch frames as the Qwen3-VL processors do.

        The frames are resized with antialiased bicubic interpolation in
        uint8 to the grid's pixel extent, normalized in FP32, and padded to a
        whole temporal patch by repeating the last frame; an image is one
        frame repeated. Each row is one patch's channels, then its temporal
        frames, then its pixel rows and columns, and rows run over each
        temporal patch in merge-block order. Every operation is the
        processors' own, in their order, so the rows equal theirs exactly.

        Raises:
            ValueError: The frames are not uint8 RGB, or their count does
                not fill the grid's temporal patches.
        """
        time, height, width = grid
        patch, merge = self.patch_size, self.downsample
        temporal = self.temporal_patch_size
        if (
            frames.ndim != 4
            or frames.dtype != torch.uint8
            or frames.shape[-1] != 3
        ):
            raise ValueError("vision frames must be uint8 RGB [F, H, W, 3]")
        if -(-frames.shape[0] // temporal) != time:
            raise ValueError("vision frames must fill the grid's time steps")

        # [F, 3, H, W] uint8, resized to the grid's pixels.
        values = tvF.resize(
            frames.permute(0, 3, 1, 2),
            [height * patch, width * patch],
            interpolation=tvF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        # The processors fold the 1/255 rescale into the statistics.
        mean = torch.tensor(self.pixels.mean) * (1.0 / (1 / 255))
        std = torch.tensor(self.pixels.std) * (1.0 / (1 / 255))
        values = tvF.normalize(values.to(torch.float32), mean, std)
        if missing := -values.shape[0] % temporal:
            values = torch.cat(
                (values, values[-1:].expand(missing, -1, -1, -1))
            )

        # [time, temporal, 3, rows, merge, patch, columns, merge, patch] ->
        # [time, rows, columns, merge, merge, 3, temporal, patch, patch].
        values = values.view(
            time,
            temporal,
            3,
            height // merge,
            merge,
            patch,
            width // merge,
            merge,
            patch,
        )
        values = values.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return values.reshape(time * height * width, 3 * temporal * patch**2)

    def pixels_layout(self, num_tokens: int) -> OutputLayout:
        shape = (
            num_tokens * self.downsample**2,
            3 * self.temporal_patch_size * self.patch_size**2,
        )
        return OutputLayout(
            shape,
            torch.float32,
            tuple(slice(0, n) for n in shape),
            variable_axes=(0,),
        )


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
        pixels: PixelConfig,
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
        self.vision = VisionEncoder(vision, pixels, dtype=dtype)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.deepstack_layers = len(vision.deepstack_visual_indexes)

    def positions(
        self,
        token_ids: Sequence[int],
        *,
        image_grids: Sequence[tuple[int, int, int]] = (),
        video_grids: Sequence[tuple[int, int, int]] = (),
    ) -> torch.Tensor:
        """Return a prompt's ``[3, tokens]`` M-RoPE coordinates.

        The image and video grids are those of the prompt's blocks in order
        (``rope_index``); a video's grid covers all of its blocks.
        """
        return rope_index(
            token_ids,
            image_grids=image_grids,
            video_grids=video_grids,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            spatial_merge_size=self.vision.downsample,
        )

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
