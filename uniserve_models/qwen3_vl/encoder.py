"""Qwen3-VL prompt encoding: vision tower and retained language layers.

``Encoder`` composes the vision encoder (``vision``) with the retained
layers of the Qwen3-VL language model (``Transformer``).
``vision.pack_pixels`` turns decoded frames into the processor's tubelet
rows, ``vision.encode`` turns those rows into merged tokens, and ``encode``
splices the tokens into a prompt at its image and video placeholders: each
token's embedding replaces the placeholder's token embedding, and its
``j``-th DeepStack feature joins the residual stream after decoder layer
``j``. ``positions`` gives the prompt's M-RoPE coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from uniserve.model import MultimodalEncoder, TubeletEncoder

from .config import PixelConfig, VisionConfig
from .positions import rope_index
from .transformer import Transformer
from .vision import VisionTower


class VisionEncoder(TubeletEncoder):
    """Qwen3-VL's vision tower with its processor's pixel packing.

    ``encode`` takes ``[time * height * width, tubelet]`` patch rows with
    their ``(time, height, width)`` grid and returns one row per merged
    token (``VisionTower``). ``pack_pixels`` produces those rows from
    decoded frames exactly as the checkpoint's Qwen3-VL image and video
    processors do on the host.
    """

    network: VisionTower

    def __init__(
        self, config: VisionConfig, pixels: PixelConfig, *, dtype: torch.dtype
    ):
        super().__init__(
            VisionTower(config),
            nn.Identity(),
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            downsample=config.spatial_merge_size,
            output_size=config.feature_size,
            output_dtype=dtype,
        )
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


class Encoder(MultimodalEncoder):
    """Encode prompts whose placeholders carry Qwen3-VL vision tokens.

    ``network`` is the Qwen3-VL language model and ``retained_layers`` the
    decoder layers this encoder runs (``uniserve.model.TextEncoder``).
    ``vision`` (``VisionEncoder``) encodes packed tubelet rows with their
    ``(time, height, width)`` grids into rows of ``vision.feature_size``
    values of ``dtype``: each token's embedding followed by one DeepStack
    feature per DeepStack merger, which ``network`` adds after its leading
    decoder layers. Prompts mark those tokens with ``image_token_id`` or
    ``video_token_id``.

    Raises:
        ValueError: The vision tokens do not match the language width, a
            DeepStack feature's decoder layer is not retained, or the two
            placeholders coincide.
    """

    network: Transformer
    vision: VisionEncoder

    def __init__(
        self,
        network: Transformer,
        retained_layers: tuple[int, ...],
        vision: VisionConfig,
        *,
        pixels: PixelConfig,
        image_token_id: int,
        video_token_id: int,
        dtype: torch.dtype,
    ):
        if vision.out_hidden_size != network.hidden_size:
            raise ValueError(
                "Qwen3-VL vision tokens must match the language width"
            )
        depth = len(vision.deepstack_visual_indexes)
        if any(layer not in retained_layers for layer in range(depth)):
            raise ValueError(
                "every DeepStack feature requires its retained decoder layer"
            )
        if image_token_id == video_token_id:
            raise ValueError("image and video placeholders must differ")
        super().__init__(
            network,
            retained_layers,
            VisionEncoder(vision, pixels, dtype=dtype),
            placeholders=(image_token_id, video_token_id),
        )
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.deepstack_depth = depth

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

    def _vision_inputs(self, rows: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Return the DeepStack features following each row's embedding.

        ``rows`` are packed ``[tokens, (1 + depth) * hidden]`` vision rows;
        the network receives their ``[tokens, depth, hidden]`` tail.
        """
        hidden = self.network.hidden_size
        if rows.shape[-1] != (1 + self.deepstack_depth) * hidden:
            raise ValueError(
                "Qwen3-VL vision rows hold an embedding and every DeepStack "
                "feature"
            )
        if not self.deepstack_depth:
            return {}
        return {
            "deepstack": rows[:, hidden:].unflatten(
                -1, (self.deepstack_depth, hidden)
            )
        }
