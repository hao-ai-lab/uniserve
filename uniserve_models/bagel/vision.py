"""BAGEL SigLIP features, connector and learned spatial positions.

BAGEL's vision path runs the ``uniserve_models.siglip`` tower, projects its
features to the language model's hidden width, and then adds a second
position table at that width. ``Model`` wraps this network in
``uniserve.model.PatchEncoder``, which packs image samples and splits the
result back per image.
"""

from __future__ import annotations

from torch import nn

from uniserve.nn.vision import MLPConnector, PositionEmbedding
from uniserve.nn.vision.position import build_abs_positions_from_grid_hw
from uniserve_models import siglip

from .config import Config


class Encoder(nn.Module):
    """BAGEL's SigLIP features, connector and learned language-width grid.

    The position table covers the same ``side x side`` patch grid as SigLIP's
    own position embedding, with ``side = image_size // patch_size``, and is
    loaded from the checkpoint's ``vit_pos_embed.pos_embed`` tensor.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.encoder = siglip.Encoder(config.vision)
        self.connector = MLPConnector(
            config.vision.encoder.hidden_size,
            config.text.hidden_size,
            config.connector_act,
        )
        side = config.vision.image_size // config.vision.patch_size
        self.position = PositionEmbedding((side, side), config.text.hidden_size)

    def forward(self, pixels, grids, grid_shapes):
        """Encode packed image patches into language-width features.

        Args:
            pixels: Patch rows or NCHW pixels, as ``siglip.Encoder`` accepts.
            grids: ``[images, 2]`` integer tensor of per-image patch grid
                ``(height, width)``.
            grid_shapes: The same grid dimensions as host integers.

        Returns:
            ``[total_patches, text_hidden]`` features, images concatenated in
            input order.
        """
        # [total_patches, text_hidden] after connector projection.
        features = self.connector(self.encoder(pixels, grids, grid_shapes))

        # The patch total comes from the host grid shapes, so position
        # construction reads no device value and stays capturable in a CUDA
        # graph. Every grid fits the ``side x side`` table because
        # ``siglip.Encoder.forward`` rejects larger ones.
        columns, rows = build_abs_positions_from_grid_hw(
            grids, total=sum(height * width for height, width in grid_shapes)
        )

        # Row-major index into the flattened position table.
        return features + self.position(
            rows * self.position.grid_size[1] + columns
        )
