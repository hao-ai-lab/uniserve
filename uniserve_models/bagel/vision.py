"""BAGEL SigLIP features, connector and learned spatial positions."""

from __future__ import annotations

from torch import nn

from uniserve.nn.vision import MLPConnector, PositionEmbedding
from uniserve.nn.vision.patching import build_abs_positions_from_grid_hw
from uniserve_models import siglip

from .config import Config


class Encoder(nn.Module):
    """BAGEL's SigLIP features, connector and learned language-width grid."""

    def __init__(self, config: Config):
        super().__init__()
        self.encoder = siglip.Encoder(config.vision)
        self.connector = MLPConnector(
            config.vision.encoder.hidden_size, config.text.hidden_size, config.connector_act
        )
        side = config.vision.image_size // config.vision.patch_size
        self.position = PositionEmbedding((side, side), config.text.hidden_size)

    def forward(self, pixels, grids, grid_shapes):
        features = self.connector(self.encoder(pixels, grids, grid_shapes))
        columns, rows = build_abs_positions_from_grid_hw(
            grids, total=sum(height * width for height, width in grid_shapes)
        )
        return features + self.position(rows * self.position.grid_size[1] + columns)
