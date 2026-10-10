"""Video Sparse Attention selection and compressed block computation."""

from .inputs import (
    DENSE_TILE,
    MAX_REGION_TILES,
    TILE_SIZES,
    BlockInput,
    Input,
    NormRope,
    Pattern,
    Regions,
    Workspace,
)

__all__ = [
    "DENSE_TILE",
    "MAX_REGION_TILES",
    "TILE_SIZES",
    "BlockInput",
    "Input",
    "NormRope",
    "Pattern",
    "Regions",
    "Workspace",
]

from .layer import Attention, BlockAttention
from .regions import RegionAttention

__all__ += ["Attention", "BlockAttention", "RegionAttention"]
