"""Video Sparse Attention selection and compressed block computation."""

from .inputs import (
    DENSE_TILE,
    BlockInput,
    Input,
    NormRope,
    Pattern,
    Segments,
    Workspace,
)

__all__ = [
    "DENSE_TILE",
    "BlockInput",
    "Input",
    "NormRope",
    "Pattern",
    "Segments",
    "Workspace",
]

from .layer import Attention, BlockAttention
from .segments import SegmentAttention

__all__ += ["Attention", "BlockAttention", "SegmentAttention"]
