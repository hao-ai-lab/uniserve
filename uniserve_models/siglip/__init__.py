"""SigLIP-NaViT visual tower and checkpoint definitions.

This package is not a catalog model. A composing model package (BAGEL) uses
``Encoder`` as its vision tower, builds ``Config`` from its own checkpoint
metadata, and merges ``assignments`` into its checkpoint mapping under the
tower's key prefix.
"""

from .config import Config, TransformerConfig
from .encoder import Attention, Encoder, TransformerLayer
from .weights import assignments

__all__ = [
    "Config",
    "TransformerConfig",
    "Encoder",
    "Attention",
    "TransformerLayer",
    "assignments",
]
