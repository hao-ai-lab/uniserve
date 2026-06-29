"""SenseNova-U1 model package (model, config, interleaved text/image denoise ops).

Re-exports ``EntryClass`` so ``models.registry`` auto-discovery picks up the
model when it imports this package.
"""
from .model import EntryClass

__all__ = ["EntryClass"]
