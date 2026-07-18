"""SenseNova-U1 family adapter and configuration package.

Re-exports ``EntryClass`` so ``models.registry`` auto-discovery picks up the
model when it imports this package.
"""

from .model import EntryClass

__all__ = ["EntryClass"]
