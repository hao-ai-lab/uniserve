"""Public model loading, checkpoint reading and numerical weight assignment."""

from .config import Config

__all__ = ["Config", "Result", "load_model", "load_weights"]


def __getattr__(name):
    if name in {"Result", "load_model", "load_weights"}:
        from . import loader

        return getattr(loader, name)
    raise AttributeError(name)
