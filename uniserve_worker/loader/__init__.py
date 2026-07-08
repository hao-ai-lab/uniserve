"""Model loader framework for the worker."""
from .base import BaseModelLoader, LoadResult, ModelBringUp
from .default import DefaultModelLoader
from .dummy import DummyModelLoader
from .registry import get_loader, get_loader_for_descriptor, register_loader
from .transformers import NativeLoadSpec, NativeTransformersLoader

__all__ = [
    "BaseModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "LoadResult",
    "ModelBringUp",
    "NativeLoadSpec",
    "NativeTransformersLoader",
    "get_loader",
    "get_loader_for_descriptor",
    "register_loader",
]
