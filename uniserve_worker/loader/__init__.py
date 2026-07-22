"""Model loader framework for the worker."""
from .base import BaseModelLoader, LoadResult
from .composite import CompositeCheckpointLoader
from .default import DefaultModelLoader
from .dummy import DummyModelLoader
from .registry import get_loader, get_loader_for_descriptor, register_loader
from .transformers import NativeTransformersLoader
from .weight_spec import (
    GraphSource,
    NativeSource,
    Rename,
    Sidecar,
    StackedParamMapping,
    TowerSplit,
    WeightSpec,
    weight_spec_of,
)

__all__ = [
    "BaseModelLoader",
    "CompositeCheckpointLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "GraphSource",
    "LoadResult",
    "NativeSource",
    "NativeTransformersLoader",
    "Rename",
    "Sidecar",
    "StackedParamMapping",
    "TowerSplit",
    "WeightSpec",
    "get_loader",
    "get_loader_for_descriptor",
    "register_loader",
    "weight_spec_of",
]
