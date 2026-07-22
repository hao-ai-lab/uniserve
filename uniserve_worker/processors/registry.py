"""Processor registry keyed by model class."""
from __future__ import annotations

import importlib
import threading
from functools import lru_cache

from ..contracts.model_family import ModelFamilyDescriptor
from ..foundation.plugins import discover_package_plugins
from .base import MultimodalProcessor

__all__ = [
    'register_processor',
    'import_processors',
    'get_processor_for_descriptor',
    'get_processor_for_model',
    'get_image_pipeline_for_descriptor',
    'get_image_pipeline_for_model',
]

class _ProcessorRegistry:
    def __init__(self) -> None:
        self._processors: list[type[MultimodalProcessor]] = []
        self._lock = threading.Lock()
        self._frozen = False

    def register(self, processor_cls: type[MultimodalProcessor]) -> None:
        with self._lock:
            if self._frozen:
                raise RuntimeError("processor registry is frozen")
            if processor_cls not in self._processors:
                self._processors.append(processor_cls)

    def freeze(self) -> None:
        with self._lock:
            self._frozen = True

    def snapshot(self) -> tuple[type[MultimodalProcessor], ...]:
        with self._lock:
            return tuple(self._processors)


_PROCESSOR_REGISTRY = _ProcessorRegistry()


def register_processor(processor_cls: type[MultimodalProcessor]) -> type[MultimodalProcessor]:
    _PROCESSOR_REGISTRY.register(processor_cls)
    return processor_cls


@lru_cache(maxsize=1)
def import_processors() -> None:
    # Processors self-register via the ``@register_processor`` decorator at
    # module-import time, so importing each plugin module is all that is needed.
    discover_package_plugins(
        importlib.import_module(__package__ or "uniserve_worker.processors"),
    )
    _PROCESSOR_REGISTRY.freeze()


def _processor_factory_for_descriptor(
    descriptor: ModelFamilyDescriptor,
) -> type[MultimodalProcessor] | None:
    import_processors()
    model_names = set(descriptor.names)
    model_names.add(descriptor.model_class.__name__)
    for processor_cls in _PROCESSOR_REGISTRY.snapshot():
        processor_names = set(getattr(processor_cls, "model_architectures", ()) or ())
        if model_names & processor_names:
            return processor_cls
    return None


def get_processor_for_descriptor(descriptor: ModelFamilyDescriptor) -> MultimodalProcessor | None:
    processor = descriptor.processor()
    if processor is not None:
        return processor
    processor_cls = _processor_factory_for_descriptor(descriptor)
    if processor_cls is None:
        return None
    return processor_cls()


def get_processor_for_model(model_cls: type) -> MultimodalProcessor | None:
    return get_processor_for_descriptor(ModelFamilyDescriptor.from_model_class(model_cls))


def get_image_pipeline_for_descriptor(descriptor: ModelFamilyDescriptor):
    processor = get_processor_for_descriptor(descriptor)
    if processor is None:
        return None
    pipeline = descriptor.image_pipeline(processor)
    if pipeline is not None:
        return pipeline
    from .image_pipeline import ImageInputPipeline

    return ImageInputPipeline(processor)


def get_image_pipeline_for_model(model_cls: type):
    return get_image_pipeline_for_descriptor(ModelFamilyDescriptor.from_model_class(model_cls))
