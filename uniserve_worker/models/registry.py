"""Model registry: architecture resolution plus the shared ``UniModelBase`` glue.

The registry owns architecture-to-class registration and the concrete contract
base shared by registered families. Family implementations stay in their
family modules.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Type

from ..contracts.caps import Caps, ExecutionConstraints
from ..contracts.forward_batch import BatchPolicy
from ..contracts.model_family import ModelFamilyDescriptor, ModelOperationSet
from ..contracts.model_protocols import UniModel
from ..foundation.errors import WorkerError, capability_mismatch, invalid_descriptor
from ..foundation.plugins import discover_package_plugins
from ..foundation.runtime_config import get_execution_config
from ..nn.quant import get_current_kv_cache_dtype, kv_store_dtype_name, resolve_kv_store_dtype
from ..nn.quant.kv_cache import KV_CACHE_NO_OVERRIDE_SENTINELS

if TYPE_CHECKING:
    import torch

    from ..contracts.resource_plan import CapsDescriptor

__all__ = [
    "ModelRegistry",
    "MODEL_REGISTRY",
    "UniModelBase",
    "import_model_classes",
    "resolve_model_cls",
    "resolve_model_descriptor",
    "detect_model_architectures",
]

logger = logging.getLogger(__name__)


class UniModelBase(UniModel):
    """Concrete glue shared by the registered model entries.

    The base carries no ``__init__`` and keeps model entries free to compose
    system execution objects around their family-specific neural adapters. It complements the
    :class:`~uniserve_worker.contracts.model_protocols.UniModel` contract rather than duplicating it.

    It owns the duplicated contract glue:

    * the KV store-dtype helpers ``_kv_store_dtype_for`` / ``_kv_dtype_name_for``
      (require an instance ``kv_cache_dtype`` attribute);
    * a single ``batch_policy()`` and ``caps()`` driven by ``_caps_descriptor``,
      both reading ``max_batch_ops`` from the same descriptor so the two stay aligned.

    Model-specific initialization retains family weights, constants, and
    placement bindings; operation and graph lifecycle remain system-owned.
    """

    kv_cache_dtype: Any

    def _kv_store_dtype_for(self, compute_dtype: "torch.dtype") -> "torch.dtype":
        return resolve_kv_store_dtype(compute_dtype, self.kv_cache_dtype)

    def _kv_dtype_name_for(self, compute_dtype: "torch.dtype") -> str:
        return kv_store_dtype_name(self._kv_store_dtype_for(compute_dtype))

    def _requested_kv_cache_dtype_for(self, config: Any | None) -> str | None:
        value = get_execution_config().kv_cache_dtype
        if value is not None and str(value).lower() not in KV_CACHE_NO_OVERRIDE_SENTINELS | {
            "null"
        }:
            return value
        return get_current_kv_cache_dtype(config)

    def _query_geometry_from(self, attn: Any) -> tuple[int, float, "torch.dtype"]:
        """Query-side decode-graph geometry read off one attention module.

        The shared extraction behind each model's
        ``query_geometry`` hook: the module's (tensor-parallel
        local) head count, its softmax scale, and the query dtype. ``q_norm``'s
        weight stays in compute dtype even under weight quantization, so it is
        the faithful dtype of the query tensor the decode kernel sees.
        """
        scale = getattr(attn, "scale", None)
        if scale is None:
            scale = attn.scaling
        return int(attn.num_heads), float(scale), attn.q_norm.weight.dtype

    def _caps_descriptor(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> "CapsDescriptor":
        raise NotImplementedError

    def batch_policy(self) -> BatchPolicy:
        descriptor = self._caps_descriptor()
        # A runner-backed model lets the system group a heterogeneous (und+gen)
        # batch by mode and drive each mode through its system driver; thin text
        # additionally fuses an all-text extend+decode window. No per-model opt-out.
        return BatchPolicy(
            max_batch_ops=descriptor.max_batch_ops,
            supports_mixed_modes=True,
        )

    def caps(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> Caps:
        descriptor = self._caps_descriptor(
            block_size=block_size,
            kv_token_capacity=kv_token_capacity,
        )
        return Caps(
            block_size=descriptor.block_size,
            num_blocks=descriptor.num_blocks,
            num_layers=descriptor.num_layers,
            scratch_capacity_tokens=descriptor.scratch_capacity_tokens,
            supported_ops=tuple(self.supported_ops),
            max_latent_size=descriptor.max_latent_size,
            latent_downsample=descriptor.latent_downsample,
            bytes_per_token=descriptor.bytes_per_token,
            supported_controls=tuple(self.supported_controls),
            adapter_mode=self.adapter_mode,
            execution_constraints=ExecutionConstraints(
                max_batch_ops=descriptor.max_batch_ops,
            ),
            resource_classes=self.resource_plan.classes(),
            attention_backend=descriptor.attention_backend,
            kv_dtype=descriptor.kv_dtype,
            encoder_cache_budget=descriptor.encoder_cache_budget,
            max_vae_grid_tokens=descriptor.max_vae_grid_tokens or descriptor.max_latent_size,
            max_vit_grid_tokens=descriptor.max_vit_grid_tokens,
            commit_marker_tokens=descriptor.commit_marker_tokens,
            gen_rope_advance=descriptor.gen_rope_advance,
            max_cfg_branches=descriptor.max_cfg_branches,
        )


class ModelRegistry:
    """Maps architecture names to :class:`UniModel` implementations."""

    def __init__(self) -> None:
        self._classes: dict[str, Type[UniModel]] = {}
        self._descriptors: dict[str, ModelFamilyDescriptor] = {}

    def register(
        self, model_cls: Type[UniModel], *, names: list[str] | tuple[str, ...] | None = None
    ) -> None:
        _validate_model_contract(model_cls)
        keys = tuple(names or (model_cls.__name__,))
        descriptor = ModelFamilyDescriptor.from_model_class(
            model_cls, names=tuple(str(key) for key in keys)
        )
        for key in keys:
            if key in self._classes:
                if self._classes[key] is model_cls:
                    continue
                raise invalid_descriptor(f"model architecture {key!r} already registered")
            self._classes[key] = model_cls
            self._descriptors[key] = descriptor

    def resolve(self, architectures: list[str] | tuple[str, ...]) -> Type[UniModel]:
        return self.resolve_descriptor(architectures).model_class

    def resolve_descriptor(
        self,
        architectures: list[str] | tuple[str, ...],
    ) -> ModelFamilyDescriptor:
        disabled = set(get_execution_config().disabled_model_archs)
        for arch in architectures:
            if arch in disabled:
                continue
            if arch in self._descriptors:
                return self._descriptors[arch]
        known = ", ".join(sorted(self._classes)) or "<none>"
        raise capability_mismatch(
            f"no UniModel registered for architectures {architectures!r}; "
            f"known architectures: {known}"
        )

    def registered_classes(self) -> tuple[Type[UniModel], ...]:
        """Distinct registered model classes (a name may map several aliases)."""
        return tuple(dict.fromkeys(self._classes.values()))


MODEL_REGISTRY = ModelRegistry()


def _register_module_models(module: ModuleType, *, strict: bool) -> None:
    entry = getattr(module, "EntryClass", None)
    if entry is None:
        return
    entries = entry if isinstance(entry, list) else [entry]
    for cls in entries:
        names = getattr(cls, "architectures", None)
        try:
            MODEL_REGISTRY.register(cls, names=names)
        except (WorkerError, ValueError):
            # Duplicate registration or contract mismatch: skip in non-strict mode.
            if strict:
                raise
            logger.error(
                "skipping model class that failed contract validation",
                extra={
                    "module_name": module.__name__,
                    "model_class": getattr(cls, "__name__", repr(cls)),
                },
                exc_info=True,
            )


@lru_cache(maxsize=1)
def import_model_classes(strict: bool | None = None) -> None:
    if strict is None:
        strict = get_execution_config().strict_model_imports
    discover_package_plugins(
        importlib.import_module(__package__ or "uniserve_worker.models"),
        strict=bool(strict),
        on_module=lambda module: _register_module_models(module, strict=bool(strict)),
    )


def resolve_model_cls(architectures: list[str] | tuple[str, ...]) -> Type[UniModel]:
    import_model_classes()
    return MODEL_REGISTRY.resolve(tuple(architectures))


def resolve_model_descriptor(
    architectures: list[str] | tuple[str, ...],
) -> ModelFamilyDescriptor:
    import_model_classes()
    return MODEL_REGISTRY.resolve_descriptor(tuple(architectures))


def detect_model_architectures(model_path: str | Path) -> list[str]:
    """Ask registered model classes whether they recognize a checkpoint path."""

    import_model_classes()
    path = Path(model_path)
    detected: list[str] = []
    for cls in MODEL_REGISTRY.registered_classes():
        recognizes = getattr(cls, "recognizes", None)
        if callable(recognizes) and bool(recognizes(path)):
            detected.extend(str(name) for name in getattr(cls, "architectures", (cls.__name__,)))
    return detected


def _validate_model_contract(model_cls: Type[UniModel]) -> None:
    if not issubclass(model_cls, UniModel):
        raise capability_mismatch(f"{model_cls.__name__} must inherit UniModel")
    ModelOperationSet.from_model_class(model_cls).validate(model_cls)
