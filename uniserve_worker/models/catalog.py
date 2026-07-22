"""Model catalog: architecture resolution plus the shared ``UniModelBase`` glue.

The ``Catalog`` is an immutable mapping from stable architecture identifiers to
concrete model constructors, built from explicit entries by the worker
composition root. This module also holds the concrete contract base shared by
cataloged families; family implementations stay in their family modules.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..contracts.caps import Caps, ExecutionConstraints
from ..contracts.forward_batch import BatchPolicy
from ..contracts.model_family import ModelFamilyDescriptor, ModelOperationSet
from ..contracts.model_protocols import UniModel
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..foundation.runtime_config import get_execution_config
from ..nn.quant import get_current_kv_cache_dtype, kv_store_dtype_name, resolve_kv_store_dtype
from ..nn.quant.kv_cache import KV_CACHE_NO_OVERRIDE_SENTINELS

if TYPE_CHECKING:
    import torch

    from ..contracts.resource_plan import CapsDescriptor

__all__ = [
    "Catalog",
    "UniModelBase",
]


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


class Catalog:
    """Immutable mapping from stable architecture identifiers to model constructors.

    Entries are explicit bootstrap data supplied at construction; each entry's
    identifiers come from the model class's ``architectures`` declaration (its
    class name when absent). Every entry passes contract validation when the
    catalog is built.
    """

    def __init__(self, model_classes: Iterable[type[UniModel]]) -> None:
        self._entries: tuple[type[UniModel], ...] = tuple(dict.fromkeys(model_classes))
        classes: dict[str, type[UniModel]] = {}
        descriptors: dict[str, ModelFamilyDescriptor] = {}
        for model_cls in self._entries:
            _validate_model_contract(model_cls)
            descriptor = ModelFamilyDescriptor.from_model_class(model_cls)
            for name in descriptor.names:
                if name in classes:
                    raise invalid_descriptor(
                        f"model architecture {name!r} already has a catalog entry"
                    )
                classes[name] = model_cls
                descriptors[name] = descriptor
        self._classes = classes
        self._descriptors = descriptors

    def resolve(self, architectures: list[str] | tuple[str, ...]) -> type[UniModel]:
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
            f"no UniModel catalog entry for architectures {architectures!r}; "
            f"known architectures: {known}"
        )

    def detect_architectures(self, model_path: str | Path) -> list[str]:
        """Ask each cataloged model class whether it recognizes a checkpoint path."""

        path = Path(model_path)
        detected: list[str] = []
        for cls in self._entries:
            recognizes = getattr(cls, "recognizes", None)
            if callable(recognizes) and bool(recognizes(path)):
                detected.extend(
                    str(name) for name in getattr(cls, "architectures", (cls.__name__,))
                )
        return detected


def _validate_model_contract(model_cls: type[UniModel]) -> None:
    if not issubclass(model_cls, UniModel):
        raise capability_mismatch(f"{model_cls.__name__} must inherit UniModel")
    ModelOperationSet.from_model_class(model_cls).validate(model_cls)
