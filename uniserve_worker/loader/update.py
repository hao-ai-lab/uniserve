"""Atomic in-place replacement of one live model weight graph."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..nn.quant.base import process_quantized_modules
from .component import CheckpointComponent
from .config import LoadFormat, LoadRequest
from .handles import TensorWeightHandle, WeightHandle
from .io import iter_weight_handles
from .loader import (
    assign_component,
    component_parameter_names,
    load_component,
    verify_checksums,
)
from .source import WeightSourceConfig, resolve_weight_sources
from .weight_set import WeightSet

if TYPE_CHECKING:
    from ..models.runtime import ExecutionModel

__all__ = ["BucketTensor", "WeightUpdater"]


@dataclass(frozen=True, slots=True)
class BucketTensor:
    """Named tensor geometry within a flattened weight-update bucket."""

    name: str
    shape: tuple[int, ...]
    offset: int
    length: int

    def __post_init__(self) -> None:
        """Validate the descriptor bounds and shape-derived element count."""

        if not self.name or self.offset < 0 or self.length < 0:
            raise ValueError("flattened bucket metadata is invalid")
        elements = 1
        for dimension in self.shape:
            if dimension < 0:
                raise ValueError("flattened bucket shapes cannot contain negative dimensions")
            elements *= dimension
        if elements != self.length:
            raise ValueError("flattened bucket length does not match its tensor shape")


class WeightUpdater:
    """Exclusive authority for installing and publishing live-weight generations."""

    def __init__(
        self,
        model: ExecutionModel,
        *,
        sources: tuple[WeightSourceConfig, ...] = (WeightSourceConfig(),),
        sidecars: tuple[str, ...] = ("config.json",),
        weights: WeightSet,
        publish: Callable[[WeightSet], None] | None = None,
        invalidate_graphs: Callable[[int], None] | None = None,
        exclusive: Callable[[], Any] | None = None,
        derived_cache_active: Callable[[], bool] | None = None,
    ) -> None:
        """Bind a model and exclusive publication callbacks to one update authority."""

        if weights.version < 0:
            raise ValueError("weight version cannot be negative")
        self.model = model
        self.sources = tuple(sources)
        self.sidecars = tuple(str(value) for value in sidecars)
        self.weights = weights
        self._publish = publish
        self._invalidate_graphs = invalidate_graphs
        self._exclusive = exclusive
        self._derived_cache_active = derived_cache_active
        self.unhealthy = False

    def update_named(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        expected_parameters: Iterable[str] | None = None,
    ) -> WeightSet:
        """Install a complete mapping of named tensors as the next live weight generation."""

        handles = tuple(
            TensorWeightHandle(name, tensor) for name, tensor in sorted(tensors.items())
        )
        return self._install(
            handles,
            expected_parameters=expected_parameters,
        )

    def update_distributed(
        self,
        received: Iterable[tuple[str, torch.Tensor]],
        *,
        expected_parameters: Iterable[str] | None = None,
    ) -> WeightSet:
        """Install streamed name/tensor pairs as the next live weight generation."""

        # Materialize the stream into a unique mapping before entering the transaction.
        tensors: dict[str, torch.Tensor] = {}
        for name, tensor in received:
            key = str(name)
            if key in tensors:
                raise ValueError(f"distributed weight source repeats tensor {key!r}")
            tensors[key] = tensor
        return self.update_named(
            tensors,
            expected_parameters=expected_parameters,
        )

    def update_flattened(
        self,
        bucket: torch.Tensor,
        metadata: Sequence[BucketTensor],
        *,
        expected_parameters: Iterable[str] | None = None,
    ) -> WeightSet:
        """Install tensor slices decoded from one flattened update bucket."""

        # Validate every descriptor while reconstructing its view into the shared bucket.
        flat = bucket.reshape(-1)
        tensors: dict[str, torch.Tensor] = {}
        for descriptor in metadata:
            end = descriptor.offset + descriptor.length
            if end > int(flat.numel()):
                raise ValueError(f"flattened bucket tensor {descriptor.name!r} exceeds the bucket")
            if descriptor.name in tensors:
                raise ValueError(f"flattened bucket repeats tensor {descriptor.name!r}")
            tensors[descriptor.name] = flat.narrow(0, descriptor.offset, descriptor.length).view(
                descriptor.shape
            )
        return self.update_named(tensors, expected_parameters=expected_parameters)

    def update_disk(
        self,
        request: LoadRequest,
        *,
        root: Path,
        repository_id: str | None = None,
    ) -> WeightSet:
        """Load a checkpoint source and atomically install it as the next weight generation."""

        # Resolve and authenticate the complete source set before touching live tensors.
        sources = resolve_weight_sources(
            request,
            sources=self.sources,
            sidecars=self.sidecars,
            root=root,
            repository_id=repository_id,
        )
        verify_checksums(sources, request.load.checksum_manifest)

        components = self.model.checkpoint_components()
        by_name = {source.source_name: source for source in sources}

        def load_remaining() -> None:
            for component in components[1:]:
                load_component(component, by_name[component.source], request)

        return self._install(
            iter_weight_handles(by_name[components[0].source], request.load),
            expected_parameters=None,
            after_primary=load_remaining if len(components) > 1 else None,
            layered=request.load.load_format is LoadFormat.LAYERED,
        )

    def _install(
        self,
        handles: Iterable[WeightHandle],
        *,
        expected_parameters: Iterable[str] | None,
        after_primary: Callable[[], None] | None = None,
        layered: bool = False,
    ) -> WeightSet:
        """Install, validate, and publish one generation as an atomic model mutation.

        Snapshot coverage expands to the full parameter graph when a secondary source
        participates. Any failure restores the selected values before leaving the
        exclusive scope; a failed restoration permanently marks the updater unhealthy.
        """

        # Reject mutations that cannot preserve the current live-weight contract.
        if self.unhealthy:
            raise RuntimeError("weight updater is unhealthy")
        if self._derived_cache_active is not None and self._derived_cache_active():
            raise RuntimeError("active derived-weight cache cannot observe in-place replacement")
        expected = (
            None if expected_parameters is None else {str(name) for name in expected_parameters}
        )

        # Determine the full rollback domain before acquiring mutation ownership.
        snapshot_names = (
            set().union(
                *(
                    _update_target_names(self.model, None, component=component)
                    for component in self.model.checkpoint_components()
                )
            )
            if after_primary is not None
            else _update_target_names(self.model, expected)
        )

        with self._exclusive_context():
            snapshot = _snapshot_tensors(self.model, snapshot_names)
            try:
                # Reset per-parameter completeness state, assign weights, then audit the
                # checkpoint contract while rollback remains available.
                _clear_packed_load_state(self.model, snapshot_names)
                component = self.model.checkpoint_components()[0]
                if expected is not None:
                    component = replace(component, included=frozenset(expected))
                report = assign_component(
                    component,
                    handles,
                    device=str(next(component.module.parameters()).device),
                    layered=layered,
                )

                # Apply secondary sources and materialize quantization-derived tensors
                # before exposing the next generation to inference.
                if after_primary is not None:
                    after_primary()
                if not layered:
                    if after_primary is None:
                        _process_loaded_modules(component.module, report.loaded)
                    else:
                        process_quantized_modules(self.model.modules())

                # Publication is the transaction's commit boundary: invalidate graph
                # captures before callers can observe the new tensor identities.
                updated = WeightSet.from_module(
                    self.model,
                    version=self.weights.version + 1,
                )
                if self._invalidate_graphs is not None:
                    self._invalidate_graphs(updated.version)
                self.weights = updated
                if self._publish is not None:
                    self._publish(updated)
                return updated
            except BaseException:
                # Restore live tensors in-place so existing module references remain valid.
                try:
                    _restore_tensors(self.model, snapshot)
                except BaseException as rollback_error:
                    self.unhealthy = True
                    raise RuntimeError(
                        "weight update failed and rollback could not restore the model"
                    ) from rollback_error
                raise

    def _exclusive_context(self) -> Any:
        """Return the configured mutation guard or a no-op context manager."""

        return self._exclusive() if self._exclusive is not None else nullcontext()


def _update_target_names(
    model: ExecutionModel,
    expected: set[str] | None,
    *,
    component: CheckpointComponent | None = None,
) -> set[str]:
    """Translate the primary component's parameter scope into model coordinates."""

    component = model.checkpoint_components()[0] if component is None else component
    names = component_parameter_names(component) if expected is None else expected
    if expected is None and component.persistent_buffers:
        serialized = component.module.state_dict()
        names.update(name for name, _ in component.module.named_buffers() if name in serialized)
    for path, module in model.named_modules():
        if module is component.module:
            prefix = f"{path}." if path else ""
            return {prefix + name for name in names}
    raise ValueError("checkpoint component has no owning model path")


def _snapshot_tensors(
    model: nn.Module,
    names: set[str],
) -> dict[str, torch.Tensor]:
    """Clone selected live tensors to CPU for transactional rollback."""

    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    missing = names.difference(tensors)
    if missing:
        raise ValueError(f"weight update selects unknown parameter {min(missing)!r}")
    return {name: tensors[name].detach().cpu().clone() for name in names}


def _restore_tensors(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Restore a CPU snapshot into the model's current parameter and buffer objects."""

    live = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    with torch.no_grad():
        for name, value in snapshot.items():
            target = live[name]
            if target.dtype != value.dtype:
                target.data = torch.empty_like(target, dtype=value.dtype)
            target.copy_(value.to(device=target.device))


def _clear_packed_load_state(model: nn.Module, names: set[str]) -> None:
    """Clear packed-shard completion markers for parameters entering an update."""

    parameters = dict(model.named_parameters())
    for name in names:
        parameter = parameters.get(name)
        if parameter is not None and hasattr(parameter, "_uniserve_checkpoint_shards"):
            delattr(parameter, "_uniserve_checkpoint_shards")


def _process_loaded_modules(model: nn.Module, loaded: set[str]) -> None:
    """Run quantization post-processing only for modules touched by the update."""

    selected: list[nn.Module] = []
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if any(name.startswith(prefix) for name in loaded):
            selected.append(module)
    process_quantized_modules(selected)
