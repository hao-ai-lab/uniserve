"""Construct process worlds and component fibers in canonical rank order."""

from __future__ import annotations

from collections.abc import Mapping

from uniserve.distributed import DeviceMesh, communication_axes
from uniserve.model import CausalLM
from uniserve.runtime.process_groups import ProcessGroups

from ..execution.component_binding import Call, ComponentBinding
from .config import ComponentConfig
from .model_loader import attention_parallel


def initialize_components(
    groups: ProcessGroups,
    components: Mapping[str, ComponentConfig],
    *,
    declarations: Mapping[str, tuple[Call, ...]] | None = None,
) -> dict[str, ComponentBinding]:
    """Bind declared components to their local meshes and process world."""
    meshes = {}
    rings = {}
    participation = {}
    for name, component in sorted(components.items()):
        if component.distribution is not None:
            # Media units are independent local invocations, so the numerical
            # mesh is this rank alone. The ranks holding consecutive units still
            # exchange the overlap between them, over a ring every rank creates
            # in the same order because group creation spans the process world.
            ring = groups.bind(
                DeviceMesh(
                    ranks=component.ranks,
                    shape=(len(component.ranks),),
                    axes=("units",),
                    rank=groups.rank,
                ),
                device=groups.device,
            )
            if groups.rank not in component.ranks:
                continue
            rings[name] = ring.get_group("units")
            ranks = (groups.rank,)
        else:
            ranks = component.ranks
        topology = DeviceMesh(
            ranks=ranks,
            shape=tuple(
                size for _, size in component.parallel_config.dimensions
            ),
            axes=tuple(
                axis for axis, _ in component.parallel_config.dimensions
            ),
            rank=groups.rank,
        )
        axes = (
            tuple((axis,) for axis in topology.axes)
            if declarations is None
            else tuple(
                sorted(
                    {
                        axes
                        for call in declarations.get(name, ())
                        for axes in (
                            *communication_axes(
                                call.module,
                                topology,
                                attention=attention_parallel(component),
                            ),
                            *call.entry_point.communication_axes(topology),
                        )
                    }
                )
            )
        )
        # Sampling broadcasts the selected token even when the numerical
        # model produces replicated logits without a tensor-parallel layer.
        if declarations is not None and any(
            isinstance(call.module, CausalLM)
            for call in declarations.get(name, ())
        ):
            axes = tuple(sorted(set(axes) | {("tp",)}))
        bound = groups.bind(topology, device=groups.device, axes=axes)
        if groups.rank in ranks:
            meshes[name] = bound
            participation[name] = tuple(
                bound.get_group(selected) for selected in axes
            )

    if not components or not any(
        groups.rank in component.ranks for component in components.values()
    ):
        raise ValueError("rank has no configured component")

    return {
        name: ComponentBinding(
            name,
            config,
            groups.process_group,
            meshes.get(name),
            groups.device,
            units=rings.get(name),
            groups=participation.get(name, ()),
        )
        for name, config in components.items()
    }
