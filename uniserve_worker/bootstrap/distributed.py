"""Bind each configured component's mesh onto the worker's process world.

``Worker.from_config`` creates the process world with
``initialize_process_groups`` and then calls ``initialize_components``, which
binds every component's ``DeviceMesh`` through ``ProcessGroups.bind`` and
returns one ``ComponentBinding`` per configured component. A fiber is the
group of ranks along a selection of mesh axes; with declarations, only the
fibers a component's calls communicate over (plus ``tp`` for a component with
a ``CausalLM`` call) are bound, and only a fiber of several ranks receives a
backend group.

Backend group creation is collective over the whole process world, so every
rank must request the same multi-rank fibers in the same order. Components are
visited in sorted name order, and ``Worker.from_config`` passes declarations
that ``prepare_worker_model`` resolves on every rank from the same
meta-device skeleton.
"""

from __future__ import annotations

from collections.abc import Mapping

from uniserve.distributed import DeviceMesh, communication_axes
from uniserve.model import CausalLM
from uniserve.runtime.process_groups import ProcessGroups
from uniserve_worker.bootstrap.components import is_host_component
from uniserve_worker.bootstrap.model_loader import attention_parallel
from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.model_executor.component_binding import (
    Call,
    ComponentBinding,
)


def initialize_components(
    groups: ProcessGroups,
    components: Mapping[str, ComponentConfig],
    *,
    declarations: Mapping[str, tuple[Call, ...]] | None = None,
) -> dict[str, ComponentBinding]:
    """Bind declared components to their local meshes and process world.

    Args:
        groups: The initialized process world of this worker.
        components: The worker's component placement.
        declarations: Each component's calls from ``validate_components``.
            When given, a component binds only the fibers its calls and their
            entry points communicate over; when ``None``, it binds each
            individual mesh axis.

    Returns:
        A binding for every configured component. Only components this rank
        is a member of have a mesh, participation groups and, for a
        temporally distributed numerical component, a ``units`` ring.

    Raises:
        ValueError: Among other causes, when this rank is a member of no
            component (checked after every rank has taken part in group
            creation) or when ``ComponentBinding`` finds a placement that
            disagrees with its mesh.
    """
    meshes = {}
    rings = {}
    participation = {}
    for name, component in sorted(components.items()):
        if component.distribution is not None and is_host_component(name):
            # A host component encodes each media unit independently and
            # exchanges nothing, so its ranks stay outside any collective.
            # Skipping it on non-members is safe because a one-rank mesh
            # creates no backend group.
            if groups.rank not in component.ranks:
                continue
            ranks: tuple[int, ...] = (groups.rank,)
        elif component.distribution is not None:
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

        # A non-distributed component's mesh is bound on every rank, members
        # and non-members alike, so the collective group creation matches. A
        # distributed component reaches here only on its members, with a
        # one-rank mesh that creates no backend group.
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
        # model produces replicated logits without a tensor-parallel layer;
        # ``Worker.from_config`` takes this ``tp`` group as its sampling group.
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
