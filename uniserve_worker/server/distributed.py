"""Runtime construction of the worker's device mesh.

Builds the :class:`DeviceMesh` for this worker process from the parallel layout:
a tensor-parallel (``tp``) axis backed by a ``torch.distributed`` collective, and
(when a modality split is requested) a ``tower`` axis. The tower axis has two
forms behind one ``AxisTransport`` surface, so the model code is identical for
both:

* **In-process tower parallel:** one worker, two devices, a
  :class:`LocalP2PTransport` over the worker's devices (``tower_devices``).
* **Cross-process tower disaggregation:** the und and gen towers are separate
  workers; the tower axis is a :class:`DataPlaneTowerTransport` over the data
  plane (``cuda_ipc`` / ``mooncake``). Each worker builds a tower view at its own
  coordinate (``tower_coord``); the und→gen KV handoff is a
  ``reshard(Pinned(primary) -> Pinned(gen))`` over the data plane.

The degenerate case (tp_size==1, no tower) returns a trivial single-device mesh
that is byte-identical to a single-rank worker.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import torch

from ..foundation.errors import distributed_setup_error
from ..nn.mesh import (
    CollectiveTransport,
    DataPlaneTowerTransport,
    DeviceMesh,
    LocalP2PTransport,
    MeshAxis,
)

__all__ = [
    'build_device_mesh',
]

logger = logging.getLogger(__name__)


def build_device_mesh(
    *,
    tp_rank: int,
    tp_size: int,
    device: str,
    tower_devices: Sequence[str] | None = None,
    tower_primary: int = 0,
    tower_coord: int | None = None,
    tower_size: int | None = None,
    tower_data_plane: Any | None = None,
    tower_gate: Any | None = None,
    tp_backend: str | None = None,
    tp_init_method: str | None = None,
) -> DeviceMesh:
    """Construct this worker's :class:`DeviceMesh`.

    In-process tower: ``tower_devices`` is the per-modality device list
    (e.g. ``["cuda:0", "cuda:1"]``); ``None`` or a single entry means no tower
    axis, and ``tower_primary`` is the coordinate holding the shared modules and
    the authoritative KV cache.

    Cross-process tower: pass ``tower_coord`` (this worker's tower coordinate,
    e.g. 0 for the und worker, 1 for the gen worker) and ``tower_size`` (the
    number of tower coordinates, default 2). The axis is then a
    :class:`DataPlaneTowerTransport` over the data plane. Inject
    ``tower_data_plane`` (a ``TensorStore``-like byte mover) and ``tower_gate``
    (the transfer-readiness gate). The model code is unchanged:
    :func:`place_towers` materializes only this worker's coordinate and the KV
    handoff routes through the data plane.
    """
    tp_rank = int(tp_rank)
    tp_size = int(tp_size)
    local_device = _resolve_local_device(device, tp_rank, tp_size)

    axes: list[MeshAxis] = []
    if tp_size > 1:
        axes.append(
            _build_tp_axis(
                tp_rank,
                tp_size,
                local_device,
                backend_override=tp_backend,
                init_method_override=tp_init_method,
            )
        )
    if tower_coord is not None:
        tower_axis = _build_cross_process_tower_axis(
            tower_coord, tower_size, tower_data_plane, tower_gate
        )
    else:
        tower_axis = _build_tower_axis(tower_devices, tower_primary)
    if tower_axis is not None:
        axes.append(tower_axis)
    return DeviceMesh.of(*axes, device=local_device)


def _resolve_local_device(device: str, tp_rank: int, tp_size: int) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and tp_size > 1:
        _set_cuda_device(dev, tp_rank)
        index = dev.index if dev.index is not None else int(tp_rank)
        return torch.device(f"cuda:{index}")
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return dev


def _build_tp_axis(
    tp_rank: int,
    tp_size: int,
    device: torch.device,
    *,
    backend_override: str | None = None,
    init_method_override: str | None = None,
) -> MeshAxis:
    if not torch.distributed.is_available():
        raise distributed_setup_error("torch.distributed is required for tp_size > 1")
    backend = _distributed_backend(device, backend_override=backend_override)
    if not torch.distributed.is_initialized():
        init_method = _init_method(init_method_override)
        logger.info(
            "initializing tensor-parallel process group",
            extra={
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "backend": backend,
                "init_method": init_method,
                "device": str(device),
            },
        )
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=tp_rank,
            world_size=tp_size,
        )
    else:
        world = int(torch.distributed.get_world_size())
        rank = int(torch.distributed.get_rank())
        if world < tp_size:
            raise distributed_setup_error(
                f"existing distributed world size {world} is smaller than tp_size {tp_size}"
            )
        if rank != tp_rank and world == tp_size:
            raise distributed_setup_error(
                f"existing distributed rank {rank} does not match requested tp_rank {tp_rank}"
            )
    group = None
    world = int(torch.distributed.get_world_size())
    if world != tp_size:
        group = torch.distributed.new_group(ranks=list(range(tp_size)), backend=backend)
    transport = CollectiveTransport(axis="tp", _size=tp_size, _coord=tp_rank, group=group)
    return MeshAxis(name="tp", size=tp_size, coord=tp_rank, transport=transport)


def _build_tower_axis(tower_devices: Sequence[str] | None, tower_primary: int) -> MeshAxis | None:
    if tower_devices is None:
        return None
    devices = tuple(torch.device(d) for d in tower_devices)
    if len(devices) <= 1:
        return None
    primary = int(tower_primary)
    if primary < 0 or primary >= len(devices):
        raise distributed_setup_error(
            f"tower_primary {primary} out of range for {len(devices)} tower devices"
        )
    transport = LocalP2PTransport(axis="tower", devices=devices, _coord=primary)
    return MeshAxis(name="tower", size=len(devices), coord=primary, transport=transport)


def _build_cross_process_tower_axis(
    tower_coord: int,
    tower_size: int | None,
    data_plane: Any | None,
    gate: Any | None,
) -> MeshAxis | None:
    """Build the cross-process tower axis at this worker's coordinate.

    The transport is the data-plane-backed :class:`DataPlaneTowerTransport`; byte
    movement (``cuda_ipc`` / ``mooncake``) and the readiness gate are injected by
    the und/gen worker builder. A size <= 1 means no tower.
    """
    size = int(tower_size) if tower_size is not None else 2
    coord = int(tower_coord)
    if size <= 1:
        return None
    if coord < 0 or coord >= size:
        raise distributed_setup_error(
            f"tower_coord {coord} out of range for {size} tower coordinates"
        )
    transport = DataPlaneTowerTransport(
        axis="tower",
        _size=size,
        _coord=coord,
        data_plane=data_plane,
        gate=gate,
    )
    return MeshAxis(name="tower", size=size, coord=coord, transport=transport)


def _set_cuda_device(device: torch.device, tp_rank: int) -> None:
    if not torch.cuda.is_available():
        raise distributed_setup_error("tp_size > 1 on cuda requires torch.cuda.is_available()")
    index = device.index if device.index is not None else int(tp_rank)
    torch.cuda.set_device(index)


def _distributed_backend(device: torch.device, *, backend_override: str | None = None) -> str:
    override = (backend_override or "").strip()
    if override:
        return override
    return "nccl" if device.type == "cuda" else "gloo"


def _init_method(init_method_override: str | None = None) -> str:
    value = (init_method_override or "").strip()
    if value:
        return value
    addr = (os.environ.get("MASTER_ADDR") or "127.0.0.1").strip() or "127.0.0.1"
    port = (os.environ.get("MASTER_PORT") or "").strip()
    if not port:
        raise distributed_setup_error(
            "MASTER_PORT or --tp-init-method is required for tp_size > 1"
        )
    return f"tcp://{addr}:{port}"
