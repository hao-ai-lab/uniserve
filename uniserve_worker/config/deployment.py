"""Immutable Python views of a rank's normalized launch configuration.

Rust resolves descriptor values, placement, transports and execution settings
before any process groups or model resources exist. These records are the
loading interface; checkpoint readers and numerical modules use ordinary
Python objects.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from uniserve.loading import Config as IOConfig
from uniserve.runtime.process_groups import Rendezvous
from uniserve_worker._uniserve_ipc import (
    ComponentConfig as ComponentConfig,
)
from uniserve_worker._uniserve_ipc import (
    ParallelConfig as ParallelConfig,
)
from uniserve_worker._uniserve_ipc import (
    SequenceConfig as SequenceConfig,
)
from uniserve_worker._uniserve_ipc import (
    parse_components as parse_components,
)
from uniserve_worker._uniserve_ipc import (
    prepare_worker_launch,
)
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.protocol.call import CallKind


@dataclass(frozen=True)
class WorkerIpcConfig:
    """Configures IPC launch settings.

    Covers the head's registration address, the payload bound and the queue
    depth, which bounds the batches this rank holds in flight. The rank names
    its own channel endpoint and reports it to the registration address; the
    head binds the channel from that report.
    """

    registration_address: str
    channel_transport: str
    # This rank's word in every chunk header it reads, unique in the instance.
    acknowledgment_slot: int
    # Acknowledgment slots of the ranks on this rank's host, across workers.
    # A host product reaches a consumer among them over shared storage and any
    # other over the rank channel; the head derives this from the placement.
    host_slots: tuple[int, ...]
    # Whether any rank that reads this rank's device products is on another
    # host. An interprocess event carries readiness within a host at no cost to
    # the producing stream; only a crossing needs a producer synchronize. The
    # head derives this from the transfer edges and the placement, which only
    # it holds.
    products_cross_hosts: bool
    max_payload_bytes: int
    queue_depth: int


@dataclass(frozen=True)
class ModelLaunchConfig:
    """Selects the checkpoint and its quantization policy."""

    path: str
    quantization_config: dict[str, object]
    # Local components supplied by the caller; otherwise use the Hub base.
    base_model: str | None = None


@dataclass(frozen=True)
class DataPlaneConfig:
    """Bind receive and required export mechanisms for a rank.

    ``WorkerProcessArgs.from_namespace`` requires ``export_backends`` to
    be a subset of ``backends``: a rank publishes only over mechanisms it also
    binds.
    """

    backends: tuple[str, ...]
    export_backends: tuple[str, ...]


@dataclass(frozen=True)
class ExpertParallelLaunch:
    """This replica's place in its deployment's expert-parallel world.

    ``rank`` is this physical rank in the union of ``size`` ranks. Leading
    ``attention_ranks`` own attention and request state; remaining ranks
    own experts. Zero selects colocated expert parallelism. The world forms
    at ``rendezvous``, whose store rank 0 serves on its inherited socket.
    """

    rank: int
    size: int
    attention_ranks: int
    rendezvous: Rendezvous


@dataclass(frozen=True)
class WorkerProcessArgs:
    """Aggregates the validated launch configuration for one worker rank."""

    worker_id: str
    supported_calls: frozenset[CallKind]
    ipc: WorkerIpcConfig
    local_rank: int
    distributed_backend: str | None
    # Where a multi-rank group forms its process world; None for one rank.
    rendezvous: Rendezvous | None
    model: ModelLaunchConfig | None
    data_plane: DataPlaneConfig
    execution: WorkerConfig
    load: IOConfig
    use_stub_model: bool
    components: tuple[tuple[str, ComponentConfig], ...] = ()
    expert_parallel: ExpertParallelLaunch | None = None

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> WorkerProcessArgs:
        """Resolve descriptor values or raise ValueError before startup."""
        return prepare_worker_launch(cls, vars(namespace))
