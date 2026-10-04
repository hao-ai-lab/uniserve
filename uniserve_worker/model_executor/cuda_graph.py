"""Prepared execution resources and fixed CUDA graph inputs.

Numerical inputs are PyTrees of dataclasses, tuples, mappings and tensors.
This module keys graph variants by an input's structure and tensor layouts
(``input_signature``), gives captured graphs their own input backing
(``clone_inputs``), and copies live inputs into that backing before each
replay (``Inputs``). ``Execution`` is the base of every ``ModelRunner``: it
owns the prepared ``ExecutionContext``, the graph buckets and the private
allocation pools charged to the worker's ``GraphStorage``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, is_dataclass
from itertools import count
from types import MappingProxyType
from typing import Any

import torch
from torch.utils import _pytree as pytree

from uniserve.runtime import CUDAGraph, ExecutionContext, Microbatches
from uniserve.runtime.resources import close_resources
from uniserve_worker.model_executor.graph_storage import GraphStorage


def _register(value):
    """Teach PyTree about numerical dataclasses on their first encounter.

    Used as the ``is_leaf`` callback of every flatten and map here, only for
    its registration side effect: it always returns False, so no node is
    treated as a leaf by this callback.
    """
    cls = type(value)
    if cls not in pytree.SUPPORTED_NODES:
        if is_dataclass(value) and not isinstance(value, type):
            pytree.register_dataclass(cls)
        elif cls is MappingProxyType:
            pytree.register_pytree_node(
                cls,
                lambda value: (list(value.values()), tuple(value)),
                lambda values, keys: MappingProxyType(dict(zip(keys, values))),
                flatten_with_keys_fn=lambda value: (
                    [
                        (pytree.MappingKey(key), item)
                        for key, item in value.items()
                    ],
                    tuple(value),
                ),
            )
    return False


def map_tensors(value, transform):
    """Preserve a numerical PyTree while transforming its tensor leaves."""
    return pytree.tree_map(
        lambda item: (
            transform(item) if isinstance(item, torch.Tensor) else item
        ),
        value,
        is_leaf=_register,
    )


def clone_inputs(value):
    """Clone backing while preserving broadcasts and repeated references.

    A tensor with zero strides clones only one element along those axes and
    is expanded again, so the copy keeps the broadcast. A tensor reachable
    through several paths is cloned once and the clones stay aliased.
    """
    copies = {}

    def clone(tensor):
        if id(tensor) not in copies:
            slices = tuple(
                slice(0, 1) if stride == 0 else slice(None)
                for stride in tensor.stride()
            )
            copies[id(tensor)] = tensor[slices].clone().expand(tensor.shape)
        return copies[id(tensor)]

    return map_tensors(value, clone)


def input_signature(value):
    """Key an exact numerical bucket by structure, static values and layouts.

    Tensors contribute device, dtype, shape, strides and an alias ordinal, so
    two inputs match only if they share the same aliasing pattern; non-tensor
    leaves contribute their values and must be hashable.
    """
    leaves, spec = pytree.tree_flatten(value, is_leaf=_register)
    # Tensor identity -> ordinal of its first appearance.
    aliases: dict[int, int] = {}
    return spec, tuple(
        (
            item.device,
            item.dtype,
            tuple(item.shape),
            tuple(item.stride()),
            aliases.setdefault(id(item), len(aliases)),
        )
        if isinstance(item, torch.Tensor)
        else item
        for item in leaves
    )


class Inputs:
    """Bind tensor correspondence once; replay visits only these known paths.

    ``value`` is the graph's captured input. Each distinct tensor in it is
    recorded once with its PyTree key path; ``copy`` reads the tensor at the
    same path in a live input of the same structure.
    """

    def __init__(self, value):
        self.value = value
        leaves, _ = pytree.tree_flatten_with_path(value, is_leaf=_register)
        targets: dict[int, tuple[pytree.KeyPath, torch.Tensor]] = {}
        for path, item in leaves:
            if isinstance(item, torch.Tensor):
                targets.setdefault(id(item), (path, item))
        self.tensors = tuple(targets.values())

    def copy(self, live):
        """Copy ``live``'s tensors into the captured input tensors.

        Only tensor leaves are read; non-tensor leaves of ``live`` are ignored.

        Raises:
            ValueError: If a tensor's shape, dtype or device differs from the
                captured one. Tensors copied before the mismatch stay copied.
        """
        for path, destination in self.tensors:
            value = pytree.key_get(live, path)
            if (
                destination.shape != value.shape
                or destination.dtype != value.dtype
                or destination.device != value.device
            ):
                raise ValueError("graph tensor shape or representation changed")

            # Inputs already staged in the graph's own backing need no copy.
            if destination.data_ptr() == value.data_ptr():
                continue

            # A broadcast destination has one backing element along each
            # zero-stride axis; copy only that slice of the live value.
            if 0 in destination.stride():
                slices = tuple(
                    slice(0, 1) if stride == 0 else slice(None)
                    for stride in destination.stride()
                )
                destination[slices].copy_(value[slices])
            else:
                destination.copy_(value)


@dataclass
class CUDAGraphRunner:
    """One executable and the numerical input addresses it retains."""

    executable: CUDAGraph
    inputs: Inputs

    @classmethod
    def capture(
        cls,
        context,
        inputs,
        call,
        *,
        pools,
        restore=None,
        warm=True,
        warmup=None,
    ):
        """Warm and capture ``call(inputs)`` on ``context``.

        With ``warm``, an eager call first warms the kernel specializations
        and prepared resources that ``CUDAGraph.capture`` requires; a caller
        that has already run the same computation at the same shapes passes
        ``warm=False``. ``warmup`` may complete numerical work outside a
        partial graph, such as its remaining expert exchanges; by default
        it runs ``call``. ``restore``, when given, returns mutated state to its
        pre-call contents after the warm call and again after capture, so
        preparation leaves live state unchanged. ``inputs`` become the
        graph's fixed input backing, retained by the runner.
        """
        if warm:
            with context.activate():
                try:
                    (call if warmup is None else warmup)(inputs)
                finally:
                    if restore is not None:
                        restore()
        executable: CUDAGraph[Any] = CUDAGraph(context=context, pools=pools)
        try:
            executable.capture(lambda: call(inputs), restore=restore)
        except BaseException:
            executable.close()
            raise
        return cls(executable, Inputs(inputs))

    def replay(self, live=None):
        """Copy ``live`` into the fixed inputs, when given, and replay.

        Returns the graph's retained output views, which the next replay
        overwrites.
        """
        with self.executable.context.activate():
            if live is not None:
                self.inputs.copy(live)
            return self.executable.replay()

    def close(self):
        self.executable.close()


# Process-wide recency counter for ``GraphBucket.last_used``.
_USES = count()


@dataclass
class GraphBucket:
    """Variants sharing one prepared numerical shape and its fixed backing.

    ``expert_layers`` names, by module identity, the expert-parallel layers
    whose exchanges the graphs replay, so a step that replays them knows
    which layers its forward reached. Expert-step variants are keyed by
    transfer capacity; independent calls use ``None``. Every variant keeps
    the same local numerical shape and input backing.
    """

    graphs: dict = field(default_factory=dict)
    last_used: int = field(default_factory=lambda: next(_USES))
    expert_layers: frozenset[int] = frozenset()

    def touch(self):
        self.last_used = next(_USES)

    def close(self):
        try:
            close_resources(*(graph.close for graph in self.graphs.values()))
        finally:
            self.graphs.clear()


class Execution:
    """Own a prepared context, graph buckets and their allocation pools.

    Capability-specific buckets may retain padding or solver staging. All
    variants retire before the context and pools supplying their resources.
    Callers drain external readers before closing this owner.

    ``pools`` maps each CUDA device in ``devices`` to this owner's private
    ``MemPool``; it is empty when no CUDA device is given, which runners
    treat as eager-only execution.
    """

    def __init__(
        self,
        context: ExecutionContext,
        *,
        devices=(),
        storage=None,
        share=None,
    ):
        # ``share`` names another owner whose pools this one borrows; see
        # ``GraphStorage.reserve``.
        self.context = context
        self.peers: tuple[Execution, ...] = (self,)
        self.microbatches = None
        self.buckets: OrderedDict[object, GraphBucket] = OrderedDict()
        self.storage = storage if storage is not None else GraphStorage()
        self.pools = self.storage.reserve(self, devices, share=share)

    def bind_microbatches(self, peers):
        """Share one host rotation across independently prepared executions.

        Each peer retains its own inputs, plans, graph pools and context.
        The first peer owns the host threads; all peers borrow the rotation
        for warmup, while serving starts it through the first peer.
        """
        peers = tuple(peers)
        if not peers or peers[0] is not self:
            raise ValueError("the first microbatch execution owns its rotation")
        owner = Microbatches([peer.context for peer in peers])
        for peer in peers:
            peer.peers, peer.microbatches = peers, owner

    def begin_expert_step(self, capacity):
        for peer in self.peers:
            peer.context.experts.begin(capacity)

    def end_expert_step(self):
        for peer in self.peers:
            peer.context.experts.end()

    def join_expert_step(self, capacity):
        """Complete one empty step over every independent expert buffer."""
        self.begin_expert_step(capacity)
        try:
            calls = [peer.context.join_expert_layers for peer in self.peers]
            if self.microbatches is None:
                with self.context.activate():
                    calls[0]()
            else:
                self.microbatches(calls)
        finally:
            self.end_expert_step()

    def warm_experts(self, call, value):
        """Warm this execution while the other microbatches join empty.

        Only this peer writes the scratch request's KV pages. All expert
        buffers still participate, including a native persistent expert
        launch that visits the complete microbatch sequence.
        """
        if self.microbatches is None:
            return call(value)
        for peer in self.peers:
            peer.context.experts.invoked.clear()
        results = self.microbatches(
            [
                (lambda: call(value))
                if peer is self
                else peer.context.join_expert_layers
                for peer in self.peers
            ]
        )
        return results[self.peers.index(self)]

    def close_bucket(self, key):
        """Retire one bucket, first synchronizing the context stream if any."""
        bucket = self.buckets.pop(key, None)
        if bucket is not None:
            if self.context.stream is not None:
                self.context.stream.synchronize()
            bucket.close()

    def close_graphs(self):
        try:
            close_resources(*(bucket.close for bucket in self.buckets.values()))
        finally:
            self.buckets.clear()

    def close(self):
        # Graphs close before the context whose resources they captured; the
        # pools are released from storage only after both.
        try:
            close_resources(
                self.close_graphs,
                *(
                    (self.microbatches.close,)
                    if self.microbatches is not None and self.peers[0] is self
                    else ()
                ),
                self.context.close,
            )
        finally:
            # Peer groups include this execution and retain its model. Break
            # the cycle after the rotation drains, including failed startup.
            self.peers = ()
            self.microbatches = None
            self.storage.release(self)
            self.pools.clear()
