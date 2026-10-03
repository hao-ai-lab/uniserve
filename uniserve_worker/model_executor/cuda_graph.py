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

from uniserve.runtime import CUDAGraph, ExecutionContext
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


def clone_inputs(value, staged=None):
    """Clone backing while preserving broadcasts and repeated references.

    A tensor with zero strides clones only one element along those axes and
    is expanded again, so the copy keeps the broadcast. A tensor reachable
    through several paths is cloned once and the clones stay aliased.
    ``staged`` maps the ids of tensors already copied elsewhere to their
    copies, which are used instead of new clones.
    """
    copies = dict(staged or {})

    def clone(tensor):
        if id(tensor) not in copies:
            slices = tuple(
                slice(0, 1) if stride == 0 else slice(None)
                for stride in tensor.stride()
            )
            copies[id(tensor)] = tensor[slices].clone().expand(tensor.shape)
        return copies[id(tensor)]

    return map_tensors(value, clone)


class SharedBacking:
    """Byte arenas that graphs replayed one at a time share for their I/O.

    The graphs of one prepared call share a pool and replay one at a time on
    one stream, and each replay's caller copies its output out before the
    next replay (``ModelRunner.execute_model``). Their fixed inputs and
    outputs therefore need not stay distinct: each graph views its tensors
    at the start of one arena per (role, device), so the arenas hold the
    largest graph's tensors rather than every graph's.

    An arena is sized by its first use, which startup makes the largest
    call (``ModelExecutor.prepare_module``); a later call whose tensors do
    not fit gets its own backing instead, so correctness never depends on
    that order. Callers allocate inside the owners' pools
    (``GraphStorage.allocate``), as they do for unshared graph backing.
    """

    ALIGNMENT = 512

    def __init__(self):
        self._arenas: dict[tuple[str, torch.device], torch.Tensor] = {}

    def views(self, role, tensors):
        """Return contiguous views packed from the start of ``role``'s arena.

        Returns one view per tensor, with its shape and dtype and unset
        contents, or ``None`` when the tensors span several devices or do
        not fit the arena an earlier call sized.
        """
        if not tensors:
            return ()
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            return None
        device = devices.pop()
        offsets, end = [], 0
        for tensor in tensors:
            offsets.append(end)
            nbytes = tensor.numel() * tensor.element_size()
            end += -(-nbytes // self.ALIGNMENT) * self.ALIGNMENT
        arena = self._arenas.get((role, device))
        if arena is None:
            arena = torch.empty(end, dtype=torch.uint8, device=device)
            self._arenas[role, device] = arena
        elif arena.numel() < end:
            return None
        return tuple(
            arena[offset : offset + tensor.numel() * tensor.element_size()]
            .view(tensor.dtype)
            .view(tensor.shape)
            for offset, tensor in zip(offsets, tensors, strict=True)
        )

    def stage_inputs(self, value):
        """Copy ``value`` into the input arena, as ``clone_inputs`` would.

        Broadcast tensors, and every tensor when they do not fit the arena,
        are cloned into their own backing; repeated references stay aliased.
        """
        leaves, _ = pytree.tree_flatten(value, is_leaf=_register)
        unique = {}
        for item in leaves:
            if isinstance(item, torch.Tensor) and 0 not in item.stride():
                unique.setdefault(id(item), item)
        views = self.views("inputs", tuple(unique.values()))
        if views is None:
            return clone_inputs(value)
        staged = dict(zip(unique, views, strict=True))
        for key, view in staged.items():
            view.copy_(unique[key])
        return clone_inputs(value, staged)

    def output_call(self, call, example):
        """Wrap ``call`` to write its tensors into the output arena.

        ``example`` is an eager result of ``call`` fixing the output
        structure and shapes; writing it warms the copies the wrapped call
        captures. Returns ``call`` itself when the outputs do not fit the
        arena, so that graph keeps its own outputs.
        """
        leaves, spec = pytree.tree_flatten(example, is_leaf=_register)
        tensors = tuple(
            item for item in leaves if isinstance(item, torch.Tensor)
        )
        views = self.views("outputs", tensors)
        if views is None:
            return call
        for view, tensor in zip(views, tensors, strict=True):
            view.copy_(tensor)

        def write(inputs):
            result, result_spec = pytree.tree_flatten(
                call(inputs), is_leaf=_register
            )
            if result_spec != spec:
                raise ValueError("graph output structure changed")
            arena = iter(views)
            written = []
            for item in result:
                if isinstance(item, torch.Tensor):
                    view = next(arena)
                    view.copy_(item)
                    item = view
                written.append(item)
            return pytree.tree_unflatten(written, spec)

        return write


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
    def capture(cls, context, inputs, call, *, pools, restore=None, warm=True):
        """Warm and capture ``call(inputs)`` on ``context``.

        With ``warm``, an eager call first warms the kernel specializations
        and prepared resources that ``CUDAGraph.capture`` requires; a caller
        that has already run the same computation at the same shapes passes
        ``warm=False``. ``restore``, when given, returns mutated state to its
        pre-call contents after the warm call and again after capture, so
        preparation leaves live state unchanged. ``inputs`` become the
        graph's fixed input backing, retained by the runner.
        """
        if warm:
            with context.activate():
                try:
                    call(inputs)
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
    """Variants sharing one prepared numerical shape and its fixed backing."""

    graphs: dict = field(default_factory=dict)
    last_used: int = field(default_factory=lambda: next(_USES))

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
        self.buckets: OrderedDict[object, GraphBucket] = OrderedDict()
        self.storage = storage if storage is not None else GraphStorage()
        self.pools = self.storage.reserve(self, devices, share=share)
        # Owners sharing pools replay one at a time on one stream, so they
        # also share their graphs' I/O backing.
        self.backing = share.backing if share is not None else SharedBacking()

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
            close_resources(self.close_graphs, self.context.close)
        finally:
            # Owners sharing pools share the backing; the last reference
            # frees its arenas before the pools are released.
            self.backing = None
            self.storage.release(self)
            self.pools.clear()
