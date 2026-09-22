"""Prepared execution resources and fixed CUDA graph inputs."""

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
    """Teach PyTree about numerical dataclasses on their first encounter."""
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
    """Clone backing while preserving broadcasts and repeated references."""
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
    """Key an exact numerical bucket by structure, static values and layouts."""
    leaves, spec = pytree.tree_flatten(value, is_leaf=_register)
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
    """Bind tensor correspondence once; replay visits only these known paths."""

    def __init__(self, value):
        self.value = value
        leaves, _ = pytree.tree_flatten_with_path(value, is_leaf=_register)
        targets: dict[int, tuple[pytree.KeyPath, torch.Tensor]] = {}
        for path, item in leaves:
            if isinstance(item, torch.Tensor):
                targets.setdefault(id(item), (path, item))
        self.tensors = tuple(targets.values())

    def copy(self, live):
        for path, destination in self.tensors:
            value = pytree.key_get(live, path)
            if (
                destination.shape != value.shape
                or destination.dtype != value.dtype
                or destination.device != value.device
            ):
                raise ValueError("graph tensor shape or representation changed")
            if destination.data_ptr() == value.data_ptr():
                continue
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
    def capture(cls, context, inputs, call, *, pools, restore=None):
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
        with self.executable.context.activate():
            if live is not None:
                self.inputs.copy(live)
            return self.executable.replay()

    def close(self):
        self.executable.close()


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
    """

    def __init__(self, context: ExecutionContext, *, devices=(), storage=None):
        self.context = context
        self.buckets: OrderedDict[object, GraphBucket] = OrderedDict()
        self.storage = storage if storage is not None else GraphStorage()
        self.pools = self.storage.reserve(self, devices)

    def close_bucket(self, key):
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
        try:
            close_resources(self.close_graphs, self.context.close)
        finally:
            self.storage.release(self)
            self.pools.clear()
