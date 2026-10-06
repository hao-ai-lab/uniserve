"""Prepared execution resources and fixed CUDA graph inputs.

Numerical inputs are PyTrees of dataclasses, tuples, mappings and tensors.
This module keys graph variants by an input's structure and tensor layouts
(``input_signature``), gives captured graphs their own input backing
(``clone_inputs``), and copies live inputs into that backing before each
replay (``Inputs``). Native ``Execution`` owns the prepared numerical context,
graph variants and allocation pools charged to the worker's ``GraphStorage``.
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from types import MappingProxyType
from typing import Any

import torch
from torch.utils import _pytree as pytree

from uniserve.runtime import CUDAGraph
from uniserve_worker._uniserve_ipc import Execution as Execution
from uniserve_worker._uniserve_ipc import GraphBucket as GraphBucket


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

            # Inputs already stored in the graph's own backing need no copy.
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
