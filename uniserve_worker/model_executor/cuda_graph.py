"""Prepared execution resources and fixed CUDA graph inputs.

Numerical inputs are PyTrees of dataclasses, tuples, mappings and tensors.
This module keys graph variants by an input's structure and tensor layouts
(``input_signature``), gives captured graphs their own input backing
(``clone_inputs``), and copies live inputs into that backing before each
replay (``GraphInputs``). Native ``Execution`` owns the prepared context, graph
variants and allocation pools charged to the worker's ``GraphStorage``.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from types import MappingProxyType

import torch
from torch.utils import _pytree as pytree

from uniserve_worker._uniserve_ipc import CUDAGraphRunner as CUDAGraphRunner
from uniserve_worker._uniserve_ipc import Execution as Execution
from uniserve_worker._uniserve_ipc import GraphBucket as GraphBucket
from uniserve_worker._uniserve_ipc import GraphInputs as GraphInputs
from uniserve_worker._uniserve_ipc import InputBatch
from uniserve_worker._uniserve_ipc import input_signature as input_signature

_BATCH_FIELDS = (
    "forward_mode",
    "inputs",
    "request_pool_indices",
    "token_selections",
    "decode_force_finish",
)


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
        elif cls is InputBatch:
            pytree.register_pytree_node(
                cls,
                lambda value: (
                    [getattr(value, name) for name in _BATCH_FIELDS],
                    None,
                ),
                lambda values, context: InputBatch(*values),
                flatten_with_keys_fn=lambda value: (
                    [
                        (pytree.GetAttrKey(name), getattr(value, name))
                        for name in _BATCH_FIELDS
                    ],
                    None,
                ),
            )
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
