"""Numerical resource ownership shared by library calls and workers."""

import torch

from uniserve_worker._uniserve_ipc import ExecutionContext as ExecutionContext
from uniserve_worker._uniserve_ipc import Scratch as Scratch


def _representation(module, cache, stream):
    reference = next(
        (value for value in module.parameters() if not value.is_meta), None
    )
    if reference is None:
        reference = next(
            (value for value in module.buffers() if not value.is_meta), None
        )
    device = (
        reference.device
        if reference is not None
        else cache.device
        if cache is not None
        else stream.device
        if stream is not None
        else torch.device("cpu")
    )

    # Source-only expert ranks retain typed meta parameters. Their dtype still
    # governs empty participation; the execution stream supplies the device.
    representation = reference
    if representation is None:
        representation = next(module.parameters(), None)
    dtype = (
        representation.dtype if representation is not None else torch.float32
    )
    return device, dtype


def _layer_representation(module, inherited):
    for value in (
        *module.parameters(recurse=False),
        *module.buffers(recurse=False),
    ):
        if value.is_floating_point() and not value.is_meta:
            return value.device, value.dtype
    return inherited


def _registered_output(parallel, *, rows, heads, head_dim, dtype):
    from .attention_storage import allocate_output_storage

    outputs = allocate_output_storage(
        (parallel,), rows=rows, heads=heads, head_dim=head_dim, dtype=dtype
    )
    buffers = outputs.views[parallel]
    return buffers, outputs.allocations, (buffers.local, buffers.receive)
