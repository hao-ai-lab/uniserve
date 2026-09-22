"""Construction boundary for records already validated by the IPC decoder.

Direct Python callers construct protocol dataclasses or use ``from_mapping``;
those paths retain their complete descriptor validation. Only decoded Rust IPC
records may enter the trusted batch factory below.
"""

from __future__ import annotations

from .batch import (
    Batch,
    BatchCommand,
    BlockTable,
    BufferAllocation,
    CachePageAllocation,
    DecodeRange,
    LatentParams,
    TensorPublication,
)
from .call import Call
from .transfer import KvTransfer


def batch_from_validated(
    batch_id: int,
    collective_seq: int,
    calls: tuple[Call, ...],
    block_tables: tuple[BlockTable, ...],
    new_cache_pages: tuple[CachePageAllocation, ...],
    forward_inputs: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[bool, ...],
    ],
    latent_params: tuple[LatentParams, ...],
    decode_ranges: tuple[DecodeRange, ...],
    buffer_allocations: tuple[BufferAllocation, ...],
    commands: tuple[BatchCommand, ...],
    input_products: tuple[TensorPublication, ...],
    kv_inputs: tuple[KvTransfer, ...],
) -> Batch:
    """Assemble a validated batch from transport-constructed members.

    Called by the Rust IPC transport, which has already decoded and validated
    every field. Construction therefore bypasses ``Batch.__init__`` so
    typed leaves are not reparsed and ``__post_init__`` validation is not
    repeated.
    """
    batch = object.__new__(Batch)
    set_field = object.__setattr__
    set_field(batch, "batch_id", batch_id)
    set_field(batch, "collective_seq", collective_seq)
    set_field(batch, "calls", calls)
    set_field(batch, "block_tables", block_tables)
    set_field(batch, "new_cache_pages", new_cache_pages)
    set_field(batch, "forward_call_indices", forward_inputs[0])
    set_field(batch, "request_pool_indices", forward_inputs[1])
    set_field(batch, "seq_lens", forward_inputs[2])
    set_field(batch, "query_lens", forward_inputs[3])
    set_field(batch, "write_kv", forward_inputs[4])

    set_field(batch, "latent_params", latent_params)
    set_field(batch, "decode_ranges", decode_ranges)
    set_field(batch, "buffer_allocations", buffer_allocations)
    set_field(batch, "commands", commands)

    set_field(batch, "input_products", input_products)
    set_field(batch, "kv_inputs", kv_inputs)
    return batch
