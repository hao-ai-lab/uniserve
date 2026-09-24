"""Trusted assembly of batches the Rust IPC decoder already validated.

The Rust codec in `uniserve_worker_ipc` decodes a submit frame and runs
`Batch::validate` on it. The PyO3 transport's `batch_to_py`
(`crates/worker-ipc-py`) then builds every member record through its ordinary
constructor and passes them to `batch_from_validated`. Python callers that
build a batch themselves construct `Batch` directly or use
`Batch.from_mapping`; both run `Batch.validate`, and `from_mapping` also
validates every call. Only records decoded by the Rust transport may enter the
factory below.
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
    every field. Construction therefore bypasses ``Batch.__init__``, so
    ``Batch.__post_init__`` does not repeat that validation.

    The argument order must match the tuple `batch_to_py` builds, and every
    `Batch` field must be assigned here: `Batch` uses slots, so reading a
    field this factory skipped raises ``AttributeError``.

    Args:
        batch_id: Logical batch identity shared by every call.
        collective_seq: Sequence number ordering collective communication.
        calls: Calls in submission order.
        block_tables: Complete KV page table per request slot and KV group.
        new_cache_pages: KV pages newly assigned by this batch.
        forward_inputs: The columnar forward inputs, in this order:
            ``forward_call_indices``, ``request_pool_indices``,
            ``seq_lens``, ``query_lens``, ``write_kv``.
        latent_params: Per-trajectory solver-step ranges and paged latent
            storage.
        decode_ranges: Bounded media-unit ranges for media calls.
        buffer_allocations: Persistent storage slices for call outputs.
        commands: Ordered lifecycle commands.
        input_products: Published tensor values feeding declared call inputs.
        kv_inputs: KV publications installed by this batch's calls.

    Returns:
        The assembled batch, without `Batch.validate` having run.
    """
    # `Batch` is frozen, so fields are written through object.__setattr__.
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
