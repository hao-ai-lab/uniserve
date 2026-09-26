"""Native paged-attention columns and optional fused dense K/V scatter."""

from uniserve_kernels.attention import paged

from uniserve_kernels import cache


def prepare(state, key, value, batch, *, lengths, offsets):
    """Read live sequence columns and commit this call's optional cache write.

    ``lengths`` and ``offsets`` receive each row's key count and their
    cumulative offsets, counted from the row's first block table column: a
    table with start pages yields keys from token ``start_page *
    block_size`` on, so kernels aligning queries to the end of their keys
    keep the same relative causal positions and history window.

    Dense contiguous backing can share a launch with metadata generation.
    Other numerical representations use the state's ordinary write call.
    The outputs borrow context workspace and are valid until its next use.
    """
    indices = batch.write_indices
    fused = False

    if indices is not None:
        if state is None:
            raise RuntimeError(
                "attention cache update requires bound prefix state"
            )
        state._validate_update(key, value, indices)
        # The fused launch writes plain same-dtype rows; conversions and
        # encoded representations take the state's own write call.
        fused = (
            indices.numel() > 0
            and cache.unsupported_paged_kv_write(
                state.key, state.value, indices.reshape(-1), key, value
            )
            is None
        )
        if not fused:
            state.update(key, value, indices=indices)

    paged.prepare(
        (
            batch.queries.values,
            batch.prefixes.values,
            batch.queries.offsets,
            batch.prefixes.offsets,
        ),
        lengths,
        offsets,
        batch_size=batch.queries.batch_size,
        start_pages=(
            None
            if batch.block_table.start_page is None
            else (batch.block_table.start_page, batch.block_table.block_size)
        ),
        write=(
            state.key,
            state.value,
            indices,
            key,
            value,
            state.initialized["key"],
            state.initialized["value"],
            state.block_size,
        )
        if fused
        else None,
    )
