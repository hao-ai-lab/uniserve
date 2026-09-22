"""Native paged-attention columns and optional fused dense K/V scatter."""

from uniserve_kernels.attention import paged

from uniserve_kernels import cache


def prepare(state, key, value, batch, *, lengths, offsets):
    """Read live sequence columns and commit this call's optional cache write.

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
        fused = cache.can_run_paged_kv_write(
            state.key, state.value, indices.reshape(-1), key, value
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
