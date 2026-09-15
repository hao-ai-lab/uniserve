"""Numerical kernels for request-slot reset and batched token advancement."""

from __future__ import annotations

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit(
        do_not_specialize=["row", "valid_cache_length", "logical_length", "sampling_position"]
    )
    def _reset_row_kernel(
        future_tokens_ptr,
        penalty_counts_ptr,
        predicates_ptr,
        logical_lengths_ptr,
        sampling_positions_ptr,
        cache_lengths_ptr,
        row,
        valid_cache_length,
        logical_length,
        sampling_position,
        continuation_width: tl.constexpr,
        vocab_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Reset one device runtime row while preserving its declared logical coordinates."""

        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)

        # Row-wide spans: continuation token slots [rows, continuation_width]
        # and per-vocabulary penalty counts [rows, vocab_size].
        tl.store(
            future_tokens_ptr + row * continuation_width + offsets,
            1,
            mask=offsets < continuation_width,
        )
        tl.store(
            penalty_counts_ptr + row * vocab_size + offsets,
            0,
            mask=offsets < vocab_size,
        )

        # Scalar per-row coordinates [rows]; only the first lane writes them.
        scalar = offsets == 0
        tl.store(predicates_ptr + row + offsets, 0, mask=scalar)
        tl.store(logical_lengths_ptr + row + offsets, logical_length, mask=scalar)
        tl.store(sampling_positions_ptr + row + offsets, sampling_position, mask=scalar)
        tl.store(cache_lengths_ptr + row + offsets, valid_cache_length, mask=scalar)

    @triton.jit(do_not_specialize=["count"])
    def _publish_decode_kernel(
        indices_ptr,
        tokens_ptr,
        predicates_in_ptr,
        future_tokens_ptr,
        predicates_out_ptr,
        logical_lengths_ptr,
        sampling_positions_ptr,
        cache_lengths_ptr,
        count,
        continuation_width: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Publish batched decode tokens and advance device-resident runtime coordinates."""

        offsets = tl.arange(0, block_size)
        mask = offsets < count
        indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
        tokens = tl.load(tokens_ptr + offsets, mask=mask, other=0).to(tl.int64)
        predicates = tl.load(predicates_in_ptr + offsets, mask=mask, other=0)

        # Publish each token into the first slot of its row's continuation span
        # [rows, continuation_width], keeping only the low 31 token bits.
        tl.store(
            future_tokens_ptr + indices * continuation_width,
            tokens & ((1 << 31) - 1),
            mask=mask,
        )
        tl.store(predicates_out_ptr + indices, predicates, mask=mask)

        # Advance the per-row runtime coordinates by the one accepted token.
        logical = tl.load(logical_lengths_ptr + indices, mask=mask, other=0)
        sampling = tl.load(sampling_positions_ptr + indices, mask=mask, other=0)
        cache = tl.load(cache_lengths_ptr + indices, mask=mask, other=0)
        tl.store(logical_lengths_ptr + indices, logical + 1, mask=mask)
        tl.store(sampling_positions_ptr + indices, sampling + 1, mask=mask)
        tl.store(cache_lengths_ptr + indices, cache + 1, mask=mask)
