"""Numerical kernels for request-slot reset and batched token advancement.

These Triton kernels back `DecodeState` (`storage.decode_state`). The state
tensors they update are indexed by request slot, with row zero reserved as
the padding sentinel. The native owner selects these CUDA kernels when
Triton can launch; other devices use indexed tensor operations.
"""

from __future__ import annotations

from uniserve_kernels.triton import tl, triton

if triton is not None:
    # Both kernels exclude per-call counts from Triton's value
    # specialization, so a new batch size does not compile another kernel
    # variant.
    @triton.jit(do_not_specialize=["count"])
    def _reset_rows_kernel(
        columns_ptr,
        future_tokens_ptr,
        penalty_counts_ptr,
        predicates_ptr,
        logical_lengths_ptr,
        sampling_positions_ptr,
        cache_lengths_ptr,
        count,
        continuation_width: tl.constexpr,
        vocab_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Reset ``count`` request rows in one launch.

        ``columns`` is an int64 ``[4, count]`` table: the rows, then their
        verified cache lengths, logical lengths and sampling positions.
        Program ``(i, j)`` resets block ``j`` of row ``i``: continuation
        tokens become 1, penalty counts and the predicate become zero, and
        the coordinates take the row's column values. The launch grid must be
        ``count`` by the ``block_size`` chunks of ``max(continuation_width,
        vocab_size)``, as the numerical `reset_rows` function sizes it.
        """
        index = tl.program_id(0)
        row = tl.load(columns_ptr + index)
        offsets = tl.program_id(1) * block_size + tl.arange(0, block_size)

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

        # Scalar per-row coordinates [rows]; only lane zero of block zero
        # writes them.
        scalar = offsets == 0
        cache_length = tl.load(columns_ptr + count + index)
        logical_length = tl.load(columns_ptr + 2 * count + index)
        sampling_position = tl.load(columns_ptr + 3 * count + index)
        tl.store(predicates_ptr + row + offsets, 0, mask=scalar)
        tl.store(
            logical_lengths_ptr + row + offsets, logical_length, mask=scalar
        )
        tl.store(
            sampling_positions_ptr + row + offsets,
            sampling_position,
            mask=scalar,
        )
        tl.store(cache_lengths_ptr + row + offsets, cache_length, mask=scalar)

    @triton.jit(do_not_specialize=["count"])
    def _commit_tokens_kernel(
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
        index_stride: tl.constexpr,
        token_stride: tl.constexpr,
        predicate_stride: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Commit batched decode tokens and advance runtime coordinates.

        Runs as a single program: ``block_size`` must be at least ``count``,
        and lanes past ``count`` are masked. Row ``indices`` must be unique,
        because the coordinate updates are unsynchronized load/store pairs.
        The native owner checks the host slot sequence before launch; the
        caller supplies the corresponding device indices.
        """
        offsets = tl.arange(0, block_size)
        mask = offsets < count
        # Result columns may be views into a wider sampled-output tensor.
        indices = tl.load(
            indices_ptr + offsets * index_stride, mask=mask, other=0
        )
        tokens = tl.load(
            tokens_ptr + offsets * token_stride, mask=mask, other=0
        ).to(tl.int64)
        predicates = tl.load(
            predicates_in_ptr + offsets * predicate_stride, mask=mask, other=0
        )

        # Store each token into the first slot of its row's continuation span
        # [rows, continuation_width], keeping only the low 31 token bits: bit
        # 31 is `TOKEN_CONTINUATION_BIT` in tagged relays, never a token bit.
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
