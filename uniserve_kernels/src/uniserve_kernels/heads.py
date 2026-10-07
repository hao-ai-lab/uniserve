"""Head-shard relayout kernels for the Ulysses head exchange.

``uniserve.distributed.tokens.HeadExchange`` trades a member's token shard
of every head for every token of its own heads. ``pack_head_shards`` writes
the outgoing payload destination-major: up to three ``[tokens, heads, dim]``
sources (query, key and value) become one ``[members, tokens, slots, dim]``
buffer whose member ``m`` row holds, for each source in order, the heads
member ``m`` owns. ``merge_head_shards`` restores the received
``[members, tokens, heads, dim]`` attention output to ``[tokens,
members * heads, dim]``. Both are pure copies, so their results equal the
tensor-operation relayouts bit for bit.

A source with fewer heads than members is replicated: member ``m`` takes
head ``m // replicas`` as its single slot. Sources and the merge destination
may use any row and head strides with contiguous features. Launchers do not
revalidate operands; callers run ``can_run_triton_*`` first.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

# Token rows each program moves for one head slot.
_BLOCK_TOKENS = 32


if triton is not None:  # pragma: no cover - depends on the accelerator stack.

    @triton.jit
    def _source_rows(
        source,
        tokens_index,
        member,
        local,
        row_stride,
        head_stride,
        columns,
        heads: tl.constexpr,
        replicas: tl.constexpr,
    ):
        """Address one member slot's rows of a source head."""
        # Partitioned heads give member m the interval [m * heads, ...);
        # replicated heads (heads == 1) give it head m // replicas.
        head = (member * heads + local) // replicas
        return (
            source
            + tokens_index[:, None] * row_stride
            + head * head_stride
            + columns[None, :]
        )

    @triton.jit
    def _pack_head_shards_kernel(
        out,
        first,
        second,
        third,
        first_row,
        first_head,
        second_row,
        second_head,
        third_row,
        third_head,
        tokens,
        first_heads: tl.constexpr,
        second_heads: tl.constexpr,
        third_heads: tl.constexpr,
        first_replicas: tl.constexpr,
        second_replicas: tl.constexpr,
        third_replicas: tl.constexpr,
        dim: tl.constexpr,
        block_dim: tl.constexpr,
        block_tokens: tl.constexpr,
    ):
        """Copy one member slot's ``[block_tokens, dim]`` rows.

        Grid: (token blocks, members * slots). Slot ``local`` of member
        ``member`` reads the source whose head range holds it.
        """
        slots_per_member: tl.constexpr = (
            first_heads + second_heads + third_heads
        )
        member = tl.program_id(1) // slots_per_member
        local = tl.program_id(1) % slots_per_member
        tokens_index = (
            tl.program_id(0) * block_tokens + tl.arange(0, block_tokens)
        ).to(tl.int64)
        columns = tl.arange(0, block_dim)
        valid = (tokens_index[:, None] < tokens) & (columns[None, :] < dim)

        if local < first_heads:
            source = _source_rows(
                first,
                tokens_index,
                member,
                local,
                first_row,
                first_head,
                columns,
                first_heads,
                first_replicas,
            )
        elif local < first_heads + second_heads:
            source = _source_rows(
                second,
                tokens_index,
                member,
                local - first_heads,
                second_row,
                second_head,
                columns,
                second_heads,
                second_replicas,
            )
        else:
            source = _source_rows(
                third,
                tokens_index,
                member,
                local - first_heads - second_heads,
                third_row,
                third_head,
                columns,
                third_heads,
                third_replicas,
            )
        values = tl.load(source, mask=valid)

        # out[member, token, local, :] in a contiguous buffer.
        destination = (
            out
            + (
                (member * tokens + tokens_index[:, None]) * slots_per_member
                + local
            )
            * dim
            + columns[None, :]
        )
        tl.store(destination, values, mask=valid)

    @triton.jit
    def _merge_head_shards_kernel(
        received,
        out,
        out_row,
        out_head,
        tokens,
        heads: tl.constexpr,
        dim: tl.constexpr,
        block_dim: tl.constexpr,
        block_tokens: tl.constexpr,
    ):
        """Copy one ``[block_tokens, dim]`` tile of one received head.

        Grid: (token blocks, members * heads). Received head ``local`` of
        member ``member`` becomes output head ``member * heads + local``.
        """
        slot = tl.program_id(1)
        member = slot // heads
        local = slot % heads
        tokens_index = (
            tl.program_id(0) * block_tokens + tl.arange(0, block_tokens)
        ).to(tl.int64)
        columns = tl.arange(0, block_dim)
        valid = (tokens_index[:, None] < tokens) & (columns[None, :] < dim)

        source = (
            received
            + ((member * tokens + tokens_index[:, None]) * heads + local) * dim
            + columns[None, :]
        )
        values = tl.load(source, mask=valid)
        destination = (
            out
            + tokens_index[:, None] * out_row
            + slot * out_head
            + columns[None, :]
        )
        tl.store(destination, values, mask=valid)


def _rows(value: torch.Tensor) -> bool:
    """Check a ``[tokens, heads, dim]`` operand with contiguous features."""
    return value.ndim == 3 and int(value.stride(-1)) == 1


def _launchable(*values: torch.Tensor) -> bool:
    device = values[0].device
    return (
        triton is not None
        and launchable(device)
        and all(value.is_cuda and value.device == device for value in values)
    )


def can_run_triton_pack_head_shards(
    sources: tuple[torch.Tensor, ...],
    heads: tuple[int, ...],
    replicas: tuple[int, ...],
    out: torch.Tensor,
) -> bool:
    """Check the operands of :func:`triton_pack_head_shards`.

    One to three ``[tokens, heads, dim]`` sources share tokens, dim and
    dtype; ``heads[i]`` is the slots source ``i`` gives each member and
    ``replicas[i]`` how many adjacent members share one of its heads (1 for
    a partitioned source). ``out`` is the contiguous
    ``[members, tokens, sum(heads), dim]`` payload.
    """
    if not 1 <= len(sources) <= 3 or not (
        len(sources) == len(heads) == len(replicas)
    ):
        return False
    first = sources[0]
    return (
        _launchable(*sources, out)
        and all(_rows(source) for source in sources)
        and all(
            source.shape[0] == first.shape[0]
            and source.shape[2] == first.shape[2]
            and source.dtype == first.dtype
            for source in sources
        )
        and out.is_contiguous()
        and out.dtype == first.dtype
        and out.ndim == 4
        and tuple(out.shape[1:]) == (first.shape[0], sum(heads), first.shape[2])
        and min(heads) >= 1
        and min(replicas) >= 1
    )


def triton_pack_head_shards(
    sources: tuple[torch.Tensor, ...],
    heads: tuple[int, ...],
    replicas: tuple[int, ...],
    out: torch.Tensor,
) -> None:
    """Write each member's head slots of every source destination-major.

    ``out[m, t, offset_i + j]`` receives ``sources[i][t, (m * heads[i] + j)
    // replicas[i]]``, where ``offset_i`` sums the earlier sources' slots.
    Callers first check :func:`can_run_triton_pack_head_shards`.
    """
    members, tokens, slots, dim = (int(size) for size in out.shape)
    # Absent sources contribute no slots; their pointer is never read.
    padded = (*sources, *(sources[0],) * (3 - len(sources)))
    counts = (*heads, *(0,) * (3 - len(heads)))
    shares = (*replicas, *(1,) * (3 - len(replicas)))
    strides = [
        stride
        for source in padded
        for stride in (int(source.stride(0)), int(source.stride(1)))
    ]
    _pack_head_shards_kernel[
        (triton.cdiv(tokens, _BLOCK_TOKENS), members * slots)
    ](
        out,
        *padded,
        *strides,
        tokens,
        *counts,
        *shares,
        dim,
        triton.next_power_of_2(dim),
        _BLOCK_TOKENS,
        num_warps=4,
    )


def can_run_triton_merge_head_shards(
    received: torch.Tensor, out: torch.Tensor
) -> bool:
    """Check the operands of :func:`triton_merge_head_shards`.

    ``received`` is the contiguous ``[members, tokens, heads, dim]`` payload
    and ``out`` a ``[tokens, members * heads, dim]`` destination with
    contiguous features.
    """
    return (
        _launchable(received, out)
        and received.ndim == 4
        and received.is_contiguous()
        and _rows(out)
        and out.dtype == received.dtype
        and tuple(out.shape)
        == (
            received.shape[1],
            received.shape[0] * received.shape[2],
            received.shape[3],
        )
    )


def triton_merge_head_shards(received: torch.Tensor, out: torch.Tensor) -> None:
    """Restore ``[members, tokens, heads, dim]`` to token-major heads.

    ``out[t, m * heads + j]`` receives ``received[m, t, j]``. Callers first
    check :func:`can_run_triton_merge_head_shards`.
    """
    members, tokens, heads, dim = (int(size) for size in received.shape)
    _merge_head_shards_kernel[
        (triton.cdiv(tokens, _BLOCK_TOKENS), members * heads)
    ](
        received,
        out,
        int(out.stride(0)),
        int(out.stride(1)),
        tokens,
        heads,
        dim,
        triton.next_power_of_2(dim),
        _BLOCK_TOKENS,
        num_warps=4,
    )
