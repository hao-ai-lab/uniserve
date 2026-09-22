"""Paged K/V state writes addressed by physical token slots."""

from __future__ import annotations

import torch


def paged_kv_write(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    *,
    cast: bool = False,
    initialized: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Scatter current K/V rows into physical token slots.

    ``k_cache`` and ``v_cache`` are ``[blocks, block_size, heads, dim]``. A
    slot is ``block * block_size + token``; ``-1`` skips a token and physical
    block zero is ordinary storage. Slots and K/V share their leading shape:
    ``[N]`` with ``[N, heads, dim]``, or ``[batch, n]`` with ``[batch, n,
    heads, dim]``. With ``cast`` the sources convert to the cache dtype;
    otherwise the dtypes must match. ``initialized`` flags receive ``True``
    for every written block. Callers authorize the write intervals;
    out-of-range slots are index errors.
    """
    from uniserve_kernels import cache

    blocks, block_size, heads, head_dim = (int(dim) for dim in k_cache.shape)
    k_source = k_current.reshape(-1, heads, head_dim)
    v_source = v_current.reshape(-1, heads, head_dim)
    slots = slots.reshape(-1)
    if initialized is not None and any(
        flags.shape != (blocks,)
        or flags.dtype != torch.bool
        or flags.device != k_cache.device
        for flags in initialized
    ):
        raise ValueError(
            "cache initialization flags must match block count and device"
        )
    if cache.can_run_paged_kv_write(
        k_cache, v_cache, slots, k_source, v_source
    ):
        cache.paged_kv_write(
            k_cache, v_cache, slots, k_source, v_source, initialized
        )
        return

    if not bool(((slots >= -1) & (slots < blocks * block_size)).all()):
        raise ValueError("paged KV write index out of bounds")
    selected = torch.nonzero(slots >= 0, as_tuple=False).reshape(-1)
    if int(selected.numel()) == 0:
        return

    slots = slots.index_select(0, selected)
    k_source = k_source.index_select(0, selected)
    v_source = v_source.index_select(0, selected)
    if cast:
        k_source = k_source.to(dtype=k_cache.dtype)
        v_source = v_source.to(dtype=v_cache.dtype)
    for target, source in (
        (k_cache.view(-1, heads, head_dim), k_source),
        (v_cache.view(-1, heads, head_dim), v_source),
    ):
        if target.dtype is torch.float8_e4m3fn:
            # index_copy_ has no float8 kernel; scatter the raw bytes instead.
            target.view(torch.uint8).index_copy_(
                0, slots, source.view(torch.uint8)
            )
        else:
            target.index_copy_(0, slots, source)

    if initialized is not None:
        for flags in initialized:
            flags.index_fill_(0, slots // block_size, True)
