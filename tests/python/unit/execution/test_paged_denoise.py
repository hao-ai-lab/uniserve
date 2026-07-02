"""Contracts for shared paged-denoise helpers."""
from __future__ import annotations

import pytest
import torch

from uniserve_worker.execution.paged_denoise import PagedDenoiseBranchSet
from uniserve_worker.runtime.paged_text_cache import PagedTextCache
from uniserve_worker.runtime.residency import ResidencyManager, ScratchKvPool

pytestmark = pytest.mark.unit


def _scratch_pool() -> ScratchKvPool:
    return ScratchKvPool(
        num_layers=1,
        num_blocks=2,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
        label="test branch scratch pool",
    )


def test_paged_denoise_branch_set_reuses_rows_and_releases_scratch_blocks():
    pool = _scratch_pool()
    cond = PagedTextCache(pool, [], num_layers=1, allocate_blocks=pool.allocate_blocks)
    text_uncond = PagedTextCache(pool, [], num_layers=1, allocate_blocks=pool.allocate_blocks)
    cond.ensure_capacity(4)
    text_uncond.ensure_capacity(4)
    branches = PagedDenoiseBranchSet(
        caches={"cond": cond, "text_uncond": text_uncond},
        positions={"cond": 5, "text_uncond": 7},
    )

    first = branches.batched_cache(("cond", "text_uncond"))
    second = branches.batched_cache(("cond", "text_uncond"))
    positions = branches.positions_tensor(("cond", "text_uncond"), device="cpu", width=3)

    assert first is second
    assert positions.tolist() == [[5, 5, 5], [7, 7, 7]]
    with pytest.raises(RuntimeError, match="test branch scratch pool exhausted"):
        pool.allocate_blocks(1)

    branches.release(ResidencyManager(scratch=pool))

    assert pool.allocate_blocks(2) == [0, 1]
