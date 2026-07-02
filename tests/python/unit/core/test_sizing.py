"""KV block sizing snapshots for shared and model-specific semantics."""
from __future__ import annotations

import pytest

from uniserve_worker.foundation.sizing import derive_cuda_kv_capacity, derive_num_blocks

pytestmark = pytest.mark.unit


def test_derive_num_blocks_preserves_shared_default_formula():
    assert derive_num_blocks(256, None) == 4096
    assert derive_num_blocks(256, 0) == 4096
    assert derive_num_blocks(256, 512) == 2
    assert derive_num_blocks(256, 1) == 1
    assert derive_num_blocks(256, None, default_blocks=64, floor=8) == 64


def test_derive_cuda_kv_capacity_uses_shared_fraction_and_floor(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1_000, 2_000))

    sizing = derive_cuda_kv_capacity(
        device="cuda:0",
        block_size=16,
        bytes_per_token=10,
        memory_fraction=0.5,
        floor=4,
    )

    assert sizing is not None
    assert sizing.free_bytes == 1_000
    assert sizing.total_bytes == 2_000
    assert sizing.memory_fraction == 0.5
    assert sizing.num_blocks == 4
    assert sizing.token_capacity == 64


def test_derive_cuda_kv_capacity_returns_none_for_non_cuda_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert (
        derive_cuda_kv_capacity(
            device="cpu",
            block_size=16,
            bytes_per_token=10,
            memory_fraction=0.5,
        )
        is None
    )


def test_sensenova_num_blocks_uses_shared_formula():
    # INC-52: sensenova's no-capacity fallback now routes through the shared
    # derive_num_blocks (4096 blocks), matching qwen3/transformers/bagel instead
    # of advertising ~16 blocks — a 256x divergence in the default production
    # (no --kv-token-capacity) path.
    from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

    model = SenseNovaU1ForUnifiedGeneration(
        config={"llm_config": {"num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 4}}
    )

    assert model.num_blocks == 4096
    assert model.caps(block_size=256, kv_token_capacity=None).num_blocks == 4096
    assert model.caps(block_size=256, kv_token_capacity=512).num_blocks == 2


def test_bagel_num_blocks_snapshot_keeps_floor_semantics():
    from uniserve_worker.models.bagel import BagelForUnifiedGeneration

    model = BagelForUnifiedGeneration(config=None, device="cpu", block_size=256)

    assert model.caps(block_size=256, kv_token_capacity=None).num_blocks == 4096
    assert model.caps(block_size=256, kv_token_capacity=512).num_blocks == 64
