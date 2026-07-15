"""KV block sizing snapshots for shared and model-specific semantics."""
from __future__ import annotations

import pytest

from uniserve_worker.foundation.sizing import (
    derive_cuda_kv_capacity,
    derive_num_blocks,
    derive_runtime_kv_capacity,
)

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


def test_derive_runtime_kv_capacity_prefers_explicit_tokens(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    capacity = derive_runtime_kv_capacity(
        block_size=16,
        kv_token_capacity=48,
        bytes_per_token=10,
        device="cuda:0",
        memory_fraction=0.5,
        floor=4,
    )

    assert capacity.num_blocks == 4
    assert capacity.token_capacity == 64
    assert capacity.cuda is None


def test_derive_runtime_kv_capacity_uses_cuda_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1_000, 2_000))

    capacity = derive_runtime_kv_capacity(
        block_size=16,
        kv_token_capacity=None,
        bytes_per_token=10,
        device="cuda:0",
        memory_fraction=0.5,
        floor=4,
    )

    assert capacity.num_blocks == 4
    assert capacity.token_capacity == 64
    assert capacity.cuda is not None


def test_derive_runtime_kv_capacity_falls_back_to_default_blocks(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    capacity = derive_runtime_kv_capacity(
        block_size=16,
        kv_token_capacity=None,
        bytes_per_token=10,
        device="cuda:0",
        memory_fraction=0.5,
    )

    assert capacity.num_blocks == 4096
    assert capacity.token_capacity == 4096 * 16
    assert capacity.cuda is None


def test_sensenova_caps_reserve_decode_graph_padding_blocks():
    from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

    model = SenseNovaU1ForUnifiedGeneration(
        config={"llm_config": {"num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 4}}
    )

    assert model.num_blocks == 4096
    assert model.caps(block_size=256, kv_token_capacity=None).num_blocks == 4095
    assert model.caps(block_size=256, kv_token_capacity=512).num_blocks == 1
    assert model.caps(block_size=64, kv_token_capacity=4096).num_blocks == 62


def test_sensenova_latent_and_scratch_capacity_scale_for_concurrent_interleave():
    from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

    model = SenseNovaU1ForUnifiedGeneration(
        config={
            "max_image_seq_len": 4096,
            "llm_config": {
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
            },
        },
        block_size=16,
        kv_token_capacity=65536,
    )

    caps = model.caps(block_size=16, kv_token_capacity=65536)

    assert caps.max_latent_size == 65536
    assert caps.max_vae_grid_tokens == 4096
    assert caps.scratch_capacity_tokens >= (4096 + (65536 // 16) * 4) * 16


def test_sensenova_caps_declare_worker_owned_encoder_residency():
    from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

    model = SenseNovaU1ForUnifiedGeneration(
        config={"llm_config": {"num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 4}}
    )
    caps = model.caps()

    assert "encoder_output" in caps.resource_classes
    assert caps.encoder_cache_budget == model.ENCODER_CACHE_BUDGET
    assert model.residency.encoder.budget == caps.encoder_cache_budget


def test_bagel_caps_reserve_decode_graph_padding_blocks():
    from uniserve_worker.models.bagel import BagelForUnifiedGeneration

    model = BagelForUnifiedGeneration(config=None, device="cpu", block_size=256)

    assert model.num_blocks == 4096
    assert model.caps(block_size=256, kv_token_capacity=None).num_blocks == 4095
    assert model.caps(block_size=256, kv_token_capacity=512).num_blocks == 63
    assert model.caps(block_size=64, kv_token_capacity=4096).num_blocks == 62


def test_bagel_latent_capacity_scales_for_concurrent_generation():
    from uniserve_worker.models.bagel import BagelConfig, BagelForUnifiedGeneration

    model = BagelForUnifiedGeneration(
        config=BagelConfig(max_latent_size=64),
        device="cpu",
        block_size=16,
        kv_token_capacity=65536,
    )

    caps = model.caps(block_size=16, kv_token_capacity=65536)

    assert caps.max_latent_size == 65536
    assert caps.max_vae_grid_tokens == model.cfg.latent_token_capacity + caps.commit_marker_tokens


def test_bagel_scratch_pool_reserves_request_kv_capacity_for_packed_staging():
    from uniserve_worker.models.bagel import BagelForUnifiedGeneration

    model = BagelForUnifiedGeneration(
        config=None,
        device="cpu",
        block_size=64,
        kv_token_capacity=65_536,
    )
    caps = model.caps(block_size=64, kv_token_capacity=65_536)

    denoise_blocks = caps.scratch_capacity_tokens // 64
    assert denoise_blocks == 1_024
    assert model._scratch_num_blocks(64) == denoise_blocks + model.num_blocks


def test_bagel_caps_bound_maximum_image_ingest_kv_writes():
    from uniserve_worker.models.bagel import BagelForUnifiedGeneration

    model = BagelForUnifiedGeneration(config=None, device="cpu", block_size=256)
    caps = model.caps(block_size=256, kv_token_capacity=None)

    max_vit_patches = (model.cfg.vit_image_size // model.cfg.vit_patch_size) ** 2
    assert caps.max_vit_grid_tokens == max_vit_patches + caps.commit_marker_tokens
    assert caps.max_vae_grid_tokens == model.cfg.latent_token_capacity + caps.commit_marker_tokens
