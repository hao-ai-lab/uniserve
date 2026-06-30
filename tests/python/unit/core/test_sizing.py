"""KV block sizing snapshots for shared and model-specific semantics."""
from __future__ import annotations

import pytest

from uniserve_worker.foundation.sizing import derive_num_blocks

pytestmark = pytest.mark.unit


def test_derive_num_blocks_preserves_shared_default_formula():
    assert derive_num_blocks(256, None) == 4096
    assert derive_num_blocks(256, 0) == 4096
    assert derive_num_blocks(256, 512) == 2
    assert derive_num_blocks(256, 1) == 1
    assert derive_num_blocks(256, None, default_blocks=64, floor=8) == 64


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
