"""KV block sizing snapshots for shared and model-specific semantics."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

from uniserve_worker.foundation.sizing import derive_num_blocks

pytestmark = pytest.mark.unit


def test_derive_num_blocks_preserves_shared_default_formula():
    assert derive_num_blocks(256, None) == 4096
    assert derive_num_blocks(256, 0) == 4096
    assert derive_num_blocks(256, 512) == 2
    assert derive_num_blocks(256, 1) == 1
    assert derive_num_blocks(256, None, default_blocks=64, floor=8) == 64


def test_qwen3_num_blocks_snapshot_uses_shared_formula():
    from uniserve_worker.models.qwen3 import Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        config={
            "vocab_size": 8,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
        }
    )

    assert model.caps(block_size=16, kv_token_capacity=None).num_blocks == 4096
    assert model.caps(block_size=16, kv_token_capacity=64).num_blocks == 4


def test_transformers_fallback_num_blocks_snapshot_keeps_its_fallback():
    from uniserve_worker.models.transformers_fallback import TransformersForCausalLM

    cfg = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        eos_token_id=2,
    )
    model = TransformersForCausalLM(
        model=nn.Linear(1, 1),
        tokenizer=SimpleNamespace(eos_token_id=2),
        config=cfg,
        device="cpu",
        block_size=16,
        kv_token_capacity=None,
    )

    assert model.caps(block_size=16, kv_token_capacity=None).num_blocks == 4096
    assert model.caps(block_size=16, kv_token_capacity=64).num_blocks == 4


def test_sensenova_num_blocks_uses_shared_formula():
    # INC-52: sensenova's no-capacity fallback now routes through the shared
    # derive_num_blocks (4096 blocks), matching qwen3/transformers/bagel instead
    # of advertising ~16 blocks — a 256x divergence in the default production
    # (no --kv-token-capacity) path.
    from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

    model = SenseNovaU1ForUnifiedGeneration(config={"llm_config": {"num_hidden_layers": 1}})

    assert model.num_blocks == 4096
    assert model.caps(block_size=256, kv_token_capacity=None).num_blocks == 4096
    assert model.caps(block_size=256, kv_token_capacity=512).num_blocks == 2


def test_bagel_num_blocks_snapshot_keeps_floor_semantics():
    from uniserve_worker.models.bagel import BagelForUnifiedGeneration

    model = BagelForUnifiedGeneration(config=None, device="cpu", block_size=256)

    assert model.caps(block_size=256, kv_token_capacity=None).num_blocks == 4096
    assert model.caps(block_size=256, kv_token_capacity=512).num_blocks == 64
