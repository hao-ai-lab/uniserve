"""Tower partial weight loading (docs/tower-parallel-and-disaggregation-plan.md §6, Phase 1).

A Mode-A und/gen worker materializes only its tower's modules; the other tower's
params stay on ``meta`` (no memory, never read by that worker's ops). The whole-
model default (``param_filter=None``) materializes everything, byte-identical to
the non-disaggregated path.
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

pytestmark = pytest.mark.integration


class _FakeTowerAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(1, 1, bias=False)
        self.qkv_proj_mot_gen = nn.Linear(1, 1, bias=False)
        self.o_proj_mot_gen = nn.Linear(1, 1, bias=False)
        self.q_norm_mot_gen = nn.Linear(1, 1, bias=False)
        self.k_norm_mot_gen = nn.Linear(1, 1, bias=False)
        self.q_norm_hw_mot_gen = nn.Linear(1, 1, bias=False)
        self.k_norm_hw_mot_gen = nn.Linear(1, 1, bias=False)


class _FakeTowerLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeTowerAttention()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(1, 1, bias=False)
        self.input_layernorm_mot_gen = nn.Linear(1, 1, bias=False)
        self.post_attention_layernorm_mot_gen = nn.Linear(1, 1, bias=False)
        self.mlp_mot_gen = nn.Linear(1, 1, bias=False)


class _FakeTowerDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Linear(1, 1, bias=False)
        self.norm = nn.Linear(1, 1, bias=False)
        self.norm_mot_gen = nn.Linear(1, 1, bias=False)
        self.layers = nn.ModuleList([_FakeTowerLayer()])


class _FakeTowerLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeTowerDecoder()


class _FakeTowerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeTowerLanguageModel()
        self.lm_head = nn.Linear(1, 1, bias=False)
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Linear(1, 1, bias=False)
        self.fm_modules = nn.ModuleDict(
            {
                "fm_head": nn.Linear(1, 1, bias=False),
                "vision_model_mot_gen": nn.Linear(1, 1, bias=False),
            }
        )


def _tower_filter(role: str | None):
    return SenseNovaU1ForUnifiedGeneration.tower_role_param_filter_from_model(
        _FakeTowerModel(),
        role,
    )


def test_partition_predicate_is_disjoint_and_complete():
    names = [
        "language_model.model.embed_tokens.weight",
        "lm_head.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.qkv_proj_mot_gen.weight",
        "language_model.model.norm_mot_gen.weight",
        "fm_modules.fm_head.weight",
        "fm_modules.vision_model_mot_gen.weight",
        "vision_model.encoder.weight",
    ]
    gen = _tower_filter("gen")
    und = _tower_filter("und")
    assert all(gen(n) ^ und(n) for n in names)  # exactly one tower per param
    assert _tower_filter(None) is None
    with pytest.raises(ValueError):
        _tower_filter("bogus")
