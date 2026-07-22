"""Tower partial weight loading (docs/tower-parallel-and-disaggregation-plan.md §6, Phase 1).

A Mode-A und/gen worker materializes only its tower's parameters, selected by
the declarative tower split in the SenseNova ``WeightSpec``; the other tower's
params stay on ``meta`` (no memory, never read by that worker's ops). The
whole-model default (no role filter) materializes everything, byte-identical to
the non-disaggregated path.
"""
from __future__ import annotations

import pytest

from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

pytestmark = pytest.mark.integration

# A representative checkpoint param-name set spanning both towers.
_PARAM_NAMES = (
    "language_model.model.embed_tokens.weight",
    "lm_head.weight",
    "language_model.model.layers.0.self_attn.q_proj.weight",
    "language_model.model.layers.0.self_attn.qkv_proj_mot_gen.weight",
    "language_model.model.norm_mot_gen.weight",
    "fm_modules.fm_head.weight",
    "fm_modules.vision_model_mot_gen.weight",
    "vision_model.encoder.weight",
)


def _tower_filter(role: str | None):
    return SenseNovaU1ForUnifiedGeneration.weight_spec.tower.role_filter(role)


def test_partition_predicate_is_disjoint_and_complete():
    gen = _tower_filter("gen")
    und = _tower_filter("und")
    assert all(gen(n) ^ und(n) for n in _PARAM_NAMES)  # exactly one tower per param
    assert _tower_filter(None) is None
    with pytest.raises(ValueError):
        _tower_filter("bogus")
