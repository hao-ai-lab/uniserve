"""Tower partial weight loading (docs/tower-parallel-and-disaggregation-plan.md §6, Phase 1).

A Mode-A und/gen worker materializes only its tower's modules; the other tower's
params stay on ``meta`` (no memory, never read by that worker's ops). The whole-
model default (``param_filter=None``) materializes everything, byte-identical to
the non-disaggregated path.
"""
from __future__ import annotations

import pytest

from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

pytestmark = pytest.mark.integration


def test_partition_predicate_is_disjoint_and_complete():
    names = [
        "language_model.model.embed_tokens.weight",
        "lm_head.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
        "language_model.model.norm_mot_gen.weight",
        "fm_modules.fm_head.weight",
        "fm_modules.vision_model_mot_gen.x.weight",
        "vision_model.encoder.x.weight",
    ]
    gen = SenseNovaU1ForUnifiedGeneration.tower_role_param_filter("gen")
    und = SenseNovaU1ForUnifiedGeneration.tower_role_param_filter("und")
    assert all(gen(n) ^ und(n) for n in names)  # exactly one tower per param
    assert SenseNovaU1ForUnifiedGeneration.tower_role_param_filter(None) is None
    with pytest.raises(ValueError):
        SenseNovaU1ForUnifiedGeneration.tower_role_param_filter("bogus")
