"""Tower partial weight loading (docs/tower-parallel-and-disaggregation-plan.md §6, Phase 1).

A Mode-A und/gen worker materializes only its tower's modules; the other tower's
params stay on ``meta`` (no memory, never read by that worker's ops). The whole-
model default (``param_filter=None``) materializes everything, byte-identical to
the non-disaggregated path.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from uniserve_worker.loader import transformers as native
from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration


class _TowerStub(nn.Module):
    """A stub model mirroring the SenseNova tower naming convention."""

    def __init__(self) -> None:
        super().__init__()
        # understanding tower
        self.embed_tokens = nn.Linear(4, 4, bias=False)
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.norm = nn.Linear(4, 4, bias=False)
        # generation tower
        self.q_proj_mot_gen = nn.Linear(4, 4, bias=False)
        self.norm_mot_gen = nn.Linear(4, 4, bias=False)
        self.fm_modules = nn.ModuleDict({"fm_head": nn.Linear(4, 4, bias=False)})


def _meta_stub() -> nn.Module:
    from accelerate import init_empty_weights

    with init_empty_weights():
        return _TowerStub()


def _materialized(model: nn.Module) -> set[str]:
    return {n for n, p in model.named_parameters() if p.device.type != "meta"}


def _stream_with_filter(model: nn.Module, weights: dict, param_filter, monkeypatch) -> None:
    from accelerate.utils import set_module_tensor_to_device

    monkeypatch.setattr(native, "resolve_weight_files", lambda _dir: ["stub"])
    monkeypatch.setattr(native, "iter_weights", lambda _files: iter(weights.items()))
    native._stream_checkpoint_weights(
        model,
        "ignored",
        device="cpu",
        dtype=torch.float32,
        set_module_tensor_to_device=set_module_tensor_to_device,
        param_filter=param_filter,
    )


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


def test_gen_role_materializes_only_gen_tower(monkeypatch):
    model = _meta_stub()
    weights = {n: torch.zeros(4, 4) for n in model.state_dict()}
    gen_filter = SenseNovaU1ForUnifiedGeneration.tower_role_param_filter("gen")
    _stream_with_filter(model, weights, gen_filter, monkeypatch)
    got = _materialized(model)
    assert got == {"q_proj_mot_gen.weight", "norm_mot_gen.weight", "fm_modules.fm_head.weight"}
    # understanding tower params stay on meta (no memory)
    assert all(p.device.type == "meta" for n, p in model.named_parameters() if n not in got)


def test_und_role_materializes_only_und_tower(monkeypatch):
    model = _meta_stub()
    weights = {n: torch.zeros(4, 4) for n in model.state_dict()}
    und_filter = SenseNovaU1ForUnifiedGeneration.tower_role_param_filter("und")
    _stream_with_filter(model, weights, und_filter, monkeypatch)
    got = _materialized(model)
    assert got == {"embed_tokens.weight", "q_proj.weight", "norm.weight"}


def test_no_filter_materializes_whole_model(monkeypatch):
    model = _meta_stub()
    weights = {n: torch.zeros(4, 4) for n in model.state_dict()}
    _stream_with_filter(model, weights, None, monkeypatch)
    assert _materialized(model) == set(model.state_dict().keys())


def test_partial_load_does_not_raise_on_out_of_scope_missing(monkeypatch):
    # The checkpoint carries every tensor, but the gen filter materializes only a
    # subset; the und tensors must NOT be reported missing.
    model = _meta_stub()
    weights = {n: torch.zeros(4, 4) for n in model.state_dict()}
    gen_filter = SenseNovaU1ForUnifiedGeneration._is_gen_tower_param
    _stream_with_filter(model, weights, gen_filter, monkeypatch)  # no RuntimeError
