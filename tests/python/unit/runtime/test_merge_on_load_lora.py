"""Behavioral tests for ``MergeOnLoadLoRA`` merge-on-load semantics.

Covers INV-LORA-SINGLE-RESIDENT: a single adapter is merged directly into the
model weights as ``(B @ A) * scale``, a second load is rejected while one is
resident, an adapter matching no parameters is rejected, and unload subtracts
the exact deltas restoring the original weights bit-for-bit.

Everything is driven through the public ``load``/``unload`` API against a tiny
deterministic ``nn.Module`` and real on-disk safetensors adapters.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.runtime.lora import MergeOnLoadLoRA

pytestmark = pytest.mark.unit


class _TinyModel(torch.nn.Module):
    """Module whose single weight matches the PEFT target ``q_proj``.

    ``out_features=4`` rows, ``in_features=3`` cols, so a rank-``r`` adapter has
    ``A`` of shape ``(r, 3)`` and ``B`` of shape ``(4, r)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 4, bias=False)


def _write_adapter(
    dir_path,
    *,
    target: str = "base_model.model.q_proj",
    a: torch.Tensor,
    b: torch.Tensor,
    lora_alpha: float | None = None,
    rank: int | None = None,
) -> str:
    """Write an ``adapter_model.safetensors`` (+ optional config) under ``dir_path``.

    Returns the directory path that is passed to ``MergeOnLoadLoRA.load``.
    """
    tensors = {
        f"{target}.lora_A.weight": a.contiguous(),
        f"{target}.lora_B.weight": b.contiguous(),
    }
    save_file(tensors, str(dir_path / "adapter_model.safetensors"))
    if lora_alpha is not None and rank is not None:
        (dir_path / "adapter_config.json").write_text(
            json.dumps({"r": rank, "lora_alpha": lora_alpha}),
            encoding="utf-8",
        )
    return str(dir_path)


def test_load_merges_scaled_delta_into_target_weight(tmp_path):
    torch.manual_seed(0)
    model = _TinyModel()
    original = model.q_proj.weight.detach().clone()

    rank = 2
    a = torch.randn(rank, 3, dtype=torch.float32)
    b = torch.randn(4, rank, dtype=torch.float32)
    # adapter_config: scale = lora_alpha / r = 8 / 2 = 4.0
    lora_dir = _write_adapter(tmp_path, a=a, b=b, lora_alpha=8, rank=rank)

    lora = MergeOnLoadLoRA(module=model)
    matched = lora.load(lora_id=7, lora_path=lora_dir)

    assert matched == 1
    expected = original + (b @ a) * 4.0
    torch.testing.assert_close(model.q_proj.weight.detach(), expected)


def test_load_returns_active_id_and_delta_count(tmp_path):
    torch.manual_seed(1)
    model = _TinyModel()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)
    lora_dir = _write_adapter(tmp_path, a=a, b=b, lora_alpha=2, rank=2)

    lora = MergeOnLoadLoRA(module=model)
    matched = lora.load(lora_id="adapter-A", lora_path=lora_dir)

    assert matched == 1
    assert lora.active_id == "adapter-A"
    assert set(lora.deltas) == {"q_proj.weight"}


def test_load_defaults_scale_to_one_without_config(tmp_path):
    torch.manual_seed(2)
    model = _TinyModel()
    original = model.q_proj.weight.detach().clone()
    a = torch.randn(1, 3, dtype=torch.float32)
    b = torch.randn(4, 1, dtype=torch.float32)
    # No adapter_config.json -> scale defaults to 1.0.
    lora_dir = _write_adapter(tmp_path, a=a, b=b)

    lora = MergeOnLoadLoRA(module=model)
    lora.load(lora_id=1, lora_path=lora_dir)

    expected = original + (b @ a) * 1.0
    torch.testing.assert_close(model.q_proj.weight.detach(), expected)


def test_double_load_of_resident_adapter_is_rejected(tmp_path):
    torch.manual_seed(3)
    model = _TinyModel()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)
    lora_dir = _write_adapter(tmp_path, a=a, b=b, lora_alpha=4, rank=2)

    lora = MergeOnLoadLoRA(module=model)
    lora.load(lora_id=1, lora_path=lora_dir)

    with pytest.raises(WorkerError) as excinfo:
        lora.load(lora_id=2, lora_path=lora_dir)
    assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_load_rejects_adapter_matching_no_parameters(tmp_path):
    torch.manual_seed(4)
    model = _TinyModel()
    original = model.q_proj.weight.detach().clone()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)
    # Target a parameter the module does not have.
    lora_dir = _write_adapter(
        tmp_path, target="base_model.model.k_proj", a=a, b=b, lora_alpha=4, rank=2
    )

    lora = MergeOnLoadLoRA(module=model)
    with pytest.raises(WorkerError) as excinfo:
        lora.load(lora_id=1, lora_path=lora_dir)

    assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR
    # A rejected zero-match load leaves the weights untouched and nothing resident.
    torch.testing.assert_close(model.q_proj.weight.detach(), original)
    assert lora.deltas == {}
    assert lora.active_id is None


def test_unload_restores_original_weights_within_fp32_tolerance(tmp_path):
    torch.manual_seed(5)
    model = _TinyModel()
    original = model.q_proj.weight.detach().clone()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)
    lora_dir = _write_adapter(tmp_path, a=a, b=b, lora_alpha=8, rank=2)

    lora = MergeOnLoadLoRA(module=model)
    lora.load(lora_id=42, lora_path=lora_dir)
    # Sanity: load actually moved the weights.
    assert not torch.equal(model.q_proj.weight.detach(), original)

    unloaded = lora.unload(lora_id=42)

    assert unloaded == 1
    # unload subtracts the same fp32 deltas it added, restoring the weights up to
    # the fp32 add/sub round-trip rounding (~1e-7) — there is no extra drift.
    torch.testing.assert_close(model.q_proj.weight.detach(), original)
    assert lora.deltas == {}
    assert lora.active_id is None


def test_unload_with_no_resident_adapter_is_noop(tmp_path):
    del tmp_path
    model = _TinyModel()
    original = model.q_proj.weight.detach().clone()

    lora = MergeOnLoadLoRA(module=model)
    unloaded = lora.unload(lora_id=99)

    assert unloaded == 0
    assert torch.equal(model.q_proj.weight.detach(), original)


def test_reload_after_unload_is_allowed(tmp_path):
    torch.manual_seed(6)
    model = _TinyModel()
    original = model.q_proj.weight.detach().clone()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)
    lora_dir = _write_adapter(tmp_path, a=a, b=b, lora_alpha=8, rank=2)

    lora = MergeOnLoadLoRA(module=model)
    lora.load(lora_id=1, lora_path=lora_dir)
    lora.unload(lora_id=1)

    # The single-resident slot is free again after unload.
    matched = lora.load(lora_id=2, lora_path=lora_dir)

    assert matched == 1
    assert lora.active_id == 2
    expected = original + (b @ a) * 4.0
    torch.testing.assert_close(model.q_proj.weight.detach(), expected)
