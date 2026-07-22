"""Behavioral tests for system-owned adapter state.

Covers the ``AdapterStore`` contract: a single adapter is resident at a time,
merged as ``(B @ A) * scale`` into the targeted parameters, load is atomic
(a rejected adapter leaves every weight untouched), unload restores the
pre-merge weights bit-for-bit, and each committed load/unload advances a
monotonic version. The worker-level half is covered too: the
``load_lora``/``unload_lora`` controls are served by ``ModelWorker`` against
the store, not by the model.

Everything is driven through the public APIs against tiny deterministic
``nn.Module`` graphs and real on-disk safetensors adapters.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from uniserve_worker.contracts.model_protocols import UniModel
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.runtime.adapter_store import AdapterStore
from uniserve_worker.server.app import dispatch
from uniserve_worker.worker.model import ModelWorker

pytestmark = pytest.mark.unit


class _TinyGraph(torch.nn.Module):
    """Module whose single weight matches the PEFT target ``q_proj``.

    ``out_features=4`` rows, ``in_features=3`` cols, so a rank-``r`` adapter has
    ``A`` of shape ``(r, 3)`` and ``B`` of shape ``(4, r)``.
    """

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 4, bias=False, dtype=dtype)


class _TwoTargetGraph(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 4, bias=False)
        self.k_proj = torch.nn.Linear(3, 4, bias=False)


def _write_adapter(
    dir_path,
    *,
    targets: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    target: str = "base_model.model.q_proj",
    lora_alpha: float | None = None,
    rank: int | None = None,
) -> str:
    """Write an ``adapter_model.safetensors`` (+ optional config) under ``dir_path``.

    Returns the directory path that is passed to ``AdapterStore.load``.
    """
    if targets is None:
        assert a is not None and b is not None
        targets = {target: (a, b)}
    tensors = {}
    for name, (a_t, b_t) in targets.items():
        tensors[f"{name}.lora_A.weight"] = a_t.contiguous()
        tensors[f"{name}.lora_B.weight"] = b_t.contiguous()
    save_file(tensors, str(dir_path / "adapter_model.safetensors"))
    if lora_alpha is not None and rank is not None:
        (dir_path / "adapter_config.json").write_text(
            json.dumps({"r": rank, "lora_alpha": lora_alpha}),
            encoding="utf-8",
        )
    return str(dir_path)


def test_load_merges_scaled_delta_into_target_weight(tmp_path):
    torch.manual_seed(0)
    model = _TinyGraph()
    original = model.q_proj.weight.detach().clone()

    rank = 2
    a = torch.randn(rank, 3, dtype=torch.float32)
    b = torch.randn(4, rank, dtype=torch.float32)
    # adapter_config: scale = lora_alpha / r = 8 / 2 = 4.0
    adapter_dir = _write_adapter(tmp_path, a=a, b=b, lora_alpha=8, rank=rank)

    store = AdapterStore(model)
    matched = store.load(7, adapter_dir)

    assert matched == 1
    expected = original + (b @ a) * 4.0
    torch.testing.assert_close(model.q_proj.weight.detach(), expected)


def test_committed_load_and_unload_advance_version_and_active_id(tmp_path):
    torch.manual_seed(1)
    model = _TinyGraph()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)
    adapter_dir = _write_adapter(tmp_path, a=a, b=b, lora_alpha=2, rank=2)

    store = AdapterStore(model)
    assert store.version == 0
    assert store.active_id is None

    store.load("adapter-A", adapter_dir)
    assert store.active_id == "adapter-A"
    assert store.version == 1

    store.unload("adapter-A")
    assert store.active_id is None
    assert store.version == 2

    store.load("adapter-B", adapter_dir)
    assert store.active_id == "adapter-B"
    assert store.version == 3


def test_load_defaults_scale_to_one_without_config(tmp_path):
    torch.manual_seed(2)
    model = _TinyGraph()
    original = model.q_proj.weight.detach().clone()
    a = torch.randn(1, 3, dtype=torch.float32)
    b = torch.randn(4, 1, dtype=torch.float32)
    # No adapter_config.json -> scale defaults to 1.0.
    adapter_dir = _write_adapter(tmp_path, a=a, b=b)

    store = AdapterStore(model)
    store.load(1, adapter_dir)

    expected = original + (b @ a) * 1.0
    torch.testing.assert_close(model.q_proj.weight.detach(), expected)


def test_double_load_of_resident_adapter_is_rejected(tmp_path):
    torch.manual_seed(3)
    model = _TinyGraph()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)
    adapter_dir = _write_adapter(tmp_path, a=a, b=b, lora_alpha=4, rank=2)

    store = AdapterStore(model)
    store.load(1, adapter_dir)

    with pytest.raises(WorkerError) as excinfo:
        store.load(2, adapter_dir)
    assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR
    # The rejected load is not a committed transition.
    assert store.active_id == 1
    assert store.version == 1


def test_load_rejects_adapter_matching_no_parameters(tmp_path):
    torch.manual_seed(4)
    model = _TinyGraph()
    original = model.q_proj.weight.detach().clone()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)
    # Target a parameter the module does not have.
    adapter_dir = _write_adapter(
        tmp_path, target="base_model.model.k_proj", a=a, b=b, lora_alpha=4, rank=2
    )

    store = AdapterStore(model)
    with pytest.raises(WorkerError) as excinfo:
        store.load(1, adapter_dir)

    assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR
    # A rejected zero-match load leaves the weights untouched and nothing resident.
    assert torch.equal(model.q_proj.weight.detach(), original)
    assert store.active_id is None
    assert store.version == 0


def test_rejected_load_leaves_every_target_untouched(tmp_path):
    """Load is atomic: one bad target aborts the merge before any weight moves."""
    torch.manual_seed(5)
    model = _TwoTargetGraph()
    q_original = model.q_proj.weight.detach().clone()
    k_original = model.k_proj.weight.detach().clone()
    good_a = torch.randn(2, 3, dtype=torch.float32)
    good_b = torch.randn(4, 2, dtype=torch.float32)
    # k_proj's B has the wrong out_features, so its delta cannot match the weight.
    bad_a = torch.randn(2, 3, dtype=torch.float32)
    bad_b = torch.randn(5, 2, dtype=torch.float32)
    adapter_dir = _write_adapter(
        tmp_path,
        targets={
            "base_model.model.q_proj": (good_a, good_b),
            "base_model.model.k_proj": (bad_a, bad_b),
        },
        lora_alpha=2,
        rank=2,
    )

    store = AdapterStore(model)
    with pytest.raises(WorkerError) as excinfo:
        store.load(1, adapter_dir)

    assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert torch.equal(model.q_proj.weight.detach(), q_original)
    assert torch.equal(model.k_proj.weight.detach(), k_original)
    assert store.active_id is None
    assert store.version == 0


def test_unload_restores_original_weights_bit_exact(tmp_path):
    # bfloat16 weights would drift under merge-then-subtract arithmetic; the
    # store restores the saved pre-merge copies, so equality is exact.
    torch.manual_seed(6)
    model = _TinyGraph(dtype=torch.bfloat16)
    original = model.q_proj.weight.detach().clone()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)
    adapter_dir = _write_adapter(tmp_path, a=a, b=b, lora_alpha=8, rank=2)

    store = AdapterStore(model)
    store.load(42, adapter_dir)
    # Sanity: load actually moved the weights.
    assert not torch.equal(model.q_proj.weight.detach(), original)

    unloaded = store.unload(42)

    assert unloaded == 1
    assert torch.equal(model.q_proj.weight.detach(), original)


def test_unload_with_no_resident_adapter_is_noop():
    model = _TinyGraph()
    original = model.q_proj.weight.detach().clone()

    store = AdapterStore(model)
    unloaded = store.unload(99)

    assert unloaded == 0
    assert torch.equal(model.q_proj.weight.detach(), original)
    assert store.version == 0


class _AdapterControlModel(UniModel):
    """Minimal adapter-capable model: the module graph is all it brings."""

    supported_ops = ("prefill_und", "decode_und")
    supported_controls = ("load_lora", "unload_lora")
    adapter_mode = "engine_wide"
    num_layers = 1
    bytes_per_token = 1

    def __init__(self, module: torch.nn.Module | None) -> None:
        self.model = module

    def forward(self, batch):  # pragma: no cover - control tests never execute
        raise AssertionError("unreachable")


def test_worker_serves_adapter_controls_against_system_store(tmp_path):
    torch.manual_seed(7)
    module = _TinyGraph()
    original = module.q_proj.weight.detach().clone()
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)
    adapter_dir = _write_adapter(tmp_path, a=a, b=b, lora_alpha=8, rank=2)

    worker = ModelWorker(_AdapterControlModel(module), block_size=256, simulation=True)
    supported = set(worker.caps().supported_controls)
    assert {"load_lora", "unload_lora"} <= supported

    resp = dispatch(
        worker, supported, {"kind": "load_lora", "lora_id": 7, "lora_path": adapter_dir}
    )
    assert resp == {"kind": "ok"}
    expected = original + (b @ a) * 4.0
    torch.testing.assert_close(module.q_proj.weight.detach(), expected)
    assert worker.adapter_store is not None
    assert worker.adapter_store.active_id == 7

    resp = dispatch(worker, supported, {"kind": "unload_lora", "lora_id": 7})
    assert resp == {"kind": "ok"}
    assert torch.equal(module.q_proj.weight.detach(), original)
    assert worker.adapter_store.active_id is None
    assert worker.adapter_store.version == 2


def test_worker_without_loaded_weights_rejects_adapter_controls():
    worker = ModelWorker(_AdapterControlModel(None), block_size=256, simulation=True)

    with pytest.raises(WorkerError) as excinfo:
        worker.load_lora(1, "/adapters/a")
    assert excinfo.value.code == ErrorCode.CAPABILITY_MISMATCH


def test_worker_rejects_adapter_controls_without_adapter_mode():
    class _NoAdapterModeModel(_AdapterControlModel):
        adapter_mode = "none"

    with pytest.raises(WorkerError) as excinfo:
        ModelWorker(_NoAdapterModeModel(_TinyGraph()), block_size=256, simulation=True)
    assert excinfo.value.code == ErrorCode.CAPABILITY_MISMATCH
