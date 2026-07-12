"""Core contract conformance for caps and worker results."""
from __future__ import annotations

import pytest
import torch

from uniserve_worker.contracts.caps import validate_caps, validate_forward_result
from uniserve_worker.contracts.forward_mode import ForwardMode, mode_for_op
from uniserve_worker.contracts.model_protocols import ModelHooks
from uniserve_worker.contracts.op_kinds import OP_KIND_TABLE, OP_KINDS
from uniserve_worker.contracts.resource_plan import ResourcePlan
from uniserve_worker.execution.runner import ModelRunner, RunnerConfig
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import PagedTextCache
from uniserve_worker.server.stub import StubEngine, StubUniModel

pytestmark = pytest.mark.contract


def test_op_kind_table_is_single_source_of_truth():
    assert OP_KINDS == set(OP_KIND_TABLE)
    for wire, spec in OP_KIND_TABLE.items():
        assert spec.wire == wire
        assert spec.mode in ForwardMode._value2member_map_
        assert callable(spec.validate)
        assert mode_for_op(wire) is ForwardMode(spec.mode)


def test_validate_caps_accepts_stub_caps():
    caps = validate_caps(StubEngine().caps(), owner="StubEngine")
    assert caps.block_size == 64
    assert "prefill_und" in caps.supported_ops
    assert "kv_block" in caps.resource_classes
    assert caps.to_wire()["adapter_mode"] == "none"


def test_runner_preserves_contextual_auto_attention_backend():
    runner = ModelRunner(
        StubUniModel(),
        config=RunnerConfig(simulation=True),
        attention_backend=None,
    )
    assert runner.attention_backend is None
    assert runner.attention_backend_name == "auto"


def test_validate_caps_accepts_qwen3_target_verify_op():
    raw = StubEngine().caps().to_wire()
    raw["supported_ops"] = ["prefill_und", "decode_und", "target_verify_und"]
    caps = validate_caps(raw, owner="Qwen3ForCausalLM")
    assert caps.supported_ops == ("prefill_und", "decode_und", "target_verify_und")


def test_validate_caps_rejects_drift_from_class_vocabulary():
    raw = StubEngine().caps().to_wire()
    raw["supported_ops"] = [*raw["supported_ops"], "not_real"]
    with pytest.raises(WorkerError) as exc:
        validate_caps(raw, owner="BadDriver")
    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_validate_forward_result_accepts_all_stub_op_shapes():
    batch = {
        "step_id": 9,
        "ops": [
            {"req_id": 1, "kind": "prefill_und"},
            {"req_id": 1, "kind": "target_verify_und"},
            {"req_id": 1, "kind": "denoise_gen"},
            {"req_id": 1, "kind": "commit_gen"},
            {"req_id": 1, "kind": "vit_encode"},
        ],
    }
    result = {
        "step_id": 9,
        "per_seq": [
            {"req_id": 1, "sampled_token_id": 123},
            {"req_id": 1, "sampled_token_id": 124, "sampled_logprob": -0.1},
            {"req_id": 1, "denoise_done": False, "num_steps_done": 1},
            {
                "req_id": 1,
                "image_png_b64": "iVBORw0KGgo=",
                "image_hw": [512, 512],
                "sampled_token_id": 456,
            },
            {"req_id": 1, "encoder_handle": 99},
        ],
    }
    assert validate_forward_result(result, batch, owner="StubEngine") is result


def test_validate_forward_result_rejects_result_count_mismatch():
    batch = {"step_id": 1, "ops": [{"req_id": 7, "kind": "decode_und"}]}
    result = {"step_id": 1, "per_seq": []}
    with pytest.raises(WorkerError) as exc:
        validate_forward_result(result, batch, owner="BadDriver")
    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_validate_forward_result_rejects_wrong_request_id():
    batch = {"step_id": 1, "ops": [{"req_id": 7, "kind": "decode_und"}]}
    result = {"step_id": 1, "per_seq": [{"req_id": 8, "sampled_token_id": 10}]}
    with pytest.raises(WorkerError) as exc:
        validate_forward_result(result, batch, owner="BadDriver")
    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_paged_transformers_cache_updates_all_layers_at_one_slot():
    pool = PagedKVPool(
        num_layers=2,
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=3,
        device="cpu",
        dtype=torch.float32,
    )
    cache = PagedTextCache(pool, [0, 1], num_layers=2)
    k0 = torch.arange(1 * 2 * 3 * 3, dtype=torch.float32).view(1, 2, 3, 3)
    v0 = k0 + 100
    out_k0, out_v0 = cache.update(k0, v0, 0)
    assert cache.get_seq_length() == 0
    assert out_k0.shape == (1, 2, 3, 3)
    assert torch.equal(out_v0, v0)

    k1 = k0 + 1000
    v1 = v0 + 1000
    out_k1, _ = cache.update(k1, v1, 1)
    assert cache.get_seq_length() == 3
    assert torch.equal(out_k1, k1)

    next_k = torch.full((1, 2, 1, 3), 7.0)
    next_v = torch.full((1, 2, 1, 3), 9.0)
    cat_k, _ = cache.update(next_k, next_v, 0)
    cache.update(next_k + 1, next_v + 1, 1)
    assert cache.get_seq_length() == 4
    assert cat_k.shape == (1, 2, 4, 3)
    assert torch.equal(cat_k[:, :, :3], k0)
    assert torch.equal(cat_k[:, :, 3:], next_k)


def test_paged_transformers_cache_can_grow_with_allocator():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=2,
        block_size=2,
        num_kv_heads=1,
        head_dim=1,
        device="cpu",
        dtype=torch.float32,
    )
    remaining = [0, 1]
    cache = PagedTextCache(
        pool,
        [],
        num_layers=1,
        allocate_blocks=lambda n: [remaining.pop(0) for _ in range(n)],
    )
    k = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1)
    v = k + 10
    out_k, out_v = cache.update(k, v, 0)
    assert cache.block_ids == [0, 1]
    assert cache.get_seq_length() == 3
    assert torch.equal(out_k, k)
    assert torch.equal(out_v, v)


def test_paged_kv_pool_fp8_storage_dequantizes_dense_reads():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=3,
        block_size=2,
        num_kv_heads=1,
        head_dim=3,
        device="cpu",
        dtype=torch.float32,
        store_dtype="fp8_e4m3",
    )
    assert pool.k.dtype == torch.float8_e4m3fn
    assert pool.dtype == torch.float32
    assert not pool.supports_paged_attention_storage

    block_ids = [0, 1]
    k = torch.tensor(
        [
            [[0.0, 0.25, -0.5]],
            [[0.75, -1.0, 1.25]],
            [[-1.5, 1.75, 2.0]],
        ],
        dtype=torch.float32,
    )
    v = k + 0.125
    pool.write(0, block_ids, start=0, k=k, v=v)
    got_k, got_v = pool.read(0, block_ids, start=0, length=3)

    assert got_k is not None and got_v is not None
    assert got_k.dtype == torch.float32
    torch.testing.assert_close(got_k, k, atol=0.08, rtol=0.08)
    torch.testing.assert_close(got_v, v, atol=0.08, rtol=0.08)

    new_k = torch.tensor([[[0.5, -0.625, 0.875]]], dtype=torch.float32)
    new_v = new_k - 0.25
    pool.write(0, block_ids, start=1, k=new_k, v=new_v)
    expected_k = k.clone()
    expected_v = v.clone()
    expected_k[1:2] = new_k
    expected_v[1:2] = new_v
    got_k, got_v = pool.read(0, block_ids, start=0, length=3)
    torch.testing.assert_close(got_k, expected_k, atol=0.08, rtol=0.08)
    torch.testing.assert_close(got_v, expected_v, atol=0.08, rtol=0.08)

    with pytest.raises(WorkerError, match="scale-aware paged attention backend") as exc:
        pool.layer_cache(0)
    assert exc.value.code == ErrorCode.CAPABILITY_MISMATCH


class CommitCapabilityModel(ModelHooks):
    resource_classes: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.inference_modes: list[bool] = []

    def decode_image(self, latent, *, req_id, state, op):
        del latent, op
        self.calls.append(req_id)
        self.inference_modes.append(torch.is_inference_mode_enabled())
        state.latent = "finished"
        state.schedule_cursor = 7
        return {"req_id": req_id, "image_hw": [8, 8]}

    def forward(self, batch):  # pragma: no cover - commit must not route here
        raise AssertionError("commit should be owned by ImageDecodeDriver")


def test_runner_commit_uses_decode_image_capability_and_resets_state():
    model = CommitCapabilityModel()
    runner = ModelRunner(model, config=RunnerConfig(simulation=True))

    result = runner.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 5, "block_ids": []}],
            "ops": [{"req_id": 5, "kind": "commit_gen"}],
        }
    )

    assert result["per_seq"] == [{"req_id": 5, "image_hw": [8, 8]}]
    assert model.calls == [5]
    assert model.inference_modes == [True]
    state = runner.request_states.get(5)
    assert state.latent is None
    assert state.schedule_cursor == 0


class CommitLogitsCapabilityModel(ModelHooks):
    resource_classes: tuple[str, ...] = ()

    def decode_image(self, latent, *, req_id, state, op):
        del latent, state, op
        return {
            "req_id": req_id,
            "image_hw": [4, 4],
            "logits": torch.tensor([0.0, 2.0, 1.0], dtype=torch.float32),
        }

    def forward(self, batch):  # pragma: no cover - commit must not route here
        raise AssertionError("commit should be owned by ImageDecodeDriver")


def test_runner_commit_samples_logits_in_image_decode_driver():
    runner = ModelRunner(
        CommitLogitsCapabilityModel(),
        config=RunnerConfig(simulation=True),
    )
    result = runner.execute(
        {
            "step_id": 2,
            "new_reqs": [{"req_id": 6, "sampling": {"temperature": 0.0, "n_logprobs": 1}}],
            "ops": [{"req_id": 6, "kind": "commit_gen", "suppress_tokens": [1]}],
        }
    )

    seq = result["per_seq"][0]
    assert seq["req_id"] == 6
    assert seq["image_hw"] == [4, 4]
    assert seq["sampled_token_id"] == 2
    assert "sampled_logprob" in seq
    assert "logits" not in seq


class TextCapabilityModel(ModelHooks):
    """HF day-zero fallback shape: no system pool, per-op ``run_text_logits_batch``.

    The system :class:`TextDriver` routes a model without a ``kv_cache_spec`` /
    KV pool through its HF fallback branch and owns the post-model sampler, so
    this fixture exercises the driver's sampling (masks/logprobs) over raw
    model logits.
    """

    resource_classes: tuple[str, ...] = ()
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.inference_modes: list[bool] = []

    def run_text_logits_batch(self, ops):
        out = []
        for op in ops:
            self.calls.append(list(op.get("token_ids") or []))
            self.inference_modes.append(torch.is_inference_mode_enabled())
            out.append(torch.tensor([0.0, 4.0, 1.0, 2.0]))
        return out


class ThinTextModel(ModelHooks):
    """Thin system-managed text model: declares KV geometry, takes a ForwardBatch.

    The system (``ForwardBatchBuilder`` + ``ResidencyManager``) owns the pool and
    builds the per-forward attention plan; this fixture records the plan it sees
    on the published context to prove the model builds none of it.
    """

    resource_classes: tuple[str, ...] = ("kv_block",)
    resource_plan = ResourcePlan(kv_block="per_block")
    supported_ops = ("prefill_und", "decode_und")
    supported_controls: tuple[str, ...] = ()
    adapter_mode = "none"
    device = "cpu"
    head_dim = 4
    num_layers = 1
    block_size = 16
    num_blocks = 8

    def __init__(self) -> None:
        self.context_metadata = None
        self.seen_req_ids: tuple[int, ...] | None = None

    def kv_cache_spec(self):
        from uniserve_worker.runtime.residency import KvCacheSpec

        return KvCacheSpec(num_layers=1, num_kv_heads=1, head_dim=4, dtype=torch.float32)

    def forward(self, input_ids, positions, forward_batch):
        from uniserve_worker.contracts.forward_context import get_forward_context

        del input_ids, positions
        self.context_metadata = get_forward_context().attention_metadata
        self.seen_req_ids = forward_batch.req_ids
        batch = forward_batch.batch_size
        logits = torch.zeros(batch, 3)
        logits[:, 1] = 5.0
        return logits


def _thin_text_runner(model):
    from uniserve_worker.runtime.residency import ResidencyManager
    from uniserve_worker.runtime.resources import ResourceRuntime

    ledger = ResourceRuntime(("kv_block",), totals={"kv_block": model.num_blocks})
    residency = ResidencyManager.build(
        model.kv_cache_spec(),
        num_blocks=model.num_blocks,
        block_size=model.block_size,
        device="cpu",
        ledger=ledger,
    )
    return ModelRunner(
        model,
        config=RunnerConfig(simulation=True),
        resource_runtime=ledger,
        residency=residency,
    )


def test_runner_text_uses_text_driver_for_sampling_masks_and_logprobs():
    model = TextCapabilityModel()
    runner = ModelRunner(model, config=RunnerConfig(simulation=True))

    result = runner.execute(
        {
            "step_id": 3,
            "new_reqs": [
                {
                    "req_id": 8,
                    "sampling": {
                        "temperature": 0.0,
                        "top_k": 0,
                        "top_p": 1.0,
                        "logit_bias": [[3, 5.0]],
                        "n_logprobs": 2,
                    },
                }
            ],
            "ops": [
                {
                    "req_id": 8,
                    "kind": "decode_und",
                    "token_ids": [11],
                    "pos_range": [0, 1],
                    "suppress_tokens": [1],
                }
            ],
        }
    )

    seq = result["per_seq"][0]
    assert seq["sampled_token_id"] == 3
    assert seq["top_logprobs"][0][0] == 3
    assert model.calls == [[11]]
    assert model.inference_modes == [True]


def test_system_builds_and_publishes_attention_plan_to_thin_model():
    from uniserve_worker.contracts.forward_context import TextAttentionMetadata

    model = ThinTextModel()
    runner = _thin_text_runner(model)

    result = runner.execute(
        {
            "step_id": 33,
            "new_reqs": [{"req_id": 8, "sampling": {"temperature": 0.0}, "block_ids": [0]}],
            "ops": [
                {
                    "req_id": 8,
                    "kind": "decode_und",
                    "token_ids": [11],
                    "pos_range": [0, 1],
                }
            ],
        }
    )

    assert result["per_seq"][0]["sampled_token_id"] == 1
    assert model.seen_req_ids == (8,)
    # The model built no metadata: the system constructed and published the plan.
    assert isinstance(model.context_metadata, TextAttentionMetadata)
    assert model.context_metadata.cache is not None


def test_runner_request_state_keeps_op_block_deltas_authoritative():
    from uniserve_worker.runtime.resources import ResourceRuntime

    model = TextCapabilityModel()
    model.resource_classes = ("kv_block",)
    runner = ModelRunner(
        model,
        config=RunnerConfig(simulation=True),
        resource_runtime=ResourceRuntime(("kv_block",), totals={"kv_block": 4}),
    )

    runner.execute(
        {
            "step_id": 4,
            "new_reqs": [{"req_id": 12, "block_ids": [1]}],
            "ops": [
                {
                    "req_id": 12,
                    "kind": "decode_und",
                    "token_ids": [5],
                    "pos_range": [0, 1],
                    "new_block_ids": [2, 3],
                }
            ],
        }
    )

    state = runner.request_states.get(12)
    assert state.block_ids == [1, 2, 3]
    assert state.resident_block_ids == {1, 2, 3}
