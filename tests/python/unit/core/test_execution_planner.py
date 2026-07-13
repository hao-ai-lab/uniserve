"""Runner-owned execution grouping (plan §6.7/§10.8, DoD 8).

The previous driver-side planner is gone: grouping is a `ModelRunner` responsibility
derived from each model's `BatchPolicy`, and every model still receives typed
single-mode `UniForwardBatch` values.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from uniserve_worker.contracts.batch_policy import BatchPolicy
from uniserve_worker.contracts.batches import CfgBatch, UniForwardBatch
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.contracts.model_protocols import ModelHooks
from uniserve_worker.contracts.resource_plan import ResourcePlan
from uniserve_worker.execution.runner import ModelRunner, RunnerConfig
from uniserve_worker.execution.text_driver import text_input_id_replacements_from_relays
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.nn.attention import RadixAttention
from uniserve_worker.runtime.request_state import RequestStateTable
from uniserve_worker.runtime.tensor_staging import TextTensorStager, stage_text_forward_batch

pytestmark = pytest.mark.unit


class RecordingModel(ModelHooks):
    resource_classes: tuple[str, ...] = ()
    whole_batch_forward = True

    def __init__(self, policy: BatchPolicy) -> None:
        self._policy = policy
        self.calls: list[tuple[ForwardMode, list[int], list[str]]] = []
        self.inference_modes: list[bool] = []

    def batch_policy(self) -> BatchPolicy:
        return self._policy

    def forward(self, batch: UniForwardBatch) -> list[dict[str, Any]]:
        self.inference_modes.append(torch.is_inference_mode_enabled())
        kinds = [str(op["kind"]) for op in batch.ops]
        req_ids = [int(op["req_id"]) for op in batch.ops]
        self.calls.append((batch.mode, req_ids, kinds))
        return [
            {"req_id": int(op["req_id"]), "kind": str(op["kind"]), "mode": batch.mode.value}
            for op in batch.ops
        ]


class RecordingEncodeCapabilityModel(RecordingModel):
    def __init__(self) -> None:
        super().__init__(BatchPolicy(max_batch_ops=4, supports_mixed_modes=False))
        self.encode_ops: list[str] = []

    def encode_image(self, pixels=None, grid=None, *, op: dict[str, Any]) -> dict[str, Any]:
        del pixels, grid
        self.encode_ops.append(str(op["kind"]))
        return {
            "req_id": int(op["req_id"]),
            "encoder_handle": 1000 + int(op["req_id"]),
            "num_tokens": 3,
        }

    def encode_latents(self, pixels=None, grid=None, *, op: dict[str, Any]) -> dict[str, Any]:
        return self.encode_image(pixels, grid, op=op)


class ThinCPUTextModel(ModelHooks):
    """Thin system-managed text model on CPU (single-token-per-op echo logits).

    Declares its KV geometry so the runtime owns the pool; the system text path
    runs each op through ``forward(input_ids, positions, forward_batch)`` (the
    gate forces per-op dense on CPU) and the driver owns sampling, relays, and
    the system spec-verify. The model records the input tokens it saw so relay
    reuse can be observed.
    """

    resource_classes: tuple[str, ...] = ("kv_block",)
    resource_plan = ResourcePlan(kv_block="per_block")
    supported_ops = ("prefill_und", "decode_und", "target_verify_und")
    supported_controls: tuple[str, ...] = ()
    adapter_mode = "none"
    device = "cpu"
    head_dim = 4
    num_layers = 1
    block_size = 16
    num_blocks = 8

    def __init__(self) -> None:
        self.input_values: list[list[int]] = []
        self.input_ptrs: list[int] = []

    def kv_cache_spec(self):
        from uniserve_worker.runtime.residency import KvCacheSpec

        return KvCacheSpec(num_layers=1, num_kv_heads=1, head_dim=4, dtype=torch.float32)

    def batch_policy(self) -> BatchPolicy:
        return BatchPolicy(max_batch_ops=8, supports_mixed_modes=True)

    def forward(self, input_ids, positions, forward_batch):
        del positions
        flat = input_ids.reshape(-1)
        self.input_values.append([int(token) for token in flat.tolist()])
        self.input_ptrs.append(int(flat.data_ptr()))
        if forward_batch.forward_mode == ForwardMode.TARGET_VERIFY:
            # Per-position logits [batch, length, vocab] with a fixed argmax (7).
            batch, length = int(input_ids.shape[0]), int(input_ids.shape[1])
            logits = torch.full((batch, length, 16), -1000.0)
            logits[:, :, 7] = 1000.0
            return logits
        rows = forward_batch.batch_size
        logits = torch.full((rows, 16), -1000.0)
        logits[:, 7] = 1000.0
        return logits


def _cpu_pool_runner(model):
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


def ops(*kinds: str) -> list[dict[str, Any]]:
    return [{"req_id": i + 1, "kind": kind, "token_ids": [101 + i], "pos_range": [i, i + 1]}
            for i, kind in enumerate(kinds)]


def execute(model: RecordingModel, submitted_ops: list[dict[str, Any]]) -> dict[str, Any]:
    req_ids = sorted({int(op["req_id"]) for op in submitted_ops})
    return ModelRunner(model, config=RunnerConfig(simulation=True)).execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": req_id, "block_ids": []} for req_id in req_ids],
            "ops": submitted_ops,
        }
    )


def test_strict_policy_preserves_contiguous_order_and_splits_by_max_batch():
    model = RecordingModel(BatchPolicy(max_batch_ops=2, supports_mixed_modes=False))
    submitted = ops("prefill_und", "prefill_und", "prefill_und", "decode_und", "prefill_und")

    result = execute(model, submitted)

    assert [r["kind"] for r in result["per_seq"]] == [op["kind"] for op in submitted]
    assert model.calls == [
        (ForwardMode.EXTEND, [1, 2], ["prefill_und", "prefill_und"]),
        (ForwardMode.EXTEND, [3], ["prefill_und"]),
        (ForwardMode.DECODE, [4], ["decode_und"]),
        (ForwardMode.EXTEND, [5], ["prefill_und"]),
    ]


def test_mixed_admission_fails_when_the_model_adapter_cannot_execute_the_batch():
    from uniserve_worker.execution.forward.step import ForwardGroupPlanner

    planner = ForwardGroupPlanner(
        BatchPolicy(max_batch_ops=8, supports_mixed_modes=True),
        log_text_mixed_split=lambda _ops, _decision: None,
        can_run_forward=lambda batch: batch.mode is not ForwardMode.MIXED,
    )
    submitted = [
        {"req_id": 1, "kind": "prefill_und", "token_ids": [1]},
        {"req_id": 2, "kind": "denoise_gen", "latent_shape": [2, 2]},
        {"req_id": 3, "kind": "prefill_und", "token_ids": [2]},
        {"req_id": 4, "kind": "denoise_gen", "latent_shape": [2, 2]},
    ]

    with pytest.raises(WorkerError, match="admitted mixed forward has no whole-batch executor"):
        planner.groups(submitted)


def test_mixed_planner_peels_encode_rows_and_keeps_text_denoise_packed():
    from uniserve_worker.execution.forward.step import ForwardGroupPlanner

    planner = ForwardGroupPlanner(
        BatchPolicy(max_batch_ops=8, supports_mixed_modes=True),
        log_text_mixed_split=lambda _ops, _decision: None,
        can_run_forward=lambda _batch: True,
    )
    submitted = [
        {"req_id": 1, "kind": "decode_und", "token_ids": [1]},
        {"req_id": 2, "kind": "vit_encode"},
        {"req_id": 3, "kind": "denoise_gen", "latent_shape": [2, 2]},
        {"req_id": 4, "kind": "prefill_und", "token_ids": [2]},
    ]

    groups = planner.groups(submitted)

    assert [[index for index, _op in group] for group in groups] == [[1], [0, 2, 3]]


def test_runner_executes_whole_batch_forward_under_inference_mode():
    model = RecordingModel(BatchPolicy(max_batch_ops=8, supports_mixed_modes=False))

    execute(model, ops("prefill_und"))

    assert model.inference_modes == [True]


def test_und_gen_mixed_policy_keeps_entire_batch_for_whole_batch_forward():
    model = ForwardHookModel(max_batch_ops=2)
    submitted = ops("decode_und", "commit_gen", "denoise_gen", "prefill_und", "decode_und")

    result = execute(model, submitted)

    assert [r["req_id"] for r in result["per_seq"]] == [1, 2, 3, 4, 5]
    assert [r["kind"] for r in result["per_seq"]] == [op["kind"] for op in submitted]
    assert model.calls == [
        (
            ForwardMode.MIXED,
            [1, 2, 3, 4, 5],
            ["decode_und", "commit_gen", "denoise_gen", "prefill_und", "decode_und"],
        )
    ]
    assert model.forward_calls == []


def test_non_thin_text_extend_decode_runs_as_unified_mixed_batch():
    # Mixed extend+decode grouping is a system-level decision; a whole-batch model receives the admitted mixed batch directly.
    model = RecordingModel(BatchPolicy(max_batch_ops=8, supports_mixed_modes=True))
    submitted = ops("decode_und", "prefill_und")

    result = execute(model, submitted)

    assert [r["mode"] for r in result["per_seq"]] == ["mixed", "mixed"]
    assert model.calls == [
        (ForwardMode.MIXED, [1, 2], ["decode_und", "prefill_und"]),
    ]


def test_text_extend_decode_route_prefers_whole_batch_forward_over_legacy_hook():
    model = ForwardHookModel(max_batch_ops=8)
    submitted = [
        {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [4, 5]},
        {"req_id": 2, "kind": "prefill_und", "token_ids": [11, 12], "pos_range": [0, 2]},
    ]

    result = ModelRunner(model, config=RunnerConfig(simulation=True)).execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 1, "block_ids": []}, {"req_id": 2, "block_ids": []}],
            "ops": submitted,
        }
    )

    assert [row["mode"] for row in result["per_seq"]] == ["mixed", "mixed"]
    assert model.calls == [
        (ForwardMode.MIXED, [1, 2], ["decode_und", "prefill_und"]),
    ]
    assert model.forward_calls == []


def test_runner_registers_text_graph_programs_with_forward_executor(monkeypatch):
    class Model(ModelHooks):
        device = "cpu"

    graph_runner = object()

    def fake_text_stack(self, model, residency):
        del self, model, residency
        return SimpleNamespace(builder=object(), gate=object(), graph_runner=graph_runner)

    monkeypatch.setattr(ModelRunner, "_build_text_execution", fake_text_stack)

    runner = ModelRunner(Model())

    assert runner.forward_graph_policy.graph_selection_delegated is False
    assert runner.forward_graph_policy.strict is True
    assert runner.text_graph_runner is graph_runner
    assert runner.text_driver.graph_runner is None
    assert runner.forward_executor.graph_runner is runner.forward_graph_runner
    assert runner.forward_graph_runner is not None
    assert [program.program_id for program in runner.forward_graph_runner.programs] == [
        "decode",
        "prefill",
        "packed_visible",
        "denoise_step",
    ]
    assert all(
        program.text_driver is runner.text_driver
        for program in runner.forward_graph_runner.programs
        if program.program_id in {"decode", "prefill"}
    )
    assert all(
        program.text_graph_runner is graph_runner
        for program in runner.forward_graph_runner.programs
        if program.program_id in {"decode", "prefill"}
    )


def test_mixed_text_build_replaces_last_sampled_placeholder_from_relay():
    states = RequestStateTable()
    state = states.get(1)
    state.decode_relay.token_tensor = torch.tensor([7], dtype=torch.long)
    batch = UniForwardBatch.from_ops(
        [
            {
                "req_id": 1,
                "kind": "decode_und",
                "token_ids": [0],
                "token_source": "last_sampled",
                "pos_range": [3, 4],
            },
            {"req_id": 2, "kind": "prefill_und", "token_ids": [11, 12], "pos_range": [0, 2]},
        ]
    )
    text = batch.as_text(allow_mixed_text=True)

    replacements = text_input_id_replacements_from_relays(text, states, torch.device("cpu"))
    flat = stage_text_forward_batch(
        text,
        "cpu",
        input_ids_replacements=replacements,
    )

    assert flat.mode == ForwardMode.MIXED
    assert flat.input_ids.tolist() == [7, 11, 12]
    assert flat.positions.tolist() == [3, 0, 1]


def test_forward_admission_keeps_extra_rows_with_und_gen_mixed_batch():
    model = ForwardHookModel(max_batch_ops=8)
    submitted = ops("decode_und", "denoise_gen", "commit_gen")

    result = execute(model, submitted)

    assert [r["mode"] for r in result["per_seq"]] == ["mixed", "mixed", "mixed"]
    assert model.calls == [
        (ForwardMode.MIXED, [1, 2, 3], ["decode_und", "denoise_gen", "commit_gen"]),
    ]
    assert model.forward_calls == []


def test_system_speculative_verify_runs_over_thin_model_forward():
    # Spec verification is system-owned now: the driver routes spec rows to the
    # system spec-verify, which builds the target_verify forward and applies the
    # accept rule over the thin model's per-position logits.
    model = ThinCPUTextModel()
    runner = _cpu_pool_runner(model)

    result = runner.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 1, "block_ids": [0], "sampling": {"temperature": 0.0}}],
            "ops": [
                {
                    "req_id": 1,
                    "kind": "decode_und",
                    "token_ids": [101],
                    "spec_token_ids": [7, 7],
                    "pos_range": [5, 6],
                }
            ],
        }
    )

    # The model's verify logits fix the argmax at 7 for every position, so both
    # draft 7s are accepted and the sampled continuation is 7.
    assert result["per_seq"] == [
        {"req_id": 1, "sampled_token_id": 7, "num_accepted_tokens": 2}
    ]
    assert runner.request_states.get(1).decode_relay.token_id == 7


def test_thin_text_runs_per_op_through_system_forward_and_samples():
    # On CPU the gate forces per-op dense; each op runs through the same thin
    # forward against a single-request system cache and the driver samples.
    model = ThinCPUTextModel()
    runner = _cpu_pool_runner(model)

    result = runner.execute(
        {
            "step_id": 1,
            "new_reqs": [
                {"req_id": 1, "block_ids": [0], "sampling": {"temperature": 0.0}},
                {"req_id": 2, "block_ids": [1], "sampling": {"temperature": 0.0}},
            ],
            "ops": [
                {"req_id": 1, "kind": "decode_und", "token_ids": [5], "pos_range": [0, 1]},
                {"req_id": 2, "kind": "decode_und", "token_ids": [6], "pos_range": [0, 1]},
            ],
        }
    )

    assert [r["sampled_token_id"] for r in result["per_seq"]] == [7, 7]
    assert model.input_values == [[5], [6]]
    # KV-length advance is system-owned (lane "text").
    assert runner.request_states.get(1).kv_lengths["text"] == 1


def test_per_op_decode_consumes_last_sampled_relay_token(monkeypatch):
    monkeypatch.setenv("UNISERVE_FORWARD_METRICS", "1")
    model = ThinCPUTextModel()
    runner = _cpu_pool_runner(model)

    runner.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 1, "block_ids": [0], "sampling": {"temperature": 0.0}}],
            "ops": [
                {"req_id": 1, "kind": "decode_und", "token_ids": [5], "pos_range": [0, 1]}
            ],
        }
    )
    # The sampled token (7) was kept on device; the next step reads it via the
    # relay rather than the wire placeholder (0).
    runner.execute(
        {
            "step_id": 2,
            "new_reqs": [],
            "ops": [
                {
                    "req_id": 1,
                    "kind": "decode_und",
                    "token_ids": [0],
                    "token_source": "last_sampled",
                    "pos_range": [1, 2],
                }
            ],
        }
    )

    assert model.input_values == [[5], [7]]


def test_per_op_last_sampled_token_source_requires_relay():
    model = ThinCPUTextModel()
    runner = _cpu_pool_runner(model)

    with pytest.raises(WorkerError) as exc:
        runner.execute(
            {
                "step_id": 1,
                "new_reqs": [{"req_id": 1, "block_ids": [0]}],
                "ops": [
                    {
                        "req_id": 1,
                        "kind": "decode_und",
                        "token_ids": [0],
                        "token_source": "last_sampled",
                        "pos_range": [1, 2],
                    }
                ],
            }
        )

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert "last_sampled" in exc.value.message


def test_denoise_batch_constructs_cfg_batch_view():
    batch = UniForwardBatch.from_ops(
        [
            {
                "req_id": 3,
                "kind": "denoise_gen",
                "timestep_idx": 2,
                "cfg": {
                    "branch_count": 3,
                    "seg_offsets": [0, 8, 16],
                    "branch_positions": [[0, 1], [0, 1], [0, 1]],
                    "branch_kv_offsets": [0, 16, 32],
                    "branch_kv_lens": [16, 16, 16],
                },
            }
        ]
    )

    view = batch.as_denoise()

    assert view.req_ids == (3,)
    assert view.timestep_indices == (2,)
    assert view.cfg_batches == (
        CfgBatch(
            branch_count=3,
            seg_offsets=[0, 8, 16],
            branch_positions=[[0, 1], [0, 1], [0, 1]],
            branch_kv_offsets=[0, 16, 32],
            branch_kv_lens=[16, 16, 16],
        ),
    )


def test_text_tensor_stager_reuses_ring_slots_and_preserves_build_text_contract():
    stager = TextTensorStager(ring_depth=2)
    first = stager.next_slot()
    a = first.long_buffer("tokens", 4, pin=False)
    a_ptr = a.data_ptr()
    second = stager.next_slot()
    b = second.long_buffer("tokens", 4, pin=False)
    assert b.data_ptr() != a_ptr
    third = stager.next_slot()
    c = third.long_buffer("tokens", 3, pin=False)
    assert c.data_ptr() == a_ptr
    grown = third.long_buffer("tokens", 8, pin=False)
    assert grown.numel() == 8
    assert grown.data_ptr() != a_ptr

    batch = stage_text_forward_batch(
        UniForwardBatch.from_ops(
            [
                {"req_id": 1, "kind": "prefill_und", "token_ids": [10, 11], "pos_range": [4, 6]},
                {"req_id": 2, "kind": "prefill_und", "token_ids": [12], "pos_range": [9, 10]},
            ]
        ).as_text(),
        torch.device("cpu"),
        stager=stager,
    )

    torch.testing.assert_close(batch.input_ids, torch.tensor([10, 11, 12], dtype=torch.long))
    torch.testing.assert_close(batch.positions, torch.tensor([4, 5, 9], dtype=torch.long))
    torch.testing.assert_close(batch.extend_start_loc, torch.tensor([0, 2], dtype=torch.long))
    torch.testing.assert_close(batch.seq_lens, torch.tensor([6, 10], dtype=torch.long))
    torch.testing.assert_close(batch.last_token_indices, torch.tensor([1, 2], dtype=torch.long))


def test_text_forward_batch_tracks_padded_token_bucket_contract():
    batch = UniForwardBatch.from_ops(
        [
            {"req_id": 1, "kind": "prefill_und", "token_ids": [10, 11], "pos_range": [4, 6]},
            {"req_id": 2, "kind": "prefill_und", "token_ids": [12], "pos_range": [9, 10]},
        ]
    )

    built = stage_text_forward_batch(batch.as_text(), torch.device("cpu"), padded_num_tokens=6)

    assert built.num_token_non_padded == 3
    assert built.padded_num_tokens == 6
    assert built.has_padding
    torch.testing.assert_close(built.input_ids, torch.tensor([10, 11, 12, 0, 0, 0], dtype=torch.long))
    torch.testing.assert_close(built.positions, torch.tensor([4, 5, 9, 0, 0, 0], dtype=torch.long))
    torch.testing.assert_close(built.extend_start_loc, torch.tensor([0, 2], dtype=torch.long))
    torch.testing.assert_close(built.last_token_indices, torch.tensor([1, 2], dtype=torch.long))

    with pytest.raises(Exception, match="padded_num_tokens"):
        stage_text_forward_batch(batch.as_text(), torch.device("cpu"), padded_num_tokens=2)


def test_attention_varlen_padding_helpers_keep_raw_tokens_only():
    q = torch.arange(6 * 2 * 4, dtype=torch.float32).reshape(6, 2, 4)
    k = q + 100
    v = q + 200

    q_run, k_run, v_run = RadixAttention._trim_padded_varlen(q, k, v, 3)
    assert q_run.shape == (3, 2, 4)
    torch.testing.assert_close(q_run, q[:3])
    torch.testing.assert_close(k_run, k[:3])
    torch.testing.assert_close(v_run, v[:3])

    restored = RadixAttention._restore_padded_varlen_output(q_run + 1, q, 3)
    assert restored.shape == q.shape
    torch.testing.assert_close(restored[:3], q[:3] + 1)
    torch.testing.assert_close(restored[3:], torch.zeros_like(restored[3:]))


def test_forward_batch_accepts_explicit_mixed_mode_view():
    batch = UniForwardBatch.from_ops(
        [
            {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [3, 4]},
            {"req_id": 2, "kind": "denoise_gen", "timestep_idx": 7},
        ]
    )

    assert batch.mode is ForwardMode.MIXED
    view = batch.as_mixed()
    assert view.modes == (ForwardMode.DECODE, ForwardMode.DENOISE)
    assert view.req_ids == (1, 2)
    assert view.kinds == ("decode_und", "denoise_gen")
    with pytest.raises(Exception, match="not text"):
        batch.as_text()


def test_forward_batch_builds_explicit_mixed_text_view():
    batch = UniForwardBatch.from_ops(
        [
            {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [3, 4]},
            {"req_id": 2, "kind": "prefill_und", "token_ids": [11, 12], "pos_range": [0, 2]},
        ]
    )

    with pytest.raises(Exception, match="not text"):
        stage_text_forward_batch(batch.as_text(), torch.device("cpu"))
    built = stage_text_forward_batch(batch.as_text(allow_mixed_text=True), torch.device("cpu"))

    assert built.mode is ForwardMode.MIXED
    assert built.req_ids == (1, 2)
    torch.testing.assert_close(built.input_ids, torch.tensor([10, 11, 12], dtype=torch.long))
    torch.testing.assert_close(built.positions, torch.tensor([3, 0, 1], dtype=torch.long))
    torch.testing.assert_close(built.extend_start_loc, torch.tensor([0, 1], dtype=torch.long))
    torch.testing.assert_close(built.extend_prefix_lens, torch.tensor([3, 0], dtype=torch.long))
    torch.testing.assert_close(built.last_token_indices, torch.tensor([0, 2], dtype=torch.long))


def test_target_verify_is_a_text_forward_mode():
    batch = UniForwardBatch.from_ops(
        [
            {
                "req_id": 1,
                "kind": "target_verify_und",
                "token_ids": [10, 11, 12],
                "pos_range": [4, 7],
            },
        ]
    )

    text = batch.as_text()
    assert batch.mode is ForwardMode.TARGET_VERIFY
    assert text.mode is ForwardMode.TARGET_VERIFY
    built = stage_text_forward_batch(text, torch.device("cpu"))
    torch.testing.assert_close(built.input_ids, torch.tensor([10, 11, 12], dtype=torch.long))
    torch.testing.assert_close(built.positions, torch.tensor([4, 5, 6], dtype=torch.long))


def test_encode_driver_handles_encode_mode_before_model_forward():
    model = RecordingEncodeCapabilityModel()
    submitted = [{"req_id": 7, "kind": "vit_encode", "mm_hash": 9}]

    result = execute(model, submitted)

    assert result["per_seq"] == [{"req_id": 7, "encoder_handle": 1007, "num_tokens": 3}]
    assert model.encode_ops == ["vit_encode"]
    assert model.calls == []


def test_batch_policy_rejects_invalid_max_batch():
    with pytest.raises(Exception, match="max_batch_ops"):
        BatchPolicy(max_batch_ops=0)


def test_forward_metrics_env_counts_modes_and_tokens(monkeypatch):
    monkeypatch.setenv("UNISERVE_FORWARD_METRICS", "1")
    model = ForwardHookModel(max_batch_ops=4)
    runner = ModelRunner(model, config=RunnerConfig(simulation=True))
    result = runner.execute(
        {
            "step_id": 7,
            "new_reqs": [{"req_id": 1, "block_ids": []}, {"req_id": 2, "block_ids": []}],
            "ops": [
                {"req_id": 1, "kind": "prefill_und", "token_ids": [11, 12, 13], "pos_range": [0, 3]},
                {
                    "req_id": 2,
                    "kind": "denoise_gen",
                    "latent_shape": [2, 2],
                    "cfg": {"branch_count": 3},
                },
            ],
        }
    )

    stats = result["forward_stats"]
    assert stats["mode_counts"] == {"extend": 1, "denoise": 1}
    assert stats["mode_tokens"] == {"extend": 3, "denoise": 12}
    assert set(stats["mode_us"]) == {"mixed"}
    assert stats["attention_launches"] == 0


class ForwardHookModel(RecordingModel):
    def __init__(self, *, max_batch_ops: int = 8) -> None:
        super().__init__(BatchPolicy(max_batch_ops=max_batch_ops, supports_mixed_modes=True))
        self.forward_calls: list[list[str]] = []

    def run_forward(self, batch, *, request_states, group):
        del request_states, group
        self.forward_calls.append([str(op["kind"]) for op in batch.ops])
        return [
            {"req_id": int(op["req_id"]), "kind": str(op["kind"]), "mode": batch.mode.value}
            for op in batch.ops
        ]


@pytest.mark.parametrize("text_kind", ["decode_und", "prefill_und"])
def test_text_denoise_route_uses_whole_batch_forward_unconditionally(text_kind):
    model = ForwardHookModel(max_batch_ops=1)
    submitted = ops(text_kind, "denoise_gen")

    req_ids = sorted({int(op["req_id"]) for op in submitted})
    result = ModelRunner(model, config=RunnerConfig(simulation=True)).execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": req_id, "block_ids": []} for req_id in req_ids],
            "ops": submitted,
        }
    )

    assert [row["mode"] for row in result["per_seq"]] == ["mixed", "mixed"]
    assert model.calls == [
        (ForwardMode.MIXED, [1, 2], [text_kind, "denoise_gen"]),
    ]
    assert model.forward_calls == []


def test_mixed_adapter_whole_batch_forward_bypasses_legacy_deferred_hook():
    class DeferredForwardHookModel(ForwardHookModel):
        def __init__(self) -> None:
            super().__init__()
            self.defer_flags: list[bool] = []

        def run_forward(self, batch, *, request_states, group, defer_text_cpu_results=False):
            self.defer_flags.append(bool(defer_text_cpu_results))
            return super().run_forward(batch, request_states=request_states, group=group)

    model = DeferredForwardHookModel()
    submitted = ops("decode_und", "denoise_gen")
    req_ids = sorted({int(op["req_id"]) for op in submitted})

    result = ModelRunner(model, config=RunnerConfig(simulation=True)).execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": req_id, "block_ids": []} for req_id in req_ids],
            "ops": submitted,
        },
        defer_text_cpu_results=True,
    )

    assert [row["mode"] for row in result["per_seq"]] == ["mixed", "mixed"]
    assert model.calls == [
        (ForwardMode.MIXED, [1, 2], ["decode_und", "denoise_gen"]),
    ]
    assert model.forward_calls == []
    assert model.defer_flags == []


@pytest.mark.parametrize("text_kind", ["decode_und", "prefill_und"])
def test_text_denoise_forward_route_uses_unified_whole_batch_without_hook(text_kind):
    model = RecordingModel(BatchPolicy(max_batch_ops=8, supports_mixed_modes=True))
    submitted = ops(text_kind, "denoise_gen")

    req_ids = sorted({int(op["req_id"]) for op in submitted})
    result = ModelRunner(model, config=RunnerConfig(simulation=True)).execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": req_id, "block_ids": []} for req_id in req_ids],
            "ops": submitted,
        }
    )

    assert [row["mode"] for row in result["per_seq"]] == ["mixed", "mixed"]
    assert model.calls == [
        (ForwardMode.MIXED, [1, 2], [text_kind, "denoise_gen"]),
    ]
