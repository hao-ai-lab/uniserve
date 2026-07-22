from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from uniserve_worker.contracts import UniModel
from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.contracts.forward_batch import (
    BatchPolicy,
    DenoiseBranchKey,
    DenoisePostprocessEntry,
    ForwardBatch,
    ForwardResult,
    TextPostprocessEntry,
)
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.execution import ExecutorConfig, ModelExecutor

pytestmark = pytest.mark.unit


class _Model(UniModel):
    resource_classes: tuple[str, ...] = ()

    def __init__(self, result: Any) -> None:
        self.result = result
        self.batches: list[ForwardBatch] = []

    def batch_policy(self) -> BatchPolicy:
        return BatchPolicy(max_batch_ops=8, supports_mixed_modes=True)

    def forward(self, batch: ForwardBatch) -> Any:
        self.batches.append(batch)
        return self.result


def _execute(model: UniModel, ops: list[dict[str, Any]]) -> dict[str, Any]:
    req_ids = [int(op["req_id"]) for op in ops]
    return ModelExecutor(model, config=ExecutorConfig(simulation=True)).execute(
        seal_batch(
            9,
            ops,
            new_reqs=[{"req_id": req_id, "block_ids": []} for req_id in req_ids],
        )
    )


def test_model_receives_the_complete_mixed_batch_once_and_results_keep_wire_order():
    ops = [
        {"req_id": 1, "kind": "decode_und", "token_ids": [4], "pos_range": [0, 1]},
        {"req_id": 2, "kind": "denoise_gen", "latent_shape": [2, 2]},
        {"req_id": 3, "kind": "commit_gen"},
    ]
    expected = [
        {"req_id": 1, "sampled_token_id": 11},
        {"req_id": 2, "denoise_done": False, "num_steps_done": 1},
        {"req_id": 3, "image_hw": [16, 16]},
    ]
    model = _Model(expected)

    result = _execute(model, ops)

    assert result["step_id"] == 9
    assert [
        {key: row[key] for key in expected_row}
        for row, expected_row in zip(result["per_seq"], expected, strict=True)
    ] == expected
    assert len(model.batches) == 1
    assert model.batches[0].mode is ForwardMode.MIXED
    assert [dict(operation) for operation in model.batches[0].ops] == seal_batch(9, ops)["ops"]


def test_typed_mixed_result_is_postprocessed_once_at_the_runner_boundary():
    ops = [
        {"req_id": 1, "kind": "decode_und", "token_ids": [4], "pos_range": [0, 1]},
        {
            "req_id": 2,
            "kind": "denoise_gen",
            "latent_shape": [1],
            "cfg": {
                "branch_count": 1,
                "text_scale": 1.0,
                "img_scale": 1.0,
                "renorm_type": "none",
                "renorm_min": 0.0,
                "interval": [0.0, 0.0],
            },
        },
    ]
    accepted: list[torch.Tensor] = []
    model = _Model(
        ForwardResult(
            text_logits=torch.tensor([[0.0, 5.0]], dtype=torch.float32),
            denoise_velocities={DenoiseBranchKey(1, 0): torch.tensor([2.0])},
            denoise_updates={
                1: DenoisePostprocessEntry(
                    row_index=1,
                    req_id=2,
                    step_index=0,
                    total_steps=1,
                    branch_names=("cond",),
                    latent=torch.tensor([1.0]),
                    t=torch.tensor(0.0),
                    t_next=torch.tensor(1.0),
                    combine_velocity=lambda velocities: velocities["cond"],
                    accept_update=accepted.append,
                )
            },
        )
    )

    result = _execute(model, ops)

    assert result["per_seq"][0]["sampled_token_id"] == 1
    assert result["per_seq"][1]["req_id"] == 2
    assert result["per_seq"][1]["denoise_done"] is True
    assert result["per_seq"][1]["num_steps_done"] == 1
    assert len(accepted) == 1
    torch.testing.assert_close(accepted[0], torch.tensor([3.0]))


class _MixedTokenFlowModel(UniModel):
    """Deterministic per-slot model: token rows emit logits, flow rows emit branch velocities."""

    resource_classes: tuple[str, ...] = ()
    _BRANCH_NAMES = ("cond", "text_uncond")

    def __init__(self) -> None:
        self.batches: list[ForwardBatch] = []
        self.accepted: dict[int, torch.Tensor] = {}

    def batch_policy(self) -> BatchPolicy:
        return BatchPolicy(max_batch_ops=8, supports_mixed_modes=True)

    def forward(self, batch: ForwardBatch) -> ForwardResult:
        self.batches.append(batch)
        text_logits_rows: list[torch.Tensor] = []
        velocities: dict[DenoiseBranchKey, torch.Tensor] = {}
        updates: dict[int, DenoisePostprocessEntry] = {}
        for row_index, op in enumerate(batch.ops):
            req_id = int(op["req_id"])
            if str(op["kind"]) == "decode_und":
                row = torch.zeros(8, dtype=torch.float32)
                row[req_id % 8] = 5.0
                text_logits_rows.append(row)
                continue
            latent = torch.full(tuple(op["latent_shape"]), float(req_id))
            for branch_id in range(len(self._BRANCH_NAMES)):
                velocities[DenoiseBranchKey(row_index, branch_id)] = torch.full_like(
                    latent, float(req_id + branch_id + 1)
                )

            def combine(values: dict[str, torch.Tensor]) -> torch.Tensor:
                return values["cond"] + 2.0 * (values["cond"] - values["text_uncond"])

            def accept(updated: torch.Tensor, key: int = req_id) -> None:
                self.accepted[key] = updated

            updates[row_index] = DenoisePostprocessEntry(
                row_index=row_index,
                req_id=req_id,
                step_index=0,
                total_steps=1,
                branch_names=self._BRANCH_NAMES,
                latent=latent,
                t=torch.tensor(0.0),
                t_next=torch.tensor(1.0),
                combine_velocity=combine,
                accept_update=accept,
            )
        return ForwardResult(
            text_logits=torch.stack(text_logits_rows) if text_logits_rows else None,
            denoise_velocities=velocities or None,
            denoise_updates=updates or None,
        )


def test_mixed_token_and_flow_rows_match_homogeneous_projections_in_one_forward():
    token_op = {"req_id": 1, "kind": "decode_und", "token_ids": [4], "pos_range": [0, 1]}
    flow_op = {
        "req_id": 2,
        "kind": "denoise_gen",
        "latent_shape": [2, 2],
        "cfg": {
            "branch_count": 2,
            "text_scale": 3.0,
            "img_scale": 1.0,
            "renorm_type": "none",
            "renorm_min": 0.0,
            "interval": [0.0, 1.0],
        },
    }

    mixed_model = _MixedTokenFlowModel()
    mixed = _execute(mixed_model, [token_op, flow_op])
    token_model = _MixedTokenFlowModel()
    token_only = _execute(token_model, [token_op])
    flow_model = _MixedTokenFlowModel()
    flow_only = _execute(flow_model, [flow_op])

    # Compatible token and flow rows execute as ONE model forward invocation.
    assert len(mixed_model.batches) == 1
    assert mixed_model.batches[0].mode is ForwardMode.MIXED

    # Each mixed row equals its homogeneous projection.
    assert (
        mixed["per_seq"][0]["sampled_token_id"]
        == token_only["per_seq"][0]["sampled_token_id"]
    )
    flow_keys = ("req_id", "denoise_done", "num_steps_done")
    assert {key: mixed["per_seq"][1][key] for key in flow_keys} == {
        key: flow_only["per_seq"][0][key] for key in flow_keys
    }
    torch.testing.assert_close(mixed_model.accepted[2], flow_model.accepted[2])
    # The committed latent is the declared euler update over the combined velocity.
    latent = torch.full((2, 2), 2.0)
    cond = torch.full((2, 2), 3.0)
    text_uncond = torch.full((2, 2), 4.0)
    expected = latent + (cond + 2.0 * (cond - text_uncond))
    torch.testing.assert_close(mixed_model.accepted[2], expected)


def test_text_forward_result_commits_recurrent_state_for_the_next_workflow_step():
    ops = [{"req_id": 1, "kind": "prefill_und", "token_ids": [4, 5, 6], "pos_range": [0, 3]}]
    cache = SimpleNamespace(length=0)
    sequence = SimpleNamespace(
        t_index=-1,
        last_logits=None,
        last_token_id=None,
    )
    model = _Model(
        ForwardResult(
            text_logits=torch.tensor([[0.0, 5.0]], dtype=torch.float32),
            text_postprocess=(
                TextPostprocessEntry(
                    row_index=0,
                    req_id=1,
                    logits_index=0,
                    position_id=3,
                    kv_new_length=3,
                    last_input_token=6,
                    program_state=SimpleNamespace(cond=sequence),
                    persistent_cache=cache,
                ),
            ),
        )
    )

    result = _execute(model, ops)

    assert result["per_seq"][0]["sampled_token_id"] == 1
    assert cache.length == 3
    assert sequence.t_index == 2
    assert sequence.last_token_id == 6
    torch.testing.assert_close(sequence.last_logits, torch.tensor([[[0.0, 5.0]]]))


def test_invalid_model_result_is_rejected_before_any_result_is_published():
    ops = [{"req_id": 1, "kind": "decode_und", "token_ids": [4], "pos_range": [0, 1]}]
    model = _Model([])

    with pytest.raises(Exception, match="output count"):
        _execute(model, ops)

    assert len(model.batches) == 1
