from __future__ import annotations

from typing import Any

import pytest
import torch

from uniserve_worker.contracts import UniModel
from uniserve_worker.contracts.forward_batch import (
    BatchPolicy,
    DenoiseBranchKey,
    DenoisePostprocessEntry,
    ForwardBatch,
    ForwardResult,
)
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.execution import ModelRunner, RunnerConfig

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
    return ModelRunner(model, config=RunnerConfig(simulation=True)).execute(
        {
            "step_id": 9,
            "new_reqs": [{"req_id": req_id, "block_ids": []} for req_id in req_ids],
            "ops": ops,
        }
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

    assert result == {"step_id": 9, "per_seq": expected}
    assert len(model.batches) == 1
    assert model.batches[0].mode is ForwardMode.MIXED
    assert model.batches[0].ops == tuple(ops)


def test_typed_mixed_result_is_postprocessed_once_at_the_runner_boundary():
    ops = [
        {"req_id": 1, "kind": "decode_und", "token_ids": [4], "pos_range": [0, 1]},
        {
            "req_id": 2,
            "kind": "denoise_gen",
            "latent_shape": [1],
            "cfg": {"branch_count": 1},
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
    assert result["per_seq"][1] == {"req_id": 2, "denoise_done": True, "num_steps_done": 1}
    assert len(accepted) == 1
    torch.testing.assert_close(accepted[0], torch.tensor([3.0]))


def test_invalid_model_result_is_rejected_before_any_result_is_published():
    ops = [{"req_id": 1, "kind": "decode_und", "token_ids": [4], "pos_range": [0, 1]}]
    model = _Model([])

    with pytest.raises(Exception, match="output count"):
        _execute(model, ops)

    assert len(model.batches) == 1
