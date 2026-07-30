from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.depth_one import (
    root_parent,
    token_operation,
    und_admission,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Batch,
    DevicePoint,
    Release,
    TokenMode,
    VersionRef,
)
from uniserve_worker.execution.executor import (
    completion_report_ready,
    finalize_completion_report,
)
from uniserve_worker.runtime.completion_store import CompletionLease
from uniserve_worker.server.stub import _next_token

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def test_same_request_continues_from_device_products_before_parent_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = "cuda:0"
    worker = execution_worker(device=device, pipeline_depth=2)
    warm_admission = und_admission(30, block_ids=(1,))
    warm_operation, warm_input = token_operation(
        warm_admission.request_key,
        op_id=100,
        parent=root_parent(warm_admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    worker.execute(
        Batch(
            step_id=100,
            admissions=(warm_admission,),
            operations=(warm_operation,),
            input_products=(warm_input,),
        )
    )
    torch.cuda.synchronize()
    worker.drop_session(30)

    admission = und_admission(31, block_ids=(0,))
    parent, parent_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )

    observe = {"enabled": False}
    ready = CompletionLease.ready

    def gated_ready(self: CompletionLease) -> bool:
        return observe["enabled"] and ready(self)

    monkeypatch.setattr(CompletionLease, "ready", gated_ready)
    parent_report = worker.execute(
        Batch(
            step_id=1,
            admissions=(admission,),
            operations=(parent,),
            input_products=(parent_input,),
        )
    )
    assert not completion_report_ready(parent_report)

    device_parent = VersionRef(
        admission.request_key,
        parent.op_id,
        DevicePoint(parent.outputs[0], parent.plan_digest),
    )
    successor, successor_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=device_parent,
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=parent.outputs[1],
    )
    successor_report = worker.execute(
        Batch(
            step_id=2,
            admissions=(),
            operations=(successor,),
            controls=(Release(admission.request_key, parent.op_id),),
            input_products=(successor_input,),
        )
    )

    observe["enabled"] = True
    torch.cuda.synchronize()
    parent_report = finalize_completion_report(parent_report)
    successor_report = finalize_completion_report(successor_report)

    first = _next_token(4)
    assert parent_report.completions[0].committed_tokens == (first,)
    assert successor_report.completions[0].committed_tokens == (_next_token(first),)
