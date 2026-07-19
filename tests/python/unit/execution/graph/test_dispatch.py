from __future__ import annotations

from typing import Any

import pytest
import torch

from uniserve_worker.contracts.forward_batch import ForwardGraphPolicy, ForwardResult
from uniserve_worker.contracts.forward_context import (
    ForwardContext,
    get_forward_context,
    use_forward_context,
)
from uniserve_worker.contracts.forward_stats import ForwardStats
from uniserve_worker.execution.engine import (
    ForwardPlanBuilder,
    UnifiedForwardBatchBuilder,
)
from uniserve_worker.execution.graph.capture import Runner, record
from uniserve_worker.execution.graph.dispatch import Dispatch, Match, Path

pytestmark = pytest.mark.unit


class _Runner(Runner):
    def __init__(self) -> None:
        self.name = "probe"
        self.default_enabled = True
        self.default_warmup = False
        self.metric_prefix = "probe_"
        self.logger = None
        self.states: dict[str, object] = {}
        self.disabled: set[str] = set()
        self._capture_pool = None
        self._graph_input_buffer_pool = {}
        self.captures = 0

    def run(self) -> ForwardResult | None:
        ctx = get_forward_context()

        def capture() -> object:
            self.captures += 1
            return object()

        return self._capture_or_replay(
            key="fixed",
            device="cpu",
            ctx=ctx,
            capture=capture,
            copy_inputs=lambda _state: None,
            replay=lambda _state: ForwardResult(
                text_logits=torch.tensor([[1.0]], dtype=torch.float32)
            ),
            record=lambda event: record(
                ctx,
                event,
                unpadded_tokens=1,
                padded_tokens=1,
            ),
            disable=lambda _exc: None,
            capture_metric="capture",
            input_copy_metric="copy",
            replay_metric="replay",
        )


class _Path(Path):
    def __init__(self, runner: _Runner) -> None:
        self.runner = runner

    def name(self, plan: Any) -> str:
        del plan
        return "probe"

    def match(self, batch: Any, plan: Any) -> Match:
        del batch, plan
        return Match(True)

    def run(self, batch: Any, plan: Any, options: Any) -> ForwardResult | None:
        del batch, plan, options
        return self.runner.run()


def _plan():
    op = {
        "req_id": 1,
        "kind": "decode_und",
        "token_ids": [7],
        "pos_range": [0, 1],
    }
    plan = ForwardPlanBuilder().build(
        [op],
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=False),
    )
    return plan, UnifiedForwardBatchBuilder().build(plan)


def test_capture_policy_reaches_the_physical_runner_and_still_allows_replay():
    plan, batch = _plan()
    runner = _Runner()
    dispatch = Dispatch((_Path(runner),))
    stats = ForwardStats()

    with use_forward_context(ForwardContext(stats=stats)):
        assert dispatch.run(batch, plan, allow_capture=False) is None
        captured = dispatch.run(batch, plan, allow_capture=True)
        replayed = dispatch.run(batch, plan, allow_capture=False)

    assert captured is not None and captured.graph is not None
    assert replayed is not None and replayed.graph is not None
    assert runner.captures == 1
    assert stats.cuda_graph_captures == 1
    assert stats.cuda_graph_replays == 2
    assert stats.cuda_graph_misses == 1
    assert stats.cuda_graph_runtime_mode_counts == {"probe": 2}
