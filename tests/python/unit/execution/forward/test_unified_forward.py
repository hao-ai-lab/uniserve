from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from uniserve_worker.contracts.forward_batch import (
    DenoiseBranchKey,
    DenoisePostprocessEntry,
    ForwardBatch,
    ForwardExecutionOptions,
    ForwardGraphPolicy,
    ForwardOutputKind,
    ForwardResult,
    StrictForwardGraphError,
    TextPostprocessEntry,
)
from uniserve_worker.contracts.forward_context import ForwardContext, use_forward_context
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.contracts.forward_stats import ForwardStats
from uniserve_worker.contracts.model_protocols import ModelHooks
from uniserve_worker.execution.graph import Dispatch, key
from uniserve_worker.execution.graph.path import Batch, Flow, Segment
from uniserve_worker.execution.runner import (
    EagerFallbackRecorder,
    EncodeDriver,
    ForwardExecutor,
    ForwardPlanBuilder,
    ForwardPostprocessor,
    TextDriver,
    WorkerForwardAdapter,
)
from uniserve_worker.execution.runner import UnifiedForwardBatchBuilder as ForwardBatchBuilder
from uniserve_worker.execution.segment import SegmentExecutor
from uniserve_worker.runtime.request_state import RequestStateTable

pytestmark = pytest.mark.unit


def test_forward_graph_policy_fails_closed_by_default():
    assert ForwardGraphPolicy().strict is True


def test_plan_builder_represents_text_generation_and_output_order():
    plan = ForwardPlanBuilder().build(
        [
            (3, {"req_id": 10, "kind": "decode_und", "token_ids": [7], "pos_range": [4, 5]}),
            (
                1,
                {
                    "req_id": 11,
                    "kind": "denoise_gen",
                    "latent_shape": [2, 2],
                    "cfg": {"branch_count": 3},
                },
            ),
            (2, {"req_id": 12, "kind": "commit_gen", "latent_shape": [2, 2]}),
        ]
    )

    assert plan.forward_mode is ForwardMode.MIXED
    assert plan.op_modes == (ForwardMode.DECODE, ForwardMode.DENOISE, ForwardMode.COMMIT)
    assert [row.original_index for row in plan.rows] == [3, 1, 2]
    assert [slot.req_id for slot in plan.output_slots] == [10, 11, 12]
    assert [slot.kind for slot in plan.output_slots] == [
        ForwardOutputKind.TEXT_TOKEN,
        ForwardOutputKind.DENOISE_STEP,
        ForwardOutputKind.COMMIT,
    ]
    assert plan.shape.branch_count == 3
    assert len(plan.segments) == 5


def test_batch_builder_flattens_mixed_text_tokens_and_segment_tables():
    plan = ForwardPlanBuilder().build(
        [
            {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [3, 4]},
            {"req_id": 2, "kind": "prefill_und", "token_ids": [11, 12], "pos_range": [0, 2]},
        ]
    )

    batch = ForwardBatchBuilder().build(plan, device="cpu")

    assert batch.forward_mode is ForwardMode.MIXED
    assert batch.op_modes == (ForwardMode.DECODE, ForwardMode.EXTEND)
    torch.testing.assert_close(batch.input_ids, torch.tensor([10, 11, 12], dtype=torch.long))
    torch.testing.assert_close(batch.positions, torch.tensor([3, 0, 1], dtype=torch.long))
    assert [segment.length for segment in batch.segments] == [1, 2]


def test_graph_capacity_key_excludes_refreshable_runtime_values():
    builder = ForwardPlanBuilder()
    batch_builder = ForwardBatchBuilder()
    plan_a = builder.build(
        [{"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}]
    )
    plan_b = builder.build(
        [{"req_id": 9, "kind": "decode_und", "token_ids": [99], "pos_range": [128, 129]}]
    )

    key_a = key(
        path="step",
        batch=batch_builder.build(plan_a),
        plan=plan_a,
    )
    key_b = key(
        path="step",
        batch=batch_builder.build(plan_b),
        plan=plan_b,
    )

    assert key_a == key_b


def test_graph_capacity_key_excludes_operation_composition():
    builder = ForwardPlanBuilder()
    batch_builder = ForwardBatchBuilder()
    homogeneous = builder.build(
        [
            {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]},
            {"req_id": 2, "kind": "decode_und", "token_ids": [11], "pos_range": [1, 2]},
        ]
    )
    heterogeneous = builder.build(
        [
            {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]},
            {"req_id": 2, "kind": "prefill_und", "token_ids": [11], "pos_range": [0, 1]},
        ]
    )

    homogeneous_key = key(
        path="segment",
        batch=batch_builder.build(homogeneous),
        plan=homogeneous,
    )
    heterogeneous_key = key(
        path="segment",
        batch=batch_builder.build(heterogeneous),
        plan=heterogeneous,
    )

    assert homogeneous_key == heterogeneous_key


def test_batch_path_replays_through_the_text_graph_executor():
    graph_object = object()
    request_states = object()

    class TextDriver:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any, Any, Any]] = []

        def forward_logits_graph(
            self,
            fb,
            states,
            model,
            *,
            graph_runner,
            defer_cpu_results,
            defer_sampling,
        ):
            assert states is request_states
            assert model == "model"
            assert graph_runner is graph_object
            assert defer_cpu_results is False
            assert defer_sampling is False
            self.calls.append((fb, states, model, graph_runner))
            return SimpleNamespace(
                logits=torch.tensor([[float(len(self.calls))]], dtype=torch.float32),
                req_ids=tuple(int(op["req_id"]) for op in fb.ops),
                cuda_ready_start_event=None,
            )

    text_driver = TextDriver()
    graph_runner = Dispatch(
        paths=(
            Batch(
                driver=text_driver,
                model="model",
                states=request_states,
                executor=graph_object,
            ),
        )
    )
    executor = ForwardExecutor(
        graph_runner=graph_runner,
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    builder = ForwardPlanBuilder()
    batch_builder = ForwardBatchBuilder()

    def build(req_id: int, token_id: int):
        op = {"req_id": req_id, "kind": "decode_und", "token_ids": [token_id], "pos_range": [0, 1]}
        plan = builder.build(
            [op],
            graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
        )
        return plan, batch_builder.build(plan)

    plan_a, batch_a = build(1, 10)
    plan_b, batch_b = build(9, 99)

    first = executor.execute(batch_a, plan_a)
    second = executor.execute(batch_b, plan_b)

    assert first.graph is not None and first.graph.path == "step"
    assert second.graph is not None and second.graph.path == "step"
    torch.testing.assert_close(first.text_logits, torch.tensor([[1.0]]))
    torch.testing.assert_close(second.text_logits, torch.tensor([[2.0]]))
    assert len(text_driver.calls) == 2


def test_dispatch_preserves_explicit_path_precedence():
    class Owner:
        def __init__(self) -> None:
            self.calls = 0

        def run_segment_graph(self, batch, plan, *, request_states, result_publisher, options):
            del batch, plan, request_states, result_publisher, options
            self.calls += 1
            return ForwardResult(text_logits=torch.tensor([[3.0]]))

        def try_run_graph_logits_batch(self, ops):
            del ops
            raise AssertionError("the batch specialization must not preempt the segment path")

    class Driver:
        def forward_graph_result(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("the batch specialization must not run")

    owner = Owner()
    states = object()
    op = {
        "req_id": 1,
        "kind": "decode_und",
        "token_ids": [7],
        "pos_range": [0, 1],
    }
    plan = ForwardPlanBuilder().build(
        [op],
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    dispatch = Dispatch(
        paths=(
            Segment(executor=owner, states=states),
            Batch(driver=Driver(), model=owner, states=states),
        )
    )

    with use_forward_context(ForwardContext(stats=ForwardStats())):
        result = dispatch.run(batch, plan)

    assert result is not None and result.graph is not None
    assert result.graph.path == "segment"
    assert owner.calls == 1


def test_general_segment_path_owns_uniform_denoise_when_available():
    class Owner:
        def __init__(self) -> None:
            self.calls = 0

        def run_segment_graph(self, batch, plan, *, request_states, result_publisher, options):
            del batch, plan, request_states, result_publisher, options
            self.calls += 1
            return ForwardResult(
                denoise_velocities={
                    DenoiseBranchKey(0, 0): torch.tensor([1.0], dtype=torch.float32)
                }
            )

    class Driver:
        def forward_result(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("the denoise fallback must not duplicate a general segment graph")

    owner = Owner()
    states = object()
    plan = ForwardPlanBuilder().build(
        [{"req_id": 1, "kind": "denoise_gen", "cfg": {"branch_count": 1}}],
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    dispatch = Dispatch(
        paths=(
            Segment(executor=owner, states=states),
            Flow(driver=Driver(), model=owner, states=states),
        )
    )

    with use_forward_context(ForwardContext(stats=ForwardStats())):
        result = dispatch.run(batch, plan)

    assert result is not None and result.graph is not None
    assert result.graph.path == "segment"
    assert owner.calls == 1


def test_segment_path_runs_graph_only_forward_result(monkeypatch):
    class RequestStates:
        def __init__(self) -> None:
            self.states = {2: object()}

        def get(self, req_id: int):
            return self.states[int(req_id)]

    class Owner:
        def __init__(self) -> None:
            self.prepared: list[tuple[Any, dict[str, Any]]] = []

        def prepare_flow(self, state, op):
            self.prepared.append((state, op))
            return SimpleNamespace(extra={})

        def packed_decoder_forward(self):
            raise AssertionError("packed graph program should require the graph path")

        def packed_text_embeddings(self):
            raise AssertionError("fake packed runner owns the graph-only result")

        def packed_text_logits(self):
            raise AssertionError("fake packed runner owns the graph-only result")

        def packed_hidden_to_velocity(self):
            raise AssertionError("fake packed runner owns the graph-only result")

        def segment_graph_attention(self):
            raise AssertionError("fake packed runner owns the graph-only result")

    owner = Owner()
    segment_executor = SegmentExecutor(owner)
    states = RequestStates()

    def fake_run(owner_arg, dispatch_batch, states_arg, denoise_steps, **kwargs):
        assert owner_arg is segment_executor
        assert states_arg is states
        assert [op["req_id"] for op in dispatch_batch.ops] == [1, 2]
        assert [row for row, _step in denoise_steps] == [1]
        assert kwargs == {
            "defer_text_cpu_results": False,
            "allow_graph": True,
            "require_graph": True,
        }
        return ForwardResult(text_logits=torch.tensor([[3.0]], dtype=torch.float32))

    segment_executor.run_segment_forward_result = lambda *args, **kwargs: fake_run(
        segment_executor, *args, **kwargs
    )
    ops = [
        {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]},
        {"req_id": 2, "kind": "denoise_gen", "cfg": {"branch_count": 1}},
    ]
    plan = ForwardPlanBuilder().build(
        ops,
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(
        graph_runner=Dispatch(paths=(Segment(executor=segment_executor, states=states),)),
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )

    result = executor.execute(batch, plan)

    assert result.graph is not None and result.graph.path == "segment"
    torch.testing.assert_close(result.text_logits, torch.tensor([[3.0]]))
    assert owner.prepared == [(states.states[2], ops[1])]


def test_segment_path_runs_decode_burst_as_graph_only_runtime_result():
    class RequestStates:
        pass

    class Owner:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def prepare_flow(self, state, op):
            raise AssertionError("burst execution must stay inside the composite graph adapter")

        def packed_decoder_forward(self):
            raise AssertionError("outer graph execution must not invoke eager decoder forward")

        def packed_text_embeddings(self):
            raise AssertionError("outer graph execution must not invoke eager text embedding")

        def packed_text_logits(self):
            raise AssertionError("outer graph execution must not invoke text projection")

        def packed_hidden_to_velocity(self):
            raise AssertionError("outer graph execution must not invoke velocity projection")

        def segment_graph_attention(self):
            raise AssertionError("outer graph execution must not inspect attention")

    owner = Owner()
    segment_executor = SegmentExecutor(owner)

    def execute(batch, *, request_states, defer_text_cpu_results=False):
        owner.calls.append(
            {
                "batch": batch,
                "request_states": request_states,
                "defer_text_cpu_results": defer_text_cpu_results,
            }
        )
        return [
            {"req_id": 1, "sampled_token_id": 17, "sampled_token_ids": [10, 17]},
            {"req_id": 2, "denoise_done": False, "num_steps_done": 1},
        ]

    segment_executor.execute = execute
    states = RequestStates()
    ops = [
        {
            "req_id": 1,
            "kind": "decode_und",
            "token_ids": [9],
            "pos_range": [4, 5],
            "decode_token_count": 2,
        },
        {"req_id": 2, "kind": "denoise_gen", "cfg": {"branch_count": 1}},
    ]
    plan = ForwardPlanBuilder().build(
        ops,
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(
        graph_runner=Dispatch(paths=(Segment(executor=segment_executor, states=states),)),
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )

    result = executor.execute(
        batch,
        plan,
        options=ForwardExecutionOptions(defer_text_cpu_results=True),
    )

    assert result.graph is not None and result.graph.path == "segment"
    assert result.runtime_outputs == (
        {"req_id": 1, "sampled_token_id": 17, "sampled_token_ids": [10, 17]},
        {"req_id": 2, "denoise_done": False, "num_steps_done": 1},
    )
    assert owner.calls == [
        {
            "batch": batch,
            "request_states": states,
            "defer_text_cpu_results": True,
        }
    ]


def test_segment_path_publishes_commit_outputs_without_eager(monkeypatch):
    class RequestStates:
        def __init__(self) -> None:
            self.states = {3: SimpleNamespace(latent="latent")}

        def get(self, req_id: int):
            return self.states[int(req_id)]

    class Owner:
        def prepare_flow(self, state, op):
            return SimpleNamespace(extra={})

        def packed_decoder_forward(self):
            raise AssertionError("fake packed runner owns the graph result")

        def packed_text_embeddings(self):
            raise AssertionError("fake packed runner owns the graph result")

        def packed_text_logits(self):
            raise AssertionError("fake packed runner owns the graph result")

        def segment_graph_attention(self):
            raise AssertionError("fake packed runner owns the graph result")

    class ImageDecodeDriver:
        def forward_result(self, items, model, *, row_indices):
            assert model is owner
            assert items == ((3, states.states[3], ops[1]),)
            assert row_indices == (1,)
            return ForwardResult(commit_outputs={1: {"image_hw": [4, 5]}})

    owner = Owner()
    segment_executor = SegmentExecutor(owner)
    states = RequestStates()
    ops = [
        {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]},
        {"req_id": 3, "kind": "commit_gen"},
    ]

    def fake_run(owner_arg, dispatch_batch, states_arg, denoise_steps, **kwargs):
        assert owner_arg is segment_executor
        assert states_arg is states
        assert [dict(op) for op in dispatch_batch.ops] == ops
        assert denoise_steps == []
        return ForwardResult(text_logits=torch.tensor([[3.0]], dtype=torch.float32))

    segment_executor.run_segment_forward_result = lambda *args, **kwargs: fake_run(
        segment_executor, *args, **kwargs
    )
    plan = ForwardPlanBuilder().build(
        ops,
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(
        graph_runner=Dispatch(
            paths=(
                Segment(
                    executor=segment_executor,
                    states=states,
                    publisher=ImageDecodeDriver(),
                ),
            )
        ),
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )

    result = executor.execute(
        batch,
        plan,
        forward_fn=lambda _batch: (_ for _ in ()).throw(
            AssertionError("graph-backed commit batch must not invoke eager forward")
        ),
    )

    assert result.graph is not None and result.graph.path == "segment"
    assert result.commit_outputs == {1: {"image_hw": [4, 5]}}


def test_denoise_path_runs_required_graph_mode():
    class RequestStates:
        def __init__(self) -> None:
            self.states = {5: object()}

        def get(self, req_id: int):
            return self.states[int(req_id)]

    class Driver:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any, dict[str, Any]]] = []

        def forward_result(self, items, model, **kwargs):
            self.calls.append((items, model, kwargs))
            return ForwardResult(
                denoise_velocities={
                    DenoiseBranchKey(0, 0): torch.tensor([1.0], dtype=torch.float32)
                }
            )

    states = RequestStates()
    driver = Driver()
    op = {"req_id": 5, "kind": "denoise_gen", "cfg": {"branch_count": 1}}
    plan = ForwardPlanBuilder().build(
        [op],
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(
        graph_runner=Dispatch(paths=(Flow(driver=driver, model="model", states=states),)),
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )

    result = executor.execute(batch, plan)

    assert result.graph is not None and result.graph.path == "flow"
    assert driver.calls == [
        (
            [(5, states.states[5], op)],
            "model",
            {"row_indices": (0,), "graph_mode": "require"},
        )
    ]


def test_owner_batch_path_uses_graph_only_driver():
    class RequestStates:
        def get(self, req_id: int):
            return {"req_id": int(req_id)}

    class TextDriver:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def forward_logits_graph(
            self,
            dispatch_batch,
            request_states,
            model,
            *,
            graph_runner,
            defer_cpu_results,
            defer_sampling,
        ):
            self.calls.append(
                {
                    "ops": tuple(dict(op) for op in dispatch_batch.ops),
                    "request_states": request_states,
                    "model": model,
                    "graph_runner": graph_runner,
                    "defer_cpu_results": defer_cpu_results,
                    "defer_sampling": defer_sampling,
                }
            )
            return SimpleNamespace(logits=torch.tensor([[8.0, 9.0]]), req_ids=(7,))

    class Model:
        def try_run_graph_logits_batch(self, _ops):
            raise AssertionError("program must route through TextDriver graph logits")

    states = RequestStates()
    text_driver = TextDriver()
    model = Model()
    op = {
        "req_id": 7,
        "kind": "prefill_und",
        "token_ids": [1, 2],
        "pos_range": [0, 2],
        "decode_token_count": 4,
    }
    plan = ForwardPlanBuilder().build(
        [op],
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(
        graph_runner=Dispatch(
            paths=(
                Batch(
                    driver=text_driver,
                    model=model,
                    states=states,
                ),
            )
        ),
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )

    result = executor.execute(batch, plan)

    assert result.graph is not None and result.graph.path == "span"
    torch.testing.assert_close(result.text_logits, torch.tensor([[8.0, 9.0]]))
    assert text_driver.calls == [
        {
            "ops": (op,),
            "request_states": states,
            "model": model,
            "graph_runner": None,
            "defer_cpu_results": False,
            "defer_sampling": False,
        }
    ]


def test_owner_batch_path_uses_runtime_result_for_multi_step_row():
    class RequestStates:
        def get(self, req_id: int):
            return {"req_id": int(req_id)}

    class TextDriver:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def forward_graph_result(
            self,
            dispatch_batch,
            request_states,
            model,
            *,
            graph_runner,
            defer_cpu_results,
            defer_sampling,
        ):
            self.calls.append(
                {
                    "ops": tuple(dict(op) for op in dispatch_batch.ops),
                    "request_states": request_states,
                    "model": model,
                    "graph_runner": graph_runner,
                    "defer_cpu_results": defer_cpu_results,
                    "defer_sampling": defer_sampling,
                }
            )
            return ForwardResult(runtime_outputs=({"req_id": 7, "sampled_token_ids": [4, 5]},))

    class Model:
        def try_run_graph_logits_batch(self, _ops):
            raise AssertionError("program must route through TextDriver graph result")

    states = RequestStates()
    text_driver = TextDriver()
    model = Model()
    op = {
        "req_id": 7,
        "kind": "decode_und",
        "token_ids": [3],
        "pos_range": [4, 5],
        "decode_token_count": 2,
    }
    plan = ForwardPlanBuilder().build(
        [op],
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(
        graph_runner=Dispatch(
            paths=(
                Batch(
                    driver=text_driver,
                    model=model,
                    states=states,
                ),
            )
        ),
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )

    result = executor.execute(batch, plan)

    assert result.graph is not None and result.graph.path == "step"
    assert result.runtime_outputs == ({"req_id": 7, "sampled_token_ids": [4, 5]},)
    assert text_driver.calls == [
        {
            "ops": (op,),
            "request_states": states,
            "model": model,
            "graph_runner": None,
            "defer_cpu_results": False,
            "defer_sampling": False,
        }
    ]


def test_text_driver_graph_result_runs_decode_burst_without_eager_fallback():
    states = RequestStateTable()
    state = states.get(7)
    state.sampling = {}

    class Model:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, Any]]] = []

        def try_run_graph_logits_batch(self, ops):
            self.calls.append([dict(op) for op in ops])
            token = 4 if len(self.calls) == 1 else 5
            logits = torch.full((1, 8), -10.0, dtype=torch.float32)
            logits[0, token] = 10.0
            return [logits]

        def run_text_logits_batch(self, _ops):
            raise AssertionError("graph result path must not use eager text logits")

    model = Model()
    op = {
        "req_id": 7,
        "kind": "decode_und",
        "token_ids": [3],
        "pos_range": [4, 5],
        "decode_token_count": 2,
    }
    driver = TextDriver()
    fb = ForwardBatch.from_ops([op])

    with use_forward_context(ForwardContext(stats=ForwardStats())):
        result = driver.forward_graph_result(
            fb,
            states,
            model,
            graph_runner=None,
            defer_cpu_results=False,
            defer_sampling=False,
        )

    assert result is not None
    assert result.runtime_outputs == (
        {"req_id": 7, "sampled_token_id": 5, "sampled_token_ids": [4, 5]},
    )
    assert len(model.calls) == 2
    assert model.calls[0] == [
        {
            "req_id": 7,
            "kind": "decode_und",
            "token_ids": [3],
            "pos_range": [4, 5],
            "decode_token_count": 1,
            "decode_stop_token_ids": [],
        }
    ]
    followup = dict(model.calls[1][0])
    token_tensor = followup.pop("token_tensor")
    assert followup == {
        "req_id": 7,
        "kind": "decode_und",
        "token_ids": [4],
        "pos_range": [5, 6],
        "decode_token_count": 1,
        "decode_stop_token_ids": [],
        "new_block_ids": [],
        "token_source": "last_sampled",
    }
    torch.testing.assert_close(token_tensor, torch.tensor([4], dtype=torch.long))
    assert state.kv_length("text") == 6
    assert state.decode_relay.token_id == 5
    torch.testing.assert_close(state.decode_relay.token_tensor, torch.tensor([5], dtype=torch.long))


def test_text_driver_scores_prompt_across_prefill_chunk_boundaries():
    states = RequestStateTable()
    state = states.get(7)
    state.sampling = {
        "temperature": 0.0,
        "return_prompt_logprobs": True,
        "n_prompt_logprobs": 1,
    }

    class Model:
        def run_text_logits(self, op):
            if op["pos_range"] == [0, 2]:
                return torch.tensor([[[0.0, 1.0, 4.0, 2.0, -1.0], [0.0, 1.0, 2.0, 5.0, -1.0]]])
            return torch.tensor([[[0.0, 1.0, 2.0, 3.0, 6.0], [0.0, 5.0, 2.0, 3.0, 1.0]]])

    driver = TextDriver()
    first = ForwardBatch.from_ops(
        [
            {
                "req_id": 7,
                "kind": "prefill_und",
                "token_ids": [1, 2],
                "pos_range": [0, 2],
                "return_all_logits": True,
            }
        ]
    )
    second = ForwardBatch.from_ops(
        [
            {
                "req_id": 7,
                "kind": "prefill_und",
                "token_ids": [3, 4],
                "pos_range": [2, 4],
                "return_all_logits": True,
            }
        ]
    )

    with use_forward_context(ForwardContext(stats=ForwardStats())):
        first_output = driver.step(first, states, Model())[0]
        second_output = driver.step(second, states, Model())[0]

    assert [position[0][0] for position in first_output.prompt_logprobs] == [2]
    assert [position[0][0] for position in second_output.prompt_logprobs] == [3, 4]
    assert all(position[0][2] == 1 for position in first_output.prompt_logprobs)
    assert all(position[0][2] == 1 for position in second_output.prompt_logprobs)


def test_executor_strict_graph_policy_rejects_eager_fallback():
    plan = ForwardPlanBuilder().build(
        [{"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}],
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True))

    with pytest.raises(StrictForwardGraphError):
        executor.execute(
            batch, plan, forward_fn=lambda _batch: ForwardResult(runtime_outputs=({"req_id": 1},))
        )


def test_executor_strict_graph_policy_preserves_graph_failure_cause():
    root_cause = RuntimeError("capture failed")

    class FailingGraphRunner:
        def run(self, *_args, **_kwargs):
            raise root_cause

    plan = ForwardPlanBuilder().build(
        [{"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}],
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(
        graph_runner=FailingGraphRunner(),
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )

    with pytest.raises(StrictForwardGraphError) as caught:
        executor.execute(batch, plan)

    assert caught.value.__cause__ is root_cause


def test_executor_delegated_graph_policy_does_not_record_fallback():
    recorder = EagerFallbackRecorder()
    plan = ForwardPlanBuilder().build(
        [{"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}],
        graph_policy=ForwardGraphPolicy(
            prefer_graph=True,
            strict=True,
            graph_selection_delegated=True,
        ),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(
        graph_policy=ForwardGraphPolicy(graph_selection_delegated=True),
        fallback_recorder=recorder,
    )

    result = executor.execute(
        batch,
        plan,
        forward_fn=lambda _batch: ForwardResult(runtime_outputs=({"req_id": 1},)),
    )

    assert result.runtime_outputs == ({"req_id": 1},)
    assert recorder.counts == {}


def test_executor_delegated_graph_policy_bypasses_installed_graph_runner():
    recorder = EagerFallbackRecorder()
    plan = ForwardPlanBuilder().build(
        [{"req_id": 1, "kind": "commit_gen"}],
        graph_policy=ForwardGraphPolicy(
            prefer_graph=True,
            strict=True,
            graph_selection_delegated=True,
        ),
    )
    batch = ForwardBatchBuilder().build(plan)

    class GraphRunner:
        def run(self, *args, **kwargs):
            raise AssertionError("delegated graph policy must not query the graph runner")

    executor = ForwardExecutor(
        graph_runner=GraphRunner(),
        graph_policy=ForwardGraphPolicy(graph_selection_delegated=True),
        fallback_recorder=recorder,
    )

    result = executor.execute(
        batch,
        plan,
        forward_fn=lambda _batch: ForwardResult(runtime_outputs=({"req_id": 1},)),
    )

    assert result.runtime_outputs == ({"req_id": 1},)
    assert recorder.counts == {}


def test_executor_invokes_adapter_forward_as_eager_surface():
    class Adapter:
        def __init__(self) -> None:
            self.batches: list[Any] = []

        def forward(self, batch):
            self.batches.append(batch)
            return ForwardResult(runtime_outputs=({"req_id": 1, "via": "adapter"},))

    adapter = Adapter()
    plan = ForwardPlanBuilder().build(
        [{"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}],
        graph_policy=ForwardGraphPolicy(graph_selection_delegated=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(
        model=adapter,
        graph_policy=ForwardGraphPolicy(graph_selection_delegated=True),
    )

    result = executor.execute(batch, plan)

    assert adapter.batches == [batch]
    assert result.runtime_outputs == ({"req_id": 1, "via": "adapter"},)


def test_postprocessor_validates_before_projection():
    plan = ForwardPlanBuilder().build(
        [{"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}],
    )
    batch = ForwardBatchBuilder().build(plan)

    with pytest.raises(Exception, match="output count"):
        ForwardPostprocessor().apply(batch, plan, ForwardResult(runtime_outputs=()))


def test_worker_adapter_text_path_returns_logits_without_request_state_mutation():
    op = {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}
    state = _FakeTextState()
    request_states = _FakeRequestStates({1: state})

    class Driver:
        def forward_logits(self, fb, states, model, *, defer_cpu_results, defer_sampling):
            del fb, model, defer_cpu_results, defer_sampling
            assert states is request_states
            assert state.kv_updates == []
            assert state.decode_relay.token_tensor is None
            return SimpleNamespace(
                logits=torch.tensor([[0.0, 4.0]], dtype=torch.float32),
                req_ids=(1,),
                cuda_ready_start_event=None,
            )

        def step(self, *args, **kwargs):
            raise AssertionError("typed text adapter path should not call TextDriver.step")

    class Model:
        device = "cpu"
        vocab_size = 2

        def forward(self, *args, **kwargs):
            raise AssertionError("fake model forward is owned by the fake text driver")

    adapter = WorkerForwardAdapter(
        model=Model(),
        request_states=request_states,
        text_driver=Driver(),
        denoise_driver=object(),
        encode_driver=object(),
        image_decode_driver=object(),
    )
    plan = ForwardPlanBuilder().build([op], request_states=request_states)
    batch = ForwardBatchBuilder().build(plan)
    result = adapter.forward(batch, plan, ForwardExecutionOptions())

    assert result.runtime_outputs is None
    torch.testing.assert_close(result.text_logits, torch.tensor([[0.0, 4.0]]))
    assert state.kv_updates == []
    assert state.decode_relay.token_tensor is None


def test_postprocessor_text_logits_samples_batched_relays_and_advances_kv_after_validation():
    ops = [
        {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]},
        {"req_id": 2, "kind": "decode_und", "token_ids": [11], "pos_range": [6, 7]},
    ]
    states = {1: _FakeTextState(), 2: _FakeTextState()}
    request_states = _FakeRequestStates(states)
    plan = ForwardPlanBuilder().build(ops, request_states=request_states)
    batch = ForwardBatchBuilder().build(plan)

    outputs = ForwardPostprocessor(request_states=request_states).apply(
        batch,
        plan,
        ForwardResult(
            text_logits=torch.tensor(
                [[0.0, 1.0, 7.0], [9.0, 1.0, 0.0]],
                dtype=torch.float32,
            )
        ),
    )

    assert outputs[0].sampled_token_id == 2
    assert outputs[1].sampled_token_id == 0
    assert states[1].kv_updates == [("text", 1)]
    assert states[2].kv_updates == [("text", 7)]
    assert states[1].decode_relay.token_id == 2
    assert states[2].decode_relay.token_id == 0
    torch.testing.assert_close(
        states[1].decode_relay.token_tensor,
        torch.tensor([2], dtype=torch.long),
    )
    torch.testing.assert_close(
        states[2].decode_relay.token_tensor,
        torch.tensor([0], dtype=torch.long),
    )
    assert states[1].decode_relay.position_id == 1
    assert states[2].decode_relay.position_id == 7
    first_position = states[1].decode_relay.position_tensor
    second_position = states[2].decode_relay.position_tensor
    torch.testing.assert_close(first_position, torch.tensor([1], dtype=torch.long))
    torch.testing.assert_close(second_position, torch.tensor([7], dtype=torch.long))
    assert (
        first_position.untyped_storage().data_ptr() == second_position.untyped_storage().data_ptr()
    )
    assert second_position.data_ptr() - first_position.data_ptr() == first_position.element_size()


def test_postprocessor_text_validation_failure_leaves_request_state_unchanged():
    op = {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}
    state = _FakeTextState()
    request_states = _FakeRequestStates({1: state})
    plan = ForwardPlanBuilder().build([op], request_states=request_states)
    batch = ForwardBatchBuilder().build(plan)

    with pytest.raises(Exception, match="row count"):
        ForwardPostprocessor(request_states=request_states).apply(
            batch,
            plan,
            ForwardResult(text_logits=torch.empty((0, 3), dtype=torch.float32)),
        )

    assert state.kv_updates == []
    assert state.decode_relay.token_tensor is None
    assert state.decode_relay.position_tensor is None


def test_worker_adapter_denoise_result_updates_latent_only_in_postprocess():
    op = {"req_id": 1, "kind": "denoise_gen", "cfg": {"branch_count": 2}, "num_steps": 1}
    state = SimpleNamespace(updated=None)
    request_states = _FakeRequestStates({1: state})

    class Driver:
        def forward_result(self, items, model, *, row_indices, graph_mode):
            del model
            assert [(req_id, item_state, item_op) for req_id, item_state, item_op in items] == [
                (1, state, op)
            ]
            assert tuple(row_indices) == (0,)
            assert graph_mode == "eager"
            assert state.updated is None
            return ForwardResult(
                denoise_velocities={
                    DenoiseBranchKey(0, 0): torch.tensor([3.0]),
                    DenoiseBranchKey(0, 1): torch.tensor([1.0]),
                },
                denoise_updates={
                    0: DenoisePostprocessEntry(
                        row_index=0,
                        req_id=1,
                        step_index=0,
                        total_steps=1,
                        branch_names=("cond", "uncond"),
                        latent=torch.tensor([0.0]),
                        t=torch.tensor(0.0),
                        t_next=torch.tensor(1.0),
                        combine_velocity=lambda velocities: (
                            velocities["cond"] - velocities["uncond"]
                        ),
                        accept_update=lambda updated: setattr(state, "updated", updated),
                    )
                },
            )

        def step_many(self, *args, **kwargs):
            raise AssertionError(
                "typed denoise adapter path should not call FlowExecutor.step_many"
            )

    class Model(ModelHooks):
        device = "cpu"

        def predict_velocity(self, ctx, t, latent, branch):
            raise AssertionError("fake driver owns branch prediction in this test")

    adapter = WorkerForwardAdapter(
        model=Model(),
        request_states=request_states,
        text_driver=object(),
        denoise_driver=Driver(),
        encode_driver=object(),
        image_decode_driver=object(),
    )
    plan = ForwardPlanBuilder().build([op], request_states=request_states)
    batch = ForwardBatchBuilder().build(plan)
    result = adapter.forward(batch, plan, ForwardExecutionOptions())

    assert result.runtime_outputs is None
    assert state.updated is None

    outputs = ForwardPostprocessor(request_states=request_states).apply(batch, plan, result)

    assert len(outputs) == 1
    assert outputs[0].req_id == 1
    assert outputs[0].denoise_done is True
    assert outputs[0].num_steps_done == 1
    torch.testing.assert_close(state.updated, torch.tensor([2.0]))


def test_worker_adapter_encode_result_is_published_by_postprocess():
    op = {"req_id": 2, "kind": "vit_encode", "mm_hash": 9}

    class Driver:
        def forward_result(self, fb, model, *, row_indices):
            del fb, model
            assert tuple(row_indices) == (0,)
            return ForwardResult(
                encode_outputs={
                    0: {"req_id": 2, "encoder_handle": 44, "num_tokens": 3, "image_hw": [8, 9]}
                }
            )

        def step(self, *args, **kwargs):
            raise AssertionError("typed encode adapter path should not call EncodeDriver.step")

    class Model(ModelHooks):
        device = "cpu"

        def encode_image(self, pixels=None, grid=None, *, op=None):
            raise AssertionError("fake driver owns encode publication in this test")

    adapter = WorkerForwardAdapter(
        model=Model(),
        request_states=_FakeRequestStates({}),
        text_driver=object(),
        denoise_driver=object(),
        encode_driver=Driver(),
        image_decode_driver=object(),
    )
    plan = ForwardPlanBuilder().build([op])
    batch = ForwardBatchBuilder().build(plan)
    result = adapter.forward(batch, plan, ForwardExecutionOptions())

    assert result.runtime_outputs is None
    outputs = ForwardPostprocessor().apply(batch, plan, result)

    assert len(outputs) == 1
    assert outputs[0].req_id == 2
    assert outputs[0].encoder_handle == 44
    assert outputs[0].num_tokens == 3
    assert outputs[0].image_hw == (8, 9)


def test_encode_driver_uses_model_batch_hook_for_aligned_outputs():
    ops = [
        {"req_id": 2, "kind": "vit_encode", "mm_hash": 9},
        {"req_id": 3, "kind": "vae_encode", "mm_hash": 10},
    ]

    class Model:
        def __init__(self):
            self.calls = 0

        def encode_many(self, submitted_ops):
            self.calls += 1
            assert tuple(submitted_ops) == tuple(ops)
            return [
                {"req_id": 2, "encoder_handle": 44, "num_tokens": 3},
                {"req_id": 3, "encoder_handle": 45, "num_tokens": 4},
            ]

        def encode_image(self, *args, **kwargs):
            raise AssertionError("batched encode must not fall back to per-op execution")

        def encode_latents(self, *args, **kwargs):
            raise AssertionError("batched encode must not fall back to per-op execution")

    model = Model()
    result = EncodeDriver().forward_result(
        ForwardBatch.from_ops(ops),
        model,
        row_indices=(4, 9),
    )

    assert model.calls == 1
    assert result.encode_outputs is not None
    assert sorted(result.encode_outputs) == [4, 9]
    assert result.encode_outputs[4].encoder_handle == 44
    assert result.encode_outputs[9].encoder_handle == 45


def test_worker_adapter_commit_result_is_sampled_by_postprocess():
    op = {"req_id": 3, "kind": "commit_gen"}
    state = _FakeTextState()
    state.sampling = {"temperature": 0.0}
    request_states = _FakeRequestStates({3: state})

    class Driver:
        def forward_result(self, items, model, *, row_indices):
            del model
            assert [(req_id, item_state, item_op) for req_id, item_state, item_op in items] == [
                (3, state, op)
            ]
            assert tuple(row_indices) == (0,)
            return ForwardResult(
                commit_outputs={
                    0: {
                        "req_id": 3,
                        "image_hw": [4, 5],
                        "logits": torch.tensor([0.0, 6.0, 2.0], dtype=torch.float32),
                    }
                }
            )

        def step(self, *args, **kwargs):
            raise AssertionError("typed commit adapter path should not call ImageDecodeDriver.step")

    class Model(ModelHooks):
        device = "cpu"

        def decode_image(self, latent, *, req_id=None, state=None, op=None):
            raise AssertionError("fake driver owns commit decode in this test")

    adapter = WorkerForwardAdapter(
        model=Model(),
        request_states=request_states,
        text_driver=object(),
        denoise_driver=object(),
        encode_driver=object(),
        image_decode_driver=Driver(),
    )
    plan = ForwardPlanBuilder().build([op], request_states=request_states)
    batch = ForwardBatchBuilder().build(plan)
    result = adapter.forward(batch, plan, ForwardExecutionOptions())

    assert result.runtime_outputs is None
    outputs = ForwardPostprocessor(request_states=request_states).apply(batch, plan, result)

    assert len(outputs) == 1
    assert outputs[0].req_id == 3
    assert outputs[0].image_hw == (4, 5)
    assert outputs[0].sampled_token_id == 1


def test_worker_adapter_private_mixed_hook_can_return_forward_result():
    ops = [
        {"req_id": 7, "kind": "decode_und", "token_ids": [4], "pos_range": [0, 1]},
        {"req_id": 8, "kind": "denoise_gen"},
    ]
    expected = ForwardResult(
        text_logits=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        denoise_velocities={DenoiseBranchKey(1, 0): torch.tensor([1.0])},
    )

    class Model(ModelHooks):
        device = "cpu"

        def _run_forward_adapter(
            self, batch, *, request_states, group, defer_text_cpu_results=False
        ):
            assert request_states is states
            assert [item[1] for item in group] == ops
            assert defer_text_cpu_results is False
            assert batch.ops == tuple(ops)
            return expected

    states = _FakeRequestStates({7: _FakeTextState(), 8: SimpleNamespace()})
    adapter = WorkerForwardAdapter(
        model=Model(),
        request_states=states,
        text_driver=object(),
        denoise_driver=object(),
        encode_driver=object(),
        image_decode_driver=object(),
    )
    plan = ForwardPlanBuilder().build(ops, request_states=states)
    batch = ForwardBatchBuilder().build(plan)
    result = adapter.forward(batch, plan, ForwardExecutionOptions())

    assert result is expected


@pytest.mark.parametrize(
    "op",
    [
        {"req_id": 7, "kind": "prefill_und", "token_ids": [4, 5], "pos_range": [0, 2]},
        {"req_id": 7, "kind": "decode_und", "token_ids": [4], "pos_range": [2, 3]},
        {"req_id": 7, "kind": "denoise_gen"},
    ],
)
def test_worker_adapter_private_hook_accepts_any_segment_group(op):
    expected = ForwardResult(runtime_outputs=({"req_id": 7},))

    class Model(ModelHooks):
        device = "cpu"

        def _run_forward_adapter(
            self, batch, *, request_states, group, defer_text_cpu_results=False
        ):
            assert request_states is states
            assert group == [(0, op)]
            assert defer_text_cpu_results is False
            assert batch.ops == (op,)
            return expected

    states = _FakeRequestStates({7: SimpleNamespace()})
    adapter = WorkerForwardAdapter(
        model=Model(),
        request_states=states,
        text_driver=object(),
        denoise_driver=object(),
        encode_driver=object(),
        image_decode_driver=object(),
    )
    plan = ForwardPlanBuilder().build([op], request_states=states)
    batch = ForwardBatchBuilder().build(plan)
    result = adapter.forward(batch, plan, ForwardExecutionOptions())

    assert result is expected


def test_postprocessor_mixed_text_entry_samples_relays_and_advances_program_state():
    text_op = {"req_id": 4, "kind": "decode_und", "token_ids": [8], "pos_range": [6, 7]}
    denoise_op = {"req_id": 5, "kind": "denoise_gen"}
    text_state = _FakeTextState()
    denoise_state = SimpleNamespace(updated=None)
    request_states = _FakeRequestStates({4: text_state, 5: denoise_state})
    image_state = SimpleNamespace(
        cond=SimpleNamespace(
            t_index=5,
            last_logits=None,
            last_token_id=None,
        )
    )
    cache = SimpleNamespace(length=6)
    plan = ForwardPlanBuilder().build(
        [text_op, denoise_op],
        request_states=request_states,
    )
    batch = ForwardBatchBuilder().build(plan)
    result = ForwardResult(
        text_logits=torch.tensor([[0.0, 2.0, 9.0]], dtype=torch.float32),
        text_postprocess=(
            TextPostprocessEntry(
                row_index=0,
                req_id=4,
                logits_index=0,
                position_id=7,
                kv_new_length=7,
                last_input_token=8,
                program_state=image_state,
                persistent_cache=cache,
            ),
        ),
        denoise_velocities={
            DenoiseBranchKey(1, 0): torch.tensor([1.0]),
        },
        denoise_updates={
            1: DenoisePostprocessEntry(
                row_index=1,
                req_id=5,
                step_index=0,
                total_steps=1,
                branch_names=("cond",),
                latent=torch.tensor([0.0]),
                t=torch.tensor(0.0),
                t_next=torch.tensor(1.0),
                combine_velocity=lambda velocities: velocities["cond"],
                accept_update=lambda updated: setattr(denoise_state, "updated", updated),
            )
        },
    )

    outputs = ForwardPostprocessor(request_states=request_states).apply(batch, plan, result)

    assert outputs[0].sampled_token_id == 2
    assert outputs[1].denoise_done is True
    assert image_state.cond.t_index == 6
    assert image_state.cond.last_token_id == 8
    assert cache.length == 7
    torch.testing.assert_close(image_state.cond.last_logits, torch.tensor([[[0.0, 2.0, 9.0]]]))
    assert text_state.decode_relay.token_id == 2
    assert text_state.decode_relay.position_id == 7
    torch.testing.assert_close(
        text_state.decode_relay.token_tensor, torch.tensor([2], dtype=torch.long)
    )
    torch.testing.assert_close(
        text_state.decode_relay.position_tensor, torch.tensor([7], dtype=torch.long)
    )
    torch.testing.assert_close(denoise_state.updated, torch.tensor([1.0]))


class _FakeTextState:
    def __init__(self) -> None:
        self.sampling: dict[str, Any] = {}
        self.decode_relay = SimpleNamespace(
            token_id=None,
            token_tensor=None,
            position_id=None,
            position_tensor=None,
        )
        self.kv_updates: list[tuple[str, int]] = []
        self.device_rngs: dict[str, torch.Generator] = {}

    def set_kv_length(self, value: int, *, lane: str) -> None:
        self.kv_updates.append((str(lane), int(value)))

    def device_rng(
        self,
        device: torch.device | str,
        *,
        stream: str = "model",
    ) -> torch.Generator:
        target = torch.device(device)
        key = f"{stream}:{target}"
        return self.device_rngs.setdefault(
            key,
            torch.Generator(device=target).manual_seed(0),
        )


class _FakeRequestStates:
    def __init__(self, states: dict[int, _FakeTextState]) -> None:
        self._states = states

    def get(self, req_id: int) -> _FakeTextState:
        return self._states[int(req_id)]
