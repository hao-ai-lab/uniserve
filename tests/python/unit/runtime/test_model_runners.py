"""Numerical batches preserve input compatibility and output lifetime."""

import pytest

from uniserve_worker._uniserve_ipc import ModelRunners
from uniserve_worker.errors import WorkerError
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    ForwardMode,
    VisionInput,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.tensor import DType, ShapeBound, TensorRef

pytestmark = pytest.mark.unit


def call(component="model", mode=ForwardMode.PREFILL):
    return Call(
        RequestKey(1, 1, 0),
        CallId(1, 0),
        CallCoordinates(),
        mode,
        Bounds(),
        component,
    )


def test_batches_preserve_row_order_and_outputs_until_their_consumers_run():
    runners = ModelRunners()
    runners.bind("model", (ForwardMode.PREFILL, ForwardMode.DECODE), "text")
    runners.bind("vision", (ForwardMode.PREFILL,), "vision")
    prefill = call()
    decode = call(mode=ForwardMode.DECODE)
    vision = call("vision")
    missing, batches = runners.group(
        (
            (prefill, prefill.kind, tuple, (), True),
            (decode, decode.kind, tuple, (), True),
            (vision, vision.kind, tuple, (16, 16), False),
            (prefill, prefill.kind, tuple, (), True),
            (vision, vision.kind, tuple, (32, 16), False),
            (call("unbound"), prefill.kind, tuple, (), True),
        )
    )
    assert missing == [5]
    assert batches == [
        ("text", (0, 3), True),
        ("text", (1,), False),
        ("vision", (2,), True),
        ("vision", (4,), False),
    ]


def test_context_segments_share_a_batch_and_uniform_modes_stay_separate():
    runners = ModelRunners()
    runners.bind("model", (ForwardMode.PREFILL,), "text")
    uniform = call()
    feature = TensorRef(
        RequestKey(1, 2, 0), CallId(1, 0), 0, 1, DType.F32, ShapeBound()
    )
    context = uniform.replace(vision_inputs=(VisionInput(1, feature),))
    missing, batches = runners.group(
        (
            (context, context.kind, tuple, (), True),
            (uniform, uniform.kind, tuple, (), True),
            (context, context.kind, tuple, (), False),
            (uniform, uniform.kind, tuple, (), False),
            (uniform, uniform.kind, list, (), False),
        )
    )
    assert missing == []
    assert [indexes for _, indexes, _ in batches] == [(0, 2), (1,), (3,), (4,)]


def test_ambiguous_binding_does_not_install_part_of_a_runner():
    runners = ModelRunners()
    runners.bind("model", (ForwardMode.PREFILL,), "first")
    with pytest.raises(WorkerError, match="multiple lane bindings"):
        runners.bind(
            "model", (ForwardMode.DECODE, ForwardMode.PREFILL), "second"
        )
    assert runners.get("model", ForwardMode.PREFILL) == "first"
    assert runners.get("model", ForwardMode.DECODE) is None
    runners.clear()
    assert runners.get("model", ForwardMode.PREFILL) is None
