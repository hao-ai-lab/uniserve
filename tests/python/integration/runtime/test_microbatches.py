"""Numerical microbatch results, stream ordering and host retirement."""

from contextlib import ExitStack
from contextvars import ContextVar

import pytest
import torch

from uniserve.model import TextSize
from uniserve.runtime import CUDAStream, ExecutionContext, Microbatches
from uniserve.runtime.microbatches import yield_microbatch

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture
def numerical_calls():
    device = torch.device("cuda", 0)
    module = torch.nn.Linear(16, 8, bias=False, device=device)
    with torch.inference_mode():
        module.weight.fill_(0.125)
    inputs = torch.ones(2, 16, device=device)
    with ExitStack() as scope:
        contexts = []
        for _ in range(2):
            stream = scope.enter_context(
                CUDAStream.external(torch.cuda.Stream(device=device))
            )
            context = scope.enter_context(
                ExecutionContext(module, stream=stream)
            )
            context.prepare(TextSize(2, 1))
            contexts.append(context)
        run = Microbatches(contexts)
        scope.callback(run.close)
        yield module, inputs, run


def test_failed_microbatch_releases_peers_and_preserves_the_exception(
    numerical_calls,
):
    module, inputs, run = numerical_calls
    failure = ValueError("invalid numerical input")

    def suspended():
        hidden = module(inputs)
        yield_microbatch()
        return hidden + 1

    def invalid():
        raise failure

    with pytest.raises(ValueError) as raised:
        run((suspended, invalid))
    assert raised.value is failure

    # Both host turns retired before the caller received the failure;
    # the same owner can run a later independent numerical invocation.
    outputs = run((suspended, lambda: module(inputs * 2)))
    torch.testing.assert_close(outputs[0].cpu(), torch.full((2, 8), 3.0))
    torch.testing.assert_close(outputs[1].cpu(), torch.full((2, 8), 4.0))


def test_uneven_calls_join_the_caller_stream_and_isolate_contexts(
    numerical_calls,
):
    module, inputs, run = numerical_calls
    request = ContextVar("request", default="unset")
    request.set("caller")

    def first():
        assert request.get() == "caller"
        request.set("first")
        hidden = module(inputs)
        for _ in range(3):
            yield_microbatch()
            hidden = hidden + 1
        return hidden, request.get()

    def second():
        hidden = module(inputs * 2)
        yield_microbatch()
        return hidden, request.get()

    caller = torch.cuda.Stream(device=inputs.device)
    caller.wait_stream(torch.cuda.current_stream(inputs.device))
    with torch.cuda.stream(caller):
        inputs.fill_(5)
        outputs = run((first, second))
        # Copy from the caller stream without a device-wide synchronization.
        # It must see writes made by both borrowed context streams.
        values = [hidden.cpu() for hidden, _ in outputs]

    torch.testing.assert_close(values[0], torch.full((2, 8), 13.0))
    torch.testing.assert_close(values[1], torch.full((2, 8), 20.0))
    assert [label for _, label in outputs] == ["first", "caller"]
    assert request.get() == "caller"


def test_running_and_closed_owners_reject_new_invocations(numerical_calls):
    module, inputs, run = numerical_calls

    def forward():
        return module(inputs)

    def reentrant():
        with pytest.raises(RuntimeError, match="already running"):
            run((forward, forward))
        with pytest.raises(RuntimeError, match="cannot close running"):
            run.close()
        yield_microbatch()
        return forward()

    with pytest.raises(ValueError, match="one numerical call"):
        run((forward,))

    outputs = run((reentrant, forward))
    for output in outputs:
        torch.testing.assert_close(output.cpu(), torch.full((2, 8), 2.0))

    run.close()
    with pytest.raises(RuntimeError, match="closed"):
        run((forward, forward))
