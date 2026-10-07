"""Numerical microbatches retire suspended calls after a host failure."""

from contextlib import ExitStack

import pytest
import torch

from uniserve.model import TextSize
from uniserve.runtime import CUDAStream, ExecutionContext, Microbatches
from uniserve.runtime.microbatches import yield_microbatch

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
def test_failed_microbatch_releases_peers_and_preserves_the_exception():
    device = torch.device("cuda", 0)
    module = torch.nn.Linear(16, 8, bias=False, device=device)
    module.weight.fill_(0.125)
    inputs = torch.ones(2, 16, device=device)
    failure = ValueError("invalid numerical input")
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
        torch.cuda.synchronize(device)
        torch.testing.assert_close(
            outputs[0], torch.full((2, 8), 3.0, device=device)
        )
        torch.testing.assert_close(
            outputs[1], torch.full((2, 8), 4.0, device=device)
        )
