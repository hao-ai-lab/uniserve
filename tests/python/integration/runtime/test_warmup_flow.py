"""Public worker startup fits guided flow within the admitted request slots."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve.processing import FlowPrompt
from uniserve_worker.config.execution import WorkerConfig

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# With a flow prompt, every guidance branch that does not reuse the request's
# conditioning reads a prefix of its own, which takes a request slot.
FLOW_PROMPT = FlowPrompt(
    user_prefix="<user>",
    user_suffix="</user>",
    assistant_suffix="<assistant>",
    conditioned_append="<image>",
    unconditional_append="<image>",
)


class _Tokenizer:
    """The external tokenizer: every framed prompt encodes to three ids."""

    def encode(self, text, *, add_special_tokens):
        return [5, 6, 7]


@pytest.mark.parametrize(
    ("request_slots", "batch_sizes"),
    # The largest capture batch holds more guided requests than there are
    # request slots for them and their guidance prefixes.
    [(8, (1, 2, 5)), (6, (1, 4))],
)
def test_guided_flow_warmup_completes_within_the_request_slots(
    request_slots, batch_sizes
):
    worker = execution_worker(
        device="cuda:0",
        max_request_pool_size=request_slots,
        execution=WorkerConfig(
            graph_policy="off",
            prefill_cuda_graph=False,
            flow_graph_batch_sizes=batch_sizes,
            flow_graph_shapes=((16, 16),),
        ),
        flow_prompt=FLOW_PROMPT,
        tokenizer=_Tokenizer(),
    )
    try:
        assert worker.info.request_slots == request_slots
        worker.warmup()

        assert worker.requests.request_ids() == ()
    finally:
        worker.close()
