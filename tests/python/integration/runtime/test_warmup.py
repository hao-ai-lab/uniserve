"""Public startup retires its requests before serving reuses their storage."""

from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.checkpoints import diffusion_gemma_checkpoint
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.integration.runtime.test_canvas_readout import _run
from uniserve.distributed import Communicator
from uniserve.loading import weights
from uniserve.processing import FlowPrompt
from uniserve.sampling import SamplingParams
from uniserve_models import loading as models
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.protocol.batch import (
    Batch,
    BlockTable,
    CacheUnitAllocation,
    CanvasSampling,
    Finish,
    GenerationParams,
    NewRequest,
    Start,
)
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    CallStatus,
    CanvasStep,
    ForwardMode,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.worker import Worker

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


@torch.inference_mode()
def test_canvas_warmup_preserves_the_first_served_block(tmp_path):
    diffusion_gemma_checkpoint(
        tmp_path, text={"vocab_size": 4096}, canvas_length=32
    )
    source = models.read_config(tmp_path)
    sampling = CanvasSampling(
        canvas_length=32,
        max_steps=2,
        entropy_bound=0.5,
        t_min=0.4,
        t_max=0.8,
        confidence_threshold=0.2,
        stability_threshold=1,
    )
    request = RequestKey(0, 1, 1)
    results = []

    for warmup in (False, True):
        model = models.load_model(
            source,
            device="cuda:0",
            weights=weights.Config(dtype=torch.bfloat16),
        ).model
        config = WorkerConfig(
            device="cuda:0",
            model_dtype="bfloat16",
            graph_policy="off",
            prefill_cuda_graph=False,
            block_size=16,
            kv_token_capacity=256,
            max_request_pool_size=4,
            max_batch_calls=4,
            max_batch_tokens=128,
            max_sequence_tokens=128,
            canvas_sampling=sampling,
        )
        with Worker(
            model,
            worker_config=config,
            sampling_group=Communicator(device=torch.device("cuda:0")),
            tokenizer=None,
            allowed_calls=None,
            queue_depth=1,
            completion_payload_bytes=1 << 16,
            attention="torch",
        ) as worker:
            if warmup:
                worker.warmup()

            # Reuse startup's first request slot, KV units and batch number.
            # Its canvas history and RNG must not affect the served request.
            assert worker.requests.request_ids() == ()
            tables = []
            unit = 1
            for group, shape in enumerate(worker.info.kv_cache.groups):
                units = tuple(range(unit, unit + shape.units_per_page))
                tables.append(BlockTable(1, group, 0, units, shape.page_tokens))
                unit += len(units)
            prompt = Call(
                request_key=request,
                call_id=CallId(1, 0),
                kind=ForwardMode.PREFILL,
                coordinates=CallCoordinates(),
                bounds=Bounds(max_tokens=3),
                input_token_ids=(7, 8, 9),
            )
            admitted = NewRequest(
                request,
                1,
                generation=GenerationParams(
                    sampling=SamplingParams(seed=11), canvas=sampling
                ),
            )
            batch = Batch(
                batch_id=1,
                calls=(prompt,),
                commands=(Start(admitted),),
                block_tables=tuple(tables),
                new_cache_units=tuple(
                    CacheUnitAllocation(1, table.group_id, table.unit_ids)
                    for table in tables
                ),
                forward_call_indices=(0,),
                request_pool_indices=(1,),
                seq_lens=(3,),
                query_lens=(3,),
                write_kv=(True,),
            )
            assert _run(worker, batch).completions[0].status is CallStatus.OK

            for step in range(sampling.max_steps):
                call = prompt.replace(
                    call_id=CallId(step + 2, 0),
                    kind=ForwardMode.TOKEN_DENOISING,
                    coordinates=CallCoordinates(3, 3, 3, 0),
                    bounds=Bounds(max_tokens=32, max_completion_bytes=128),
                    input_token_ids=(),
                    canvas=CanvasStep(0, step),
                )
                (result,) = _run(
                    worker,
                    batch.replace(
                        batch_id=step + 2,
                        calls=(call,),
                        commands=(),
                        new_cache_units=(),
                        seq_lens=(35,),
                        query_lens=(32,),
                        write_kv=(False,),
                    ),
                ).completions
                assert result.status is CallStatus.OK
                if result.committed_tokens:
                    results.append(result.committed_tokens)
                    break

            _run(worker, Batch(batch_id=4, commands=(Finish(request),)))

    assert len(results) == 2
    assert len(results[0]) == 32
    assert results[0] == results[1]
