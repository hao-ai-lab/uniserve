"""A batch that retires a request answers without waiting for later batches.

A DiffusionGemma worker on one CUDA device runs batches ahead: a batch is
launched while the batches before it still run. The result of a batch that
carries a ``Finish`` acknowledges the finished request's release, and must be
delivered once that batch's own device work is done, while device work
queued after it still runs. A request admitted into the released slot right
after that acknowledgement, with later work still queued, must read the same
candidate log-probabilities as a request admitted into a fresh slot.
"""

import math
import time

import pytest
import torch

from tests.python.fixtures.checkpoints import (
    diffusion_gemma_checkpoint,
    load_diffusion_gemma,
)
from tests.python.fixtures.worker_config import stub_worker_config
from uniserve.distributed.mesh import Communicator
from uniserve.sampling import SamplingParams
from uniserve_models import loading as models
from uniserve_worker.protocol.batch import (
    Batch,
    BlockTable,
    CacheUnitAllocation,
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
    ForwardMode,
    Readout,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.worker import Worker

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# Tokens per page in both cache groups, the sliding-window and the full one.
PAGE = 4
# Units each request slot owns in each cache group.
SLOT_UNITS = 16
CANVAS = [4, 11, 4, 12, 4, 13, 6, 0, 0]
SLOTS = ((0, (30, 31)), (2, (40, 41, 42)), (4, (50,)))
# GPU cycles of the kernels that hold the device before the retiring batch
# and stand in for long work queued behind it (about half a second and a
# second on current devices).
EARLIER_WORK_CYCLES = 1_000_000_000
LATER_WORK_CYCLES = 2_000_000_000


def _worker(root):
    """A worker on ``cuda:0`` that holds up to three batches at once."""
    loaded = models.read_config(root)
    config = stub_worker_config(PAGE, max_batch_tokens=256).replace(
        device="cuda:0",
        model_dtype="bfloat16",
        max_sequence_tokens=128,
        graph_policy="off",
        prefill_cuda_graph=False,
    )
    return Worker(
        # CUDA expert kernels take BF16 weights.
        load_diffusion_gemma(root).to("cuda:0", dtype=torch.bfloat16),
        worker_config=config,
        image_processor=loaded.image_processor,
        sampling_group=Communicator(device=torch.device("cuda:0")),
        tokenizer=None,
        allowed_calls=None,
        queue_depth=3,
        completion_payload_bytes=1 << 16,
        attention="torch",
        host_slots=(0, 1),
    )


def _drain(worker, submission):
    """Drive the worker until a submitted batch's report is ready."""
    deadline = time.monotonic() + 120.0
    while True:
        worker.advance()
        report = worker.poll(submission)
        if report is not None:
            return report
        if time.monotonic() >= deadline:
            raise TimeoutError("worker batch did not complete")
        time.sleep(0.0001)


def _tables(slot, tokens):
    """Each group's table of ``slot``, covering ``tokens`` on its own units."""
    pages = math.ceil(tokens / PAGE)
    return tuple(
        BlockTable(
            slot,
            group,
            0,
            tuple(
                range(
                    1 + 64 * group + SLOT_UNITS * slot,
                    1 + 64 * group + SLOT_UNITS * slot + pages,
                )
            ),
            pages * PAGE,
        )
        for group in (0, 1)
    )


def _prefill(batch, request, slot, prompt, *, finish=()):
    """Admit ``request`` into ``slot`` and prefill its whole prompt.

    The batch also carries a ``Finish`` for each request key in ``finish``.
    """
    count = len(prompt)
    call = Call(
        request_key=request,
        call_id=CallId(batch, 0),
        coordinates=CallCoordinates(0, 0, 0),
        kind=ForwardMode.PREFILL,
        bounds=Bounds(max_tokens=count),
        input_token_ids=tuple(prompt),
    )
    tables = _tables(slot, count)
    return Batch(
        batch_id=batch,
        collective_seq=batch,
        calls=(call,),
        commands=(
            Start(
                NewRequest(
                    request,
                    slot,
                    generation=GenerationParams(sampling=SamplingParams()),
                )
            ),
            *(Finish(key) for key in finish),
        ),
        block_tables=tables,
        new_cache_units=tuple(
            CacheUnitAllocation(slot, table.group_id, table.unit_ids)
            for table in tables
        ),
        forward_call_indices=(0,),
        request_pool_indices=(slot,),
        seq_lens=(count,),
        query_lens=(count,),
        write_kv=(True,),
    )


def _readout(batch, request, slot, prompt_length):
    """One canvas read over the request's cached ``prompt_length`` tokens."""
    slot_tokens, offsets, candidates = [], [0], []
    for position, ids in SLOTS:
        slot_tokens.append(position)
        candidates.extend(ids)
        offsets.append(len(candidates))
    call = Call(
        request_key=request,
        call_id=CallId(batch, 0),
        coordinates=CallCoordinates(
            prompt_length, prompt_length, prompt_length
        ),
        kind=ForwardMode.TOKEN_DENOISING,
        bounds=Bounds(
            max_tokens=len(CANVAS), max_completion_bytes=4 * len(candidates)
        ),
        input_token_ids=tuple(CANVAS),
        readout=Readout(tuple(slot_tokens), tuple(offsets), tuple(candidates)),
    )
    return Batch(
        batch_id=batch,
        collective_seq=batch,
        calls=(call,),
        block_tables=_tables(slot, prompt_length),
        forward_call_indices=(0,),
        request_pool_indices=(slot,),
        seq_lens=(prompt_length + len(CANVAS),),
        query_lens=(len(CANVAS),),
        write_kv=(False,),
    )


def _logprobs(report):
    (record,) = report.completions
    assert record.status is CallStatus.OK
    return torch.tensor(record.candidate_logprobs)


def test_a_retiring_batch_answers_before_later_work_and_its_slot_is_reusable(
    tmp_path,
):
    diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(41)
    prompt = torch.randint(7, 58, (13,), generator=generator).tolist()
    first = RequestKey(0, 7, 1)
    second = RequestKey(0, 8, 1)
    third = RequestKey(0, 9, 1)
    reused = RequestKey(0, 10, 1)
    stream = torch.cuda.current_stream(torch.device("cuda:0"))

    worker = _worker(tmp_path)
    with worker:
        _drain(worker, worker.submit(_prefill(1, first, 1, prompt)))
        expected = _logprobs(
            _drain(worker, worker.submit(_readout(2, first, 1, len(prompt))))
        )

        # One retirement runs with nothing queued behind it, so the kernels
        # it first launches, whose loading orders the whole device, are
        # loaded before the measured retirement.
        _drain(
            worker,
            worker.submit(_prefill(3, second, 2, prompt, finish=(first,))),
        )
        warm = _logprobs(
            _drain(worker, worker.submit(_readout(4, second, 2, len(prompt))))
        )

        # The device is held first, so the retiring batch is still queued
        # when the batch behind it and a long stretch of later device work
        # are submitted, as when a worker runs batches ahead.
        torch.cuda._sleep(EARLIER_WORK_CYCLES)
        retiring = worker.submit(
            _prefill(5, third, 3, prompt, finish=(second,))
        )
        later = worker.submit(_readout(6, third, 3, len(prompt)))
        torch.cuda._sleep(LATER_WORK_CYCLES)

        report = _drain(worker, retiring)
        later_work_running = not stream.query()
        assert report.completions[0].status is CallStatus.OK

        # A request admitted into the released slot while the later work
        # still runs reads its own prompt, not the retired request's state.
        readmitted = worker.submit(_prefill(7, reused, 2, prompt))
        reused_readout = worker.submit(_readout(8, reused, 2, len(prompt)))
        queued = _logprobs(_drain(worker, later))
        _drain(worker, readmitted)
        reused_values = _logprobs(_drain(worker, reused_readout))

    assert later_work_running
    torch.testing.assert_close(warm, expected, rtol=0, atol=0)
    torch.testing.assert_close(queued, expected, rtol=0, atol=0)
    torch.testing.assert_close(reused_values, expected, rtol=0, atol=0)
