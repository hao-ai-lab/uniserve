"""A worker steps resident generation canvases over the context it cached.

A DiffusionGemma worker prefills a prompt into its paged KV cache, then runs
token-denoising calls that each take one denoising step of the canvas its
request keeps in its slot. A step reports no tokens until it finishes the
block; the finishing step reports the block's tokens, which the engine
commits to the context as a prefill before the next block starts at step
zero. The tokens and the step that finishes each block must equal the
public model driven directly by the block-diffusion sampler of
``uniserve.diffusion.canvas``: a prompt pass that writes a prefix cache,
then per step a canvas pass that reads it with the previous step's
self-conditioning, in FP32 on the CPU with the torch attention backend. A
step may be queued behind the one before it, predicated on that step's
completion, as the engine queues steps; queued behind the finishing step it
is predicated and leaves the canvas as the finishing step left it. A step
that does not continue its slot's canvas, or of a canvas whose sampling is
not the one the worker serves, is refused.
"""

from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.checkpoints import (
    VOCAB,
    diffusion_gemma_checkpoint,
    load_diffusion_gemma,
)
from tests.python.integration.runtime.test_canvas_readout import (
    PAGE,
    PAGES,
    REQUEST,
    SLOT,
    _attention,
    _call,
    _drain,
    _paged,
    _prefill,
    _run,
    _tables,
    _worker,
)
from uniserve.diffusion import canvas as sampler
from uniserve.model import CanvasInput, TextInput, TextSize
from uniserve.nn.attention import (
    BlockTable as Table,
)
from uniserve.nn.attention import (
    SegmentedInput,
    SequenceLengths,
)
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve.runtime.prefix_cache import plan_units
from uniserve.sampling import SamplingParams
from uniserve_worker.protocol.batch import (
    Batch,
    CacheUnitAllocation,
    CanvasSampling,
    Finish,
    GenerationParams,
    NewRequest,
    Start,
)
from uniserve_worker.protocol.call import (
    Bounds,
    CallStatus,
    CanvasStep,
    ForwardMode,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.tensor import DType, ShapeBound, TensorRef

pytestmark = pytest.mark.integration

CANVAS = 16
SEED = 11
ADMITTED = CanvasSampling(
    canvas_length=CANVAS,
    max_steps=4,
    entropy_bound=0.5,
    t_min=0.4,
    t_max=0.8,
    confidence_threshold=0.2,
    stability_threshold=1,
)
# The sampler constants of ``ADMITTED`` with the fixture's generation
# config: end of sequence 1 and 6, padding 0.
SAMPLING = sampler.CanvasSampling(
    steps=4,
    entropy_bound=0.5,
    t_min=0.4,
    t_max=0.8,
    confidence=0.2,
    stability=1,
    eos_ids=(1, 6),
    pad_id=0,
)
# Sampling whose blocks stop at their first step: every argmax canvas is
# stable without history, and every mean entropy is below the threshold.
EARLY = ADMITTED.replace(confidence_threshold=100.0, stability_threshold=0)
EARLY_SAMPLING = replace(SAMPLING, confidence=100.0, stability=0)


def _admission(canvas=ADMITTED):
    return Start(
        NewRequest(
            REQUEST,
            SLOT,
            generation=GenerationParams(
                sampling=SamplingParams(seed=SEED), canvas=canvas
            ),
        )
    )


def _step(batch, context, block, step, predicate=None):
    """One canvas step over the ``context``-token cached prefix.

    The step's completion output reports whether its block continues; a
    queued step is predicated on the ``predicate`` of the step before it.
    """
    call = _row_step(batch, 0, REQUEST, context, block, step, predicate)
    return Batch(
        batch_id=batch,
        collective_seq=batch,
        calls=(call,),
        block_tables=_tables(context),
        forward_call_indices=(0,),
        request_pool_indices=(SLOT,),
        seq_lens=(context + CANVAS,),
        query_lens=(CANVAS,),
        write_kv=(False,),
    )


class _Reference:
    """The public model and sampler stepping one canvas directly."""

    def __init__(self, root, sampling=SAMPLING):
        self.sampling = sampling
        self.model = load_diffusion_gemma(root)
        config = self.model.text.cache_config
        tables = len(plan_units(config, block_size=PAGE).tables)
        self.cache = PrefixCache(
            config, num_units=tables * PAGES, block_size=PAGE, device="cpu"
        )
        self.context = ExecutionContext(
            self.model, cache=self.cache, attention="torch"
        )
        self.cache.__enter__()
        self.context.__enter__()
        self.context.prepare(TextSize(128, 1))
        backbone = self.model.denoiser.backbone
        self.state = sampler.CanvasState.empty(
            1,
            CANVAS,
            backbone.embedding.embedding_dim,
            stability=sampling.stability,
            dtype=backbone.embedding.weight.dtype,
        )

    def close(self):
        self.context.__exit__(None, None, None)
        self.cache.__exit__(None, None, None)

    def extend(self, tokens, start):
        """Write ``tokens`` into the cache causally after ``start``."""
        stop = start + len(tokens)
        batch = _attention(
            self.cache,
            build=lambda units, page: _paged(
                units, page, start=start, stop=stop, causal=True
            ),
        )
        self.context.bind_attention(batch)
        self.model.text(
            TextInput(torch.tensor(tokens), torch.arange(start, stop), batch)
        )

    def step(self, context, block, step):
        """Run step ``step`` of block ``block`` over the cached context."""
        state = self.state
        state.seed.fill_(SEED)
        state.block.fill_(block)
        state.step.fill_(step)
        sampler.start_canvas(state, vocab_size=VOCAB)
        batch = _attention(
            self.cache,
            build=lambda units, page: SegmentedInput(
                SequenceLengths.from_lengths((CANVAS,), device="cpu"),
                SequenceLengths.from_lengths((context,), device="cpu"),
                Table(torch.tensor([units], dtype=torch.int32), page),
                None,
                torch.full((1, CANVAS), CANVAS, dtype=torch.int32),
                True,
            ),
        )
        self.context.bind_attention(batch)
        denoiser = self.model.denoiser
        hidden = denoiser(
            CanvasInput(
                state.canvas.view(-1),
                torch.arange(context, context + CANVAS),
                batch,
                self_conditioning=state.self_conditioning,
            )
        )
        logits = denoiser.compute_logits(
            hidden, token_indices=torch.arange(CANVAS)
        ).gather()
        decision = sampler.CanvasDecision.empty(1, CANVAS)
        sampler.denoise_canvas(
            logits.view(1, CANVAS, VOCAB),
            denoiser.backbone.embedding.weight,
            denoiser.backbone.embedding_scale,
            state,
            self.sampling,
            scores=sampler.CanvasScores.empty(1, CANVAS),
            decision=decision,
            workspace=sampler.CanvasWorkspace.empty(
                1,
                CANVAS,
                VOCAB,
                state.self_conditioning.shape[1],
                dtype=state.self_conditioning.dtype,
            ),
        )
        done = bool(decision.finished[0, 0])
        return tuple(decision.tokens[0].tolist()) if done else ()


def _block(run, context, block, first_batch):
    """Step block ``block`` until it finishes; return its steps' tokens."""
    reported = []
    for step in range(ADMITTED.max_steps):
        tokens = run(first_batch + step, context, block, step)
        reported.append(tokens)
        if tokens:
            break
    return reported


def _expected_blocks(root, prompt, sampling=SAMPLING):
    """The reference's steps' tokens of two blocks after ``prompt``."""
    reference = _Reference(root, sampling)
    try:
        with torch.no_grad():
            reference.extend(prompt, 0)
            expected = _block(
                lambda _batch, context, block, step: reference.step(
                    context, block, step
                ),
                13,
                0,
                0,
            )
            reference.extend(list(expected[-1]), 13)
            expected += _block(
                lambda _batch, context, block, step: reference.step(
                    context, block, step
                ),
                13 + CANVAS,
                1,
                0,
            )
    finally:
        reference.close()
    return expected


def _admitted_prompt(worker, prompt, canvas=ADMITTED):
    """Admit the request with ``canvas`` sampling and prefill ``prompt``."""
    prefill = _prefill(1, 0, tokens=prompt).replace(
        commands=(_admission(canvas),),
        block_tables=_tables(13 + 2 * CANVAS),
        new_cache_units=tuple(
            CacheUnitAllocation(SLOT, table.group_id, table.unit_ids)
            for table in _tables(13 + 2 * CANVAS)
        ),
    )
    assert _run(worker, prefill).completions[0].status is CallStatus.OK


def test_canvas_steps_follow_the_public_model_and_sampler(tmp_path):
    """Two blocks, the second over the committed first, match the reference.

    Every step before a block's finishing step reports no tokens, the
    finishing step reports the block, and the step counts agree.
    """
    diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(37)
    prompt = torch.randint(7, 58, (13,), generator=generator).tolist()
    expected = _expected_blocks(tmp_path, prompt)

    worker = _worker(tmp_path, canvas_sampling=ADMITTED)
    with worker:
        _admitted_prompt(worker, prompt)

        def run(batch, context, block, step):
            (record,) = _run(
                worker, _step(batch, context, block, step)
            ).completions
            assert record.status is CallStatus.OK
            assert (record.position, record.kv_visible_len) == (
                context,
                context,
            )
            return tuple(record.committed_tokens)

        actual = _block(run, 13, 0, 2)
        commit = _prefill(10, 13, tokens=list(actual[-1])).replace(
            block_tables=_tables(13 + 2 * CANVAS),
        )
        assert _run(worker, commit).completions[0].kv_visible_len == 13 + CANVAS
        actual += _block(run, 13 + CANVAS, 1, 11)
        _run(worker, Batch(batch_id=20, commands=(Finish(REQUEST),)))

    assert actual == expected
    # Each block ends with the one step that reports all of its tokens.
    assert [len(tokens) for tokens in expected if tokens] == [CANVAS, CANVAS]


def _queued_block(worker, context, block, first_batch):
    """Step block ``block`` with each step queued behind the one in flight.

    Each step after the first is submitted before the step it follows
    reports, predicated on that step's completion. Returns the records of
    every step through the finishing one, then the record of the step
    queued behind it, if the step limit allows one.
    """
    calls, submissions, records = [], [], []

    def submit(step):
        predicate = calls[-1].completion_output if calls else None
        batch = _step(first_batch + step, context, block, step, predicate)
        calls.append(batch.calls[0])
        submissions.append(worker.submit(batch))

    submit(0)
    step = 0
    while True:
        if step + 1 < EARLY.max_steps:
            submit(step + 1)
        (record,) = _drain(worker, submissions[step]).completions
        records.append(record)
        if record.committed_tokens or step + 1 == EARLY.max_steps:
            break
        step += 1
    if len(submissions) > len(records):
        records.append(_drain(worker, submissions[-1]).completions[0])
    return records


def test_queued_steps_give_the_blocks_of_steps_that_wait(tmp_path):
    """Steps queued back to back match the reference.

    With sampling whose blocks stop at their first step, the steps through
    each block's finishing one report as the reference steps do, and the
    step queued behind the finishing one reports predicated, with no
    tokens: it does not run the finished block again. The next block, over
    the committed first, still matches the reference.
    """
    diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(37)
    prompt = torch.randint(7, 58, (13,), generator=generator).tolist()
    expected = _expected_blocks(tmp_path, prompt, EARLY_SAMPLING)

    worker = _worker(tmp_path, queue_depth=2, canvas_sampling=EARLY)
    actual, queued = [], []
    with worker:
        _admitted_prompt(worker, prompt, EARLY)
        batch = 2
        for block in (0, 1):
            context = 13 + block * CANVAS
            records = _queued_block(worker, context, block, batch)
            batch += len(records)
            finishing = next(
                index
                for index, record in enumerate(records)
                if record.committed_tokens
            )
            for record in records[: finishing + 1]:
                assert record.status is CallStatus.OK
                actual.append(tuple(record.committed_tokens))
            queued.extend(records[finishing + 1 :])
            if block == 0:
                commit = _prefill(batch, 13, tokens=list(actual[-1])).replace(
                    block_tables=_tables(13 + 2 * CANVAS),
                )
                assert (
                    _run(worker, commit).completions[0].kv_visible_len
                    == 13 + CANVAS
                )
                batch += 1
        _run(worker, Batch(batch_id=batch, commands=(Finish(REQUEST),)))

    assert actual == expected
    assert len(queued) == 2
    for record in queued:
        assert record.status is CallStatus.PREDICATED
        assert not record.committed_tokens


def _first_step_status(root, admitted, step):
    """Admit with ``admitted`` sampling; the status of canvas step ``step``.

    The worker serves ``ADMITTED``.
    """
    worker = _worker(root, canvas_sampling=ADMITTED)
    with worker:
        prefill = _prefill(1, 0, tokens=list(range(7, 20))).replace(
            commands=(_admission(admitted),),
            new_cache_units=tuple(
                CacheUnitAllocation(SLOT, table.group_id, table.unit_ids)
                for table in _tables(13)
            ),
        )
        _run(worker, prefill)
        (record,) = _run(worker, _step(2, 13, 0, step)).completions
        _run(worker, Batch(batch_id=3, commands=(Finish(REQUEST),)))
    return record.status


def test_a_step_that_skips_its_canvas_is_refused(tmp_path):
    """A slot's canvas starts at step zero and advances one step per call."""
    diffusion_gemma_checkpoint(tmp_path)
    assert _first_step_status(tmp_path, ADMITTED, 1) is CallStatus.ERROR


def test_a_canvas_with_other_sampling_is_refused(tmp_path):
    """A worker steps only canvases of the sampling it serves."""
    diffusion_gemma_checkpoint(tmp_path)
    other = ADMITTED.replace(max_steps=ADMITTED.max_steps + 1)
    assert _first_step_status(tmp_path, other, 0) is CallStatus.ERROR
    assert _first_step_status(tmp_path, ADMITTED, 0) is CallStatus.OK


# A second request, in its own slot and cache units.
SECOND = RequestKey(0, 8, 1)
SECOND_SLOT = 2


def _second_tables(tokens):
    """Each group's table of the second request's slot, covering ``tokens``.

    Group ``g`` holds units ``33 + 64 * g ...``, clear of the first
    request's units.
    """
    return tuple(
        replace(
            table,
            request_pool_idx=SECOND_SLOT,
            unit_ids=tuple(unit + 32 for unit in table.unit_ids),
        )
        for table in _tables(tokens)
    )


def _row_step(batch, row, request, context, block, step, predicate=None):
    """Row ``row`` of batch ``batch``: request ``request``'s canvas step."""
    return _call(
        batch,
        ForwardMode.TOKEN_DENOISING,
        context,
        bounds=Bounds(max_tokens=CANVAS, max_completion_bytes=4 * CANVAS),
        canvas=CanvasStep(block, step),
        completion_output=TensorRef(
            request_key=request,
            producer_call_id=CallId(batch, row),
            output_index=0,
            generation=batch,
            dtype=DType.U8,
            shape_bound=ShapeBound(),
        ),
        predicate=predicate,
    ).replace(request_key=request, call_id=CallId(batch, row))


def _steps_batch(batch, rows):
    """One batch of canvas steps, one row per entry of ``rows``.

    Each entry holds a row's call, request slot, block tables and cached
    context length, in row order.
    """
    return Batch(
        batch_id=batch,
        collective_seq=batch,
        calls=tuple(call for call, _slot, _tables, _context in rows),
        block_tables=tuple(
            table for _call, _slot, tables, _context in rows for table in tables
        ),
        forward_call_indices=tuple(range(len(rows))),
        request_pool_indices=tuple(slot for _call, slot, _tables, _c in rows),
        seq_lens=tuple(context + CANVAS for _call, _s, _t, context in rows),
        query_lens=(CANVAS,) * len(rows),
        write_kv=(False,) * len(rows),
    )


def test_a_batch_of_canvas_steps_reports_each_rows_outcome(tmp_path):
    """Rows stepped in one batch each report their own outcome.

    The first request's block stops at its last step in the same batch as
    the second request's block takes its first step. The stopped row, the
    batch's second, reports its block as the reference does; the other row
    reports no tokens. Each row's completion carries its own outcome: a
    call predicated on the stopped row's completion is skipped, and the
    continuing request's next step, predicated on its row's completion,
    runs.
    """
    diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(37)
    first_prompt = torch.randint(7, 58, (13,), generator=generator).tolist()
    second_prompt = torch.randint(7, 58, (13,), generator=generator).tolist()
    expected = _expected_blocks(tmp_path, first_prompt)
    stop = ADMITTED.max_steps - 1
    assert [bool(tokens) for tokens in expected[: stop + 1]] == [
        False
    ] * stop + [True]

    worker = _worker(tmp_path, canvas_sampling=ADMITTED)
    with worker:
        _admitted_prompt(worker, first_prompt)
        second = _prefill(2, 0, tokens=second_prompt)
        admitted = second.replace(
            calls=(second.calls[0].replace(request_key=SECOND),),
            commands=(
                Start(
                    NewRequest(
                        SECOND,
                        SECOND_SLOT,
                        generation=GenerationParams(
                            sampling=SamplingParams(seed=SEED), canvas=ADMITTED
                        ),
                    )
                ),
            ),
            block_tables=_second_tables(13 + CANVAS),
            new_cache_units=tuple(
                CacheUnitAllocation(SECOND_SLOT, table.group_id, table.unit_ids)
                for table in _second_tables(13 + CANVAS)
            ),
            request_pool_indices=(SECOND_SLOT,),
        )
        assert _run(worker, admitted).completions[0].status is CallStatus.OK

        # The first request steps alone up to its block's last step.
        for step in range(stop):
            (record,) = _run(worker, _step(3 + step, 13, 0, step)).completions
            assert record.status is CallStatus.OK
            assert not record.committed_tokens

        batch = 3 + stop
        continuing = _row_step(batch, 0, SECOND, 13, 0, 0)
        stopping = _row_step(batch, 1, REQUEST, 13, 0, stop)
        report = _run(
            worker,
            _steps_batch(
                batch,
                (
                    (continuing, SECOND_SLOT, _second_tables(13 + CANVAS), 13),
                    (stopping, SLOT, _tables(13 + CANVAS), 13),
                ),
            ),
        )
        records = {record.request_key: record for record in report.completions}
        assert records[SECOND].status is CallStatus.OK
        assert not records[SECOND].committed_tokens
        assert records[REQUEST].status is CallStatus.OK
        assert tuple(records[REQUEST].committed_tokens) == expected[stop]

        # A commit gated on the stopped row's completion is skipped; the
        # continuing request's next step, gated on its row's, runs.
        gated_commit = _prefill(batch + 1, 13, tokens=list(expected[stop]))
        (record,) = _run(
            worker,
            gated_commit.replace(
                calls=(
                    gated_commit.calls[0].replace(
                        predicate=stopping.completion_output,
                    ),
                ),
                block_tables=_tables(13 + CANVAS),
            ),
        ).completions
        assert record.status is CallStatus.PREDICATED
        next_step = _row_step(
            batch + 2,
            0,
            SECOND,
            13,
            0,
            1,
            predicate=continuing.completion_output,
        )
        (record,) = _run(
            worker,
            _steps_batch(
                batch + 2,
                ((next_step, SECOND_SLOT, _second_tables(13 + CANVAS), 13),),
            ),
        ).completions
        assert record.status is CallStatus.OK
        assert not record.committed_tokens
        _run(
            worker,
            Batch(
                batch_id=batch + 3,
                commands=(Finish(REQUEST), Finish(SECOND)),
            ),
        )
