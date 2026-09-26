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
step that does not continue its slot's canvas is refused.
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


def _admission():
    return Start(
        NewRequest(
            REQUEST,
            SLOT,
            generation=GenerationParams(
                sampling=SamplingParams(seed=SEED), canvas=ADMITTED
            ),
        )
    )


def _step(batch, context, block, step):
    """One canvas step over the ``context``-token cached prefix."""
    call = _call(
        batch,
        ForwardMode.TOKEN_DENOISING,
        context,
        bounds=Bounds(max_tokens=CANVAS, max_completion_bytes=4 * CANVAS),
        canvas=CanvasStep(block, step),
    )
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

    def __init__(self, root):
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
            stability=SAMPLING.stability,
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
            SAMPLING,
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
    for step in range(SAMPLING.steps):
        tokens = run(first_batch + step, context, block, step)
        reported.append(tokens)
        if tokens:
            break
    return reported


def test_canvas_steps_follow_the_public_model_and_sampler(tmp_path):
    """Two blocks, the second over the committed first, match the reference.

    Every step before a block's finishing step reports no tokens, the
    finishing step reports the block, and the step counts agree.
    """
    diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(37)
    prompt = torch.randint(7, 58, (13,), generator=generator).tolist()

    reference = _Reference(tmp_path)
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

    worker = _worker(
        tmp_path, canvas_history_depth=ADMITTED.stability_threshold
    )
    with worker:
        prefill = replace(
            _prefill(1, 0, tokens=prompt),
            commands=(_admission(),),
            block_tables=_tables(13 + 2 * CANVAS),
            new_cache_units=tuple(
                CacheUnitAllocation(SLOT, table.group_id, table.unit_ids)
                for table in _tables(13 + 2 * CANVAS)
            ),
        )
        assert _run(worker, prefill).completions[0].status is CallStatus.OK

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
        commit = replace(
            _prefill(10, 13, tokens=list(actual[-1])),
            block_tables=_tables(13 + 2 * CANVAS),
        )
        assert _run(worker, commit).completions[0].kv_visible_len == 13 + CANVAS
        actual += _block(run, 13 + CANVAS, 1, 11)
        _run(worker, Batch(batch_id=20, commands=(Finish(REQUEST),)))

    assert actual == expected
    # Each block ends with the one step that reports all of its tokens.
    assert [len(tokens) for tokens in expected if tokens] == [CANVAS, CANVAS]


def test_a_step_that_skips_its_canvas_is_refused(tmp_path):
    """A slot's canvas starts at step zero and advances one step per call."""
    diffusion_gemma_checkpoint(tmp_path)
    worker = _worker(
        tmp_path, canvas_history_depth=ADMITTED.stability_threshold
    )
    with worker:
        prefill = replace(
            _prefill(1, 0, tokens=list(range(7, 20))),
            commands=(_admission(),),
            new_cache_units=tuple(
                CacheUnitAllocation(SLOT, table.group_id, table.unit_ids)
                for table in _tables(13)
            ),
        )
        _run(worker, prefill)
        (record,) = _run(worker, _step(2, 13, 0, 1)).completions
        _run(worker, Batch(batch_id=3, commands=(Finish(REQUEST),)))

    assert record.status is CallStatus.ERROR
