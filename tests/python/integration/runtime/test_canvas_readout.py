"""A worker reads canvas candidates over the prompt it cached.

A DiffusionGemma worker prefills a prompt into its paged KV cache, then
serves a token-denoising call: each canvas row attends to the cached prompt
and to its own tokens without writing the cache, and the call reports each
answer slot's candidate log-probabilities under the full-vocabulary
softmax. The reported values must equal the public model evaluated
directly, a prompt pass that writes a prefix cache followed by a canvas pass
that reads it, in FP32 on the CPU with the torch attention backend. Image
soft tokens enter the prompt between its image markers.
"""

import base64
import io
import logging
import math
import threading
import time
from dataclasses import replace
from functools import partial

import pytest
import torch
from PIL import Image

from tests.python.fixtures.checkpoints import (
    BEGIN_IMAGE,
    END_IMAGE,
    VOCAB,
    diffusion_gemma_checkpoint,
    load_diffusion_gemma,
)
from tests.python.fixtures.worker_config import stub_worker_config
from uniserve.distributed.mesh import Communicator
from uniserve.model import (
    CanvasInput,
    EmbeddingReplacement,
    TextInput,
    TextSize,
    VisionInput,
)
from uniserve.nn.attention import (
    AttentionBatch,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.nn.attention import (
    BlockTable as Table,
)
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve.runtime.prefix_cache import plan_units
from uniserve.sampling import SamplingParams
from uniserve_models import loading as models
from uniserve_worker.model_executor.image_inputs import prepare_image
from uniserve_worker.protocol.batch import (
    Batch,
    BlockTable,
    BufferAllocation,
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
    ErrorCode,
    ForwardMode,
    MediaCall,
    Readout,
)
from uniserve_worker.protocol.call import VisionInput as VisionBlock
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.tensor import (
    DeviceDim,
    DType,
    ShapeBound,
    TensorRef,
)
from uniserve_worker.worker import Worker

pytestmark = pytest.mark.integration

# Tokens per page in both cache groups, the sliding-window and the full one.
PAGE = 4
# Reference cache pages per numerical table.
PAGES = 32
SLOT = 1
REQUEST = RequestKey(0, 7, 1)


def _worker(root, queue_depth=1, **settings):
    """A CPU worker serving the checkpoint at ``root``.

    It holds up to ``queue_depth`` batches at once; ``settings`` replace
    further ``WorkerConfig`` fields.
    """
    loaded = models.read_config(root)
    config = replace(
        stub_worker_config(PAGE, max_batch_tokens=256),
        model_dtype="float32",
        max_sequence_tokens=128,
        **settings,
    )
    return Worker(
        load_diffusion_gemma(root),
        worker_config=config,
        image_processor=loaded.image_processor,
        sampling_group=Communicator(device=torch.device("cpu")),
        tokenizer=None,
        allowed_calls=None,
        queue_depth=queue_depth,
        completion_payload_bytes=1 << 16,
        attention="torch",
        host_slots=(0, 1),
    )


def _run(worker, batch):
    """Submit one batch and drive the worker until its report is ready."""
    return _drain(worker, worker.submit(batch))


def _drain(worker, submission):
    """Drive the worker until a submitted batch's report is ready."""
    deadline = time.monotonic() + 60.0
    while True:
        worker.advance()
        report = worker.poll(submission)
        if report is not None:
            return report
        if time.monotonic() >= deadline:
            raise TimeoutError("worker batch did not complete")
        time.sleep(0.0001)


def _tables(tokens):
    """Each group's table of the request slot, covering ``tokens``.

    Group ``g`` owns units ``1 + 64 * g ...`` of the shared pool.
    """
    pages = math.ceil(tokens / PAGE)
    return tuple(
        BlockTable(
            SLOT,
            group,
            0,
            tuple(range(1 + 64 * group, 1 + 64 * group + pages)),
            pages * PAGE,
        )
        for group in (0, 1)
    )


def _call(batch, kind, start, **fields):
    """A call of batch ``batch`` entering at prompt position ``start``."""
    return Call(
        request_key=REQUEST,
        call_id=CallId(batch, 0),
        coordinates=CallCoordinates(start, start, start),
        kind=kind,
        **fields,
    )


def _prefill(batch, start, *, tokens=()):
    """A prefill batch writing prompt tokens after ``start``."""
    count = len(tokens)
    call = _call(
        batch,
        ForwardMode.PREFILL,
        start,
        bounds=Bounds(max_tokens=count),
        input_token_ids=tuple(tokens),
    )
    return Batch(
        batch_id=batch,
        collective_seq=batch,
        calls=(call,),
        block_tables=_tables(start + count),
        forward_call_indices=(0,),
        request_pool_indices=(SLOT,),
        seq_lens=(start + count,),
        query_lens=(count,),
        write_kv=(True,),
    )


def _readout(batch, prompt, canvases, slots):
    """A token-denoising batch over the ``prompt``-token cached prefix.

    ``slots`` lists each canvas's ``(position, candidates)`` pairs.
    """
    tokens, slot_tokens, offsets, candidates = [], [], [0], []
    for canvas, row in zip(canvases, slots, strict=True):
        for position, ids in row:
            slot_tokens.append(len(tokens) + position)
            candidates.extend(ids)
            offsets.append(len(candidates))
        tokens.extend(canvas)
    call = _call(
        batch,
        ForwardMode.TOKEN_DENOISING,
        prompt,
        bounds=Bounds(
            max_tokens=len(tokens), max_completion_bytes=4 * len(candidates)
        ),
        input_token_ids=tuple(tokens),
        readout=Readout(tuple(slot_tokens), tuple(offsets), tuple(candidates)),
    )
    return Batch(
        batch_id=batch,
        collective_seq=batch,
        calls=(call,),
        block_tables=_tables(prompt),
        forward_call_indices=(0,) * len(canvases),
        request_pool_indices=(SLOT,) * len(canvases),
        seq_lens=tuple(prompt + len(canvas) for canvas in canvases),
        query_lens=tuple(len(canvas) for canvas in canvases),
        write_kv=(False,) * len(canvases),
    )


def _completion(report):
    (record,) = report.completions
    assert record.status is CallStatus.OK
    return record


def _admission(input_images=0, *, sampling=None):
    return Start(
        NewRequest(
            REQUEST,
            SLOT,
            generation=GenerationParams(sampling=sampling or SamplingParams()),
            input_images=input_images,
        )
    )


def _attention(cache, *, build):
    """One attention entry per numerical table over its own pages."""
    entries = {}
    for table in range(len(cache.tables)):
        units = tuple(range(table * PAGES, (table + 1) * PAGES))
        page = cache.groups[cache.tables[table].group].page_tokens
        entry = build(units, page)
        if entries:
            entry = replace(
                entry,
                queries=entries[0].queries,
                prefixes=entries[0].prefixes,
            )
        entries[table] = entry
    return AttentionBatch(entries, entries[0].queries)


def _paged(units, page, *, start, stop, causal):
    """A paged prompt segment ``[start, stop)`` over one table's pages."""
    return PagedInput.from_blocks(
        query_lengths=(stop - start,),
        prefix_lengths=(start,),
        blocks=(units,),
        block_size=page,
        causal=causal,
        device="cpu",
    )


def _reference(root, prompt, segments, canvases, slots, features=None):
    """Candidate log-probabilities of the public model evaluated directly.

    ``segments`` are the prompt's ``(start, stop, causal)`` passes; a
    segment whose start keys ``features`` takes those embeddings.
    """
    model = load_diffusion_gemma(root)
    config = model.text.cache_config
    tables = len(plan_units(config, block_size=PAGE).tables)
    cache = PrefixCache(
        config, num_units=tables * PAGES, block_size=PAGE, device="cpu"
    )
    context = ExecutionContext(model, cache=cache, attention="torch")
    values, prompt_values = [], []
    preceding = None
    with cache, context, torch.no_grad():
        context.prepare(TextSize(128, 1))
        for start, stop, causal in segments:
            batch = _attention(
                cache,
                build=partial(_paged, start=start, stop=stop, causal=causal),
            )
            context.bind_attention(batch)
            replacement = None
            if features is not None and start in features:
                embeddings = features[start]
                replacement = EmbeddingReplacement(
                    embeddings, torch.ones(stop - start, dtype=torch.bool)
                )
            hidden = model.text(
                TextInput(
                    prompt[start:stop],
                    torch.arange(start, stop),
                    batch,
                    replacement,
                )
            )
            logits = model.text.compute_logits(
                hidden, token_indices=torch.arange(stop - start)
            ).gather()
            if causal:
                scores = (
                    logits[:-1]
                    if preceding is None
                    else torch.cat((preceding[None], logits[:-1]))
                )
                targets = prompt[start + int(preceding is None) : stop]
                prompt_values.append(
                    scores.log_softmax(-1).gather(1, targets[:, None])[:, 0]
                )
            preceding = logits[-1]

        length = prompt.numel()
        for canvas, row in zip(canvases, slots, strict=True):
            count = len(canvas)
            batch = _attention(
                cache,
                build=lambda units, page, count=count: SegmentedInput(
                    SequenceLengths.from_lengths((count,), device="cpu"),
                    SequenceLengths.from_lengths((length,), device="cpu"),
                    Table(torch.tensor([units], dtype=torch.int32), page),
                    None,
                    torch.full((1, count), count, dtype=torch.int32),
                    True,
                ),
            )
            context.bind_attention(batch)
            hidden = model.denoiser(
                CanvasInput(
                    torch.tensor(canvas),
                    torch.arange(length, length + count),
                    batch,
                )
            )
            logits = model.denoiser.compute_logits(
                hidden, token_indices=torch.arange(count)
            ).gather()
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            for position, ids in row:
                values.append(logprobs[position, list(ids)])
    return torch.cat(values), torch.cat(prompt_values), preceding


def test_canvas_rows_read_candidates_over_the_cached_prompt(tmp_path):
    """Two canvases of one call read their slots over one cached prompt.

    The prompt spans more than the sliding window, so the sliding layer's
    canvas reads only the prompt's recent tokens. A slot whose candidates
    cover the whole vocabulary reports a distribution that sums to one, and
    the pass leaves the request's coordinates where the prompt left them.
    """
    diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(29)
    prompt = torch.randint(7, 58, (13,), generator=generator)
    canvases = (
        [4, 11, 4, 12, 4, 13, 6, 0, 0],
        [4, 20, 4, 21, 6, 0, 0, 0, 0],
    )
    slots = (
        ((0, (30, 31)), (2, tuple(range(VOCAB))), (4, (40, 41, 42))),
        ((2, (50,)),),
    )

    worker = _worker(tmp_path)
    with worker:
        prefill = _prefill(1, 0, tokens=prompt.tolist())
        prefill = replace(
            prefill,
            commands=(_admission(),),
            new_cache_units=tuple(
                CacheUnitAllocation(SLOT, table.group_id, table.unit_ids)
                for table in prefill.block_tables
            ),
        )
        assert _completion(_run(worker, prefill)).kv_visible_len == 13

        record = _completion(_run(worker, _readout(2, 13, canvases, slots)))
        _run(worker, Batch(batch_id=3, commands=(Finish(REQUEST),)))

    expected, _, _ = _reference(
        tmp_path, prompt, ((0, 13, True),), canvases, slots
    )
    actual = torch.tensor(record.candidate_logprobs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    whole = actual[2 : 2 + VOCAB]
    torch.testing.assert_close(
        torch.logsumexp(whole, dim=0), torch.tensor(0.0), atol=1e-5, rtol=0
    )
    assert (record.position, record.kv_visible_len) == (13, 13)
    assert record.kv_computed_len == 13


def _png(generator):
    pixels = torch.randint(0, 256, (24, 36, 3), generator=generator)
    buffer = io.BytesIO()
    Image.fromarray(pixels.to(torch.uint8).numpy()).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize("score_prompt", (False, True))
@pytest.mark.parametrize(
    ("trailing_text", "sample"), ((True, False), (True, True), (False, True))
)
def test_image_context_preserves_readout_and_next_token(
    tmp_path, score_prompt, trailing_text, sample
):
    """An encoded image's features enter the prompt before the readout.

    Image features attend to each other in both directions and take
    consecutive positions. The final text or image row predicts the next
    token, and prompt scoring excludes the image placeholders.
    """
    diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(31)
    image = _png(generator)
    loaded = models.read_config(tmp_path)
    prepared = prepare_image(
        loaded.image_processor,
        MediaCall.VISION_ENCODING,
        image,
        device=torch.device("cpu"),
        input_images=1,
    )
    rows, columns = prepared.grid_shape
    soft = rows * columns // 9

    head = [*torch.randint(7, 58, (3,), generator=generator).tolist()]
    tail = [*torch.randint(7, 58, (4,), generator=generator).tolist()]
    before = [*head, BEGIN_IMAGE]
    after = [END_IMAGE, *tail] if trailing_text else []
    length = len(before) + soft + len(after)
    canvases = ([4, 11, 4, 12, 6, 0, 0, 0, 0],)
    slots = (((0, (30, 31, 32)), (2, (40, 41))),)

    # The encoder's entry holds at most 70 soft tokens of 32 BF16 values.
    feature = TensorRef(
        request_key=REQUEST,
        producer_call_id=CallId(1, 0),
        output_index=0,
        generation=1,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(70 * 32),)),
    )
    token = TensorRef(
        request_key=REQUEST,
        producer_call_id=CallId(2, 0),
        output_index=0,
        generation=1,
        dtype=DType.I64,
        shape_bound=ShapeBound(),
    )
    worker = _worker(tmp_path)
    with worker:
        encode = Batch(
            batch_id=1,
            collective_seq=1,
            commands=(
                _admission(
                    input_images=1,
                    sampling=SamplingParams(
                        return_prompt_logprobs=score_prompt
                    ),
                ),
            ),
            block_tables=_tables(length),
            new_cache_units=tuple(
                CacheUnitAllocation(SLOT, table.group_id, table.unit_ids)
                for table in _tables(length)
            ),
            calls=(
                Call(
                    request_key=REQUEST,
                    call_id=CallId(1, 0),
                    coordinates=CallCoordinates(0, 0, 0),
                    kind=MediaCall.VISION_ENCODING,
                    bounds=Bounds(
                        max_tokens=soft, max_latent_bytes=feature.max_bytes
                    ),
                    input_image=image,
                    encoder_output=feature,
                ),
            ),
            buffer_allocations=(
                BufferAllocation(feature.buffer_id, 0, feature.max_bytes),
            ),
        )
        _completion(_run(worker, encode))

        # One context call contains all dependent text and image segments.
        query_lengths = (len(before), soft) + (
            (len(after),) if trailing_text else ()
        )
        sequence_lengths = (len(before), len(before) + soft) + (
            (length,) if trailing_text else ()
        )
        context = Batch(
            batch_id=2,
            collective_seq=2,
            calls=(
                _call(
                    2,
                    ForwardMode.PREFILL,
                    0,
                    bounds=Bounds(max_tokens=length, max_completion_bytes=4096),
                    input_token_ids=tuple(before + after),
                    vision_inputs=(VisionBlock(len(before), feature),),
                    token_output=token if sample else None,
                ),
            ),
            block_tables=_tables(length),
            forward_call_indices=(0,) * len(query_lengths),
            request_pool_indices=(SLOT,) * len(query_lengths),
            seq_lens=sequence_lengths,
            query_lens=query_lengths,
            write_kv=(True,) * len(query_lengths),
            buffer_allocations=(
                (
                    BufferAllocation(
                        token.buffer_id, feature.max_bytes, token.max_bytes
                    ),
                )
                if sample
                else ()
            ),
        )
        written = _completion(_run(worker, context))
        assert written.kv_visible_len == length
        assert written.position == length
        record = _completion(_run(worker, _readout(3, length, canvases, slots)))
        _run(worker, Batch(batch_id=4, commands=(Finish(REQUEST),)))

    model = load_diffusion_gemma(tmp_path)
    with torch.no_grad():
        (embeddings,) = model.vision_encoder.encode(
            VisionInput(
                (prepared.pixels,), (prepared.grid,), (prepared.grid_shape,)
            )
        )
    prompt = torch.tensor([*before, *[0] * soft, *after])
    start = len(before) + soft
    expected, expected_prompt, next_logits = _reference(
        tmp_path,
        prompt,
        (
            (0, len(before), True),
            (len(before), start, False),
        )
        + (((start, length, True),) if trailing_text else ()),
        canvases,
        slots,
        features={len(before): embeddings},
    )
    torch.testing.assert_close(
        torch.tensor(record.candidate_logprobs),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )
    if sample:
        assert written.committed_tokens == (int(next_logits.argmax()),)
    if score_prompt:
        assert tuple(
            position[0][0] for position in written.prompt_logprobs
        ) == (
            *before[1:],
            *after,
        )
        torch.testing.assert_close(
            torch.tensor(
                [position[0][1] for position in written.prompt_logprobs]
            ),
            expected_prompt,
            rtol=1e-5,
            atol=1e-6,
        )


def _encode(batch, image, soft):
    """A vision-encoding batch of ``REQUEST`` for an inline ``image``."""
    feature = TensorRef(
        request_key=REQUEST,
        producer_call_id=CallId(batch, 0),
        output_index=0,
        generation=1,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(soft * 32),)),
    )
    call = _call(
        batch,
        MediaCall.VISION_ENCODING,
        4,
        bounds=Bounds(max_tokens=soft, max_latent_bytes=feature.max_bytes),
        input_image=image,
        encoder_output=feature,
    )
    return Batch(
        batch_id=batch,
        collective_seq=batch,
        calls=(call,),
        buffer_allocations=(
            BufferAllocation(feature.buffer_id, 0, feature.max_bytes),
        ),
    )


def _admitted_prefill(batch, tokens, input_images):
    """``REQUEST``'s admission with the prefill of its first ``tokens``."""
    tables = _tables(len(tokens))
    return replace(
        _prefill(batch, 0, tokens=tokens),
        commands=(_admission(input_images=input_images),),
        new_cache_units=tuple(
            CacheUnitAllocation(SLOT, table.group_id, table.unit_ids)
            for table in tables
        ),
    )


def test_a_later_text_batch_completes_while_an_image_is_preparing(tmp_path):
    """Preparing an inline image holds only the batch that encodes it.

    The image is prepared on the rank's host lane. While every lane thread
    is busy, an independent request's prefill submitted after the encode
    runs to completion; the encode completes once the lane frees.
    """
    diffusion_gemma_checkpoint(tmp_path)
    image = _png(torch.Generator().manual_seed(5))
    other = RequestKey(0, 8, 1)
    worker = _worker(tmp_path, queue_depth=4)
    lane = worker.host_tasks
    release = threading.Event()
    with worker:
        try:
            _completion(
                _run(worker, _admitted_prefill(1, [7, 8, 9, BEGIN_IMAGE], 1))
            )

            # Occupy the lane's threads, leaving one task of capacity for the
            # image, which then queues behind them.
            while lane.max_inflight - lane.reserved > 1:
                lane.reserve().submit(release.wait)
            encode = worker.submit(_encode(2, image, soft=70))
            worker.advance()

            # Group g of the other request owns units 33 + 64 * g.
            tables = tuple(
                BlockTable(2, group, 0, (33 + 64 * group,), PAGE)
                for group in (0, 1)
            )
            prefill = Batch(
                batch_id=3,
                collective_seq=3,
                commands=(
                    Start(
                        NewRequest(
                            other,
                            2,
                            generation=GenerationParams(
                                sampling=SamplingParams()
                            ),
                        )
                    ),
                ),
                calls=(
                    Call(
                        request_key=other,
                        call_id=CallId(3, 0),
                        coordinates=CallCoordinates(0, 0, 0),
                        kind=ForwardMode.PREFILL,
                        bounds=Bounds(max_tokens=3),
                        input_token_ids=(7, 8, 9),
                    ),
                ),
                block_tables=tables,
                new_cache_units=tuple(
                    CacheUnitAllocation(2, table.group_id, table.unit_ids)
                    for table in tables
                ),
                forward_call_indices=(0,),
                request_pool_indices=(2,),
                seq_lens=(3,),
                query_lens=(3,),
                write_kv=(True,),
            )
            _completion(_run(worker, prefill))
            assert worker.poll(encode) is None
        finally:
            release.set()

        _completion(_drain(worker, encode))


def test_an_undecodable_image_fails_its_encode_with_the_reason(
    tmp_path, caplog
):
    """A payload that is not an image fails the encoding batch as invalid."""
    diffusion_gemma_checkpoint(tmp_path)
    payload = base64.b64encode(b"not an image").decode()
    worker = _worker(tmp_path, queue_depth=2)
    with worker, caplog.at_level(logging.WARNING):
        _completion(
            _run(worker, _admitted_prefill(1, [7, 8, 9, BEGIN_IMAGE], 1))
        )
        (record,) = _run(worker, _encode(2, payload, soft=70)).completions

    assert record.status is CallStatus.ERROR
    assert record.error_code is ErrorCode.INVALID_CALL
    assert "inline image payload is not a valid encoded image" in caplog.text
