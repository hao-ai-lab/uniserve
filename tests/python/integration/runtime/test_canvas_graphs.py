"""Canvas passes replay CUDA graphs captured at worker startup.

A DiffusionGemma worker on CUDA captures its canvas row buckets at startup:
a readout graph per bucket and, when it generates canvases, a canvas step
graph per bucket for the sampling it serves. On a model with the released
checkpoints' attention shapes, five canvases replay the six-canvas bucket,
so every replay carries padding, and a replayed call copies no device value
to the host. The eager reference is the same worker's canvas pass run
without a graph over a copy of each request, prefilled in the same call:

- replayed readouts of full and shorter canvases return the eager pass's
  log-probabilities, within the rounding their different shapes allow;
- replayed canvas steps return the eager steps' results and leave their
  canvas state bit for bit, where each eager step runs the copies with a
  sixth canvas, so both calls have one shape and the replay's padding row
  is the only difference.

A readout reading more slots than the largest readout tail graph holds
replays the tails in chunks and returns the eager pass's log-probabilities.
A model whose final layer exchanges tokens across an expert group keeps its
readout tail eager.
After startup, a call of more canvases than every bucket fails instead of
running eagerly. A worker whose KV unit pool holds fewer requests than its
call bound starts, and replays calls of as many canvases as the pool holds.
"""

from __future__ import annotations

import socket
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch
import torch.multiprocessing as mp

from tests.python.fixtures.checkpoints import diffusion_gemma_checkpoint
from tests.python.integration.runtime.test_prefill_graphs import (
    NATIVE_TEXT,
    NATIVE_VISION,
    SM100,
)
from uniserve.distributed import Communicator
from uniserve.loading import weights
from uniserve.math import ceil_div
from uniserve.nn.moe import FusedMoE
from uniserve.runtime import PrefixCache
from uniserve.runtime.process_groups import (
    Rendezvous,
    initialize_process_groups,
)
from uniserve_models import loading as models
from uniserve_worker.bootstrap.cache import cache_info
from uniserve_worker.bootstrap.capacity import (
    graph_table_widths,
    input_buffer_config,
)
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.errors import ResourceError
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.input_batch import (
    CanvasRow,
    CanvasStepRow,
    TokenRow,
)
from uniserve_worker.protocol.batch import CanvasSampling
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    ForwardMode,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.storage.canvas_slots import (
    CanvasSlots,
    generating_denoiser,
)
from uniserve_worker.storage.kv_cache import KVCacheManager

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# The canvas sampler's CUDA kernels take canvases of a multiple of 32
# tokens over a vocabulary of a multiple of 4096.
CANVAS, VOCAB = 32, 4096
PROMPT, SEQUENCE, UNITS, SLOTS = 40, 256, 2048, 12
# A pool of 16 units, whose sentinel leaves 15 allocatable: seven requests
# of one page of every cache group, fewer than ``SLOTS``. Its prompts take
# one page of every group.
SMALL_UNITS, PAGE = 16, 16
# Five canvases replay the six-canvas bucket, so every call carries padding.
ROWS = 5
# Requests 1..ROWS replay graphs; their copies ROWS + 1..2 * ROWS, with the
# extra request after them, run eagerly.
COPY, EXTRA = ROWS, 2 * ROWS + 1
SAMPLING = CanvasSampling(
    canvas_length=CANVAS,
    max_steps=3,
    entropy_bound=0.5,
    t_min=0.4,
    t_max=0.8,
    confidence_threshold=0.2,
    stability_threshold=1,
)


@contextmanager
def _worker(
    root,
    units=UNITS,
    *,
    device="cuda:0",
    experts=None,
    graphs=True,
    split=None,
    microbatches=1,
):
    """Bind a DiffusionGemma worker with graphs generating ``SAMPLING``.

    The KV pool holds ``units`` units. Yields the executor, the unit pool's
    manager and the canvas state.
    """
    source = models.read_config(root)
    expert_only = split is not None and split.rank > 0
    paths = frozenset()
    if split is not None:
        with torch.device("meta"):
            description = source.model_class(source.model)
        paths = frozenset(
            path
            for path, module in description.named_modules()
            if isinstance(module, FusedMoE)
        )
        experts = (
            Communicator((1,), 0, "experts", torch.device(device))
            if expert_only
            else None
        )
    model = models.load_model(
        source,
        device=device,
        weights=weights.Config(dtype=torch.bfloat16),
        experts=experts,
        modules=paths if expert_only else None,
        exclude_modules=paths
        if split is not None and not expert_only
        else frozenset(),
    ).model
    if expert_only:
        model = torch.nn.ModuleList(
            module for module in model.modules() if isinstance(module, FusedMoE)
        )
    config = WorkerConfig(
        device=device,
        model_dtype="bfloat16",
        block_size=16,
        max_batch_calls=SLOTS,
        max_request_pool_size=SLOTS,
        max_batch_tokens=512,
        max_sequence_tokens=SEQUENCE,
        prefill_graph_token_sizes=(64, 256),
        decode_graph_batch_sizes=(1,),
        canvas_sampling=SAMPLING,
        graph_policy="auto" if graphs else "off",
        role="experts" if expert_only else "model",
        expert_exchange="alltoall" if split is None else "deepep",
        expert_microbatches=microbatches,
    )
    processor = source.image_processor
    runner = ModelExecutor(
        model,
        config,
        image_processor=None if expert_only else processor,
        expert_group=split,
        attention_ranks=0 if split is None else 1,
        bindings={} if expert_only else None,
        entry_points={} if expert_only else None,
    )
    if expert_only:
        try:
            runner.configure_experts()
            runner.capture(tokenizer=None, latents=None)
            runner.complete_startup()
            yield runner, None, None
        except BaseException:
            runner.close(aborted=True)
            raise
        else:
            runner.close()
        return
    cache = PrefixCache(
        model.text.cache_config,
        num_units=units,
        block_size=16,
        device=device,
    )
    manager = KVCacheManager(
        cache,
        info=cache_info(model.text, config, num_units=units),
        request_pool_size=SLOTS,
        table_width=64,
    )
    slots = None
    try:
        runner.configure_inputs(
            input_config=input_buffer_config(
                model, config, processor=processor
            ),
            kv_cache=manager,
            latent_pool=None,
            decode_predicates=torch.zeros(
                SLOTS + 1, dtype=torch.bool, device=device
            ),
            max_calls=SLOTS,
            request_slots=SLOTS,
            latent_capacity_units=0,
            table_widths=graph_table_widths(model, config, manager),
            max_inflight=1,
        )
        canvas_runner = runner.canvas_runner
        slots = CanvasSlots.for_denoiser(
            generating_denoiser(canvas_runner.model),
            request_pool_size=SLOTS,
            sampling=SAMPLING,
            device=device,
        )
        runner.bind_canvas_slots(slots)
        runner.capture(tokenizer=None, latents=None)
        runner.complete_startup()
        yield runner, manager, slots
    except BaseException:
        runner.close(aborted=True)
        raise
    else:
        runner.close()
    finally:
        if slots is not None:
            slots.close()
        manager.close()


def _install(manager, count, tokens=SEQUENCE):
    """Install tables of every group covering ``tokens`` per slot."""
    entries, unit = [], 1
    for slot in range(1, count + 1):
        for group, shape in enumerate(manager.shapes):
            pages = ceil_div(tokens, shape.page_tokens)
            units = tuple(range(unit, unit + pages * shape.units_per_page))
            unit += len(units)
            entries.append((slot, group, 0, units, pages * shape.page_tokens))
    manager.block_tables.install(tuple(entries))


def _run(runner, manager, rows, kind):
    """Run one staged group of rows as calls of ``kind``."""
    calls = tuple(
        Call(
            request_key=RequestKey(1, row.request_pool_idx, 0),
            call_id=CallId(1, index),
            coordinates=CallCoordinates(),
            kind=kind,
            bounds=Bounds(),
        )
        for index, row in enumerate(rows)
    )
    return runner.run_forward_group(
        rows,
        calls=calls,
        cache=manager,
        tables=manager.block_tables,
        states=None,
    )


def _eager(runner, manager, rows):
    """Run canvas rows through the canvas runner without a graph."""
    entry = runner.canvas_runner
    batch = entry.prepare_inputs(
        rows,
        forward_mode=ForwardMode.TOKEN_DENOISING,
        cache=manager,
        tables=manager.block_tables,
    )
    return entry.eager_batch(batch, entry.batch_forward)


def _request(slot):
    """The request slot ``slot`` copies, when it holds a copy."""
    return slot - COPY if COPY < slot <= 2 * COPY else slot


def _scenario(slots, prompt_tokens=PROMPT):
    """Prompt rows and, per request slot, its readout row and step rows.

    Slots ``1..EXTRA`` hold requests of ``prompt_tokens`` prompts; a copy's
    prompt, canvas and seed are its request's. Readout canvases read
    differing numbers of slots and candidates; one is half a canvas long,
    as a compact readout canvas is.
    """
    generator = torch.Generator().manual_seed(911)
    requests = {}
    for request in (*range(1, ROWS + 1), EXTRA):
        length = CANVAS // 2 if request == 2 else CANVAS
        reads = 1 + request % 3
        requests[request] = (
            torch.randint(7, VOCAB, (prompt_tokens,), generator=generator),
            torch.randint(7, VOCAB, (length,), generator=generator),
            tuple(range(reads)),
            tuple(
                torch.randint(
                    7, VOCAB, (2 * reads,), generator=generator
                ).tolist()
            ),
        )

    prompt, readout, steps = [], {}, {}
    for slot in range(1, EXTRA + 1):
        tokens, canvas, reads, candidates = requests[_request(slot)]
        prompt.append(
            TokenRow(
                forward_mode=ForwardMode.PREFILL,
                token_ids=tokens,
                positions=torch.arange(prompt_tokens),
                selection=TokenSelection.LAST_LOGITS,
                request_pool_idx=slot,
                seq_len=0,
                write_kv=True,
                causal=True,
            )
        )
        readout[slot] = CanvasRow(
            forward_mode=ForwardMode.TOKEN_DENOISING,
            token_ids=canvas,
            positions=torch.arange(
                prompt_tokens, prompt_tokens + canvas.numel()
            ),
            request_pool_idx=slot,
            seq_len=prompt_tokens,
            write_kv=False,
            causal=False,
            slot_tokens=reads,
            candidate_offsets=tuple(range(0, 2 * len(reads) + 1, 2)),
            candidate_ids=candidates,
        )
        steps[slot] = tuple(
            CanvasStepRow(
                forward_mode=ForwardMode.TOKEN_DENOISING,
                positions=torch.arange(prompt_tokens, prompt_tokens + CANVAS),
                request_pool_idx=slot,
                seq_len=prompt_tokens,
                write_kv=False,
                causal=False,
                canvas_length=CANVAS,
                seed=_request(slot),
                block=0,
                step=step,
                sampling=slots.constants,
            )
            for step in range(SAMPLING.max_steps)
        )
    return tuple(prompt), readout, steps


def _values(output):
    return tuple(value.cpu() for value in output.materialize().values)


def _checkpoint(root):
    diffusion_gemma_checkpoint(
        root,
        text={**NATIVE_TEXT, "vocab_size": VOCAB},
        vision=NATIVE_VISION,
        unit_scores=True,
        canvas_length=CANVAS,
    )


@torch.inference_mode()
def _expert_reads(rank, root, port):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    with initialize_process_groups(
        rank=0,
        local_rank=rank,
        world_size=1,
        device=device,
        experts=(rank, 2, Rendezvous("127.0.0.1", port)),
    ) as groups:
        results = []
        for graphs in (False, True):
            observations = []
            with _worker(
                root, device=device, experts=groups.experts, graphs=graphs
            ) as (runner, manager, slots):
                _install(manager, EXTRA)
                prompt, readout, _ = _scenario(slots)
                _run(runner, manager, prompt[:6], ForwardMode.PREFILL)
                _run(runner, manager, prompt[6:], ForwardMode.PREFILL)

                # One rank keeps the same small prefill while its peer
                # changes canvas batches. Reverse the capabilities too:
                # a compact readout must retain its result alongside a
                # peer's longer prefill. All rows have disjoint KV slots.
                for count in (1, 5, 11):
                    rows = (
                        prompt[:1]
                        if rank == 0
                        else tuple(
                            readout[slot] for slot in range(1, count + 1)
                        )
                    )
                    kind = (
                        ForwardMode.PREFILL
                        if rank == 0
                        else ForwardMode.TOKEN_DENOISING
                    )
                    observations.append(
                        _values(_run(runner, manager, rows, kind))
                    )
                for count in (1, 3, 6):
                    rows = (readout[2],) if rank == 0 else prompt[:count]
                    kind = (
                        ForwardMode.TOKEN_DENOISING
                        if rank == 0
                        else ForwardMode.PREFILL
                    )
                    observations.append(
                        _values(_run(runner, manager, rows, kind))
                    )

                # Repeated slot reads can make the final expert layer's
                # sender larger than the compact canvas that produced it.
                repeated = replace(
                    readout[2],
                    slot_tokens=(0,) * 96,
                    candidate_offsets=tuple(range(0, 193, 2)),
                    candidate_ids=readout[2].candidate_ids[:2] * 96,
                )
                rows = (repeated,) if rank == 0 else prompt[:1]
                kind = (
                    ForwardMode.TOKEN_DENOISING
                    if rank == 0
                    else ForwardMode.PREFILL
                )
                observations.append(_values(_run(runner, manager, rows, kind)))
                if rank == 0:
                    observations.append(
                        _values(
                            _run(
                                runner, manager, prompt[:2], ForwardMode.PREFILL
                            )
                        )
                    )

                # Independent callers can finish at different times. The
                # completed rank serves its peer's remaining expert work
                # until both release, as the worker service does on close.
                while not runner.experts.released:
                    runner.join_expert_step(leaving=True)
            results.append(observations)

        # The existing native BF16 graph/eager contract permits rounding
        # from different batch shapes, including expert route reductions.
        eager, replayed = results
        for expected_rows, actual_rows in zip(eager, replayed, strict=True):
            for expected, actual in zip(
                expected_rows, actual_rows, strict=True
            ):
                torch.testing.assert_close(
                    actual, expected, rtol=2e-2, atol=2e-2
                )


@SM100
def test_expert_peers_preserve_results_across_unequal_numerical_batches(
    tmp_path,
):
    """Mixed prefill/readout ranks return the eager numerical results."""
    if torch.cuda.device_count() < 2:
        pytest.fail("expert-parallel canvas calls need two GPUs")
    _checkpoint(tmp_path)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_expert_reads, (tmp_path, port), nprocs=2, join=True)


def _canvas_observations(runner, manager, slots):
    """Read tails and advance resident canvases after the same prompt cache."""
    _install(manager, 3)
    prompt, readout, steps = _scenario(slots)
    _run(runner, manager, prompt[:3], ForwardMode.PREFILL)
    results = []
    for count in (1, 3, 2):
        results.append(
            _values(
                _run(
                    runner,
                    manager,
                    tuple(readout[slot] for slot in range(1, count + 1)),
                    ForwardMode.TOKEN_DENOISING,
                )
            )
        )
    repeated = replace(
        readout[2],
        slot_tokens=(0,) * 96,
        candidate_offsets=tuple(range(0, 193, 2)),
        candidate_ids=readout[2].candidate_ids[:2] * 96,
    )
    results.append(
        _values(_run(runner, manager, (repeated,), ForwardMode.TOKEN_DENOISING))
    )
    for step in range(SAMPLING.max_steps):
        results.append(
            _values(
                _run(
                    runner,
                    manager,
                    tuple(steps[slot][step] for slot in (1, 2, 3)),
                    ForwardMode.TOKEN_DENOISING,
                )
            )
        )
    return results


@torch.inference_mode()
def _split_canvases(rank, root, port):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=0,
        local_rank=rank,
        world_size=1,
        device=device,
        experts=(rank, 2, Rendezvous("127.0.0.1", port)),
    ) as groups:
        if rank == 0:
            with _worker(root, device=device, graphs=False) as values:
                expected = _canvas_observations(*values)
        with _worker(
            root,
            device=device,
            split=groups.experts,
            microbatches=2,
        ) as (runner, manager, slots):
            if rank == 0:
                observed = _canvas_observations(runner, manager, slots)
                for actual_rows, expected_rows in zip(
                    observed, expected, strict=True
                ):
                    for actual, wanted in zip(
                        actual_rows, expected_rows, strict=True
                    ):
                        torch.testing.assert_close(
                            actual, wanted, rtol=2e-2, atol=2e-2
                        )
            while not runner.experts.released:
                runner.join_expert_step(leaving=True)


@SM100
def test_disaggregated_microbatches_preserve_readout_and_canvas_steps(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.fail("disaggregated canvas calls require two GPUs")
    _checkpoint(tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_split_canvases, (tmp_path, port), nprocs=2, join=True)


@SM100
@torch.inference_mode()
def test_canvas_graph_replay_matches_eager_execution(tmp_path):
    """Replayed readouts and steps equal eager passes, copying nothing.

    The replays run under CUDA synchronization checks, which fail on any
    device-to-host copy.
    """
    _checkpoint(tmp_path)
    graphs = range(1, ROWS + 1)
    copies = (*range(COPY + 1, 2 * COPY + 1), EXTRA)
    with _worker(tmp_path) as (runner, manager, slots):
        _install(manager, EXTRA)
        prompt, readout, steps = _scenario(slots)
        _run(runner, manager, prompt[:6], ForwardMode.PREFILL)
        _run(runner, manager, prompt[6:], ForwardMode.PREFILL)

        def replay(rows):
            torch.cuda.set_sync_debug_mode("error")
            try:
                output = _run(
                    runner, manager, rows, ForwardMode.TOKEN_DENOISING
                )
            finally:
                torch.cuda.set_sync_debug_mode("default")
            return _values(output)

        # A batch consisting entirely of short canvases must retain the
        # same distributions as an ordinary numerical pass. Include both
        # one row and two independent copies of that row's request.
        for selected in ((2,), (2, COPY + 2)):
            rows = tuple(readout[slot] for slot in selected)
            replayed = replay(rows)
            eager = _values(_eager(runner, manager, rows))
            for actual, expected in zip(replayed, eager, strict=True):
                torch.testing.assert_close(
                    actual, expected, rtol=2e-2, atol=2e-2
                )

        replayed = replay(tuple(readout[slot] for slot in graphs))
        eager = _values(
            _eager(runner, manager, tuple(readout[slot] for slot in copies))
        )
        for actual, expected in zip(replayed, eager[:ROWS], strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

        for step in range(SAMPLING.max_steps):
            replayed = replay(tuple(steps[slot][step] for slot in graphs))
            eager = _values(
                _eager(
                    runner,
                    manager,
                    tuple(steps[slot][step] for slot in copies),
                )
            )
            for actual, expected in zip(replayed, eager[:ROWS], strict=True):
                assert torch.equal(actual, expected)

        for bank in slots.banks.values():
            assert torch.equal(
                bank[1 : ROWS + 1], bank[COPY + 1 : 2 * COPY + 1]
            )


@SM100
@torch.inference_mode()
def test_a_readout_answers_every_slot_of_many_canvases(tmp_path):
    """A readout returns the candidates of every requested slot.

    Every token of nine canvases, one of them half a canvas long, is a slot
    with two candidates. The result equals the eager pass's log-probabilities
    within the rounding their different shapes allow.
    """
    _checkpoint(tmp_path)
    with _worker(tmp_path) as (runner, manager, slots):
        _install(manager, EXTRA)
        prompt, readout, _ = _scenario(slots)
        _run(runner, manager, prompt[:6], ForwardMode.PREFILL)
        _run(runner, manager, prompt[6:], ForwardMode.PREFILL)

        generator = torch.Generator().manual_seed(17)

        def every_token(row):
            tokens = row.query_tokens
            return replace(
                row,
                slot_tokens=tuple(range(tokens)),
                candidate_offsets=tuple(range(0, 2 * tokens + 1, 2)),
                candidate_ids=tuple(
                    torch.randint(
                        7, VOCAB, (2 * tokens,), generator=generator
                    ).tolist()
                ),
            )

        rows = tuple(every_token(readout[slot]) for slot in range(3, EXTRA + 1))
        torch.cuda.set_sync_debug_mode("error")
        try:
            output = _run(runner, manager, rows, ForwardMode.TOKEN_DENOISING)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        replayed = _values(output)
        eager = _values(_eager(runner, manager, rows))
        for actual, expected in zip(replayed, eager, strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@SM100
@torch.inference_mode()
def test_a_call_exceeding_the_configured_row_capacity_fails(tmp_path):
    """Calls beyond the deployment's row capacity are rejected."""
    _checkpoint(tmp_path)
    with _worker(tmp_path) as (runner, manager, slots):
        _install(manager, 1)
        _, readout, _ = _scenario(slots)
        rows = (readout[1],) * (SLOTS + 1)
        with pytest.raises(ResourceError):
            _run(runner, manager, rows, ForwardMode.TOKEN_DENOISING)


@SM100
@torch.inference_mode()
def test_a_small_pool_replays_calls_of_every_canvas_it_holds(tmp_path):
    """A pool of fewer requests than the call bound starts and replays.

    Every live request holds a page of every cache group of its own, so the
    pool's allocatable units bound the canvases one call reads. A worker on
    such a pool starts, and a readout and a canvas step of that many
    canvases, each over a one-page prompt, replay a graph.
    """
    _checkpoint(tmp_path)
    with _worker(tmp_path, units=SMALL_UNITS) as (runner, manager, slots):
        rows = (manager.info.num_units - 1) // manager.row_units
        # The pool, not the call or request slot bound, bounds the rows.
        assert rows < SLOTS
        _install(manager, rows, tokens=PAGE)
        prompt, readout, steps = _scenario(slots, prompt_tokens=PAGE)
        for row in prompt[:rows]:
            _run(runner, manager, (row,), ForwardMode.PREFILL)

        for canvases in (
            tuple(readout[slot] for slot in range(1, rows + 1)),
            tuple(steps[slot][0] for slot in range(1, rows + 1)),
        ):
            output = _run(
                runner, manager, canvases, ForwardMode.TOKEN_DENOISING
            )
            values = _values(output)
            assert len(values) == len(canvases)
            eager = _values(_eager(runner, manager, canvases))
            for actual, expected in zip(values, eager, strict=True):
                torch.testing.assert_close(
                    actual, expected, rtol=2e-2, atol=2e-2
                )
