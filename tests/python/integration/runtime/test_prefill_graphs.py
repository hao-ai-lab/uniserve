"""Prefill and decode calls replay CUDA graphs captured at worker startup.

A worker with prefill graphs, the default on CUDA, captures its prefill
buckets at startup and replays one for every prefill call. On a
DiffusionGemma model with the released checkpoints' attention shapes, whose
sliding tables start after retired pages, replays of causal text chunks,
non-causal image-block rows and 256-token commit rows return the outputs and
write the KV of eager execution of the same calls, and a replayed call copies
no device value to the host. After startup, a prefill or decode call no
captured graph holds fails instead of running eagerly, and the worker
reports the most rows its prefill and decode graphs hold, at most the rows
its KV unit pool holds a page of each cache group for. On pools of few
units, whose cache groups have pages of different sizes, startup captures
every prefill bucket a batch the pool holds selects, and each replays. A
worker whose prefills select no output replays graphs that only write the
K/V cache, and leaves the cache of the eager pass that evaluates outputs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from tests.python.fixtures.checkpoints import diffusion_gemma_checkpoint
from uniserve.distributed import Communicator
from uniserve.loading import weights
from uniserve.math import ceil_div
from uniserve.runtime import PrefixCache
from uniserve_models import loading as models
from uniserve_worker.bootstrap.cache import cache_info
from uniserve_worker.bootstrap.capacity import (
    graph_table_widths,
    input_buffer_config,
)
from uniserve_worker.bootstrap.components import supported_calls
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.errors import ResourceError
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.graph_inputs import (
    prefill_captures,
    prefill_units,
)
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    ForwardMode,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.storage.kv_cache import KVCacheManager
from uniserve_worker.worker import Worker

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

SM100 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10,
    reason="native DiffusionGemma attention requires an SM100 GPU",
)

# Attention shapes of the released DiffusionGemma checkpoints (see
# test_diffusion_gemma.py): sliding layers of width 256 over a 41-token
# window and full layers of width 512, which the native trtllm-gen and
# prefix-block kernels serve.
NATIVE_TEXT = {
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_global_key_value_heads": 2,
    "sliding_window": 41,
    "top_k_experts": 6,
}
NATIVE_VISION = {"hidden_size": 144, "head_dim": 72}

# Every measured call follows a 120-token prompt; its first query reads the
# sliding window from token 79 on, so the sliding tables retire their first
# two 16-token pages before it.
PROMPT, RETIRED, SEQUENCE = 120, 2, 512
UNITS = 1024


@contextmanager
def _diffusion_gemma_worker(
    root, *, graphs, units=UNITS, slots=4, outputs=True
):
    """Bind a DiffusionGemma worker's staged calls over a fresh unit pool.

    Startup captures prefill graphs when ``graphs`` is set and runs prefill
    eagerly otherwise; without ``outputs`` the worker's prefills select no
    output. The pool holds ``units`` units and the worker ``slots`` request
    slots, which also bound its calls. Yields the executor, the pool's
    manager and the worker configuration.
    """
    source = models.read_config(root)
    model = models.load_model(
        source,
        device="cuda:0",
        weights=weights.Config(dtype=torch.bfloat16),
    ).model
    config = WorkerConfig(
        device="cuda:0",
        model_dtype="bfloat16",
        block_size=16,
        max_batch_calls=slots,
        max_request_pool_size=slots,
        max_batch_tokens=768,
        max_sequence_tokens=SEQUENCE,
        prefill_cuda_graph=graphs,
        prefill_outputs=outputs,
        prefill_graph_token_sizes=(64, 256),
        decode_graph_batch_sizes=(1,),
    )
    processor = source.image_processor
    cache = PrefixCache(
        model.text.cache_config,
        num_units=units,
        block_size=16,
        device="cuda:0",
    )
    manager = KVCacheManager(
        cache,
        info=cache_info(model.text, config, num_units=units),
        request_pool_size=slots,
        table_width=64,
    )
    runner = ModelExecutor(model, config, image_processor=processor)
    try:
        runner.configure_inputs(
            input_config=input_buffer_config(
                model, config, processor=processor
            ),
            kv_cache=manager,
            latent_pool=None,
            decode_predicates=torch.zeros(
                slots + 1, dtype=torch.bool, device="cuda:0"
            ),
            max_calls=slots,
            request_slots=slots,
            latent_capacity_units=0,
            table_widths=graph_table_widths(model, config, manager),
            max_inflight=1,
        )
        runner.capture(tokenizer=None, latents=None)
        runner.complete_startup()
        yield runner, manager, config
    finally:
        runner.close()
        manager.close()


def _install(manager, slots, *, retired):
    """Install each slot's tables of every group over its own units.

    Every table covers ``SEQUENCE`` tokens; a sliding group's table starts
    ``retired`` pages in, as the scheduler retires pages its window left.
    """
    entries, unit = [], 1
    for slot in slots:
        for group, shape in enumerate(manager.shapes):
            pages = ceil_div(SEQUENCE, shape.page_tokens)
            units = tuple(range(unit, unit + pages * shape.units_per_page))
            unit += len(units)
            start = retired if shape.window is not None else 0
            entries.append(
                (
                    slot,
                    group,
                    start,
                    units[start * shape.units_per_page :],
                    pages * shape.page_tokens,
                )
            )
    manager.block_tables.install(tuple(entries))


def _call(runner, manager, rows):
    """Run one staged group of token rows and return its gathered outputs."""
    calls = tuple(
        Call(
            request_key=RequestKey(1, row.request_pool_idx, 0),
            call_id=CallId(1, index),
            coordinates=CallCoordinates(),
            kind=row.forward_mode,
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


def _rows(
    tokens,
    *,
    prefix,
    selections,
    causal=True,
    embeddings=None,
    mode=ForwardMode.PREFILL,
):
    """Build ``mode`` rows over slots ``1..`` appending after ``prefix``."""
    return tuple(
        TokenRow(
            forward_mode=mode,
            token_ids=value,
            token_embeddings=None if embeddings is None else embeddings[index],
            positions=torch.arange(prefix, prefix + value.numel()),
            selection=selection,
            request_pool_idx=index + 1,
            seq_len=prefix,
            write_kv=True,
            causal=causal[index] if isinstance(causal, tuple) else causal,
        )
        for index, (value, selection) in enumerate(
            zip(tokens, selections, strict=True)
        )
    )


def _scenario(kind):
    """Return a prompt call and the measured call that follows it.

    ``causal`` appends text chunks of different lengths selecting final
    logits, every token's logits and hidden states; ``image_block`` appends
    non-causal image rows whose embeddings are replaced, selecting hidden
    states and final logits; ``commit`` appends two 256-token committed
    blocks selecting final logits.
    """
    generator = torch.Generator().manual_seed(517)

    def tokens(*lengths):
        return tuple(
            torch.randint(7, 58, (length,), generator=generator)
            for length in lengths
        )

    last, every, hidden = (
        TokenSelection.LAST_LOGITS,
        TokenSelection.ALL_LOGITS,
        TokenSelection.HIDDEN,
    )
    if kind == "causal":
        lengths, selections = (40, 7, 90), (last, every, hidden)
        measured = {"selections": selections}
    elif kind in {"image_block", "mixed_context"}:
        lengths, selections = (70, 12), (hidden, last)
        measured = {
            "selections": selections,
            "causal": (True, False) if kind == "mixed_context" else False,
            "embeddings": tuple(
                torch.randn((length, 32), generator=generator).to(
                    torch.bfloat16
                )
                for length in lengths
            ),
        }
    else:
        lengths, selections = (256, 256), (last, last)
        measured = {"selections": selections}
    prompt = _rows(
        tokens(*(PROMPT,) * len(lengths)),
        prefix=0,
        selections=(last,) * len(lengths),
    )
    return prompt, _rows(tokens(*lengths), prefix=PROMPT, **measured)


def _outputs(output):
    return tuple(value.float().cpu() for value in output.materialize().values)


def _cache_values(manager):
    cache = manager.cache
    return tuple(
        tensor.float().cpu()
        for name in cache.config.layers
        for tensor in (cache.state(name).key, cache.state(name).value)
    )


@SM100
@torch.inference_mode()
@pytest.mark.parametrize(
    "kind", ["causal", "image_block", "mixed_context", "commit"]
)
def test_prefill_graph_replay_matches_eager_execution(tmp_path, kind):
    """A replayed prefill equals the eager call and copies nothing to host.

    Both workers run the same prompt call and the same measured call over
    identical unit tables, whose sliding tables start after retired pages;
    the graph worker's calls run under CUDA synchronization checks, which
    fail on any device-to-host copy.
    """
    diffusion_gemma_checkpoint(
        tmp_path, text=NATIVE_TEXT, vision=NATIVE_VISION, unit_scores=True
    )
    prompt, measured = _scenario(kind)
    results = {}
    for graphs in (True, False):
        with _diffusion_gemma_worker(tmp_path, graphs=graphs) as (
            runner,
            manager,
            _,
        ):
            slots = range(1, len(prompt) + 1)
            _install(manager, slots, retired=0)
            outputs = []
            for rows in (prompt, measured):
                if rows is measured:
                    _install(manager, slots, retired=RETIRED)
                if graphs:
                    torch.cuda.set_sync_debug_mode("error")
                try:
                    output = _call(runner, manager, rows)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                if graphs:
                    assert output.stats.cuda_graph_replays == 1
                outputs.append(_outputs(output))
            results[graphs] = outputs, _cache_values(manager)

    (graph_outputs, graph_cache), (eager_outputs, eager_cache) = (
        results[True],
        results[False],
    )
    for graph_call, eager_call in zip(
        graph_outputs, eager_outputs, strict=True
    ):
        for actual, expected in zip(graph_call, eager_call, strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    for actual, expected in zip(graph_cache, eager_cache, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@SM100
@torch.inference_mode()
@pytest.mark.parametrize(
    "kind", ["causal", "image_block", "mixed_context", "commit"]
)
def test_cache_only_prefill_writes_the_cache_of_the_complete_pass(
    tmp_path, kind
):
    """Prefills that select no output write what the complete pass writes.

    A worker whose prefills select no output replays graphs that stop at the
    final layer's cache write, and returns an empty value per row. Over the
    same calls, the cache it leaves equals that of an eager worker whose
    rows select their outputs, within the rounding their different shapes
    allow.
    """
    diffusion_gemma_checkpoint(
        tmp_path, text=NATIVE_TEXT, vision=NATIVE_VISION, unit_scores=True
    )
    prompt, measured = _scenario(kind)
    caches = {}
    for graphs in (True, False):
        with _diffusion_gemma_worker(
            tmp_path, graphs=graphs, outputs=not graphs
        ) as (runner, manager, _):
            slots = range(1, len(prompt) + 1)
            _install(manager, slots, retired=0)
            for rows in (prompt, measured):
                if rows is measured:
                    _install(manager, slots, retired=RETIRED)
                if graphs:
                    rows = tuple(
                        replace(row, selection=TokenSelection.CACHE)
                        for row in rows
                    )
                output = _call(runner, manager, rows)
                if graphs:
                    assert output.stats.cuda_graph_replays == 1
                    assert all(
                        value.numel() == 0
                        for value in output.materialize().values
                    )
            caches[graphs] = _cache_values(manager)

    for actual, expected in zip(caches[True], caches[False], strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _qwen_checkpoint(root):
    config = Qwen3Config(
        vocab_size=37,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=64,
        rope_theta=10000,
    )
    torch.manual_seed(94)
    Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(root)


@torch.inference_mode()
def test_sealed_prefill_rejects_calls_no_captured_graph_holds(tmp_path):
    """After startup, a prefill no captured graph holds fails, writing no KV.

    Forty admitted calls exceed the 31 rows the widest prefill bucket holds;
    the worker reports that bound, replays a graph for 31 rows, and rejects
    35 rows, and non-causal rows its text-only model captured no graph for,
    without running them.
    """
    _qwen_checkpoint(tmp_path)
    model = models.load_model(
        models.read_config(tmp_path), device="cuda:0"
    ).model
    config = WorkerConfig(
        device="cuda:0",
        block_size=16,
        kv_token_capacity=4096,
        max_sequence_tokens=32,
        max_batch_calls=40,
        max_batch_tokens=64,
        max_request_pool_size=40,
        prefill_graph_token_sizes=(16,),
        decode_graph_batch_sizes=(1,),
    )
    with Worker(
        model,
        worker_config=config,
        sampling_group=Communicator(device=torch.device("cuda:0")),
        tokenizer=None,
        allowed_calls=supported_calls(model),
        queue_depth=2,
        completion_payload_bytes=1 << 16,
    ) as worker:
        worker.warmup()
        assert worker.info.max_prefill_calls == 31

        manager = worker.kv_cache
        manager.block_tables.install(
            tuple((slot, 0, 0, (slot,), 16) for slot in range(1, 36))
        )

        def rows(count, *, causal=True):
            return _rows(
                tuple(torch.tensor([slot % 30 + 1]) for slot in range(count)),
                prefix=0,
                selections=(TokenSelection.LAST_LOGITS,) * count,
                causal=causal,
            )

        output = _call(worker.runner, manager, rows(31))
        assert output.stats.cuda_graph_replays == 1
        assert len(output.materialize().values) == 31

        before = _cache_values(manager)
        for rejected in (rows(35), rows(2, causal=False)):
            with pytest.raises(ResourceError, match="no prefill graph"):
                _call(worker.runner, manager, rejected)
        torch.cuda.synchronize()
        for actual, expected in zip(
            _cache_values(manager), before, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_unit_pool_bounds_the_rows_of_prefill_graphs(tmp_path):
    """A pool of few units bounds prefill rows, and startup still captures.

    Every row of a prefill call holds a page of the model's one cache
    group, so a 96-token unit pool holds fewer rows than the configured
    prefill buckets would. Startup captures every prefill graph within the
    pool, the worker reports the rows its allocatable units hold, and a
    prefill of that many rows replays a graph.
    """
    _qwen_checkpoint(tmp_path)
    model = models.load_model(
        models.read_config(tmp_path), device="cuda:0"
    ).model
    config = WorkerConfig(
        device="cuda:0",
        block_size=16,
        kv_token_capacity=96,
        max_sequence_tokens=32,
        max_batch_calls=16,
        max_batch_tokens=64,
        max_request_pool_size=16,
        prefill_graph_token_sizes=(16,),
        decode_graph_batch_sizes=(1,),
    )
    with Worker(
        model,
        worker_config=config,
        sampling_group=Communicator(device=torch.device("cuda:0")),
        tokenizer=None,
        allowed_calls=supported_calls(model),
        queue_depth=2,
        completion_payload_bytes=1 << 16,
    ) as worker:
        worker.warmup()
        # Unit zero is the pool's sentinel; each row holds one unit, and the
        # pool holds fewer rows than the 31 the widest bucket would.
        allocatable = worker.info.kv_cache.num_units - 1
        assert allocatable < 31
        assert worker.info.max_prefill_calls == allocatable

        manager = worker.kv_cache
        manager.block_tables.install(
            tuple(
                (slot, 0, 0, (slot,), 16) for slot in range(1, allocatable + 1)
            )
        )
        output = _call(
            worker.runner,
            manager,
            _rows(
                tuple(
                    torch.tensor([slot % 30 + 1]) for slot in range(allocatable)
                ),
                prefix=0,
                selections=(TokenSelection.LAST_LOGITS,) * allocatable,
            ),
        )
        assert output.stats.cuda_graph_replays == 1
        assert len(output.materialize().values) == allocatable


@torch.inference_mode()
def test_sealed_decode_rejects_calls_no_captured_graph_holds(tmp_path):
    """Decode capacity follows the unit pool; wider decode calls fail.

    Decode graphs of up to 16 rows are configured, and capturing one stages
    a page of the model's one cache group per row on the unit pool, whose
    96 tokens hold fewer. The worker reports the largest configured size its
    allocatable units hold, replays a graph for a decode call of that many
    rows, and rejects one more row without running it.
    """
    _qwen_checkpoint(tmp_path)
    model = models.load_model(
        models.read_config(tmp_path), device="cuda:0"
    ).model
    sizes = (1, 2, 4, 8, 16)
    config = WorkerConfig(
        device="cuda:0",
        block_size=16,
        kv_token_capacity=96,
        max_sequence_tokens=32,
        max_batch_calls=16,
        max_batch_tokens=64,
        max_request_pool_size=16,
        prefill_graph_token_sizes=(16,),
        decode_graph_batch_sizes=sizes,
    )
    with Worker(
        model,
        worker_config=config,
        sampling_group=Communicator(device=torch.device("cuda:0")),
        tokenizer=None,
        allowed_calls=supported_calls(model),
        queue_depth=2,
        completion_payload_bytes=1 << 16,
    ) as worker:
        worker.warmup()
        # Unit zero is the pool's sentinel; each row stages one unit.
        allocatable = worker.info.kv_cache.num_units - 1
        capacity = worker.info.max_decode_calls
        assert capacity == max(size for size in sizes if size <= allocatable)
        assert capacity < max(sizes)

        # Each row owns one unit; the row beyond the capacity shares one,
        # which its rejected call never reaches.
        manager = worker.kv_cache
        manager.block_tables.install(
            tuple(
                (slot, 0, 0, ((slot - 1) % allocatable + 1,), 16)
                for slot in range(1, capacity + 2)
            )
        )

        def rows(count):
            return _rows(
                tuple(torch.tensor([slot % 30 + 1]) for slot in range(count)),
                prefix=0,
                selections=(TokenSelection.LAST_LOGITS,) * count,
                mode=ForwardMode.DECODE,
            )

        output = _call(worker.runner, manager, rows(capacity))
        assert output.stats.cuda_graph_replays == 1
        assert len(output.materialize().values) == capacity

        before = _cache_values(manager)
        with pytest.raises(ResourceError, match="no decode graph"):
            _call(worker.runner, manager, rows(capacity + 1))
        torch.cuda.synchronize()
        for actual, expected in zip(
            _cache_values(manager), before, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@SM100
@torch.inference_mode()
@pytest.mark.parametrize("units", [12, *range(17, 24), 28])
def test_a_small_pool_replays_every_prefill_bucket(tmp_path, units):
    """Every prefill bucket a small pool's batches select captures and replays.

    The test model's cache groups have 16- and 32-token pages of one unit
    each, and twelve request slots bound its calls, so these pools hold
    fewer rows and tokens than its buckets' configured sizes. The worker
    starts, and for every bucket of ``prefill_captures`` over this pool, the
    batch of the fewest rows and tokens it serves, with its tokens spread
    evenly over its rows, fits the pool and replays a graph.
    """
    diffusion_gemma_checkpoint(
        tmp_path, text=NATIVE_TEXT, vision=NATIVE_VISION, unit_scores=True
    )
    with _diffusion_gemma_worker(
        tmp_path, graphs=True, units=units, slots=12
    ) as (runner, manager, config):
        pages = tuple(
            (shape.page_tokens, shape.units_per_page)
            for shape in manager.shapes
        )
        shapes = prefill_captures(
            config,
            max_rows=min(12, (units - 1) // manager.row_units),
            max_tokens=min(768, manager.token_capacity),
            image_builder=False,
            feature_injection=True,
            device_causality=True,
            pool=(pages, units - 1),
        )
        assert shapes

        generator = torch.Generator().manual_seed(29)
        previous = {}
        for shape in shapes:
            # A bucket serves batches of at least its live rows and more
            # tokens than the next smaller bucket of its rows and kind.
            kind = (shape.row_bucket, shape.causal, shape.embeddings)
            rows = shape.live_rows
            tokens = max(rows, previous.get(kind, 0) + 1)
            previous[kind] = shape.token_bucket
            lengths = [tokens // rows] * rows
            for index in range(tokens % rows):
                lengths[index] += 1
            assert prefill_units(pages, rows, tokens) <= units - 1
            assert sum(map(manager.page_units, lengths)) <= units - 1

            entries, unit = [], 1
            for slot, length in enumerate(lengths, start=1):
                for group, page in enumerate(manager.shapes):
                    count = ceil_div(length, page.page_tokens)
                    held = count * page.units_per_page
                    entries.append(
                        (
                            slot,
                            group,
                            0,
                            tuple(range(unit, unit + held)),
                            count * page.page_tokens,
                        )
                    )
                    unit += held
            manager.block_tables.install(tuple(entries))

            batch = _rows(
                tuple(
                    torch.randint(7, 58, (length,), generator=generator)
                    for length in lengths
                ),
                prefix=0,
                selections=(TokenSelection.LAST_LOGITS,) * rows,
                causal=shape.causal,
                embeddings=tuple(
                    torch.randn((length, 32), generator=generator).to(
                        torch.bfloat16
                    )
                    for length in lengths
                )
                if shape.embeddings
                else None,
            )
            output = _call(runner, manager, batch)
            assert output.stats.cuda_graph_replays == 1, shape
