"""SM100 prefix-block attention reads exactly each row's visible keys.

Every query row of a block attends to its block (all of it, or with causal
ordering the rows up to its own) and to a window of the sequence's paged
prefix. Expected values come from an FP32 softmax over keys gathered
explicitly through the block table. Cache contents that no row may see, the
sentinel page, and packed rows after the last sequence hold NaN, so any read
of them reaches the output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import accumulate

import pytest
import torch
from uniserve_kernels.attention import prefix_block

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not prefix_block.available(),
        reason="requires the CUTLASS DSL and a compute capability 10.x GPU",
    ),
]

# (query heads, KV heads, head dim) of the model's sliding-window and
# full-attention layers.
_SLIDING = (16, 8, 256)
_FULL = (16, 2, 512)
# Packed rows after the last sequence, as in a graph-captured buffer.
_PADDING_ROWS = 9
_UNWRITTEN = 7.0
_TOLERANCE = {"rtol": 2e-2, "atol": 2e-2}
# Causal launches run one after another on one stream, so they share a
# workspace sized for the largest batch below.
_MAX_SEQUENCES = 16
_WORKSPACE = {}


def _workspace():
    if "counters" not in _WORKSPACE:
        _WORKSPACE["counters"] = prefix_block.new_workspace(
            torch.device("cuda"), _MAX_SEQUENCES
        )
    return _WORKSPACE["counters"]


def _reset(workspace):
    # The launch that precedes a causal launch on its stream zeroes the
    # ticket counter.
    workspace[:1].zero_()


@dataclass(frozen=True)
class Batch:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    query_offsets: torch.Tensor
    prefix_lengths: torch.Tensor
    start_page: torch.Tensor | None = None
    prefix_start: torch.Tensor | None = None


def _normalized(shape, generator):
    # Per-head RMS-normalized projections, as the model's Q/K norms produce.
    values = torch.randn(shape, generator=generator, device="cuda")
    scale = torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (values * scale).to(torch.bfloat16)


def _int32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _batch(
    lengths,
    prefixes,
    *,
    page_tokens,
    heads=_SLIDING,
    start_pages=None,
    prefix_starts=None,
    seed=0,
    pages=None,
    table_width=None,
):
    """Build packed blocks and a NaN-filled paged cache.

    Physical page 0 is a NaN sentinel that fills unused block table
    entries. Sequence ``b`` owns logical pages ``[start_pages[b],
    ceil(P_b / page_tokens))`` on distinct random physical pages; only its
    prefix tokens are written, so unwritten page tails and unowned pages
    stay NaN.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    starts = start_pages or (0,) * len(lengths)
    tokens = sum(lengths)
    query_heads, kv_heads, head_dim = heads

    query = _normalized(
        (tokens + _PADDING_ROWS, query_heads, head_dim), generator
    )
    key = _normalized((tokens + _PADDING_ROWS, kv_heads, head_dim), generator)
    value = torch.randn(key.shape, generator=generator, device="cuda")
    value = value.to(torch.bfloat16)
    for tensor in (query, key, value):
        tensor[tokens:] = float("nan")

    owned = [
        max(0, math.ceil(prefix / page_tokens) - start)
        for prefix, start in zip(prefixes, starts, strict=True)
    ]
    pages = pages or 1 + sum(owned) + 3
    shape = (pages, page_tokens, kv_heads, head_dim)
    key_cache = torch.full(shape, float("nan"), device="cuda")
    value_cache = torch.full(shape, float("nan"), device="cuda")
    key_cache = key_cache.to(torch.bfloat16)
    value_cache = value_cache.to(torch.bfloat16)

    physical = 1 + torch.randperm(
        pages - 1, generator=torch.Generator().manual_seed(seed)
    )
    table = torch.zeros(
        (len(lengths), table_width or max(owned + [1]) + 2),
        dtype=torch.int32,
    )
    used = 0
    for row, (prefix, start, count) in enumerate(
        zip(prefixes, starts, owned, strict=True)
    ):
        for column in range(count):
            page = int(physical[used])
            used += 1
            table[row, column] = page
            written = min(page_tokens, prefix - (start + column) * page_tokens)
            rows = (written, kv_heads, head_dim)
            key_cache[page, :written] = _normalized(rows, generator)
            value_cache[page, :written] = torch.randn(
                rows, generator=generator, device="cuda"
            ).to(torch.bfloat16)

    return Batch(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=table.cuda(),
        query_offsets=_int32(tuple(accumulate(lengths, initial=0))),
        prefix_lengths=_int32(prefixes),
        start_page=None if start_pages is None else _int32(start_pages),
        prefix_start=None if prefix_starts is None else _int32(prefix_starts),
    )


def _reference(batch, *, window, query_window, scale, causal=False):
    """FP32 attention over explicitly gathered visible keys.

    Returns the output ``[tokens, Hq, D]`` and the natural-log LSE
    ``[tokens, Hq]`` of every packed row inside a block.
    """
    offsets = batch.query_offsets.tolist()
    page_tokens = batch.key_cache.shape[1]
    _tokens, query_heads, head_dim = batch.query.shape
    group = query_heads // batch.key.shape[1]
    kv_of_head = torch.arange(query_heads, device="cuda") // group
    output = torch.empty((offsets[-1], query_heads, head_dim), device="cuda")
    lse = torch.empty((offsets[-1], query_heads), device="cuda")

    for row, prefix in enumerate(batch.prefix_lengths.tolist()):
        begin, end = offsets[row], offsets[row + 1]
        positions = torch.arange(end - begin, device="cuda")
        lower = torch.zeros_like(positions)
        if batch.prefix_start is not None:
            lower += int(batch.prefix_start[row])
        if window is not None:
            history = prefix - window + (positions if query_window else 0)
            lower = torch.maximum(lower, torch.as_tensor(history).cuda())
        first = int(lower.clamp(0, prefix).min())
        # Gather, through the table, the written prefix tokens that any row
        # can see.
        tokens = torch.arange(first, prefix, device="cuda")
        start = 0 if batch.start_page is None else int(batch.start_page[row])
        pages = batch.block_table[row, tokens // page_tokens - start].long()
        slots = tokens % page_tokens
        keys = torch.cat(
            (batch.key_cache[pages, slots], batch.key[begin:end])
        ).float()
        values = torch.cat(
            (batch.value_cache[pages, slots], batch.value[begin:end])
        ).float()
        block = torch.ones(
            (end - begin, end - begin), dtype=torch.bool, device="cuda"
        )
        if causal:
            block = block.tril()
        visible = torch.cat((tokens[None, :] >= lower[:, None], block), dim=1)
        scores = torch.einsum(
            "qhd,khd->hqk",
            batch.query[begin:end].float(),
            keys[:, kv_of_head],
        )
        scores = (scores * scale).masked_fill(~visible, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output[begin:end] = torch.einsum(
            "hqk,khd->qhd", weights, values[:, kv_of_head]
        )
        lse[begin:end] = torch.logsumexp(scores, dim=-1).T
    return output, lse


def _launch(
    batch,
    *,
    window,
    query_window,
    scale,
    out,
    lse,
    lse_base2=False,
    max_query_len=280,
    causal=False,
):
    workspace = None
    if causal:
        workspace = _workspace()
        _reset(workspace)
    prefix_block.prefix_block_attention(
        batch.query,
        batch.key,
        batch.value,
        batch.key_cache,
        batch.value_cache,
        batch.block_table,
        batch.query_offsets,
        batch.prefix_lengths,
        max_query_len=max_query_len,
        workspace=workspace,
        window=window,
        query_window=query_window,
        causal=causal,
        start_page=batch.start_page,
        prefix_start=batch.prefix_start,
        scale=scale,
        out=out,
        lse=lse,
        lse_base2=lse_base2,
    )


def _outputs(batch):
    out = torch.full_like(batch.query, _UNWRITTEN)
    lse = torch.full(batch.query.shape[:2], _UNWRITTEN, device="cuda")
    return out, lse


def _assert_matches(
    batch, out, lse, *, window, query_window, scale, base2, causal=False
):
    expected, expected_lse = _reference(
        batch,
        window=window,
        query_window=query_window,
        scale=scale,
        causal=causal,
    )
    tokens = expected.shape[0]
    if base2:
        expected_lse = expected_lse / math.log(2.0)
    torch.testing.assert_close(out[:tokens].float(), expected, **_TOLERANCE)
    torch.testing.assert_close(lse[:tokens], expected_lse, **_TOLERANCE)
    # Rows after the packed sequences are not written.
    assert bool((out[tokens:] == _UNWRITTEN).all())
    assert bool((lse[tokens:] == _UNWRITTEN).all())


def _history_start_pages(prefixes, page_tokens):
    # Pages entirely before the 1023-token window are retired, so the
    # table's first column is a later logical page.
    return tuple(max(0, prefix - 1023) // page_tokens for prefix in prefixes)


@pytest.mark.parametrize("page_tokens", [16, 32])
@torch.inference_mode()
def test_canvas_reads_fixed_history_window(page_tokens):
    # Empty prefix, prefixes shorter than, at and around the window, one
    # that is not a page multiple, and a block shorter than a key tile.
    lengths = (256, 256, 256, 256, 256, 256, 17)
    prefixes = (0, 5, 1022, 1023, 1024, 1025, 3001)
    batch = _batch(
        lengths,
        prefixes,
        page_tokens=page_tokens,
        start_pages=_history_start_pages(prefixes, page_tokens),
        seed=page_tokens,
    )
    out, lse = _outputs(batch)

    _launch(
        batch, window=1023, query_window=False, scale=1 / 16, out=out, lse=lse
    )

    _assert_matches(
        batch,
        out,
        lse,
        window=1023,
        query_window=False,
        scale=1 / 16,
        base2=False,
    )


@pytest.mark.parametrize("page_tokens", [16, 32])
@torch.inference_mode()
def test_image_block_window_follows_each_query(page_tokens):
    lengths = (280, 280, 280, 280, 3)
    prefixes = (0, 300, 1023, 1500, 2000)
    batch = _batch(
        lengths,
        prefixes,
        page_tokens=page_tokens,
        start_pages=_history_start_pages(prefixes, page_tokens),
        seed=7 + page_tokens,
    )
    out, lse = _outputs(batch)

    _launch(
        batch, window=1023, query_window=True, scale=1 / 16, out=out, lse=lse
    )

    _assert_matches(
        batch,
        out,
        lse,
        window=1023,
        query_window=True,
        scale=1 / 16,
        base2=False,
    )


@pytest.mark.parametrize("page_tokens", [16, 32])
@pytest.mark.parametrize("query_window", [False, True])
@torch.inference_mode()
def test_small_batch_reads_history_window(page_tokens, query_window):
    # A batch too small to occupy the GPU with the widest work tiles (two
    # sequences of at most 256 queries): a full block over a prefix just past
    # the window and a short block over a prefix that is not a page multiple.
    lengths = (256, 17)
    prefixes = (1025, 3001)
    batch = _batch(
        lengths,
        prefixes,
        page_tokens=page_tokens,
        start_pages=_history_start_pages(prefixes, page_tokens),
        seed=41 + page_tokens,
    )
    out, lse = _outputs(batch)

    _launch(
        batch,
        window=1023,
        query_window=query_window,
        scale=1 / 16,
        out=out,
        lse=lse,
        max_query_len=max(lengths),
    )

    _assert_matches(
        batch,
        out,
        lse,
        window=1023,
        query_window=query_window,
        scale=1 / 16,
        base2=False,
    )


@pytest.mark.parametrize("heads", [_SLIDING, _FULL], ids=["hd256", "hd512"])
@pytest.mark.parametrize("query_window", [False, True])
@torch.inference_mode()
def test_short_window_bounds_are_exact(heads, query_window):
    # With a few visible keys per row every key carries a large share of
    # the softmax, so a key admitted or dropped at either window edge moves
    # the output well past the tolerance. Windows cut pages mid-way.
    lengths = (5, 3, 9)
    prefixes = (20, 37, 3)
    batch = _batch(lengths, prefixes, page_tokens=16, heads=heads, seed=23)
    out, lse = _outputs(batch)

    _launch(
        batch,
        window=3,
        query_window=query_window,
        scale=1.0,
        out=out,
        lse=lse,
    )

    _assert_matches(
        batch,
        out,
        lse,
        window=3,
        query_window=query_window,
        scale=1.0,
        base2=False,
    )


@pytest.mark.parametrize("page_tokens", [32, 64])
@torch.inference_mode()
def test_full_layer_reads_whole_prefix(page_tokens):
    # Head dim 512 with 8 query heads per KV head: canvas blocks and image
    # blocks read the whole prefix; lengths and prefixes are not tile or page
    # multiples.
    lengths = (256, 256, 280, 17)
    prefixes = (0, 1025, 300, 4097)
    batch = _batch(
        lengths, prefixes, page_tokens=page_tokens, heads=_FULL, seed=31
    )
    out, lse = _outputs(batch)

    _launch(
        batch,
        window=None,
        query_window=False,
        scale=512**-0.5,
        out=out,
        lse=lse,
    )

    _assert_matches(
        batch,
        out,
        lse,
        window=None,
        query_window=False,
        scale=512**-0.5,
        base2=False,
    )


@pytest.mark.parametrize("page_tokens", [32, 64])
@torch.inference_mode()
def test_causal_chunks_continue_their_history(page_tokens):
    # Head dimension 512 prefill chunks: a prompt without history, chunks
    # after long prefixes that are not page multiples, and a short final
    # chunk. Block lengths are not key-tile multiples, so the diagonal cuts
    # key tiles at every offset of the 16-query work tiles.
    lengths = (2048, 1023, 300, 17, 130)
    prefixes = (0, 300, 14336, 4097, 64)
    batch = _batch(
        lengths, prefixes, page_tokens=page_tokens, heads=_FULL, seed=61
    )
    out, lse = _outputs(batch)

    _launch(
        batch,
        window=None,
        query_window=False,
        scale=512**-0.5,
        out=out,
        lse=lse,
        max_query_len=max(lengths),
        causal=True,
    )

    _assert_matches(
        batch,
        out,
        lse,
        window=None,
        query_window=False,
        scale=512**-0.5,
        base2=False,
        causal=True,
    )


@pytest.mark.parametrize("lse_base2", [False, True])
@torch.inference_mode()
def test_causal_diagonal_is_exact(lse_base2):
    # Early rows of a chunk without history see one or two keys, so a key
    # admitted past a row's own position or dropped at it moves the output
    # well past the tolerance. Short chunks after prefixes that cut pages.
    lengths = (5, 3, 9, 200)
    prefixes = (20, 37, 3, 0)
    batch = _batch(lengths, prefixes, page_tokens=16, heads=_FULL, seed=67)
    out, lse = _outputs(batch)

    _launch(
        batch,
        window=None,
        query_window=False,
        scale=512**-0.5,
        out=out,
        lse=lse,
        lse_base2=lse_base2,
        causal=True,
    )

    _assert_matches(
        batch,
        out,
        lse,
        window=None,
        query_window=False,
        scale=512**-0.5,
        base2=lse_base2,
        causal=True,
    )


@torch.inference_mode()
def test_causal_graph_replay_reads_updated_lengths():
    # One capture of the counter reset and a causal launch whose bound
    # grid has more tiles than clusters, replayed with changed lengths and
    # prefixes: short chunks leave many first-wave grid tiles empty, so
    # their clusters go straight to the dynamically claimed tiles.
    pages, width = 800, 200
    batches = [
        _batch(
            (1024, 17, 600, 1, 300),
            (4097, 0, 1500, 64, 12000),
            page_tokens=64,
            heads=_FULL,
            seed=83,
            pages=pages,
            table_width=width,
        ),
        _batch(
            (5, 1024, 130, 700, 1024),
            (300, 9000, 0, 2049, 64),
            page_tokens=64,
            heads=_FULL,
            seed=89,
            pages=pages,
            table_width=width,
        ),
    ]
    rows = max(batch.query.shape[0] for batch in batches)

    def resized(tensor):
        grown = torch.full(
            (rows, *tensor.shape[1:]),
            float("nan"),
            dtype=tensor.dtype,
            device="cuda",
        )
        grown[: tensor.shape[0]] = tensor
        return grown

    live = replace(
        batches[0],
        query=resized(batches[0].query),
        key=resized(batches[0].key),
        value=resized(batches[0].value),
    )
    out, lse = _outputs(live)
    options = {"window": None, "query_window": False, "scale": 512**-0.5}

    def launch():
        _launch(
            live, out=out, lse=lse, max_query_len=1024, causal=True, **options
        )

    # Compile outside the capture, then record the reset and the launch.
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for batch in batches:
        for name in ("query", "key", "value"):
            getattr(live, name).copy_(resized(getattr(batch, name)))
        for name in (
            "key_cache",
            "value_cache",
            "block_table",
            "query_offsets",
            "prefix_lengths",
        ):
            getattr(live, name).copy_(getattr(batch, name))
        out.fill_(_UNWRITTEN)
        lse.fill_(_UNWRITTEN)

        graph.replay()

        _assert_matches(live, out, lse, base2=False, causal=True, **options)


@pytest.mark.parametrize("lse_base2", [False, True])
@torch.inference_mode()
def test_whole_prefix_from_explicit_start(lse_base2):
    # Without a window every row reads the prefix from its start column; the
    # model's unit softmax scale.
    lengths = (256, 280, 64)
    prefixes = (300, 1000, 64)
    batch = _batch(
        lengths, prefixes, page_tokens=16, prefix_starts=(100, 0, 63), seed=11
    )
    out, lse = _outputs(batch)

    _launch(
        batch,
        window=None,
        query_window=False,
        scale=1.0,
        out=out,
        lse=lse,
        lse_base2=lse_base2,
    )

    _assert_matches(
        batch,
        out,
        lse,
        window=None,
        query_window=False,
        scale=1.0,
        base2=lse_base2,
    )


@torch.inference_mode()
def test_graph_replay_reads_updated_lengths_and_tables():
    # Two batches of different lengths, prefixes, start pages and tables
    # replayed through one capture sized for both.
    pages, width = 400, 80
    batches = [
        _batch(
            (280, 256),
            (1500, 700),
            page_tokens=16,
            start_pages=(29, 0),
            seed=3,
            pages=pages,
            table_width=width,
        ),
        _batch(
            (17, 280),
            (2000, 0),
            page_tokens=16,
            start_pages=(61, 0),
            seed=4,
            pages=pages,
            table_width=width,
        ),
    ]
    rows = max(batch.query.shape[0] for batch in batches)

    def resized(tensor):
        grown = torch.full(
            (rows, *tensor.shape[1:]),
            float("nan"),
            dtype=tensor.dtype,
            device="cuda",
        )
        grown[: tensor.shape[0]] = tensor
        return grown

    live = replace(
        batches[0],
        query=resized(batches[0].query),
        key=resized(batches[0].key),
        value=resized(batches[0].value),
    )
    out, lse = _outputs(live)

    def launch():
        _launch(
            live,
            window=1023,
            query_window=True,
            scale=1 / 16,
            out=out,
            lse=lse,
        )

    # Compile outside the capture, then record the launch.
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for batch in batches:
        for name in ("query", "key", "value"):
            getattr(live, name).copy_(resized(getattr(batch, name)))
        for name in (
            "key_cache",
            "value_cache",
            "block_table",
            "query_offsets",
            "prefix_lengths",
            "start_page",
        ):
            getattr(live, name).copy_(getattr(batch, name))
        out.fill_(_UNWRITTEN)
        lse.fill_(_UNWRITTEN)

        graph.replay()

        _assert_matches(
            live,
            out,
            lse,
            window=1023,
            query_window=True,
            scale=1 / 16,
            base2=False,
        )


@torch.inference_mode()
def test_capture_at_another_batch_size_after_one_call():
    # One eager call of a configuration (here 64-token pages, used by no
    # other head-dim-256 test) prepares it for every batch size: a graph
    # captured at a much smaller batch compiles nothing during capture.
    prefixes = (1025,) * 8
    large = _batch(
        (256,) * 8,
        prefixes,
        page_tokens=64,
        start_pages=_history_start_pages(prefixes, 64),
        seed=51,
    )
    small = _batch(
        (256, 17),
        (1025, 3001),
        page_tokens=64,
        start_pages=_history_start_pages((1025, 3001), 64),
        seed=52,
    )
    options = {"window": 1023, "query_window": False, "scale": 1 / 16}

    out, lse = _outputs(large)
    _launch(large, out=out, lse=lse, max_query_len=256, **options)
    torch.cuda.synchronize()

    out, lse = _outputs(small)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(small, out=out, lse=lse, max_query_len=256, **options)
    graph.replay()

    _assert_matches(small, out, lse, base2=False, **options)


def test_eligibility_reports_unsupported_page_size():
    batch = _batch((256,), (100,), page_tokens=16)
    arguments = [
        batch.query,
        batch.key,
        batch.value,
        batch.key_cache,
        batch.value_cache,
        batch.block_table,
        batch.query_offsets,
        batch.prefix_lengths,
    ]
    assert prefix_block.can_run(*arguments, window=1023)

    wide = batch.key_cache.new_zeros((4, 128, *batch.key_cache.shape[2:]))
    arguments[3:5] = [wide, wide]
    assert not prefix_block.can_run(*arguments, window=1023)
    with pytest.raises(ValueError, match="page_tokens"):
        prefix_block.prefix_block_attention(*arguments, max_query_len=256)


def test_eligibility_reports_unsupported_causal_configuration():
    # Causal ordering serves full-attention layers only: head dimension 512
    # without a history window.
    sliding = _batch((256,), (100,), page_tokens=16)
    full = _batch((256,), (100,), page_tokens=16, heads=_FULL)

    def arguments(batch):
        return (
            batch.query,
            batch.key,
            batch.value,
            batch.key_cache,
            batch.value_cache,
            batch.block_table,
            batch.query_offsets,
            batch.prefix_lengths,
        )

    assert prefix_block.can_run(*arguments(full), causal=True)
    assert not prefix_block.can_run(*arguments(sliding), causal=True)
    assert not prefix_block.can_run(*arguments(full), causal=True, window=1023)
    with pytest.raises(ValueError, match="causal"):
        prefix_block.prefix_block_attention(
            *arguments(sliding),
            max_query_len=256,
            workspace=_workspace(),
            causal=True,
        )
