"""Automatic attention evaluates block-diffusion calls natively on SM100.

A two-group unit pool holds a sliding layer (16 query and 8 KV heads of
width 256 reading a history window) and a full layer (16 query and 2 KV
heads of width 512) whose pages hold twice as many tokens. Through automatic
provider selection each call kind of a block-diffusion model reads its paged
history and matches the portable torch provider within the established BF16
attention tolerance:

- causal rows (prompt chunks and committed blocks) append their K/V and
  attend within the history window;
- non-causal block rows (image blocks) append their K/V and attend to their
  whole block and their windowed history;
- segmented canvas rows read a read-only prefix window and their whole
  canvas;
- one prefill call mixes causal and non-causal rows, including consecutive
  rows of one sequence.

Sliding tables start after retired pages wherever the window allows, and
padding rows (a sequence over the sentinel unit whose writes are skipped,
then an empty sequence) follow the live rows, as a graph bucket pads them.
Unit bytes that no row may read hold finite noise, as reused units do.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace

import pytest
import torch
from torch.nn import functional as F

from uniserve.cache import Config, mha
from uniserve.model import TextSize
from uniserve.nn.attention import (
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.runtime import PrefixCache, TensorBuffers
from uniserve.runtime.backends.attention import resolve
from uniserve.runtime.backends.attention.torch import Backend as TorchBackend

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 10,
        reason="native block-diffusion attention requires an SM100 GPU",
    ),
]

# (query heads, KV heads, head dim) of the sliding and full layers.
_HEADS = {"sliding": (16, 8, 256), "full": (16, 2, 512)}
_TOLERANCE = {"rtol": 2e-2, "atol": 2e-2}
_UNITS = 1024


@dataclass(frozen=True)
class Row:
    """One sequence of a call: absolute prefix and query token counts.

    Rows naming the same ``sequence`` are consecutive chunks of one
    sequence and share its units; other rows own theirs.
    """

    prefix: int
    queries: int
    causal: bool = True
    sequence: int | None = None


@contextmanager
def _pool(block_size, window):
    layers = Config(
        {
            "sliding": mha.Config(
                8, 256, tuple(range(8)), torch.bfloat16, window
            ),
            "full": mha.Config(2, 512, (0, 1), torch.bfloat16),
        }
    )
    with PrefixCache(
        layers, num_units=_UNITS, block_size=block_size, device="cuda"
    ) as cache:
        yield cache


@contextmanager
def _operators(state, *, layer, window, size):
    """Prepare the automatic and the portable operator of one layer."""
    num_heads, num_kv_heads, head_dim = _HEADS[layer]
    arguments = {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": torch.bfloat16,
        "size": size,
        "cache": state,
        "window": window,
    }
    with ExitStack() as scope:
        operators = []
        for backend in (
            resolve("auto", device=torch.device("cuda", 0)),
            TorchBackend(),
        ):
            requirements = backend.workspace_buffers(**arguments)
            buffers = scope.enter_context(
                TensorBuffers.allocate(requirements, device="cuda")
            )
            operator = backend.prepare(
                **arguments, workspace=buffers.view(requirements)
            )
            scope.callback(operator.close)
            operators.append(operator)
        yield operators


def _pages(row, *, window, page, segmented):
    """Return the row's first and end logical pages.

    The first page holds the row's lowest visible history key, so earlier
    pages are retired; the end page follows its last key: its prefix for a
    segmented read, its appended queries otherwise.
    """
    first = 0 if window is None else max(row.prefix - window, 0) // page
    end = row.prefix + (0 if segmented else row.queries)
    return first, max(-(-end // page), first + 1)


def _history(state, rows, *, window, units, generator, segmented):
    """Write each sequence's history; return every row's units and start.

    A sequence owns units for its logical pages from its first row's start
    page to its last row's end page, and holds its history up to its first
    row's prefix; later rows of the sequence read keys the call appends.
    """
    page = state.block_size
    kv_heads, head_dim = state.key.shape[2:]
    extents = {}
    for index, row in enumerate(rows):
        name = index if row.sequence is None else row.sequence
        first, end = _pages(row, window=window, page=page, segmented=segmented)
        previous = extents.get(name, (first, end, row.prefix))
        extents[name] = (previous[0], max(previous[1], end), previous[2])

    owned = {}
    for name, (first, end, prefix) in extents.items():
        owned[name] = tuple(units[: end - first])
        del units[: end - first]
        count = prefix - first * page
        if count:
            keys, values = (
                torch.randn(
                    (count, kv_heads, head_dim),
                    device="cuda",
                    generator=generator,
                ).bfloat16()
                for _ in range(2)
            )
            state.write(owned[name], start=0, key=keys, value=values)

    blocks, starts = [], []
    for index, row in enumerate(rows):
        name = index if row.sequence is None else row.sequence
        first, end = _pages(row, window=window, page=page, segmented=segmented)
        origin = extents[name][0]
        blocks.append(owned[name][first - origin : end - origin])
        starts.append(first)
    return tuple(blocks), tuple(starts)


def _call(state, rows, *, kind, window, units, generator, padding):
    """Build one call's input over fresh units of ``state``.

    Returns the numerical input and the number of live query tokens, which
    precede the padding rows' tokens.
    """
    segmented = kind == "canvas"
    blocks, starts = _history(
        state,
        rows,
        window=window,
        units=units,
        generator=generator,
        segmented=segmented,
    )
    live = sum(row.queries for row in rows)
    queries = tuple(row.queries for row in rows)
    prefixes = tuple(row.prefix for row in rows)
    causal = tuple(row.causal for row in rows)
    if padding:
        # One sequence of padding tokens over the sentinel unit, then an
        # empty sequence; neither writes the cache.
        blocks += ((0,), (0,))
        starts += (0, 0)
        queries += (padding, 0)
        prefixes += (0, 0)
        causal += (causal[0],) * 2

    if segmented:
        width = max(map(len, blocks))
        indices = torch.tensor(
            [row + (0,) * (width - len(row)) for row in blocks],
            dtype=torch.int32,
            device="cuda",
        )
        table = BlockTable(
            indices,
            state.block_size,
            None
            if window is None
            else torch.tensor(starts, dtype=torch.int32, device="cuda"),
            None if window is None else starts,
        )
        lengths = SequenceLengths.from_lengths(queries, device="cuda")
        ends = torch.zeros(
            (len(queries), max(queries)), dtype=torch.int32, device="cuda"
        )
        return (
            SegmentedInput(
                lengths,
                SequenceLengths.from_lengths(prefixes, device="cuda"),
                table,
                None,
                ends,
                True,
            ),
            live,
        )

    paged = PagedInput.from_blocks(
        blocks=blocks,
        query_lengths=queries,
        prefix_lengths=prefixes,
        block_size=state.block_size,
        causal=causal,
        device="cuda",
        start_pages=None if window is None else starts,
    )
    writes = paged.write_indices.clone()
    writes[live:] = -1
    return replace(paged, write_indices=writes), live


def _widen(batch, width):
    """Pad a call's block table with sentinel columns to ``width``."""
    table = batch.block_table
    extra = width - table.indices.shape[1]
    return replace(
        batch,
        block_table=replace(table, indices=F.pad(table.indices, (0, extra))),
    )


def _projections(tokens, layer, generator):
    """Per-head RMS-normalized Q/K and unit-scale V, as model norms give."""
    num_heads, num_kv_heads, head_dim = _HEADS[layer]

    def normalized(heads):
        values = torch.randn(
            (tokens, heads, head_dim), device="cuda", generator=generator
        )
        scale = torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (values * scale).bfloat16()

    value = torch.randn(
        (tokens, num_kv_heads, head_dim), device="cuda", generator=generator
    )
    return normalized(num_heads), normalized(num_kv_heads), value.bfloat16()


def _reference(reference, q, k, v, batch, out):
    """Evaluate ``batch`` portably over the cache the native call wrote."""
    if isinstance(batch, PagedInput):
        batch = replace(batch, write_indices=None)
    return reference(q, k, v, batch, scale=1.0, out=out)


# Live rows per call kind. The first sequence's history exceeds the
# production window, so its sliding table starts after retired pages; the
# 280-token blocks exceed the short window, so their queries read block keys
# beyond it.
_ROWS = {
    "causal": (Row(1500, 37), Row(300, 256), Row(0, 19)),
    "block": (Row(1500, 37, False), Row(300, 280, False), Row(0, 19, False)),
    "canvas": (Row(1500, 256), Row(300, 256), Row(0, 256)),
    "mixed": (
        Row(1500, 37, True, 0),
        Row(1537, 280, False, 0),
        Row(1817, 21, True, 0),
        Row(300, 64, False),
    ),
}

# Sliding layers at the production window and at a window shorter than a
# block; full layers read the whole history.
_LAYERS = (("sliding", 1023), ("sliding", 40), ("full", None))


@torch.inference_mode()
@pytest.mark.parametrize("kind", tuple(_ROWS))
@pytest.mark.parametrize(("layer", "window"), _LAYERS)
@pytest.mark.parametrize("block_size", [16, 32])
def test_calls_match_the_portable_reference(kind, layer, window, block_size):
    generator = torch.Generator(device="cuda").manual_seed(block_size)
    rows = _ROWS[kind]
    padding = 0 if kind == "mixed" else 13
    with _pool(block_size, window if layer == "sliding" else 1023) as cache:
        state = cache.state(layer)
        state.key.normal_(generator=generator)
        state.value.normal_(generator=generator)
        batch, live = _call(
            state,
            rows,
            kind=kind,
            window=window,
            units=list(range(1, _UNITS)),
            generator=generator,
            padding=padding,
        )
        tokens = live + padding
        q, k, v = _projections(tokens, layer, generator)
        size = TextSize(tokens, batch.queries.batch_size)
        with _operators(state, layer=layer, window=window, size=size) as (
            native,
            reference,
        ):
            actual, expected = torch.empty_like(q), torch.empty_like(q)
            native.bind(batch)
            native(q, k, v, batch, scale=1.0, out=actual)
            _reference(reference, q, k, v, batch, expected)

    torch.testing.assert_close(actual[:live], expected[:live], **_TOLERANCE)


# Captured and replayed calls: equal row and token counts, other lengths,
# prefixes, pages and start pages.
_REPLAYS = {
    "causal": (
        (Row(1500, 100), Row(40, 156)),
        (Row(2600, 200), Row(700, 56)),
    ),
    "block": (
        (Row(1500, 100, False), Row(40, 156, False)),
        (Row(2600, 200, False), Row(700, 56, False)),
    ),
    "canvas": (
        (Row(1500, 256), Row(40, 256)),
        (Row(2600, 256), Row(1100, 256)),
    ),
}


@torch.inference_mode()
@pytest.mark.parametrize("kind", tuple(_REPLAYS))
@pytest.mark.parametrize(
    ("layer", "window"), (("sliding", 1023), ("full", None))
)
def test_graph_replay_reads_changed_lengths_and_tables(kind, layer, window):
    generator = torch.Generator(device="cuda").manual_seed(11)
    captured_rows, replayed_rows = _REPLAYS[kind]
    padding = 7
    with _pool(16, 1023) as cache:
        state = cache.state(layer)
        state.key.normal_(generator=generator)
        state.value.normal_(generator=generator)
        # Disjoint units keep the captured call's writes out of the
        # replayed call's history.
        units = list(range(1, _UNITS))
        calls = [
            _call(
                state,
                rows,
                kind=kind,
                window=window,
                units=units,
                generator=generator,
                padding=padding,
            )
            for rows in (captured_rows, replayed_rows)
        ]
        (captured, live), (replayed, replayed_live) = calls
        assert live == replayed_live
        # The captured table spans the widest table a replay may read.
        width = max(
            call.block_table.indices.shape[1] for call in (captured, replayed)
        )
        captured = _widen(captured, width)

        tokens = live + padding
        q, k, v = _projections(tokens, layer, generator)
        size = TextSize(tokens, captured.queries.batch_size)
        with _operators(state, layer=layer, window=window, size=size) as (
            native,
            reference,
        ):
            actual = torch.empty_like(q)
            native.bind(captured)
            # The eager call compiles every kernel specialization first.
            native(q, k, v, captured, scale=1.0, out=actual)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                native(q, k, v, captured, scale=1.0, out=actual)

            # Replay reads the captured buffers with the other call's values.
            _assign(captured, replayed)
            for tensor, value in zip(
                (q, k, v), _projections(tokens, layer, generator), strict=True
            ):
                tensor.copy_(value)
            actual.zero_()
            graph.replay()

            expected = torch.empty_like(q)
            _reference(reference, q, k, v, replayed, expected)
            torch.cuda.synchronize()
            graph.reset()

    torch.testing.assert_close(actual[:live], expected[:live], **_TOLERANCE)


def _assign(target, source):
    """Copy ``source``'s device metadata into ``target``'s buffers."""
    table, values = target.block_table, source.block_table
    table.indices.zero_()
    table.indices[:, : values.indices.shape[1]].copy_(values.indices)
    pairs = [
        (target.queries.values, source.queries.values),
        (target.queries.offsets, source.queries.offsets),
        (target.prefixes.values, source.prefixes.values),
        (target.prefixes.offsets, source.prefixes.offsets),
    ]
    if table.start_page is not None:
        pairs.append((table.start_page, values.start_page))
    if getattr(target, "write_indices", None) is not None:
        pairs.append((target.write_indices, source.write_indices))
    for destination, value in pairs:
        destination.copy_(value)


@torch.inference_mode()
def test_windowed_calls_without_a_native_kernel_raise_when_bound():
    # Head dimension 512 has no windowed TensorRT-LLM context kernel, and
    # the prefix-block kernel reads only non-causal rows.
    with _pool(16, 7) as cache:
        state = cache.state("full")
        batch = PagedInput.from_blocks(
            blocks=((1, 2), (3,)),
            query_lengths=(20, 4),
            prefix_lengths=(30, 0),
            block_size=state.block_size,
            causal=True,
            device="cuda",
        )
        with (
            _operators(state, layer="full", window=7, size=TextSize(24, 2)) as (
                native,
                _,
            ),
            pytest.raises(
                ValueError,
                match=(
                    r"paged attention of 2 rows.*causal rows.*16 query and 2 "
                    r"KV heads of dimension 512.*bfloat16.*32-token pages.*"
                    r"7-token history window"
                ),
            ),
        ):
            native.bind(batch)


def test_windowed_layers_without_native_storage_raise_when_prepared():
    # Native windowed kernels compute in half precision only.
    with pytest.raises(
        ValueError,
        match=r"no native attention kernel.*dimension 256.*float32.*1023",
    ):
        resolve("auto", device=torch.device("cuda", 0)).prepare(
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            dtype=torch.float32,
            size=TextSize(8, 1),
            cache=None,
            workspace={},
            window=1023,
        )
