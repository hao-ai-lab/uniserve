"""Token rows stage the attention columns their slots' resident tables imply.

``AttentionBuffers.stage_rows`` gathers every numerical table's units, first
staged pages and slot mapping from the request slots' resident tables.
Over random calls of a model whose windowed and full-attention groups share
one unit pool (six numerical tables, as DiffusionGemma has), its staging
equals the layout the tables define: every installed page of a full table
and, for a windowed table, the pages from the first one the row's first
query reaches back to through its last query; slot mapping at
``unit * block_size + offset``, or -1 for rows that do not write. Padding
the call to a graph bucket leaves its live rows unchanged. The calls
cover appending rows, mixed appending and read-only rows, read-only
non-causal rows, rows whose windowed tables retired their first pages, and
padding past the live rows.
"""

from __future__ import annotations

import itertools
import random

import pytest
import torch

from tests.python.fixtures.hybrid import hybrid_model, hybrid_pool
from tests.python.fixtures.worker_config import cpu_worker_config
from uniserve.math import ceil_div
from uniserve.nn.attention import PagedInput
from uniserve_worker.model_executor.attention import first_page
from uniserve_worker.model_executor.graph_inputs import pad_text
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.model_executor.input_buffers import (
    TokenBufferConfig,
    TokenBuffers,
)
from uniserve_worker.protocol.call import ForwardMode
from uniserve_worker.sampling.metadata import TokenSelection

pytestmark = pytest.mark.integration

SLOTS, WIDTH, TOKENS, UNITS = 6, 16, 256, 512


def _install(manager, generator):
    """Install random tables for every slot; return each slot's extent.

    Windowed groups may start past page zero, as a request whose window
    retired its first pages does; every slot's units are distinct.
    """
    tables = manager.block_tables
    entries, extents, unit = [], {}, 1
    for slot in range(1, SLOTS + 1):
        extent = generator.randint(1, 96)
        extents[slot] = extent
        for group, shape in enumerate(tables.groups):
            pages = ceil_div(extent, shape.block_size)
            start = (
                generator.randint(0, max(0, pages - 2))
                if shape.window is not None
                else 0
            )
            count = (pages - start) * shape.units_per_page
            units = tuple(range(unit, unit + count))
            unit += count
            entries.append((slot, group, start, units, extent))
    tables.install(tuple(entries))
    return extents


def _rows(manager, extents, generator, *, write, causal):
    """Random rows over distinct slots whose reads avoid retired pages."""
    tables = manager.block_tables
    count = generator.randint(1, SLOTS)
    rows = []
    for slot in generator.sample(range(1, SLOTS + 1), count):
        extent = extents[slot]
        for _ in range(100):
            query = generator.randint(1, extent)
            prefix = (
                extent - query
                if write
                else generator.randint(0, extent - query)
            )
            groups = tuple(
                tables.table(slot, group) for group in range(len(tables.groups))
            )
            if all(
                table.shape.window is None
                or first_page(table, prefix) >= table.start_page
                for table in groups
            ):
                break
        else:
            continue
        # A mixed call's first row appends, so the call stays paged.
        writes = (
            write if write is not None else not rows or generator.random() < 0.5
        )
        rows.append(
            TokenRow(
                forward_mode=ForwardMode.PREFILL,
                token_ids=torch.randint(0, 8, (query,)),
                positions=torch.arange(prefix, prefix + query),
                selection=TokenSelection.LAST_LOGITS,
                request_pool_idx=slot,
                seq_len=prefix,
                write_kv=writes,
                causal=causal,
            )
        )
    return tuple(rows)


def _expected(rows, manager):
    """The attention columns the installed tables define for ``rows``.

    Returns the same keys as ``_columns``, computed from the host mirrors of
    the tables alone: per numerical table, the staged units of each row
    (zero past them, to the widest row), first pages and slot mapping.
    """
    tables = manager.block_tables
    queries = tuple(row.query_tokens for row in rows)
    prefixes = tuple(row.seq_len for row in rows)
    appending = any(row.write_kv for row in rows)
    values = {
        "queries": torch.tensor(queries, dtype=torch.int32),
        "query_offsets": torch.tensor(
            (0, *itertools.accumulate(queries)), dtype=torch.int32
        ),
        "query_host": queries,
    }
    for group, shape in enumerate(tables.groups):
        for position in range(shape.units_per_page):
            number = tables.first_table[group] + position
            staged, firsts, slot_mapping = [], [], []
            for row in rows:
                table = tables.table(row.request_pool_idx, group)
                first = (
                    first_page(table, row.seq_len)
                    if shape.window is not None
                    else 0
                )
                end = table.end_page
                if shape.window is not None:
                    end = min(
                        end,
                        ceil_div(
                            row.seq_len + row.query_tokens, shape.block_size
                        ),
                    )
                units = table.row(position)
                staged.append(
                    units[first - table.start_page : end - table.start_page]
                )
                firsts.append(first)
                for token in range(row.seq_len, row.seq_len + row.query_tokens):
                    page = token // shape.block_size - table.start_page
                    slot_mapping.append(
                        table.row(position)[page] * shape.block_size
                        + token % shape.block_size
                        if row.write_kv
                        else -1
                    )
            width = max(1, *map(len, staged))
            values[number, "kind"] = (
                "PagedInput" if appending else "SegmentedInput"
            )
            values[number, "prefixes"] = torch.tensor(
                prefixes, dtype=torch.int32
            )
            values[number, "prefix_offsets"] = torch.tensor(
                (0, *itertools.accumulate(prefixes)), dtype=torch.int32
            )
            values[number, "prefix_host"] = prefixes
            values[number, "table"] = torch.tensor(
                [[*units, *(0,) * (width - len(units))] for units in staged],
                dtype=torch.int32,
            )
            values[number, "block_size"] = shape.block_size
            values[number, "start"] = (
                None
                if shape.window is None
                else torch.tensor(firsts, dtype=torch.int32)
            )
            values[number, "start_host"] = (
                None if shape.window is None else tuple(firsts)
            )
            values[number, "slot_mapping"] = (
                torch.tensor(slot_mapping, dtype=torch.int32)
                if appending
                else None
            )
            if appending:
                values[number, "causal"] = tuple(row.causal for row in rows)
    return values


def _columns(batch):
    """Every staged attention column of a batch, on the host."""
    attention = batch.inputs.attention
    values = {
        "queries": attention.queries.values,
        "query_offsets": attention.queries.offsets,
        "query_host": attention.queries.host,
    }
    for number, entry in attention.entries.items():
        values[number, "kind"] = type(entry).__name__
        values[number, "prefixes"] = entry.prefixes.values
        values[number, "prefix_offsets"] = entry.prefixes.offsets
        values[number, "prefix_host"] = entry.prefixes.host
        values[number, "table"] = entry.block_table.indices
        values[number, "block_size"] = entry.block_table.block_size
        values[number, "start"] = entry.block_table.start_page
        values[number, "start_host"] = entry.block_table.start_page_host
        values[number, "slot_mapping"] = entry.slot_mapping
        if isinstance(entry, PagedInput):
            values[number, "causal"] = entry.causal
    return {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def _compare(columns, expected):
    assert columns.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(columns[key].to(value.dtype), value), key
        else:
            assert columns[key] == value, key


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
@pytest.mark.parametrize(
    ("write", "causal"),
    ((True, True), (None, True), (False, False)),
    ids=("appending", "mixed", "read_only"),
)
@torch.inference_mode()
def test_rows_stage_the_columns_their_resident_tables_define(
    device, write, causal
):
    model = hybrid_model()
    config = cpu_worker_config(device=device, block_size=16, max_model_len=128)
    manager = hybrid_pool(
        model, config, num_units=UNITS, request_pool_size=SLOTS
    )
    assert manager.block_tables.table_count == 6
    buffer_config = TokenBufferConfig(
        max_rows=2 * SLOTS,
        max_tokens=TOKENS,
        max_text_tokens=TOKENS,
        table_widths=(WIDTH,) * 6,
        hidden_size=0,
    )
    staged = TokenBuffers(config=buffer_config, device=device)
    generator = random.Random(1 + (write is None) + 2 * (not causal))
    try:
        extents = _install(manager, generator)
        for _ in range(12):
            rows = _rows(
                manager, extents, generator, write=write, causal=causal
            )
            if not rows:
                continue
            actual = staged.prepare_inputs(
                rows,
                forward_mode=ForwardMode.PREFILL,
                cache=manager,
                tables=manager.block_tables,
            )
            expected = _expected(rows, manager)
            _compare(_columns(actual), expected)
            if not causal:
                # Read-only canvases pad through the canvas runner instead.
                continue

            # Padding to a bucket of spare rows and tokens keeps the live
            # rows' columns, as a graph replay reads them.
            tokens = actual.inputs.input_ids.numel()
            bucket = (len(rows) + 1, tokens + 1, (WIDTH,) * 6, False)
            padded = _columns(pad_text(actual, *bucket, staging=staged))
            live = _columns(actual)
            width = {
                number: live[number, "table"].shape[1] for number in range(6)
            }
            assert padded["query_host"][: len(rows)] == live["query_host"]
            for number in range(6):
                assert torch.equal(
                    padded[number, "table"][: len(rows), : width[number]],
                    live[number, "table"],
                )
                if live[number, "slot_mapping"] is not None:
                    live_slot_mapping = live[number, "slot_mapping"]
                    assert torch.equal(
                        padded[number, "slot_mapping"][
                            : live_slot_mapping.numel()
                        ],
                        live_slot_mapping,
                    )
    finally:
        if device != "cpu":
            torch.cuda.synchronize()
        staged.close()
        manager.close()
