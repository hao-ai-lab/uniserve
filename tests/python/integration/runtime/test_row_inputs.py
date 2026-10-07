"""Token rows prepare the attention columns their host page layout describes.

Native input preparation gathers every numerical table's units, first
gathered pages and write addresses from the request slots' resident tables.
Over random calls of a model whose windowed and full-attention groups share
one unit pool (six numerical tables, as DiffusionGemma has), its attention
inputs, padded to a graph bucket, equal those of the same call's host layout
(``attention.from_tables`` of ``attention.table_pages``): appending rows,
mixed appending and read-only rows, read-only non-causal rows, rows whose
windowed tables retired their first pages, and padding past the live rows.
"""

from __future__ import annotations

import random

import pytest
import torch

from tests.python.fixtures.hybrid import hybrid_model, hybrid_pool
from uniserve.math import ceil_div
from uniserve.nn.attention import PagedInput
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.model_executor.attention import (
    from_tables,
    table_pages,
)
from uniserve_worker.model_executor.graph_inputs import pad_text
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.model_executor.input_buffers import (
    InputBuffers,
    TokenBufferConfig,
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
            pages = ceil_div(extent, shape.page_tokens)
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
                or max(0, prefix - table.shape.window)
                // table.shape.page_tokens
                >= table.start_page
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


def _host_layout(rows, manager):
    """The call's attention from its host page layout."""
    queries = tuple(row.query_tokens for row in rows)
    prefixes = tuple(row.seq_len for row in rows)
    return from_tables(
        table_pages(
            tuple(
                tuple(
                    manager.block_tables.table(row.request_pool_idx, group)
                    for group in range(len(manager.block_tables.groups))
                )
                for row in rows
            ),
            prefix_lengths=prefixes,
            query_lengths=queries,
        ),
        query_lengths=queries,
        prefix_lengths=prefixes,
        causal=tuple(row.causal for row in rows),
        write=tuple(row.write_kv for row in rows),
    )


def _columns(batch, padding_from=None):
    """Every attention input column of a batch, on the host.

    With ``padding_from``, the batch is padded to a bucket: its tables are
    compared from that row on, since the columns of live rows past their
    gathered pages are never read and hold whatever earlier calls left.
    """
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
        table, start = entry.block_table.indices, entry.block_table.start_page
        if padding_from is not None:
            table = table[padding_from:]
            start = None if start is None else start[padding_from:]
        values[number, "table"] = table
        values[number, "block_size"] = entry.block_table.block_size
        values[number, "start"] = start
        values[number, "start_host"] = entry.block_table.start_page_host
        values[number, "writes"] = entry.write_indices
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
            assert torch.equal(columns[key], value), key
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
def test_rows_gather_the_columns_their_host_layout_describes(
    device, write, causal
):
    model = hybrid_model()
    config = WorkerConfig(device=device, block_size=16, max_sequence_tokens=128)
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
    buffers = InputBuffers(
        ForwardMode.PREFILL, config=buffer_config, device=device
    )
    reference = InputBuffers(
        ForwardMode.PREFILL, config=buffer_config, device=device
    )
    generator = random.Random(1 + (write is None) + 2 * (not causal))
    try:
        extents = _install(manager, generator)
        for _ in range(12):
            rows = _rows(
                manager, extents, generator, write=write, causal=causal
            )
            if not rows:
                continue
            actual = buffers.prepare_inputs(
                rows,
                forward_mode=ForwardMode.PREFILL,
                cache=manager,
                tables=manager.block_tables,
            )
            expected = reference.prepare_inputs(
                rows,
                forward_mode=ForwardMode.PREFILL,
                attention=_host_layout(rows, manager),
            )
            _compare(_columns(actual), _columns(expected))
            if not causal:
                # Read-only canvases pad through the canvas runner instead.
                continue

            # Pad both to a bucket of spare rows and tokens, as a graph does.
            tokens = actual.inputs.input_ids.numel()
            bucket = (len(rows) + 1, tokens + 1, (WIDTH,) * 6, False)
            padded, padded_expected = (
                pad_text(batch, *bucket, buffers=inputs)
                for batch, inputs in (
                    (actual, buffers),
                    (expected, reference),
                )
            )
            _compare(
                _columns(padded, len(rows)),
                _columns(padded_expected, len(rows)),
            )
    finally:
        if device != "cpu":
            torch.cuda.synchronize()
        buffers.close()
        reference.close()
        manager.close()


@pytest.mark.gpu
def test_verification_inputs_keep_the_continuation_on_device():
    from uniserve_worker.execution.token import token_values

    current = torch.tensor([71], dtype=torch.long, device="cuda:0")
    mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        tokens, positions = token_values((72, 73), 12, current=current)
    finally:
        torch.cuda.set_sync_debug_mode(mode)

    # Reading values is allowed only after numerical preparation has returned.
    assert tokens.device == current.device
    assert tokens.cpu().tolist() == [71, 72, 73]
    assert positions.tolist() == [12, 13, 14]


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ("temporal", "sequential"))
def test_vision_features_keep_host_coordinates_under_a_device_context(layout):
    from uniserve.processing import (
        FeatureInjection,
        FeatureLayout,
        PositionLayout,
    )
    from uniserve_worker.execution.image import vision_values

    features = torch.arange(8, device="cuda:0").reshape(2, 4).float()
    injection = FeatureInjection(
        FeatureLayout.FRAMED,
        PositionLayout(layout),
        start_token_id=7,
        end_token_id=9,
    )
    with torch.device(features.device):
        tokens, embeddings, mask, positions = vision_values(
            features,
            16,
            16,
            12,
            input_images=1,
            close_image=False,
            injection=injection,
            transform=None,
        )

    assert tokens.device.type == positions.device.type == "cpu"
    assert tokens.tolist() == [7, 1, 1, 9]
    assert positions.tolist() == (
        [12, 12, 12, 12] if layout == "temporal" else [12, 13, 14, 15]
    )
    assert mask.device == features.device
    assert mask.tolist() == [False, True, True, False]
    torch.testing.assert_close(embeddings[1:3], features)
