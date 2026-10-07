"""A hybrid model's K/V export, installation and attention inputs.

The fixture model has a windowed group (window 8, 16-token pages of five
units) and a full-attention group (32-token pages of one unit) in one unit
pool; see ``tests.python.fixtures.hybrid``.
"""

from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.hybrid import hybrid_model, hybrid_pool
from tests.python.fixtures.transport import make_transport
from uniserve.runtime import EventPool
from uniserve_worker.bootstrap.cache import plan_cache, table_widths
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.errors import WorkerError
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.model_executor.input_buffers import (
    InputBuffers,
    TokenBufferConfig,
)
from uniserve_worker.protocol.call import ForwardMode
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.storage.decode_state import DecodeState

pytestmark = pytest.mark.integration

WINDOW = 8
# The producer has retired windowed page 0: it holds windowed pages 1 and 2
# (five units each, page-major) and full pages 0 and 1.
PRODUCER = (
    (1, 1, (11, 12, 13, 14, 15, 16, 17, 18, 19, 20), 48),
    (1, 0, (1, 2), 64),
)
# The consumer needs windowed history from token 32 only, on page 2.
CONSUMER = ((1, 2, (21, 22, 23, 24, 25), 48), (1, 0, (3, 4), 64))


def _config(device):
    return WorkerConfig(device=device, block_size=16, max_sequence_tokens=128)


def _install(pool, tables):
    pool.block_tables.install(
        tuple(
            (slot, group, start, units, allocated)
            for group, (slot, start, units, allocated) in enumerate(tables)
        )
    )


def _layers(pool):
    """Yield every layer's name and its group table position on slot 1."""
    columns = pool.cache.planes.columns
    for group, values in enumerate(pool.cache.groups):
        table = pool.block_tables.table(1, group)
        for index, name in enumerate(values.layers):
            yield name, table, index // columns


def test_attention_pages_retain_assignments_and_bound_reads_and_writes():
    pool = hybrid_pool(hybrid_model(), _config("cpu"), num_units=64)
    try:
        _install(pool, PRODUCER)
        selected = pool.prepare_attention(((1, 40, 1, True),))
        expected = [((unit,),) for unit in range(16, 21)] + [((1, 2),)]
        assert [table.rows for table in selected] == expected

        # Read-only canvases can extend beyond capacity; they consume the
        # retained prefix and keep their current tokens outside the cache.
        readonly = pool.prepare_attention(((1, 48, 24, False),))
        assert [table.rows for table in readonly] == expected
        with pytest.raises(WorkerError, match="scheduler block table"):
            pool.prepare_attention(((1, 48, 1, True),))
        with pytest.raises(WorkerError, match="retired window pages"):
            pool.prepare_attention(((1, 23, 1, True),))
        for prefix, query in ((40, 0), (40, -1), (-1, 1)):
            with pytest.raises(WorkerError, match="lengths are invalid"):
                pool.prepare_attention(((1, prefix, query, True),))

        _install(pool, CONSUMER)
        current = pool.prepare_attention(((1, 40, 1, True),))
        assert [table.rows for table in current] == [
            ((unit,),) for unit in range(21, 26)
        ] + [((3, 4),)]
        assert [table.rows for table in selected] == expected
    finally:
        pool.close()


def _history(pool, device):
    """Write distinct K/V to every layer's held tokens; return it by layer."""
    generator = torch.Generator().manual_seed(47)
    written = {}
    for name, table, row in _layers(pool):
        state = pool.cache.state(name)
        first = table.start_page * table.shape.page_tokens
        length = 40 - first
        shape = (length, *state.key.shape[2:])
        key = torch.randn(shape, generator=generator).to(device)
        value = torch.randn(shape, generator=generator).to(device)
        state.write(table.row(row), start=0, key=key, value=value)
        written[name] = (first, key, value)
    return written


def _buffer(call: int) -> BufferId:
    return BufferId(RequestKey(1, 1, 1), CallId(call, 0), 0, call)


@pytest.mark.parametrize(
    "backend,device",
    (
        ("local", "cpu"),
        pytest.param("cuda_vmm", "cuda:0", marks=pytest.mark.gpu),
    ),
)
def test_export_carries_each_group_from_its_first_needed_token(backend, device):
    model = hybrid_model(window=WINDOW)
    producer = hybrid_pool(model, _config(device), num_units=32)
    consumer = hybrid_pool(model, _config(device), num_units=32)
    events = [EventPool(), EventPool()]
    transports = [
        make_transport(
            backend, byte_capacity=1 << 20, ticket_capacity=16, event_pool=event
        )
        for event in events
    ]
    source, write, export = _buffer(1), None, None
    try:
        _install(producer, PRODUCER)
        _install(consumer, CONSUMER)
        written = _history(producer, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

        export = producer.export(
            request_pool_idx=1,
            visible_length=40,
            destination="consumer",
            buffer=source,
            transports={backend: transports[0]},
        )
        # A reader at token 40 needs the windowed group's last eight tokens
        # and the full group's whole history, each over its group's layers.
        windowed, full = export.groups
        assert (windowed.start, windowed.page_tokens) == (32, 16)
        assert windowed.tensors[0].shape == (8, 10, 4, 4)
        assert (full.start, full.page_tokens) == (0, 32)
        assert full.tensors[0].shape == (40, 2, 1, 8)
        producer.validate_exports(((source, export),), ())
        producer.apply_exports(((source, export),), ())

        write = consumer.prepare_install(
            export,
            request_pool_idx=1,
            tables=tuple(
                consumer.block_tables.table(1, group) for group in range(2)
            ),
            initialized_units=(21, 22, 23, 24, 25, 3, 4),
            transports={backend: transports[1]},
        )
        write.result(timeout=30)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        installed = _buffer(101)
        result = consumer.install(installed_buffer=installed, write=write)
        consumer.validate_exports((), ((source, installed, result),))
        consumer.apply_exports((), ((source, installed, result),))

        # Each layer holds exactly the carried tokens at its own units.
        for name, table, row in _layers(consumer):
            first, key, value = written[name]
            start = table.start_page * table.shape.page_tokens
            actual = consumer.cache.state(name).read(
                table.row(row), start=0, length=40 - start
            )
            for got, wanted in zip(actual, (key, value), strict=True):
                torch.testing.assert_close(
                    got, wanted[start - first :], rtol=0, atol=0
                )
    finally:
        if write is not None:
            consumer.imports.abandon(write)
        if export is not None:
            for tensor in export.tensors:
                for location in tensor.locations:
                    transports[0].release(location)
            producer.release_buffers((source,))
        for pool in (producer, consumer):
            pool.imports.stop()
        for transport in transports:
            transport.close()
        for pool in (producer, consumer):
            pool.close()
        for event in events:
            event.close()


def _buffers(device, planes, *, rows, tokens):
    return InputBuffers(
        ForwardMode.PREFILL,
        config=TokenBufferConfig(
            max_rows=rows,
            max_tokens=tokens,
            max_text_tokens=tokens,
            table_widths=table_widths(
                planes, max_sequence_tokens=128, max_query_tokens=tokens
            ),
            hidden_size=0,
        ),
        device=device,
    )


def test_attention_inputs_build_one_entry_per_table_from_each_row_start():
    model = hybrid_model(window=WINDOW)
    pool = hybrid_pool(model, _config("cpu"), num_units=32)
    buffers = _buffers(
        "cpu", plan_cache(model, _config("cpu")), rows=1, tokens=4
    )
    try:
        _install(pool, PRODUCER)
        row = TokenRow(
            forward_mode=ForwardMode.PREFILL,
            token_ids=torch.tensor([1, 2, 3, 4]),
            positions=torch.arange(40, 44),
            selection=TokenSelection.LAST_LOGITS,
            request_pool_idx=1,
            seq_len=40,
            write_kv=True,
        )
        batch = buffers.prepare_inputs(
            (row,),
            forward_mode=ForwardMode.PREFILL,
            cache=pool,
            tables=pool.block_tables,
        )
        entries = batch.inputs.attention.entries
        assert sorted(entries) == list(range(6))
        for position in range(5):
            # Queries from token 40 read windowed history from token 32:
            # page 2 alone, the second page the slot holds.
            entry = entries[position]
            unit = PRODUCER[0][2][5 + position]
            assert entry.block_table.block_size == 16
            assert entry.block_table.start_page.tolist() == [2]
            assert entry.block_table.start_page_host == (2,)
            assert entry.block_table.indices.tolist() == [[unit]]
            assert entry.write_indices.tolist() == [
                unit * 16 + offset for offset in range(8, 12)
            ]
        # The full table stages every held page from page zero.
        full = entries[5]
        assert full.block_table.block_size == 32
        assert full.block_table.start_page is None
        assert full.block_table.indices.tolist() == [[1, 2]]
        assert full.write_indices.tolist() == [
            2 * 32 + offset for offset in range(8, 12)
        ]
        assert full.prefixes.values.tolist() == [40]
    finally:
        buffers.close()
        pool.close()


@pytest.mark.gpu
def test_indexed_decode_gathers_every_table_in_one_batch():
    device = "cuda:0"
    model = hybrid_model(window=WINDOW)
    config = _config(device)
    pool = hybrid_pool(model, config, num_units=48, request_pool_size=2)
    planes = plan_cache(model, config)
    indexed = _buffers(device, planes, rows=2, tokens=2)
    reference = _buffers(device, planes, rows=2, tokens=2)
    states = DecodeState(
        request_pool_size=2, vocab_size=32, continuation_width=1, device=device
    )
    enabled = torch.ones(1, dtype=torch.bool, device=device)
    try:
        # Slot 2 holds a short sequence whose windowed pages start at zero.
        _install(pool, PRODUCER)
        pool.block_tables.install(
            ((2, 0, 0, tuple(range(26, 36)), 32), (2, 1, 0, (5,), 32))
        )
        lengths = (41, 20)
        pool.block_tables.set_verified(
            torch.tensor([1, 2], device=device),
            torch.tensor(lengths, device=device),
        )
        for slot, token in ((1, 7), (2, 11)):
            states.apply_tokens(
                (slot,),
                tokens=torch.tensor([token], device=device),
                predicates=enabled,
                valid=enabled,
                active=enabled,
                penalty_bases=(None,),
                logical_position=lengths[slot - 1],
                sampling_position=lengths[slot - 1],
            )

        def rows(indexed_decode):
            return tuple(
                TokenRow(
                    forward_mode=ForwardMode.DECODE,
                    request_pool_idx=slot,
                    seq_len=lengths[slot - 1],
                    write_kv=True,
                    selection=TokenSelection.LAST_LOGITS,
                    request_indexed_decode=indexed_decode,
                    token_ids=None
                    if indexed_decode
                    else torch.tensor([token], device=device),
                    positions=None
                    if indexed_decode
                    else torch.tensor([lengths[slot - 1]], device=device),
                )
                for slot, token in ((2, 11), (1, 7))
            )

        gathered = indexed.prepare_inputs(
            rows(True),
            forward_mode=ForwardMode.DECODE,
            cache=pool,
            tables=pool.block_tables,
            states=states,
        )
        # The host path builds the same attention from the installed tables.
        host_batch = reference.prepare_inputs(
            rows(False),
            forward_mode=ForwardMode.DECODE,
            cache=pool,
            tables=pool.block_tables,
        )
        assert gathered.inputs.input_ids.tolist() == [11, 7]
        entries = gathered.inputs.attention.entries
        assert sorted(entries) == sorted(host_batch.inputs.attention.entries)
        for number, entry in entries.items():
            other = host_batch.inputs.attention.entries[number]
            width = other.block_table.indices.shape[1]
            assert (
                entry.block_table.indices[:, :width].tolist()
                == other.block_table.indices.tolist()
            )
            assert entry.write_indices.tolist() == other.write_indices.tolist()
            assert entry.prefixes.values.tolist() == list(lengths[::-1])
            if other.block_table.start_page is None:
                assert entry.block_table.start_page is None
            else:
                assert (
                    entry.block_table.start_page.tolist()
                    == other.block_table.start_page.tolist()
                )
                assert (
                    entry.block_table.start_page_host
                    == other.block_table.start_page_host
                )
    finally:
        torch.cuda.synchronize()
        indexed.close()
        reference.close()
        pool.close()
