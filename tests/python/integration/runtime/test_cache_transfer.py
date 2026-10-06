"""KV transfer values and storage lifetime over real transports."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.cache import mha_pool
from tests.python.fixtures.shm_export import serve_pending_export
from tests.python.fixtures.transport import make_transport
from uniserve.runtime import EventPool
from uniserve_worker.errors import WorkerError, WorkerErrorCode
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.transfer import (
    MAX_TRANSFER_HANDLE_BYTES,
    KvGroupTransfer,
    KvTransfer,
    Locator,
    TensorTransfer,
)
from uniserve_worker.storage.block_tables import GroupTable
from uniserve_worker.transport import make_transports

pytestmark = pytest.mark.integration


def test_cancelled_import_releases_pages_after_pending_read_retires() -> None:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    shape = (256, 2, 1, 2)
    process = context.Process(target=serve_pending_export, args=(child, shape))
    pool = mha_pool(
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        total_layers=2,
        total_kv_heads=1,
        num_pages=3,
        page_size=256,
        device="cpu",
        request_pool_size=2,
        table_width=1,
        import_capacity=2,
    )
    exports = pool
    events = EventPool()
    consumer = make_transport(
        "shm", byte_capacity=8192, ticket_capacity=2, event_pool=events
    )
    write = None
    process.start()
    child.close()
    try:
        assert parent.poll(30), "shared-storage publisher did not start"
        locator = Locator.from_mapping(parent.recv())
        source = _buffer(1)
        field = TensorTransfer(shape=shape, locations=(locator,))
        export = KvTransfer(
            groups=(KvGroupTransfer(0, 256, (field, field)),),
            source=source,
            destination="consumer",
            base=None,
            base_extent=0,
            exported_extent=256,
            compute_dtype="float32",
        )
        write = exports.prepare_install(
            export,
            request_pool_idx=1,
            tables=_tables(pool, (1,), 256),
            initialized_units=(1,),
            transports={consumer.name: consumer},
        )
        # A pending producer cannot complete the read. The destination remains
        # reserved while an unrelated page is usable.
        assert not write.done()
        assert not pool.retirement_ready(requests=(source.owner,))
        with pytest.raises(WorkerError, match="import destination"):
            pool.zero_units((1,))

        # Refusing another installation of this buffer must preserve the
        # first import's destination and its ability to finish cancellation.
        with pytest.raises(WorkerError) as repeated:
            exports.prepare_install(
                export,
                request_pool_idx=1,
                tables=_tables(pool, (1,), 256),
                initialized_units=(1,),
                transports={consumer.name: consumer},
            )
        assert repeated.value.code is WorkerErrorCode.INVALID_DESCRIPTOR
        with pytest.raises(WorkerError, match="import destination"):
            pool.zero_units((1,))

        independent = torch.full((256, 1, 2), 7.0)
        pool.cache.state(pool.cache.groups[0].layers[0]).write(
            (2,), start=0, key=independent, value=independent
        )
        for actual in pool.cache.state(pool.cache.groups[0].layers[0]).read(
            (2,), start=0, length=256
        ):
            torch.testing.assert_close(actual, independent, rtol=0, atol=0)

        pool.imports.cancel_requests(frozenset((source.owner,)))
        with pytest.raises(WorkerError, match="cancelled"):
            write.result(timeout=5)
        write.retirement.result(timeout=5)
        assert pool.retirement_ready(requests=(source.owner,))
        parent.send("settled")
        assert parent.poll(5) and parent.recv()
        pool.zero_units((1,))
        for actual in pool.cache.state(pool.cache.groups[0].layers[0]).read(
            (1,), start=0, length=256
        ):
            assert torch.count_nonzero(actual).item() == 0
        parent.send("exit")
        process.join(30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()
        if write is not None:
            pool.imports.abandon(write)
        pool.imports.stop()
        consumer.close()
        pool.close()
        events.close()


def test_shutdown_retires_running_and_queued_import_destinations() -> None:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    shape = (256, 2, 1, 2)
    process = context.Process(target=serve_pending_export, args=(child, shape))
    pool = mha_pool(
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        total_layers=2,
        total_kv_heads=1,
        num_pages=6,
        page_size=256,
        device="cpu",
        request_pool_size=5,
        table_width=1,
        import_capacity=5,
    )
    events = EventPool()
    consumer = make_transport(
        "shm", byte_capacity=65536, ticket_capacity=16, event_pool=events
    )
    process.start()
    child.close()
    try:
        assert parent.poll(30), "shared-storage publisher did not start"
        locator = Locator.from_mapping(parent.recv())
        field = TensorTransfer(shape=shape, locations=(locator,))
        writes = []
        for slot in range(1, 6):
            export = KvTransfer(
                groups=(KvGroupTransfer(0, 256, (field, field)),),
                source=replace(_buffer(slot), owner=RequestKey(1, slot, 1)),
                destination="consumer",
                base=None,
                base_extent=0,
                exported_extent=256,
                compute_dtype="float32",
            )
            writes.append(
                pool.prepare_install(
                    export,
                    request_pool_idx=slot,
                    tables=_tables(pool, (slot,), 256),
                    initialized_units=(slot,),
                    transports={consumer.name: consumer},
                )
            )

        # Five pending copies exceed the import lane's four execution threads.
        # Cancelling copy tasks leaves running readers for import shutdown to
        # drain. Queued copies never enter the numerical action at all.
        assert all(not write.done() for write in writes)
        for write in writes:
            if write.cancel():
                assert write.cancelled
        pool.imports.stop()
        consumer.close()
        for write in writes:
            assert write.done()
            write.retirement.result(timeout=5)
        pool.imports.require_retired()
        pool.zero_units(tuple(range(1, 6)))

        parent.send("settled")
        assert parent.poll(5) and parent.recv()
        parent.send("exit")
        process.join(30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()
        pool.imports.stop()
        consumer.close()
        pool.close()
        events.close()


def _tables(pool, units, allocated):
    """Return the one-group destination tables of an installation."""
    return (GroupTable(pool.shapes[0], 0, tuple(units), int(allocated)),)


def _merged(shards):
    """Merge rank descriptors of one logical value into one descriptor."""
    return replace(
        shards[0],
        groups=tuple(
            replace(
                group,
                tensors=tuple(
                    replace(
                        field,
                        locations=tuple(
                            location
                            for shard in shards
                            for location in shard.groups[number]
                            .tensors[index]
                            .locations
                        ),
                    )
                    for index, field in enumerate(group.tensors)
                ),
            )
            for number, group in enumerate(shards[0].groups)
        ),
    )


def _buffer(call: int) -> BufferId:
    return BufferId(
        owner=RequestKey(1, 1, 1),
        producer_call_id=CallId(call, 0),
        output_index=0,
        generation=call,
    )


def test_failed_kv_export_releases_partial_views_and_source_reservation() -> (
    None
):
    pool = mha_pool(
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float32,
        total_layers=1,
        total_kv_heads=1,
        num_pages=2,
        page_size=4,
        device="cpu",
        request_pool_size=1,
        table_width=1,
    )
    pool.block_tables.install(((1, 0, 0, (1,), 4),))
    pool.cache.planes_of(0, "key").fill_(3.0)
    pool.cache.planes_of(0, "value").fill_(7.0)
    events = EventPool()
    # Four float32 keys fit; exporting their values exhausts the transport.
    transport = make_transport(
        "local", byte_capacity=16, ticket_capacity=2, event_pool=events
    )
    source = _buffer(1)
    try:
        with pytest.raises(WorkerError) as failure:
            pool.export(
                request_pool_idx=1,
                visible_length=4,
                destination="consumer",
                buffer=source,
                transports={"local": transport},
            )
        assert failure.value.code is WorkerErrorCode.RESOURCE_ERROR
        pool.prepare_attention(((1, 0, 4, True),))

        # Retrying the same buffer with a smaller extent requires both the
        # cache reservation and the first export's transport bytes to be free.
        exported = pool.export(
            request_pool_idx=1,
            visible_length=2,
            destination="consumer",
            buffer=source,
            transports={"local": transport},
        )
        for tensor, expected in zip(exported.tensors, (3.0, 7.0), strict=True):
            for locator in tensor.locations:
                ticket = transport.fetch(locator, device=torch.device("cpu"))
                try:
                    value = ticket.result()
                    if isinstance(value, tuple):
                        value = torch.cat(value, dim=0)
                    torch.testing.assert_close(
                        value,
                        torch.full((2, 1, 1, 1), expected),
                        rtol=0,
                        atol=0,
                    )
                finally:
                    ticket.close()
    finally:
        transport.close()
        pool.close()
        events.close()


def test_kv_exports_isolate_request_epochs() -> None:
    pool = mha_pool(
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float32,
        total_layers=1,
        total_kv_heads=1,
        num_pages=2,
        page_size=4,
        device="cpu",
        request_pool_size=1,
        table_width=1,
    )
    tables = pool.block_tables
    tables.install(((1, 0, 0, (1,), 4),))
    exports = pool
    first = _buffer(1)
    second = replace(
        first,
        owner=replace(first.owner, request_epoch=first.owner.request_epoch + 1),
    )
    try:
        for source in (first, second):
            # An empty extent is a valid export: its identity and installed
            # base still belong to one exact request incarnation.
            export = exports.export(
                request_pool_idx=1,
                visible_length=0,
                destination="consumer",
                buffer=source,
                transports={},
            )
            exports.validate_exports(((source, export),), ())
            exports.apply_exports(((source, export),), ())
            write = exports.prepare_install(
                export,
                request_pool_idx=1,
                tables=_tables(pool, (1,), 4),
                initialized_units=(),
                transports={},
            )
            installed = replace(source, producer_call_id=CallId(3, 0))
            result = exports.install(
                installed_buffer=installed,
                write=write,
            )
            exports.validate_exports((), ((source, installed, result),))
            exports.apply_exports((), ((source, installed, result),))
            assert exports.get_export(installed) == export
        with pytest.raises(WorkerError, match="another request"):
            exports.validate_conditioning(
                second.owner,
                first,
                request_pool_idx=1,
                visible_length=0,
            )
    finally:
        pool.close()


@pytest.mark.parametrize(
    "backend,device",
    (
        ("local", "cpu"),
        ("shm", "cpu"),
        pytest.param("shm", "cuda", marks=pytest.mark.gpu),
        pytest.param("cuda_vmm", "cuda:0", marks=pytest.mark.gpu),
    ),
)
@pytest.mark.parametrize(
    "source_dtype,target_dtype,page_size",
    (
        ("bfloat16", "bfloat16", 2),
        ("float8_e4m3fn", "float8_e4m3fn", 4),
        ("float8_e4m3fn", "float8_e4m3fn", 2),
        ("float8_e4m3fn", "bfloat16", 2),
        ("bfloat16", "float8_e4m3fn", 2),
        ("float8_e4m3fn", "float32", 2),
    ),
)
def test_incremental_kv_import_preserves_values_in_reserved_pages(
    backend: str,
    device: str,
    source_dtype: str,
    target_dtype: str,
    page_size: int,
) -> None:
    pools = [
        mha_pool(
            num_layers=2,
            num_kv_heads=1,
            head_dim=1,
            dtype=torch.bfloat16,
            store_dtype=getattr(torch, dtype),
            total_layers=2,
            total_kv_heads=1,
            num_pages=6,
            page_size=size,
            device=device,
            request_pool_size=1,
            table_width=4,
        )
        for size, dtype in ((4, source_dtype), (page_size, target_dtype))
    ]
    tables = [pool.block_tables for pool in pools]
    pages = ((4, 1), (3, 1, 4, 5) if page_size == 2 else (3, 1))
    for table, pool, assigned in zip(tables, pools, pages, strict=True):
        table.install(
            ((1, 0, 0, assigned, len(assigned) * pool.shapes[0].page_tokens),)
        )
    exports = [pool for pool, table in zip(pools, tables, strict=True)]
    events = [EventPool(), EventPool()]
    transports = [
        make_transport(
            backend, byte_capacity=16384, ticket_capacity=16, event_pool=event
        )
        for event in events
    ]
    # The smaller source values expose BF16 rounding after FP8 dequantization.
    rounding = target_dtype == "float32"
    prefix = (1.0, 2.0, 3.0) if rounding else (112.0, 224.0, 448.0)
    suffix = (
        (6.0, -6.0, 0.375, 0.75, 1.5)
        if rounding
        else (896.0, -896.0, 56.0, 112.0, 224.0)
    )
    if rounding:
        expected = (0.96484375, 1.9296875, 3.0, 6.0, -6.0, 0.375, 0.75, 1.5)
    else:
        expected = (*prefix, *suffix)
    locators = []
    writes = []
    try:
        for call, (start, values) in enumerate(
            ((0, prefix), (3, suffix)), start=1
        ):
            tensor = torch.tensor(
                values, dtype=torch.bfloat16, device=device
            ).view(-1, 1, 1)
            for layer in range(2):
                pools[0].cache.state(
                    pools[0].cache.groups[0].layers[layer]
                ).write(
                    pages[0],
                    start=start,
                    key=tensor * 2**layer,
                    value=-tensor * 2**layer / 2,
                )
            extent = start + len(values)
            source = _buffer(call)
            installed = _buffer(100 + call)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            export = exports[0].export(
                request_pool_idx=1,
                visible_length=extent,
                destination="consumer",
                buffer=source,
                transports={transports[0].name: transports[0]},
            )
            locators.extend(
                location
                for field in export.tensors
                for location in field.locations
            )
            exports[0].validate_exports(((source, export),), ())
            exports[0].apply_exports(((source, export),), ())
            if start:
                for destination_pages, initialized in (
                    (pages[1], (pages[1][0],)),
                    (tuple(reversed(pages[1])), ()),
                ):
                    with pytest.raises(
                        WorkerError, match="replace its installed base units"
                    ):
                        exports[1].prepare_install(
                            export,
                            request_pool_idx=1,
                            tables=_tables(
                                pools[1],
                                destination_pages,
                                len(pages[1]) * page_size,
                            ),
                            initialized_units=initialized,
                            transports={transports[1].name: transports[1]},
                        )
            write = exports[1].prepare_install(
                export,
                request_pool_idx=1,
                tables=_tables(pools[1], pages[1], len(pages[1]) * page_size),
                initialized_units=pages[1] if start == 0 else (),
                transports={transports[1].name: transports[1]},
            )
            writes.append(write)
            write.result(timeout=30)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            result = exports[1].install(
                installed_buffer=installed,
                write=write,
            )
            exports[1].validate_exports((), ((source, installed, result),))
            exports[1].apply_exports((), ((source, installed, result),))
            for layer in range(2):
                key, value = (
                    pools[1]
                    .cache.state(pools[1].cache.groups[0].layers[layer])
                    .read(pages[1], start=0, length=extent)
                )
                dtype = (
                    torch.bfloat16
                    if target_dtype == "float8_e4m3fn"
                    else getattr(torch, target_dtype)
                )
                wanted = (
                    torch.tensor(
                        expected[:extent], dtype=dtype, device=device
                    ).view(-1, 1, 1)
                    * 2**layer
                )
                torch.testing.assert_close(key, wanted, rtol=0, atol=0)
                torch.testing.assert_close(value, -wanted / 2, rtol=0, atol=0)
                untouched = (
                    pools[1]
                    .cache.state(pools[1].cache.groups[0].layers[layer])
                    .read((2,), start=0, length=page_size)
                )
                for field in untouched:
                    assert field is not None
                    assert torch.count_nonzero(field).item() == 0
    finally:
        for write in writes:
            pools[1].imports.abandon(write)
        for location in locators:
            transports[0].release(location)
        for pool in pools:
            pool.imports.stop()
        for transport in transports:
            transport.close()
        for pool in pools:
            pool.close()
        for event in events:
            event.close()


@pytest.mark.parametrize(
    "source_ranks,target_ranks,replicated,source_stages,target_stages",
    (
        (1, 2, False, 1, 1),
        (2, 1, False, 1, 1),
        (3, 2, False, 1, 1),
        (2, 3, True, 1, 1),
        pytest.param(2, 1, False, 2, 3, id="layers-two-to-three"),
        pytest.param(1, 2, False, 3, 2, id="layers-three-to-two"),
    ),
)
@pytest.mark.parametrize(
    "source_dtype,target_dtype",
    (
        ("bfloat16", "bfloat16"),
        ("float8_e4m3fn", "float8_e4m3fn"),
        ("float8_e4m3fn", "bfloat16"),
        ("bfloat16", "float8_e4m3fn"),
    ),
)
@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
@pytest.mark.parametrize("target_page_size", (4, 8))
def test_kv_delivery_reshards_logical_heads_and_source_scale_groups(
    source_ranks: int,
    target_ranks: int,
    replicated: bool,
    source_stages: int,
    target_stages: int,
    source_dtype: str,
    target_dtype: str,
    device: str,
    target_page_size: int,
) -> None:
    """Consumer layer/head regions gather required producer pages and scales."""
    total_layers = 5 if max(source_stages, target_stages) > 1 else 2
    source_count = source_ranks * source_stages
    total_heads = 6
    source_heads = total_heads if replicated else total_heads // source_ranks
    target_heads = total_heads // target_ranks
    pools = [
        mha_pool(
            num_layers=layer_end - layer_start,
            total_layers=total_layers,
            layer_offset=layer_start,
            num_kv_heads=heads,
            total_kv_heads=total_heads,
            kv_head_offset=offset,
            head_dim=1,
            dtype=torch.bfloat16,
            store_dtype=getattr(torch, dtype),
            num_pages=4,
            page_size=page_size,
            device=device,
            request_pool_size=1,
            table_width=2,
        )
        for heads, offset, dtype, page_size, layer_start, layer_end in (
            *(
                (
                    source_heads,
                    0 if replicated else rank * source_heads,
                    source_dtype,
                    4,
                    total_layers * stage // source_stages,
                    total_layers * (stage + 1) // source_stages,
                )
                for stage in range(source_stages)
                for rank in range(source_ranks)
            ),
            *(
                (
                    target_heads,
                    rank * target_heads,
                    target_dtype,
                    target_page_size,
                    total_layers * stage // target_stages,
                    total_layers * (stage + 1) // target_stages,
                )
                for stage in range(target_stages)
                for rank in range(target_ranks)
            ),
        )
    ]
    tables = [pool.block_tables for pool in pools]
    pages = (3, 1)
    for pool, table in zip(pools, tables, strict=True):
        table.install(((1, 0, 0, pages, 2 * pool.shapes[0].page_tokens),))
    owners = [pool for pool, table in zip(pools, tables, strict=True)]
    events = [EventPool() for _ in pools]
    backend = "cuda_vmm" if device.startswith("cuda") else "shm"
    transports = [
        make_transport(
            backend, byte_capacity=32768, ticket_capacity=32, event_pool=event
        )
        for event in events
    ]
    token_values = torch.tensor(
        (112, 224, 448, 112, 224, 448), dtype=torch.bfloat16, device=device
    )
    head_values = torch.tensor(
        [2**head for head in range(total_heads)],
        dtype=torch.bfloat16,
        device=device,
    )
    wanted = token_values[:, None, None] * head_values[None, :, None]
    exports = []
    writes = []
    try:
        for call, (start, extent) in enumerate(((0, 3), (3, 6)), start=1):
            source = _buffer(call)
            shards = []
            for pool, owner, transport in zip(
                pools[:source_count],
                owners[:source_count],
                transports[:source_count],
                strict=True,
            ):
                values = wanted[
                    start:extent,
                    pool.info.groups[0].kv_head_offset : pool.info.groups[
                        0
                    ].kv_head_offset
                    + pool.info.groups[0].num_kv_heads,
                ]
                for layer in range(len(pool.info.groups[0].layer_ids)):
                    logical_layer = pool.info.groups[0].layer_ids[0] + layer
                    pool.cache.state(pool.cache.groups[0].layers[layer]).write(
                        pages,
                        start=start,
                        key=values * 2**logical_layer,
                        value=-values * 2**logical_layer / 2,
                    )
                shard = owner.export(
                    request_pool_idx=1,
                    visible_length=extent,
                    destination="consumer",
                    buffer=source,
                    transports={backend: transport},
                )
                exports.append((pool, transport, source, shard))
                owner.validate_exports(((source, shard),), ())
                owner.apply_exports(((source, shard),), ())
                shards.append(shard)
            # Rank descriptors identify one logical value with distributed
            # physical coverage.
            merged = _merged(shards)
            for rank, (pool, owner, transport) in enumerate(
                zip(
                    pools[source_count:],
                    owners[source_count:],
                    transports[source_count:],
                    strict=True,
                )
            ):
                write = owner.prepare_install(
                    merged,
                    request_pool_idx=1,
                    tables=_tables(pool, pages, 2 * pool.shapes[0].page_tokens),
                    initialized_units=pages if start == 0 else (),
                    transports={backend: transport},
                )
                writes.append((pool, write))
                write.result(timeout=30)
                installed = _buffer(100 + call)
                value = owner.install(
                    installed_buffer=installed,
                    write=write,
                )
                owner.validate_exports((), ((source, installed, value),))
                owner.apply_exports((), ((source, installed, value),))
                for layer in range(len(pool.info.groups[0].layer_ids)):
                    logical_layer = pool.info.groups[0].layer_ids[0] + layer
                    key, value = pool.cache.state(
                        pool.cache.groups[0].layers[layer]
                    ).read(pages, start=0, length=extent)
                    expected = (
                        wanted[
                            :extent,
                            pool.info.groups[
                                0
                            ].kv_head_offset : pool.info.groups[
                                0
                            ].kv_head_offset
                            + pool.info.groups[0].num_kv_heads,
                        ]
                        * 2**logical_layer
                    )
                    torch.testing.assert_close(key, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        value, -expected / 2, rtol=0, atol=0
                    )
    finally:
        for pool, write in writes:
            pool.imports.abandon(write)
        for pool, transport, buffer, export in exports:
            for tensor in export.tensors:
                for location in tensor.locations:
                    transport.release(location)
            pool.release_buffers((buffer,))
        for pool in pools:
            pool.imports.stop()
        for transport in transports:
            transport.close()
        for pool in pools:
            pool.close()
        for event in events:
            event.close()


@pytest.mark.parametrize("dtype", ("bfloat16", "float8_e4m3fn"))
@pytest.mark.parametrize(
    "mechanisms,device",
    (
        (("local", "shm"), "cpu"),
        pytest.param(("local", "channel"), "cpu", id="channel"),
        pytest.param(("local", "cuda_vmm"), "cuda:0", marks=pytest.mark.gpu),
    ),
)
def test_deep_kv_export_fits_its_descriptor_and_installs(
    mechanisms: tuple[str, str], device: str, dtype: str
) -> None:
    """A deep model's KV stays within the transfer descriptor bound.

    Two head shards each publish every layer over both mechanisms their rank
    binds, as a tensor-parallel producer does, and the head merges their
    reports into one descriptor. A consumer on another rank installs every
    layer exactly from that descriptor.
    """
    total_layers = 128
    total_heads = 2
    extent = 6
    pages = (3, 1)
    pools = [
        mha_pool(
            num_layers=total_layers,
            total_layers=total_layers,
            num_kv_heads=heads,
            total_kv_heads=total_heads,
            kv_head_offset=offset,
            head_dim=1,
            dtype=torch.bfloat16,
            store_dtype=getattr(torch, dtype),
            num_pages=4,
            page_size=4,
            device=device,
            request_pool_size=1,
            table_width=2,
        )
        for heads, offset in ((1, 0), (1, 1), (total_heads, 0))
    ]
    sources, target = pools[:2], pools[2]
    for pool in pools:
        pool.block_tables.install(
            ((1, 0, 0, pages, 2 * pool.shapes[0].page_tokens),)
        )
    events = [EventPool() for _ in pools]
    transports = [
        make_transports(
            mechanisms,
            byte_capacity=1 << 20,
            ticket_capacity=64,
            event_pool=event,
        )
        for event in events
    ]

    # Power-of-two ratios within each page keep FP8 encoding and re-encoding
    # exact, and a sign and scale per layer distinguish the layers.
    token_values = torch.tensor(
        (112, 224, 448, 112, 224, 448), dtype=torch.bfloat16, device=device
    )
    head_values = torch.tensor(
        (1, 2), dtype=torch.bfloat16, device=device
    ).view(1, total_heads, 1)
    wanted = tuple(
        token_values.view(extent, 1, 1)
        * head_values
        * (-1) ** layer
        * 2 ** (layer % 3)
        for layer in range(total_layers)
    )

    source = _buffer(1)
    shards = []
    write = None
    try:
        for pool, rank_transports in zip(
            sources, transports[: len(sources)], strict=True
        ):
            heads = slice(
                pool.info.groups[0].kv_head_offset,
                pool.info.groups[0].kv_head_offset
                + pool.info.groups[0].num_kv_heads,
            )
            for layer, name in enumerate(pool.cache.groups[0].layers):
                pool.cache.state(name).write(
                    pages,
                    start=0,
                    key=wanted[layer][:, heads],
                    value=-wanted[layer][:, heads] / 2,
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            shard = pool.export(
                request_pool_idx=1,
                visible_length=extent,
                destination="consumer",
                buffer=source,
                transports=rank_transports,
            )
            pool.validate_exports(((source, shard),), ())
            pool.apply_exports(((source, shard),), ())
            shards.append(shard)

        merged = _merged(shards)
        assert merged.encoded_size_bound() <= MAX_TRANSFER_HANDLE_BYTES

        # The consumer reads each layer's region over the peer mechanism.
        peer = mechanisms[1]
        write = target.prepare_install(
            merged,
            request_pool_idx=1,
            tables=_tables(target, pages, 2 * target.shapes[0].page_tokens),
            initialized_units=pages,
            transports={peer: transports[2][peer]},
        )
        write.result(timeout=60)
        target.install(installed_buffer=_buffer(101), write=write)
        for layer, name in enumerate(target.cache.groups[0].layers):
            key, value = target.cache.state(name).read(
                pages, start=0, length=extent
            )
            torch.testing.assert_close(key, wanted[layer], rtol=0, atol=0)
            torch.testing.assert_close(
                value, -wanted[layer] / 2, rtol=0, atol=0
            )
    finally:
        if write is not None:
            target.imports.abandon(write)
        for shard, rank_transports in zip(
            shards, transports[: len(shards)], strict=True
        ):
            for tensor in shard.tensors:
                for location in tensor.locations:
                    rank_transports[location.backend].release(location)
        for pool in sources[: len(shards)]:
            pool.release_buffers((source,))
        for pool in pools:
            pool.imports.stop()
        for rank_transports in transports:
            for transport in rank_transports.values():
                transport.close()
        for pool in pools:
            pool.close()
        for event in events:
            event.close()


def test_fp8_append_preserves_installed_scale_when_producer_head_group_changes() -> (  # noqa: E501
    None
):
    pools = [
        mha_pool(
            num_layers=1,
            num_kv_heads=heads,
            total_kv_heads=4,
            head_dim=1,
            dtype=torch.bfloat16,
            store_dtype=torch.float8_e4m3fn,
            total_layers=1,
            num_pages=2,
            page_size=4,
            device="cpu",
            request_pool_size=1,
            table_width=1,
        )
        for heads in (2, 4, 2)
    ]
    tables = [pool.block_tables for pool in pools]
    for table in tables:
        table.install(((1, 0, 0, (1,), 4),))
    owners = [pool for pool, table in zip(pools, tables, strict=True)]
    events = [EventPool() for _ in pools]
    transports = [
        make_transport(
            "local", byte_capacity=4096, ticket_capacity=16, event_pool=event
        )
        for event in events
    ]
    values = (
        torch.tensor((112, 224, 448, 896), dtype=torch.bfloat16)
        .reshape(1, 4, 1)
        .expand(4, -1, -1)
    )
    exports = []
    writes = []
    try:
        for index in (0, 1):
            pools[index].cache.state(
                pools[index].cache.groups[0].layers[0]
            ).write(
                (1,),
                start=0,
                key=values[:2, : pools[index].info.groups[0].num_kv_heads],
                value=-values[:2, : pools[index].info.groups[0].num_kv_heads],
            )
        for call, source_index, extent in ((1, 0, 2), (2, 1, 4)):
            source = _buffer(call)
            if call == 2:
                pools[1].cache.state(pools[1].cache.groups[0].layers[0]).write(
                    (1,), start=2, key=values[2:], value=-values[2:]
                )
            export = owners[source_index].export(
                request_pool_idx=1,
                visible_length=extent,
                destination="consumer",
                buffer=source,
                transports={"local": transports[source_index]},
            )
            exports.append((source_index, source, export))
            owners[source_index].validate_exports(((source, export),), ())
            owners[source_index].apply_exports(((source, export),), ())
            write = owners[2].prepare_install(
                export,
                request_pool_idx=1,
                tables=_tables(pools[2], (1,), 4),
                initialized_units=(1,) if call == 1 else (),
                transports={"local": transports[2]},
            )
            writes.append(write)
            write.result(timeout=10)
            installed = _buffer(100 + call)
            result = owners[2].install(
                installed_buffer=installed,
                write=write,
            )
            owners[2].validate_exports((), ((source, installed, result),))
            owners[2].apply_exports((), ((source, installed, result),))
            key, value = (
                pools[2]
                .cache.state(pools[2].cache.groups[0].layers[0])
                .read((1,), start=0, length=extent)
            )
            torch.testing.assert_close(key, values[:extent, :2], rtol=0, atol=0)
            torch.testing.assert_close(
                value, -values[:extent, :2], rtol=0, atol=0
            )
            if call == 1:
                # A replica publishes the same buffer identity with a wider
                # quantization group before producing the next suffix.
                other = source
                replica = owners[1].export(
                    request_pool_idx=1,
                    visible_length=extent,
                    destination="consumer",
                    buffer=other,
                    transports={"local": transports[1]},
                )
                exports.append((1, other, replica))
                owners[1].validate_exports(((other, replica),), ())
                owners[1].apply_exports(((other, replica),), ())
    finally:
        for write in writes:
            pools[2].imports.abandon(write)
        for index, buffer, export in exports:
            for tensor in export.tensors:
                for location in tensor.locations:
                    transports[index].release(location)
            pools[index].release_buffers((buffer,))
        for pool in pools:
            pool.imports.stop()
        for transport in transports:
            transport.close()
        for pool in pools:
            pool.close()
        for event in events:
            event.close()
