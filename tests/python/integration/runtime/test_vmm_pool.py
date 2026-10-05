"""Export capacity follows physical producer and remote-reader completion."""

from __future__ import annotations

import os
import random
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tests.python.fixtures.cuda_stream import blocked_stream
from uniserve.runtime import EventPool
from uniserve_worker.errors import WorkerError
from uniserve_worker.transport.descriptor_grants import DescriptorGrants, fetch
from uniserve_worker.transport.vmm_pool import (
    ACK_WORD_BYTES,
    ACKNOWLEDGED,
    CLAIMED,
    HEADER_BYTES,
    PoolExhaustedError,
    VmmPool,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture
def pool():
    value = VmmPool(torch.device("cuda:0"), capacity_bytes=8 << 20)
    try:
        yield value
    finally:
        value.close()


@pytest.fixture
def events():
    value = EventPool()
    try:
        yield value
    finally:
        value.close()


def _producer(events):
    event = events.acquire(torch.device("cuda:0"))
    events.record(event, torch.device("cuda:0"))
    return event


def _reap(pool):
    deadline = time.monotonic() + 5
    while pool.awaiting_acknowledgment():
        pool.reap()
        assert time.monotonic() < deadline, "readers did not retire"
        time.sleep(0.001)


def test_one_exported_mapping_addresses_disjoint_payloads(pool):
    first = pool.reserve(4096)
    second = pool.reserve(4096)
    first.storage.fill_(11)
    second.storage.fill_(23)

    torch.testing.assert_close(
        first.storage, torch.full_like(first.storage, 11)
    )
    torch.testing.assert_close(
        second.storage, torch.full_like(second.storage, 23)
    )
    assert first.payload_offset == first.offset + HEADER_BYTES
    assert len(pool.handle) in (4, 64)


def test_capacity_returns_only_after_named_readers_finish(pool, events):
    chunk = pool.reserve(pool.capacity - HEADER_BYTES)
    # A consumer knows only the exported chunk offset and its own rank slot.
    words = pool.mapping.view(torch.uint8)[
        chunk.offset : chunk.payload_offset
    ].view(torch.int32)
    words[3] = CLAIMED
    words[7] = CLAIMED
    pool.retire(chunk, (3, 7), _producer(events))
    pool.reap()
    torch.cuda.synchronize()
    pool.reap()

    with pytest.raises(PoolExhaustedError):
        pool.reserve(1)
    words[3] = ACKNOWLEDGED
    torch.cuda.synchronize()
    pool.reap()
    with pytest.raises(PoolExhaustedError):
        pool.reserve(1)

    # An unnamed reader word does not retain this export.
    words[5] = CLAIMED
    words[7] = ACKNOWLEDGED
    torch.cuda.synchronize()
    _reap(pool)
    replacement = pool.reserve(pool.capacity - HEADER_BYTES)
    assert replacement.storage.numel() == pool.capacity - HEADER_BYTES
    assert replacement.acknowledgments.count_nonzero().item() == 0


def test_an_unclaimed_reader_does_not_retain_a_retired_export(pool, events):
    chunk = pool.reserve(pool.capacity - HEADER_BYTES)
    pool.retire(chunk, (9,), _producer(events))
    _reap(pool)
    assert pool.reserve(pool.capacity - HEADER_BYTES).storage.numel() > 0


def test_descriptor_grant_ends_with_chunk_retirement(pool, events):
    endpoint = f"uniserve-exports-{uuid.uuid4().hex}"
    export = uuid.uuid4().hex
    grants = DescriptorGrants(endpoint)

    try:
        with tempfile.TemporaryFile() as source:
            source.write(b"allocation")
            source.flush()
            grants.register(export, source.fileno())

        chunk = pool.reserve(4096)
        chunk.acknowledgments[1] = CLAIMED
        pool.retire(chunk, (1,), _producer(events), grants, export)
        del grants
        pool.reap()
        torch.cuda.synchronize()
        pool.reap()

        with os.fdopen(fetch(endpoint, export), "rb") as received:
            received.seek(0)
            assert received.read() == b"allocation"

        chunk.acknowledgments[1] = ACKNOWLEDGED
        torch.cuda.synchronize()
        _reap(pool)
        with pytest.raises(WorkerError):
            fetch(endpoint, export)
    finally:
        pool.close()


@pytest.mark.parametrize("consumers", ((), (1,)))
def test_pending_producer_retirement_does_not_block_the_host(
    pool, events, consumers
):
    chunk = pool.reserve(pool.capacity - HEADER_BYTES)
    torch.cuda.synchronize()

    with ThreadPoolExecutor(max_workers=1) as threads:
        with blocked_stream("cuda:0") as stream:
            with torch.cuda.stream(stream):
                chunk.storage.fill_(19)
                producer = _producer(events)
            pool.retire(chunk, consumers, producer)
            threads.submit(pool.reap).result(timeout=5)
            with pytest.raises(PoolExhaustedError):
                pool.reserve(1)

        _reap(pool)
    assert pool.reserve(pool.capacity - HEADER_BYTES).storage.numel() > 0


@pytest.mark.parametrize("grow_readback", (False, True))
def test_reaping_does_not_wait_for_later_work_on_the_calling_stream(
    pool, events, grow_readback
):
    if grow_readback:
        chunk = pool.reserve(4096)
        pool.retire(chunk, (1,), _producer(events))
        _reap(pool)

    for _ in range(2):
        chunk = pool.reserve(4096)
        pool.retire(chunk, (1,), _producer(events))
    torch.cuda.synchronize()

    def reap(stream):
        with torch.cuda.stream(stream):
            pool.reap()

    with ThreadPoolExecutor(max_workers=1) as threads:
        with blocked_stream("cuda:0") as stream:
            threads.submit(reap, stream).result(timeout=5)
        _reap(pool)


def test_released_handles_cannot_free_a_replacement(pool):
    first = pool.reserve(pool.capacity - HEADER_BYTES)
    torch.cuda.synchronize()
    pool.release(first)
    replacement = pool.reserve(pool.capacity - HEADER_BYTES)
    pool.release(first)
    with pytest.raises(PoolExhaustedError):
        pool.reserve(1)
    replacement.storage.fill_(31)
    torch.testing.assert_close(
        replacement.storage, torch.full_like(replacement.storage, 31)
    )


def test_a_tail_reservation_does_not_overwrite_a_live_payload(pool):
    unit = pool.capacity // 8
    low = pool.reserve(2 * unit - HEADER_BYTES)
    high = pool.reserve(4 * unit - HEADER_BYTES)
    torch.cuda.synchronize()
    pool.release(high)

    tail = pool.reserve(5 * unit - HEADER_BYTES)
    tail.storage.fill_(0xAB)
    last = pool.reserve(unit - HEADER_BYTES)
    last.storage.fill_(0xCD)
    low.storage.fill_(0xEF)
    torch.testing.assert_close(
        tail.storage, torch.full_like(tail.storage, 0xAB)
    )


def test_live_ranges_remain_disjoint_under_fragmentation(pool):
    generator = random.Random(0)
    unit = pool.capacity // 16
    live = []
    for _ in range(5000):
        if live and generator.random() < 0.5:
            chunk = live.pop(generator.randrange(len(live)))
            # Header initialization is an asynchronous producer access too.
            torch.cuda.current_stream().synchronize()
            pool.release(chunk)
        else:
            units = generator.randint(1, 12)
            spans = sorted(
                (value.offset, value.offset + value.nbytes) for value in live
            )
            try:
                live.append(pool.reserve(units * unit - HEADER_BYTES))
            except PoolExhaustedError:
                gaps = [
                    end - start
                    for start, end in zip(
                        [0, *(stop for _, stop in spans)],
                        [*(start for start, _ in spans), pool.capacity],
                        strict=True,
                    )
                ]
                assert max(gaps) < units * unit

        spans = sorted(
            (value.offset, value.offset + value.nbytes) for value in live
        )
        assert all(0 <= start < end <= pool.capacity for start, end in spans)
        assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


def test_header_words_match_the_wire_representation(pool):
    chunk = pool.reserve(4096)
    slot = 9
    word = pool.mapping[chunk.offset + slot * ACK_WORD_BYTES :][
        :ACK_WORD_BYTES
    ].view(torch.int32)
    word.fill_(CLAIMED)
    assert chunk.acknowledgments[slot].item() == CLAIMED
