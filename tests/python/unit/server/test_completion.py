from __future__ import annotations

import pytest
import torch

from uniserve_worker.foundation.errors import ErrorCode, ResourceError, WorkerError
from uniserve_worker.server.completion import CompletionArena

pytestmark = pytest.mark.unit


def test_observing_every_row_releases_capacity_with_a_new_generation() -> None:
    arena = CompletionArena(depth=1, token_capacity=4)
    lease = arena.reserve(2)
    tokens = lease.capture(torch.tensor([17, 29], dtype=torch.long))
    lease.seal()

    assert tokens.values() == (17, 29)
    generation = lease.generation
    copy_us, ready_to_observed_us = lease.observe(0, generation)
    assert copy_us >= 0
    assert ready_to_observed_us >= 0

    with pytest.raises(ResourceError):
        arena.reserve(1)

    lease.observe(1, generation)
    assert arena.reserve(1).generation != generation


def test_stale_generation_is_rejected_without_releasing_the_live_lease() -> None:
    arena = CompletionArena(depth=1, token_capacity=1)
    lease = arena.reserve(1)
    lease.seal()

    with pytest.raises(WorkerError) as raised:
        lease.observe(0, lease.generation + 1)

    assert raised.value.code == ErrorCode.INVARIANT_VIOLATION
    assert raised.value.fatal
    lease.observe(0, lease.generation)
    assert arena.reserve(1).generation != lease.generation


def test_capture_respects_the_lease_byte_bound() -> None:
    arena = CompletionArena(depth=1, token_capacity=2)
    lease = arena.reserve(1)
    lease.capture(torch.tensor([7], dtype=torch.long))

    with pytest.raises(ResourceError):
        lease.capture_bytes(torch.ones(9, dtype=torch.uint8))

    lease.abandon()


def test_token_and_image_captures_materialize_from_disjoint_ranges() -> None:
    arena = CompletionArena(depth=1, token_capacity=4)
    lease = arena.reserve(1)
    tokens = lease.capture(torch.tensor([17, 29], dtype=torch.long))
    pixels = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.uint8)
    image = lease.capture_bytes(pixels)
    lease.seal()

    assert tokens.values() == (17, 29)
    assert torch.equal(image.tensor(), pixels)
    lease.observe(0, lease.generation)


def test_partition_leases_share_one_bounded_capacity() -> None:
    arena = CompletionArena(depth=3, token_capacity=4, total_token_capacity=4)
    first = arena.reserve(1, token_capacity=2)
    second = arena.reserve(1, token_capacity=2)
    first.seal()
    second.seal()

    with pytest.raises(ResourceError):
        arena.reserve(1, token_capacity=1)

    first.observe(0, first.generation)
    successor = arena.reserve(1, token_capacity=2)
    assert successor.row_count == 1


def test_abandoned_ready_completion_returns_capacity() -> None:
    arena = CompletionArena(depth=1, token_capacity=1)
    lease = arena.reserve(1)
    lease.seal()
    lease.abandon()

    assert arena.reserve(1).generation != lease.generation
