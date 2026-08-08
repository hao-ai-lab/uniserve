from __future__ import annotations

import gc
import weakref

import pytest
import torch

from uniserve_worker.foundation.errors import ErrorCode, ResourceError, WorkerError
from uniserve_worker.runtime.completion_store import CompletionArena

pytestmark = pytest.mark.unit


def test_completion_observation_releases_a_new_physical_generation() -> None:
    arena = CompletionArena(depth=1, token_capacity=4)
    lease = arena.reserve(2)
    capture = lease.capture(torch.tensor([17, 29], dtype=torch.long))
    lease.seal()

    assert capture.ready()
    assert capture.values() == (17, 29)
    first_generation = lease.generation
    copy_us, ready_to_observed_us = lease.observe(0, first_generation)
    assert copy_us >= 0
    assert ready_to_observed_us >= 0
    lease.observe(1, first_generation)

    successor = arena.reserve(1)
    assert successor.generation != first_generation


def test_completion_capacity_reports_backpressure_until_observation() -> None:
    arena = CompletionArena(depth=1, token_capacity=1)
    lease = arena.reserve(1)
    lease.seal()

    with pytest.raises(ResourceError):
        arena.reserve(1)

    lease.observe(0, lease.generation)
    assert arena.reserve(1).generation != lease.generation


def test_completion_generation_is_validated_at_observation() -> None:
    arena = CompletionArena(depth=1, token_capacity=1)
    lease = arena.reserve(1)
    lease.seal()

    with pytest.raises(WorkerError) as raised:
        lease.observe(0, lease.generation + 1)

    assert raised.value.code == ErrorCode.INVARIANT_VIOLATION
    assert raised.value.fatal
    lease.observe(0, lease.generation)


def test_execution_tensor_retires_before_host_completion_observation() -> None:
    arena = CompletionArena(depth=1, token_capacity=2)
    lease = arena.reserve(1)
    execution_tensor = torch.tensor([5, 8], dtype=torch.long)
    retired: list[bool] = []
    weakref.finalize(execution_tensor, retired.append, True)
    capture = lease.capture(execution_tensor)
    lease.seal()

    del execution_tensor
    gc.collect()

    assert retired == [True]
    assert capture.values() == (5, 8)
    lease.observe(0, lease.generation)


def test_completion_capture_cannot_exceed_the_preallocated_slab() -> None:
    arena = CompletionArena(depth=1, token_capacity=2)
    lease = arena.reserve(1)

    with pytest.raises(ResourceError):
        lease.capture(torch.tensor([1, 2, 3], dtype=torch.long))

    lease.abandon()


def test_abandoned_query_ready_completion_returns_capacity() -> None:
    arena = CompletionArena(depth=1, token_capacity=1)
    lease = arena.reserve(1)
    lease.seal()
    lease.abandon()

    assert arena.reserve(1).generation != lease.generation


def test_partition_leases_share_one_bounded_completion_byte_pool() -> None:
    arena = CompletionArena(depth=3, token_capacity=4, total_token_capacity=4)
    first = arena.reserve(1, token_capacity=2)
    second = arena.reserve(1, token_capacity=2)
    first.seal()
    second.seal()

    with pytest.raises(ResourceError):
        arena.reserve(1, token_capacity=1)

    first.observe(0, first.generation)
    successor = arena.reserve(1, token_capacity=2)
    assert successor.generation >= 1


def test_completion_generation_owns_disjoint_token_and_image_byte_ranges() -> None:
    arena = CompletionArena(depth=1, token_capacity=4)
    lease = arena.reserve(1)
    tokens = lease.capture(torch.tensor([17, 29], dtype=torch.long))
    pixels = torch.tensor(
        [[[1, 2, 3], [4, 5, 6]]],
        dtype=torch.uint8,
    )
    image = lease.capture_bytes(pixels)
    lease.seal()

    assert tokens.values() == (17, 29)
    assert torch.equal(image.tensor(), pixels)
    generation = lease.generation
    lease.observe(0, generation)

    assert arena.reserve(1).generation != generation


def test_completion_generation_enforces_one_shared_token_and_byte_bound() -> None:
    arena = CompletionArena(depth=1, token_capacity=2)
    lease = arena.reserve(1)
    lease.capture(torch.tensor([7], dtype=torch.long))

    with pytest.raises(ResourceError):
        lease.capture_bytes(torch.ones(9, dtype=torch.uint8))

    lease.abandon()
