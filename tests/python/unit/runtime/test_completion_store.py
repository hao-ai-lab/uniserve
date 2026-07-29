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


def test_abandoned_query_ready_completion_returns_capacity() -> None:
    arena = CompletionArena(depth=1, token_capacity=1)
    lease = arena.reserve(1)
    lease.seal()
    lease.abandon()

    assert arena.reserve(1).generation != lease.generation
