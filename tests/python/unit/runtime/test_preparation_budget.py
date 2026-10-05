"""A bound graph budget holds everything startup preparation leaves resident."""

from __future__ import annotations

import gc
import weakref

import pytest

from tests.python.fixtures.device_storage import (
    DeviceStorage,
    install_device_storage,
)
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve_worker.model_executor.graph_storage import GraphStorage

pytestmark = pytest.mark.unit


def test_bound_budget_charges_storage_prepared_outside_the_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = DeviceStorage(total=100_000, free=50_000)
    storage.held = 50_000
    install_device_storage(monkeypatch, storage)
    graphs = GraphStorage()

    graphs.set_budget("cuda:0", 10_000)

    # Capture leaves graph executables, communicator buffers and modules
    # outside every pool; the budget holds them all the same.
    storage.held = 60_000
    graphs.check()

    storage.held = 60_001
    with pytest.raises(CUDAGraphError, match="exceeds its byte budget"):
        graphs.check()

    # Once startup is sealed, serving growth such as resident products
    # belongs to the worker's grant, not to graph residency.
    graphs.seal()
    graphs.check()


def test_graph_storage_releases_unreachable_execution_owners():
    class Execution:
        pass

    storage = GraphStorage()
    owner = Execution()
    owner.storage = storage
    storage.reserve(owner, ("cpu",))
    observed = weakref.ref(owner)
    del owner, storage

    gc.collect()
    assert observed() is None


def test_owner_finalization_can_observe_closed_storage():
    storage = GraphStorage()
    observed = []

    class Execution:
        def __del__(self):
            try:
                observed.append(storage.pool_bytes())
            except Exception as error:
                observed.append(error)

    owner = Execution()
    storage.reserve(owner, ("cpu",))
    del owner
    storage.close()

    assert observed == [{}]
