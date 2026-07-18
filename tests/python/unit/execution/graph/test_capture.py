"""Unit coverage for the shared graph capture lifecycle."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from uniserve_worker.contracts.forward_context import ForwardContext
from uniserve_worker.execution.graph.capture import Runner

pytestmark = pytest.mark.unit


def test_new_capture_reclaims_warmup_cache_before_graph_allocation(monkeypatch):
    events: list[str] = []

    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_stream(self, other: "Stream") -> None:
            events.append(f"{self.name}.wait({other.name})")

    class Graph:
        def capture_begin(self, *args, **kwargs) -> None:
            del args, kwargs
            events.append("capture_begin")

        def capture_end(self) -> None:
            events.append("capture_end")

    class GraphContext:
        def __init__(self, graph: Graph, **kwargs) -> None:
            self.graph = graph
            events.append(f"graph_context(pool={kwargs.get('pool')})")

        def __enter__(self) -> None:
            torch.cuda.empty_cache()
            self.graph.capture_begin()

        def __exit__(self, *args) -> None:
            del args
            self.graph.capture_end()

    class TestRunner(Runner):
        def capture_pool(self):
            return "shared_pool"

    current = Stream("current")
    warmup = Stream("warmup")
    graph = Graph()
    state = SimpleNamespace(graph=graph, logits=None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: current)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: warmup)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "graph", GraphContext)

    TestRunner()._capture_graph_state(
        device="cuda:0",
        state=state,
        run=lambda: object(),
        copy_inputs=lambda _state: events.append("copy"),
        before_run=lambda _state: events.append("prepare"),
    )

    assert "empty_cache" in events
    assert "graph_context(pool=shared_pool)" in events
    assert events.index("empty_cache") < events.index("capture_begin")
    assert events.count("capture_begin") == 1
    assert events.count("capture_end") == 1


def test_capture_reclaims_cached_blocks_under_device_memory_pressure(monkeypatch):
    events: list[str] = []

    class TestRunner(Runner):
        def __init__(self) -> None:
            self.name = "test"
            self.logger = None
            self.states = {}

    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (32 * 1024**2, 16 * gib))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 4 * gib)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 8 * gib)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))

    runner = TestRunner()
    result = runner._capture_or_replay(
        key="capacity",
        device="cuda:0",
        ctx=ForwardContext(),
        capture=lambda: events.append("capture") or object(),
        copy_inputs=lambda _state: None,
        replay=lambda _state: "result",
        record=lambda _event: None,
        disable=lambda _exc: None,
        capture_metric="capture",
        input_copy_metric="copy",
        replay_metric="replay",
    )

    assert result == "result"
    assert events == ["synchronize", "empty_cache", "capture"]

    events.clear()
    assert (
        runner._capture_or_replay(
            key="capacity",
            device="cuda:0",
            ctx=ForwardContext(),
            capture=lambda: events.append("capture") or object(),
            copy_inputs=lambda _state: None,
            replay=lambda _state: "replayed",
            record=lambda _event: None,
            disable=lambda _exc: None,
            capture_metric="capture",
            input_copy_metric="copy",
            replay_metric="replay",
        )
        == "replayed"
    )
    assert events == []


def test_capture_keeps_allocator_cache_when_device_has_headroom(monkeypatch):
    events: list[str] = []

    class TestRunner(Runner):
        def __init__(self) -> None:
            self.name = "test"
            self.logger = None
            self.states = {}

    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (4 * gib, 16 * gib))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 4 * gib)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 8 * gib)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))

    runner = TestRunner()
    result = runner._capture_or_replay(
        key="capacity",
        device="cuda:0",
        ctx=ForwardContext(),
        capture=lambda: events.append("capture") or object(),
        copy_inputs=lambda _state: None,
        replay=lambda _state: "result",
        record=lambda _event: None,
        disable=lambda _exc: None,
        capture_metric="capture",
        input_copy_metric="copy",
        replay_metric="replay",
    )

    assert result == "result"
    assert events == ["capture"]
