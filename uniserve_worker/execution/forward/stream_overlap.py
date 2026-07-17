"""Intra-worker plan/forward CUDA stream overlap.

Implements ``specs/intra-worker-stream-overlap.md`` for the unified forward
path. The worker's host loop already dispatches batch N+1 while the GPU runs
batch N (host pipeline depth), but N+1's prepare work — staging H2D copies and
attention-plan metadata tensors — enqueues on the forward stream, where it
queues behind N's kernels. :class:`PlanStreamOverlap` runs that prepare phase
on a dedicated ``plan_stream`` so the copies execute concurrently with the
previous forward, then makes the forward stream wait on a recorded plan event
before consuming the prepared batch.

Safety model (the WAR fences the spec calls out):

* The text staging ring (:class:`~uniserve_worker.runtime.tensor_staging.
  TextTensorStager`) reuses one pinned-host and one device buffer set per slot
  with period ``ring_depth``. Before preparing batch K the coordinator
  host-syncs on plan-done(K - R) — so the CPU never refills a pinned buffer
  whose H2D copy is still pending — and makes ``plan_stream`` wait on
  forward-done(K - R) — so a slot's device buffers are never rewritten while a
  previous forward may still read them.
* Tensors allocated under ``plan_stream`` are freed back to that stream's
  allocator pool, so a released batch could be reused by a later prepare while
  the forward still reads it. The coordinator therefore retains each prepared
  batch until ``plan_stream`` has waited on that batch's forward-done event,
  which orders any reuse after the read.

FlashInfer ``plan()`` itself stays on the forward stream (it runs inside the
first attention layer of an eager forward), so the shared plan workspace stays
stream-ordered against the kernels that read it and needs no double buffering
under this seam. Graph replay refreshes its stable buffers on the forward
stream after the plan event, so captured graphs are likewise unaffected.

The coordinator is constructed only when ``UNISERVE_STREAM_OVERLAP=1`` and the
model runs on CUDA; otherwise the runner keeps the fully serial path.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import torch

__all__ = ["PlanStreamOverlap", "PreparedOnPlanStream"]

_T = TypeVar("_T")


@dataclass(frozen=True)
class PreparedOnPlanStream:
    """One prepare-phase result plus the event that fences its consumption."""

    value: Any
    plan_done: torch.cuda.Event


@dataclass(frozen=True)
class _InflightForward:
    plan_done: torch.cuda.Event
    forward_done: torch.cuda.Event
    retained: Any


class PlanStreamOverlap:
    """Runs batch preparation on a dedicated stream, fenced against reuse.

    ``max_inflight`` must equal the staging-ring depth: it is the reuse period
    of the pinned/device staging buffers, and the coordinator's fences are what
    make that reuse safe across streams.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        max_inflight: int,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("PlanStreamOverlap requires a CUDA device")
        self.device = device
        self.plan_stream = torch.cuda.Stream(device=device)
        self.max_inflight = max(1, int(max_inflight))
        self._inflight: deque[_InflightForward] = deque()

    def prepare(self, build: Callable[[], _T]) -> PreparedOnPlanStream:
        """Run ``build`` under ``plan_stream`` and record its completion event.

        Applies the ring-reuse fences before running: host-sync on the oldest
        retired prepare (pinned-buffer WAR) and a device-side wait on its
        forward (device-buffer WAR), then releases that batch's references.
        """

        while len(self._inflight) >= self.max_inflight:
            oldest = self._inflight.popleft()
            oldest.plan_done.synchronize()
            self.plan_stream.wait_event(oldest.forward_done)
        with torch.cuda.stream(self.plan_stream):
            value = build()
            plan_done = torch.cuda.Event()
            plan_done.record(self.plan_stream)
        return PreparedOnPlanStream(value=value, plan_done=plan_done)

    def launch(
        self,
        prepared: PreparedOnPlanStream,
        run: Callable[[], _T],
        *,
        retain: Any,
    ) -> _T:
        """Launch the forward on the current stream after the plan event.

        ``retain`` (the prepared batch) is held until the ring-reuse fence for
        its slot passes, keeping plan-stream allocations alive while the
        forward may still read them.
        """

        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(prepared.plan_done)
        result = run()
        forward_done = torch.cuda.Event()
        forward_done.record(stream)
        self._inflight.append(
            _InflightForward(
                plan_done=prepared.plan_done,
                forward_done=forward_done,
                retained=retain,
            )
        )
        return result

    def drain(self) -> None:
        """Retire every in-flight forward (host-blocking); used by tests."""

        while self._inflight:
            entry = self._inflight.popleft()
            entry.plan_done.synchronize()
            entry.forward_done.synchronize()
