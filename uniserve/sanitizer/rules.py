"""Temporal execution rules over correlated profiler events."""

import re

from . import Event, Hint, Scope

RULES = (
    "host-sync",
    "gil-held-wait",
    "sync-copy",
    "pageable-copy",
    "device-roundtrip",
)

_HOST_WAITS = {
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuEventSynchronize",
}


def is_host_wait(name: str) -> bool:
    """Recognize CUDA host waits, including versioned and stream ABI names."""
    return _api_name(name) in _HOST_WAITS


def _api_name(name: str) -> str:
    return re.sub(r"(?:_v\d+|_ptsz|_ptds)+$", "", name)


def inspect_events(events: list[Event]) -> list[Hint]:
    """Apply advisory rules, retaining the original events as evidence.

    No duration threshold proves a wait unnecessary. Even a short wait can
    prevent schedule-ahead. Conversely, a long wait may belong to an explicit
    result consumer or startup. Scope, stack and workload decide necessity.
    """
    hints = []
    readbacks: dict[tuple[Scope, int], Event] = {}
    for event in events:
        # Symbols carry ABI and per-thread-default-stream suffixes.
        name = _api_name(event.name)
        if name in _HOST_WAITS:
            hints.append(
                Hint(
                    "host-sync",
                    f"Host synchronization API {event.name} "
                    f"inside {event.scope.name}.",
                    "Consider querying completion or using a stream dependency "
                    "to keep submitting ready work. Result consumption, "
                    "startup, shutdown and cleanup may require this wait.",
                    (event,),
                )
            )

        holds = [state for state in event.gil if state.name == "Holding GIL"]
        if holds:
            held_ns = sum(
                min(event.end_ns, state.end_ns)
                - max(event.start_ns, state.start_ns)
                for state in holds
            )
            waiters = {
                state.global_tid
                for state in event.gil
                if state.name == "Waiting for GIL"
            }
            contention = (
                f" {len(waiters)} other thread(s) in this process were "
                "waiting for the GIL during that overlap."
                if waiters
                else " No overlapping GIL waiter was captured."
            )
            hints.append(
                Hint(
                    "gil-held-wait",
                    f"{event.name} overlapped GIL ownership on its thread "
                    f"for {held_ns} ns." + contention,
                    "Inspect the native binding or resource destructor. "
                    "A necessary GPU wait can still release the GIL. "
                    "The overlap alone does not prove deadlock or a latency "
                    "bottleneck; interpreter identity is not in this trace.",
                    (event,),
                )
            )

        for copy in event.copies:
            if (
                "Memcpy" in name
                and "Async" not in name
                and copy.direction in ("Host-to-Device", "Device-to-Host")
            ):
                hints.append(
                    Hint(
                        "sync-copy",
                        f"Synchronous host/device copy: {event.name}.",
                        "Consider an asynchronous copy with pinned storage and "
                        "a completion dependency. Immediate host consumption "
                        "or startup can justify a synchronous copy.",
                        (event,),
                    )
                )

            if "Pageable" in (copy.source_memory, copy.destination_memory):
                hints.append(
                    Hint(
                        "pageable-copy",
                        f"{copy.direction}: {copy.bytes} bytes through "
                        "pageable host memory. An Async API may block or "
                        "stage the copy.",
                        "Consider reusing pinned host buffers on the "
                        "execution path. One-time loading and host-only copies "
                        "may not justify pinned memory.",
                        (event,),
                    )
                )

            key = event.scope, copy.device
            if copy.direction == "Device-to-Host":
                readbacks[key] = event
            elif copy.direction == "Host-to-Device":
                previous = readbacks.pop(key, None)
                if previous is None:
                    continue
                ready = max(c.end_ns for c in previous.copies)
                if ready > event.start_ns:
                    continue

                hints.append(
                    Hint(
                        "device-roundtrip",
                        "A D-to-H copy completed before an H-to-D submission "
                        "on the same thread, device and execution scope.",
                        "Check whether intermediate values or decisions can "
                        "stay on the device. The trace does not establish "
                        "that these copies carry the same data; independent "
                        "inputs or CPU computation can explain this ordering.",
                        (previous, event),
                    )
                )

    return hints
