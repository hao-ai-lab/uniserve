"""Offline execution hints from profiler evidence.

Hints identify work to inspect, not correctness failures or proven speedups.
The analyzer never runs in the serving process.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scope:
    """A synchronous NVTX range, in nanoseconds on the capture timeline."""

    name: str
    start_ns: int
    end_ns: int
    global_tid: int
    batch_id: int | None = None


@dataclass(frozen=True)
class Copy:
    """Device copy activity correlated with its host CUDA call."""

    start_ns: int
    end_ns: int
    device: int
    stream: int
    bytes: int
    direction: str
    source_memory: str
    destination_memory: str


@dataclass(frozen=True)
class Event:
    """One actual CUDA API call, with optional stack and GPU copy evidence."""

    name: str
    start_ns: int
    end_ns: int
    global_tid: int
    correlation_id: int
    scope: Scope
    stack: tuple[str, ...] = ()
    copies: tuple[Copy, ...] = ()
    gil: tuple[Scope, ...] = ()


@dataclass(frozen=True)
class Hint:
    rule: str
    observation: str
    suggestion: str
    evidence: tuple[Event, ...]


@dataclass
class Report:
    hints: list[Hint] = field(default_factory=list)
    coverage: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
