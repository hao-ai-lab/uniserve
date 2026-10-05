"""Supplement timeline hints with measured Python/native call edges."""

from __future__ import annotations

import pstats
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CallHint:
    caller: str
    callee: str
    direction: str
    calls: int
    total_seconds: float


@dataclass(frozen=True)
class CallReport:
    hints: list[CallHint]
    native_functions: list[str]
    notes: tuple[str, ...]


def analyze(filename: str, *, native_names: tuple[str, ...]) -> CallReport:
    """Find repeated call edges involving selected native function names.

    Accepts standard cProfile output. Substrings match builtin function labels,
    not source filenames: callers need no installation-path assumptions. The
    call graph cannot establish per-batch ordering or count unprofiled C API
    operations, including many attribute accesses and extension constructors.
    """
    # The standard-library stub omits Stats.stats, which carries caller edges.
    stats: Any = pstats.Stats(filename)
    native = {
        function
        for function in stats.stats
        if function[0] == "~"
        and any(name in function[2] for name in native_names)
    }
    hints = []
    for callee, (_, _, _, _, callers) in stats.stats.items():
        for caller, (calls, _, _, total) in callers.items():
            if calls < 2 or (caller in native) == (callee in native):
                continue
            peer = callee if caller in native else caller
            if peer[0] == "~":
                continue

            hints.append(
                CallHint(
                    _name(caller),
                    _name(callee),
                    "native-to-python"
                    if caller in native
                    else "python-to-native",
                    calls,
                    total,
                )
            )

    hints.sort(key=lambda hint: (-hint.total_seconds, -hint.calls))
    notes = [
        "Repeated edges are inspection candidates: batch numerical calls are "
        "expected, while per-row metadata conversions may be avoidable.",
        "cProfile has no timeline or batch attribution. Its thread coverage "
        "depends on capture; foreign threads and many C API accesses "
        "are absent.",
        "Inclusive edge times overlap. Do not sum them or treat them as FFI "
        "overhead; they include work performed by the callee.",
    ]
    if not native:
        notes.append(
            "No selected native functions recorded; coverage is unknown."
        )
    return CallReport(
        hints,
        sorted(_name(function) for function in native),
        tuple(notes),
    )


def _name(function: tuple[str, int, str]) -> str:
    filename, line, name = function
    return name if filename == "~" else f"{filename}:{line}({name})"
