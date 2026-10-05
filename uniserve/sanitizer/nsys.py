"""Read CUDA and NVTX evidence from a Nsight Systems SQLite export."""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import replace
from urllib.parse import quote

from . import Copy, Event, Report, Scope
from .rules import inspect_events


def analyze(
    filename: str | os.PathLike[str], *, scope_prefix: str = "uniserve."
) -> Report:
    """Inspect captured calls inside synchronous ranges with this prefix.

    Reads an existing export without modifying it. Times are nanoseconds;
    global thread IDs and correlation IDs can be looked up in Nsight Systems.
    Missing activities are reported as coverage gaps, never a clean bill of
    health. SQL/schema errors propagate to the caller.
    """
    filename = quote(os.path.abspath(filename), safe="/")
    with closing(sqlite3.connect(f"file:{filename}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        report = Report()
        tables = {
            row[0] for row in db.execute("SELECT name FROM sqlite_master")
        }
        required = {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME"}
        missing = required - tables
        if missing:
            report.notes.append(
                "Missing trace activities: " + ", ".join(sorted(missing))
            )
            return report

        scopes = _scopes(db, scope_prefix)
        report.coverage["scopes"] = sum(
            scope.name.startswith(scope_prefix)
            for thread_scopes in scopes.values()
            for scope in thread_scopes
        )
        report.coverage["batch_scopes"] = sum(
            scope.batch_id is not None
            for thread_scopes in scopes.values()
            for scope in thread_scopes
        )
        if not report.coverage["scopes"]:
            report.notes.append(
                f"No closed NVTX ranges match {scope_prefix!r}."
            )
        elif not report.coverage["batch_scopes"]:
            report.notes.append(
                "No native batch stages recorded; preparation and completion "
                "coverage is unknown."
            )

        events = _events(db, tables, scopes, scope_prefix, report)
        events = _copies(db, tables, events, report)
        report.hints = inspect_events(events)
        report.notes.append(
            "Hints are candidates for inspection, not errors. Trace timing "
            "includes profiler overhead; it is not a speedup measurement."
        )
        return report


def _scopes(db: sqlite3.Connection, prefix: str) -> dict[int, list[Scope]]:
    scopes: dict[int, list[Scope]] = defaultdict(list)
    rows = db.execute(
        """
        SELECT n.*, coalesce(n.text, s.value) AS name
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId
        WHERE n.end IS NOT NULL AND n.end >= n.start
          AND (n.endGlobalTid IS NULL OR n.endGlobalTid = n.globalTid)
          AND (instr(coalesce(n.text, s.value), ?) = 1
               OR instr(coalesce(n.text, s.value), 'uniserve.worker.') = 1)
        ORDER BY n.start, n.end DESC
        """,
        (prefix,),
    )
    for row in rows:
        batch_id = None
        if row["name"].startswith("uniserve.worker."):
            batch_id = row["uint64Value"]

        scopes[row["globalTid"]].append(
            Scope(
                row["name"],
                row["start"],
                row["end"],
                row["globalTid"],
                batch_id,
            )
        )

    return scopes


def _events(
    db: sqlite3.Connection,
    tables: set[str],
    scopes: dict[int, list[Scope]],
    prefix: str,
    report: Report,
) -> list[Event]:
    # Nsight exports Driver and Runtime calls in the same activity table.
    # CUDA backtrace records repeat a call's interval/correlation ID; they
    # supply stacks, not additional API invocations.
    classes = {
        row["name"]: row["id"]
        for row in db.execute("SELECT id, name FROM ENUM_NSYS_EVENT_CLASS")
    }
    runtime = classes["TRACE_PROCESS_EVENT_CUDA_RUNTIME"]
    driver = classes["TRACE_PROCESS_EVENT_CUDA_DRIVER"]
    backtrace = classes["TRACE_PROCESS_EVENT_CUDABACKTRACE"]
    stacks = {
        (row["globalTid"], row["correlationId"]): row["callchainId"]
        for row in db.execute(
            "SELECT globalTid, correlationId, callchainId "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE eventClass = ?",
            (backtrace,),
        )
    }
    report.coverage.update(runtime_calls=0, driver_calls=0, scoped_calls=0)

    positions: dict[int, int] = defaultdict(int)
    active: dict[int, list[Scope]] = defaultdict(list)
    stack_cache: dict[int, tuple[str, ...]] = {}
    events = []
    rows = db.execute(
        """
        SELECT a.*, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id = a.nameId
        WHERE a.eventClass IN (?, ?)
        ORDER BY a.start, a.end DESC
        """,
        (runtime, driver),
    )
    for row in rows:
        kind = (
            "runtime_calls" if row["eventClass"] == runtime else "driver_calls"
        )
        report.coverage[kind] += 1
        tid = row["globalTid"]
        candidates = scopes.get(tid, ())
        position = positions[tid]
        while position < len(candidates):
            scope = candidates[position]
            if scope.start_ns > row["start"]:
                break
            active[tid].append(scope)
            position += 1
        positions[tid] = position
        active[tid] = [s for s in active[tid] if s.end_ns >= row["start"]]
        enclosing = [s for s in active[tid] if s.end_ns >= row["end"]]
        selected = [s for s in enclosing if s.name.startswith(prefix)]
        if not selected:
            continue

        # The innermost numerical range inherits its enclosing batch number.
        scope = selected[-1]
        if scope.batch_id is None:
            batch = next(
                (
                    s.batch_id
                    for s in reversed(enclosing)
                    if s.batch_id is not None
                ),
                None,
            )
            scope = replace(scope, batch_id=batch)

        chain = row["callchainId"]
        if chain is None:
            chain = stacks.get((tid, row["correlationId"]))
        if chain is not None and "CUDA_CALLCHAINS" in tables:
            if chain not in stack_cache:
                stack_cache[chain] = _stack(db, chain)
        stack = stack_cache.get(chain, ())
        events.append(
            Event(
                row["name"],
                row["start"],
                row["end"],
                tid,
                row["correlationId"],
                scope,
                stack,
            )
        )

    report.coverage["scoped_calls"] = len(events)
    report.coverage["scoped_calls_with_stack"] = sum(
        bool(e.stack) for e in events
    )
    for kind in ("runtime_calls", "driver_calls"):
        if report.coverage[kind] == 0:
            report.notes.append(
                f"No {kind} recorded; this API coverage is unknown."
            )
    if not stack_cache:
        report.notes.append("No CUDA call stacks in selected scopes.")
    return events


def _stack(db: sqlite3.Connection, chain: int) -> tuple[str, ...]:
    rows = db.execute(
        """
        SELECT s.value AS symbol, m.value AS module
        FROM CUDA_CALLCHAINS c
        LEFT JOIN StringIds s ON s.id = c.symbol
        LEFT JOIN StringIds m ON m.id = c.module
        WHERE c.id = ? ORDER BY c.stackDepth
        """,
        (chain,),
    )
    return tuple(
        f"{row['symbol'] or '<unresolved>'} ({row['module'] or '?'})"
        for row in rows
    )


def _copies(
    db: sqlite3.Connection,
    tables: set[str],
    events: list[Event],
    report: Report,
) -> list[Event]:
    if "CUPTI_ACTIVITY_KIND_MEMCPY" not in tables:
        report.notes.append(
            "No GPU copy activities; copy rules have no evidence."
        )
        return events

    # Nsight globalTid packs the process above the low 24 thread-ID bits.
    # Correlation IDs are process-local; include the process when joining ranks.
    keys = {(e.global_tid & ~0xFFFFFF, e.correlation_id) for e in events}
    copies: dict[tuple[int, int], list[Copy]] = defaultdict(list)
    rows = db.execute(
        """
        SELECT c.*, d.label AS direction,
               s.label AS source, t.label AS destination
        FROM CUPTI_ACTIVITY_KIND_MEMCPY c
        LEFT JOIN ENUM_CUDA_MEMCPY_OPER d ON d.id = c.copyKind
        LEFT JOIN ENUM_CUDA_MEM_KIND s ON s.id = c.srcKind
        LEFT JOIN ENUM_CUDA_MEM_KIND t ON t.id = c.dstKind
        ORDER BY c.start
        """
    )
    for row in rows:
        key = row["globalPid"], row["correlationId"]
        if key not in keys:
            continue
        copies[key].append(
            Copy(
                row["start"],
                row["end"],
                row["deviceId"],
                row["streamId"],
                row["bytes"],
                row["direction"] or "Unknown",
                row["source"] or "Unknown",
                row["destination"] or "Unknown",
            )
        )

    report.coverage["scoped_copies"] = sum(map(len, copies.values()))
    return [
        replace(
            e,
            copies=tuple(
                copies.get((e.global_tid & ~0xFFFFFF, e.correlation_id), ())
            ),
        )
        for e in events
    ]
