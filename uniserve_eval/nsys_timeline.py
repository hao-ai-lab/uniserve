"""Fixed Nsight Systems SQLite normalization for UniServe benchmark timelines."""

from __future__ import annotations

import re
import sqlite3
from array import array
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

_PROCESS_SHIFT = 24
_FIELD = re.compile(r"(?:^|\s)(step|partition|request|op|work|rank)=([^\s]+)")
_INVALID_CAPTURE = re.compile(
    r"(?:events? (?:were )?dropped|buffer overflow|trace data is incomplete|collection was truncated)",
    re.IGNORECASE,
)


def transform_timeline(
    source: Path,
    destination: Path,
    *,
    point_name: str,
    require_cuda: bool = True,
) -> None:
    """Create a stable interval/link database from one complete nsys export."""

    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"timeline output already exists: {destination}")
    with sqlite3.connect(source) as raw, sqlite3.connect(destination) as out:
        raw.row_factory = sqlite3.Row
        _create_schema(out)
        strings = _string_ids(raw)
        processes = _processes(raw)
        threads = _thread_names(raw, strings)
        tables = _tables(raw)
        if "NVTX_EVENTS" not in tables:
            raise RuntimeError("nsys export is missing NVTX intervals")
        if require_cuda and "CUPTI_ACTIVITY_KIND_RUNTIME" not in tables:
            raise RuntimeError("nsys export is missing CUDA runtime intervals")

        ranges_by_thread: dict[int, list[dict[str, Any]]] = defaultdict(list)
        ranges_by_process: dict[int, list[dict[str, Any]]] = defaultdict(list)
        correlation: dict[tuple[int, int], tuple[int, dict[str, Any]]] = {}
        gpu_intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)

        for row in _rows(raw, "NVTX_EVENTS"):
            start = _integer(row, "start")
            end = _integer(row, "end", default=start)
            text = _text(row, strings)
            global_tid = _optional_integer(row, "globalTid")
            global_pid, pid, process_name = _process_identity(row, processes)
            fields = _fields(text)
            category = _nvtx_category(text, end == start)
            event_id = _insert_event(
                out,
                point_name=point_name,
                category=category,
                process_id=pid,
                process_name=process_name,
                rank=fields.get("rank"),
                global_tid=global_tid,
                thread_name=threads.get(global_tid),
                start_ns=start,
                end_ns=end,
                name=text.split(maxsplit=1)[0] if text else "nvtx",
                nvtx_text=text,
                request_id=fields.get("request"),
                step_id=fields.get("step"),
                partition_id=fields.get("partition"),
                operation_id=fields.get("op"),
                work_kind=fields.get("work"),
                source_table="NVTX_EVENTS",
            )
            record = {
                "event_id": event_id,
                "start": start,
                "end": end,
                "rank": fields.get("rank"),
                "request": fields.get("request"),
                "step": fields.get("step"),
                "partition": fields.get("partition"),
                "op": fields.get("op"),
                "work": fields.get("work"),
            }
            if global_tid is not None:
                ranges_by_thread[global_tid].append(record)
            if global_pid is not None:
                ranges_by_process[global_pid].append(record)

        _enrich_nvtx(out, ranges_by_thread)
        thread_ranges = {
            identity: _RangeIndex(ranges) for identity, ranges in ranges_by_thread.items()
        }
        process_ranges = {
            identity: _RangeIndex(ranges) for identity, ranges in ranges_by_process.items()
        }

        runtime_ids: dict[tuple[int, int], int] = {}
        for row in (
            _rows(raw, "CUPTI_ACTIVITY_KIND_RUNTIME")
            if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables
            else ()
        ):
            start, end = _interval(row)
            global_tid = _optional_integer(row, "globalTid")
            global_pid, pid, process_name = _process_identity(row, processes)
            parent = _containing(thread_ranges.get(global_tid or -1), start, end)
            inherited = _inherited(parent)
            correlation_id = _optional_integer(row, "correlationId")
            event_id = _insert_event(
                out,
                point_name=point_name,
                category="cuda_api",
                process_id=pid,
                process_name=process_name,
                global_tid=global_tid,
                thread_name=threads.get(global_tid),
                start_ns=start,
                end_ns=end,
                correlation_id=correlation_id,
                name=_name(row, strings, "nameId", fallback="cuda_api"),
                source_table="CUPTI_ACTIVITY_KIND_RUNTIME",
                **inherited,
            )
            _link_parent(out, parent, event_id)
            if global_pid is not None and correlation_id is not None:
                correlation[(global_pid, correlation_id)] = (event_id, inherited)
                runtime_ids[(global_pid, correlation_id)] = event_id

        standard = {
            "NVTX_EVENTS",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
            "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
            "OSRT_API",
            "DIAGNOSTIC_EVENT",
        }
        _insert_gpu_table(
            raw,
            out,
            tables,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "cuda_kernel",
            point_name,
            strings,
            processes,
            correlation,
            gpu_intervals,
        )
        _insert_gpu_table(
            raw,
            out,
            tables,
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "memcpy",
            point_name,
            strings,
            processes,
            correlation,
            gpu_intervals,
        )
        _insert_gpu_table(
            raw,
            out,
            tables,
            "CUPTI_ACTIVITY_KIND_MEMSET",
            "memset",
            point_name,
            strings,
            processes,
            correlation,
            gpu_intervals,
        )
        _insert_gpu_table(
            raw,
            out,
            tables,
            "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
            "cuda_sync",
            point_name,
            strings,
            processes,
            correlation,
            gpu_intervals,
        )

        if "OSRT_API" in tables:
            for row in _rows(raw, "OSRT_API"):
                start, end = _interval(row)
                global_tid = _optional_integer(row, "globalTid")
                global_pid, pid, process_name = _process_identity(row, processes)
                parent = _containing(thread_ranges.get(global_tid or -1), start, end)
                event_id = _insert_event(
                    out,
                    point_name=point_name,
                    category="osrt",
                    process_id=pid,
                    process_name=process_name,
                    global_tid=global_tid,
                    thread_name=threads.get(global_tid),
                    start_ns=start,
                    end_ns=end,
                    name=_name(row, strings, "nameId", fallback="osrt"),
                    source_table="OSRT_API",
                    **_inherited(parent),
                )
                _link_parent(out, parent, event_id)

        for table in sorted(set(tables) - standard):
            upper = table.upper()
            columns = tables[table]
            if "start" not in columns or "end" not in columns:
                continue
            if "GRAPH" in upper:
                category = "cuda_graph"
            elif "NCCL" in upper:
                category = "nccl"
            elif "SCHED" in upper or "THREAD_STATE" in upper:
                category = "cpu_schedule"
            elif upper.startswith("CUPTI_ACTIVITY_KIND_"):
                category = "cuda_activity"
            else:
                continue
            _insert_generic_table(
                raw,
                out,
                table,
                category,
                point_name,
                strings,
                processes,
                threads,
                thread_ranges,
                process_ranges,
                correlation,
                gpu_intervals,
            )

        _insert_gpu_idle(out, point_name, gpu_intervals)
        _copy_diagnostics(raw, out, tables, strings, processes)
        _copy_metadata(raw, out, tables, point_name, source)
        _create_indexes(out)
        kernel_count = out.execute(
            "SELECT count(*) FROM events WHERE category = 'cuda_kernel'"
        ).fetchone()[0]
        if require_cuda and int(kernel_count) < 1:
            raise RuntimeError("nsys capture contains no CUDA kernel intervals")
        invalid = out.execute(
            "SELECT text FROM diagnostics WHERE text REGEXP ? LIMIT 1",
            (_INVALID_CAPTURE.pattern,),
        ).fetchone()
        if invalid is not None:
            raise RuntimeError(f"nsys reported an incomplete capture: {invalid[0]}")
        out.execute("INSERT INTO metadata(key, value) VALUES('status', 'complete')")
        out.commit()


def _create_schema(db: sqlite3.Connection) -> None:
    db.create_function(
        "REGEXP", 2, lambda pattern, value: bool(re.search(pattern, value or "", re.I))
    )
    db.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE events(
            event_id INTEGER PRIMARY KEY,
            point_name TEXT NOT NULL,
            category TEXT NOT NULL,
            process_id INTEGER,
            process_name TEXT,
            rank INTEGER,
            global_tid INTEGER,
            thread_name TEXT,
            device_id INTEGER,
            stream_id INTEGER,
            start_ns INTEGER NOT NULL,
            end_ns INTEGER NOT NULL,
            duration_ns INTEGER NOT NULL,
            correlation_id INTEGER,
            byte_count INTEGER,
            name TEXT NOT NULL,
            nvtx_text TEXT,
            request_id TEXT,
            step_id INTEGER,
            partition_id INTEGER,
            operation_id INTEGER,
            work_kind TEXT,
            source_table TEXT NOT NULL
        );
        CREATE TABLE event_links(
            parent_event_id INTEGER NOT NULL REFERENCES events(event_id),
            child_event_id INTEGER NOT NULL REFERENCES events(event_id),
            relation TEXT NOT NULL,
            PRIMARY KEY(parent_event_id, child_event_id, relation)
        );
        CREATE TABLE diagnostics(
            timestamp_ns INTEGER NOT NULL,
            severity TEXT NOT NULL,
            source TEXT NOT NULL,
            process_id INTEGER,
            text TEXT NOT NULL
        );
        """
    )


def _create_indexes(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE INDEX events_time ON events(start_ns, end_ns);
        CREATE INDEX events_gpu ON events(device_id, stream_id, start_ns);
        CREATE INDEX events_operation ON events(request_id, operation_id, start_ns);
        CREATE INDEX events_correlation ON events(process_id, correlation_id);
        """
    )


def _insert_gpu_table(
    raw: sqlite3.Connection,
    out: sqlite3.Connection,
    tables: dict[str, set[str]],
    table: str,
    category: str,
    point_name: str,
    strings: dict[int, str],
    processes: dict[int, tuple[int, str]],
    correlation: dict[tuple[int, int], tuple[int, dict[str, Any]]],
    gpu_intervals: dict[int, list[tuple[int, int]]],
) -> None:
    if table not in tables:
        return
    enum_copy = _enum(raw, "ENUM_CUDA_MEMCPY_OPER")
    enum_sync = _enum(raw, "ENUM_CUPTI_SYNC_TYPE")
    for row in _rows(raw, table):
        start, end = _interval(row)
        global_pid, pid, process_name = _process_identity(row, processes)
        correlation_id = _optional_integer(row, "correlationId")
        parent_id = None
        inherited: dict[str, Any] = {}
        if global_pid is not None and correlation_id is not None:
            linked = correlation.get((global_pid, correlation_id))
            if linked is not None:
                parent_id, inherited = linked
        if category == "cuda_kernel":
            name = _name(
                row,
                strings,
                "demangledName",
                "shortName",
                "mangledName",
                fallback="cuda_kernel",
            )
        elif category == "memcpy":
            name = enum_copy.get(_optional_integer(row, "copyKind") or -1, "memcpy")
        elif category == "cuda_sync":
            name = enum_sync.get(_optional_integer(row, "syncType") or -1, "cuda_sync")
        else:
            name = category
        device = _optional_integer(row, "deviceId")
        event_id = _insert_event(
            out,
            point_name=point_name,
            category=category,
            process_id=pid,
            process_name=process_name,
            device_id=device,
            stream_id=_optional_integer(row, "streamId"),
            start_ns=start,
            end_ns=end,
            correlation_id=correlation_id,
            byte_count=_optional_integer(row, "bytes"),
            name=name,
            source_table=table,
            **inherited,
        )
        if parent_id is not None:
            _link(out, parent_id, event_id, "cuda_correlation")
        if device is not None and category in {
            "cuda_kernel",
            "memcpy",
            "memset",
            "cuda_graph",
            "nccl",
        }:
            gpu_intervals[device].append((start, end))


def _insert_generic_table(
    raw: sqlite3.Connection,
    out: sqlite3.Connection,
    table: str,
    category: str,
    point_name: str,
    strings: dict[int, str],
    processes: dict[int, tuple[int, str]],
    threads: dict[int, str],
    ranges_by_thread: dict[int, _RangeIndex],
    ranges_by_process: dict[int, _RangeIndex],
    correlation: dict[tuple[int, int], tuple[int, dict[str, Any]]],
    gpu_intervals: dict[int, list[tuple[int, int]]],
) -> None:
    for row in _rows(raw, table):
        start, end = _interval(row)
        global_tid = _optional_integer(row, "globalTid", "threadId")
        global_pid, pid, process_name = _process_identity(row, processes)
        correlation_id = _optional_integer(row, "correlationId")
        parent = _containing(ranges_by_thread.get(global_tid or -1), start, end)
        if parent is None and global_pid is not None:
            parent = _containing(ranges_by_process.get(global_pid), start, end)
        inherited = _inherited(parent)
        correlation_parent = None
        if global_pid is not None and correlation_id is not None:
            linked = correlation.get((global_pid, correlation_id))
            if linked is not None:
                correlation_parent, inherited = linked
        device = _optional_integer(row, "deviceId", "gpuId")
        event_id = _insert_event(
            out,
            point_name=point_name,
            category=category,
            process_id=pid,
            process_name=process_name,
            global_tid=global_tid,
            thread_name=threads.get(global_tid),
            device_id=device,
            stream_id=_optional_integer(row, "streamId"),
            start_ns=start,
            end_ns=end,
            correlation_id=correlation_id,
            byte_count=_optional_integer(row, "bytes", "bytesProcessed"),
            name=_name(
                row,
                strings,
                "nameId",
                "demangledName",
                "shortName",
                fallback=table.lower(),
            ),
            source_table=table,
            **inherited,
        )
        if correlation_parent is not None:
            _link(out, correlation_parent, event_id, "cuda_correlation")
        else:
            _link_parent(out, parent, event_id)
        if device is not None and category in {"cuda_graph", "nccl", "cuda_activity"}:
            gpu_intervals[device].append((start, end))


def _insert_gpu_idle(
    out: sqlite3.Connection,
    point_name: str,
    gpu_intervals: dict[int, list[tuple[int, int]]],
) -> None:
    for device, intervals in gpu_intervals.items():
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        for previous, current in zip(merged, merged[1:]):
            if current[0] <= previous[1]:
                continue
            _insert_event(
                out,
                point_name=point_name,
                category="gpu_idle",
                device_id=device,
                start_ns=previous[1],
                end_ns=current[0],
                name="gpu_idle",
                source_table="derived",
            )


def _copy_diagnostics(
    raw: sqlite3.Connection,
    out: sqlite3.Connection,
    tables: dict[str, set[str]],
    strings: dict[int, str],
    processes: dict[int, tuple[int, str]],
) -> None:
    if "DIAGNOSTIC_EVENT" not in tables:
        return
    severity = _enum(raw, "ENUM_DIAGNOSTIC_SEVERITY_LEVEL")
    sources = _enum(raw, "ENUM_DIAGNOSTIC_SOURCE_TYPE")
    for row in _rows(raw, "DIAGNOSTIC_EVENT"):
        _, pid, _ = _process_identity(row, processes)
        out.execute(
            "INSERT INTO diagnostics VALUES(?, ?, ?, ?, ?)",
            (
                _integer(row, "timestamp"),
                severity.get(_optional_integer(row, "severity") or 0, "Unknown"),
                sources.get(_optional_integer(row, "source") or 0, "Unknown"),
                pid,
                _name(row, strings, "text", fallback=""),
            ),
        )


def _copy_metadata(
    raw: sqlite3.Connection,
    out: sqlite3.Connection,
    tables: dict[str, set[str]],
    point_name: str,
    source: Path,
) -> None:
    values = {"point_name": point_name, "source_sqlite": source.name}
    for table in ("META_DATA_CAPTURE", "META_DATA_EXPORT"):
        if table not in tables:
            continue
        for row in _rows(raw, table):
            values[f"{table.lower()}.{row['name']}"] = str(row["value"])
    out.executemany("INSERT INTO metadata(key, value) VALUES(?, ?)", sorted(values.items()))


def _insert_event(db: sqlite3.Connection, **values: Any) -> int:
    columns = (
        "point_name",
        "category",
        "process_id",
        "process_name",
        "rank",
        "global_tid",
        "thread_name",
        "device_id",
        "stream_id",
        "start_ns",
        "end_ns",
        "correlation_id",
        "byte_count",
        "name",
        "nvtx_text",
        "request_id",
        "step_id",
        "partition_id",
        "operation_id",
        "work_kind",
        "source_table",
    )
    start = int(values["start_ns"])
    end = max(start, int(values["end_ns"]))
    values["start_ns"] = start
    values["end_ns"] = end
    numeric = {"rank", "step_id", "partition_id", "operation_id"}
    row = []
    for column in columns:
        value = values.get(column)
        if column in numeric and value is not None:
            value = int(value)
        row.append(value)
    cursor = db.execute(
        f"INSERT INTO events({', '.join(columns)}, duration_ns) "
        f"VALUES({', '.join('?' for _ in columns)}, ?)",
        (*row, end - start),
    )
    return int(cursor.lastrowid)


def _link_parent(db: sqlite3.Connection, parent: dict[str, Any] | None, child: int) -> None:
    if parent is not None:
        _link(db, int(parent["event_id"]), child, "nvtx_contains")


def _link(db: sqlite3.Connection, parent: int, child: int, relation: str) -> None:
    db.execute(
        "INSERT OR IGNORE INTO event_links VALUES(?, ?, ?)",
        (parent, child, relation),
    )


def _inherited(parent: dict[str, Any] | None) -> dict[str, Any]:
    if parent is None:
        return {}
    return {
        "rank": parent.get("rank"),
        "request_id": parent.get("request"),
        "step_id": parent.get("step"),
        "partition_id": parent.get("partition"),
        "operation_id": parent.get("op"),
        "work_kind": parent.get("work"),
    }


def _enrich_nvtx(
    db: sqlite3.Connection,
    ranges_by_thread: dict[int, list[dict[str, Any]]],
) -> None:
    fields = ("rank", "request", "step", "partition", "op", "work")
    columns = {
        "rank": "rank",
        "request": "request_id",
        "step": "step_id",
        "partition": "partition_id",
        "op": "operation_id",
        "work": "work_kind",
    }
    for ranges in ranges_by_thread.values():
        ordered = sorted(
            ranges,
            key=lambda item: (item["start"], -item["end"], item["event_id"]),
        )
        stack: list[dict[str, Any]] = []
        for item in ordered:
            while stack and not (
                stack[-1]["start"] <= item["start"] and stack[-1]["end"] >= item["end"]
            ):
                stack.pop()
            parent = stack[-1] if stack else None
            inherited = (
                {name: parent.get(name) for name in fields if parent.get(name) is not None}
                if parent is not None
                else {}
            )
            inherited.update(
                {name: item.get(name) for name in fields if item.get(name) is not None}
            )
            item.update(inherited)
            if inherited:
                assignments = ", ".join(f"{columns[name]} = ?" for name in inherited)
                db.execute(
                    f"UPDATE events SET {assignments} WHERE event_id = ?",
                    (*inherited.values(), item["event_id"]),
                )
            if parent is not None:
                _link(db, int(parent["event_id"]), int(item["event_id"]), "nvtx_contains")
            stack.append(item)


class _RangeIndex:
    """Logarithmic innermost-container lookup for thread-nested NVTX ranges."""

    def __init__(self, ranges: Iterable[dict[str, Any]]) -> None:
        self.items = sorted(
            ranges,
            key=lambda item: (item["start"], -item["end"], item["event_id"]),
        )
        self.starts = [int(item["start"]) for item in self.items]
        size = 1
        while size < len(self.items):
            size <<= 1
        self.size = size
        self.max_end = array("q", [-1]) * (2 * size)
        for index, item in enumerate(self.items):
            self.max_end[size + index] = int(item["end"])
        for index in range(size - 1, 0, -1):
            self.max_end[index] = max(self.max_end[index * 2], self.max_end[index * 2 + 1])

    def containing(self, start: int, end: int) -> dict[str, Any] | None:
        limit = bisect_right(self.starts, int(start))
        index = self._rightmost(1, 0, self.size, limit, int(end))
        return None if index < 0 or index >= len(self.items) else self.items[index]

    def _rightmost(
        self,
        node: int,
        left: int,
        right: int,
        limit: int,
        minimum_end: int,
    ) -> int:
        if left >= limit or self.max_end[node] < minimum_end:
            return -1
        if right - left == 1:
            return left
        midpoint = (left + right) // 2
        selected = self._rightmost(node * 2 + 1, midpoint, right, limit, minimum_end)
        if selected >= 0:
            return selected
        return self._rightmost(node * 2, left, midpoint, limit, minimum_end)


def _containing(ranges: _RangeIndex | None, start: int, end: int) -> dict[str, Any] | None:
    return None if ranges is None else ranges.containing(start, end)


def _fields(text: str) -> dict[str, str]:
    return dict(_FIELD.findall(text))


def _nvtx_category(text: str, marker: bool) -> str:
    lower = text.lower()
    if "nccl" in lower:
        return "nccl"
    if text.startswith("uniserve.h3.collective"):
        return "collective"
    return "nvtx_marker" if marker else "nvtx"


def _tables(db: sqlite3.Connection) -> dict[str, set[str]]:
    names = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    return {
        name: {row[1] for row in db.execute(f"PRAGMA table_info({_quote(name)})")} for name in names
    }


def _rows(db: sqlite3.Connection, table: str) -> Iterable[sqlite3.Row]:
    return db.execute(f"SELECT * FROM {_quote(table)}")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _string_ids(db: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in _tables(db):
        return {}
    return {int(row[0]): str(row[1]) for row in db.execute("SELECT id, value FROM StringIds")}


def _processes(db: sqlite3.Connection) -> dict[int, tuple[int, str]]:
    if "PROCESSES" not in _tables(db):
        return {}
    return {
        int(row[0]): (int(row[1]), str(row[2]))
        for row in db.execute("SELECT globalPid, pid, name FROM PROCESSES")
    }


def _thread_names(db: sqlite3.Connection, strings: dict[int, str]) -> dict[int, str]:
    if "ThreadNames" not in _tables(db):
        return {}
    return {
        int(row[2]): strings.get(int(row[0]), str(row[0]))
        for row in db.execute("SELECT nameId, priority, globalTid FROM ThreadNames")
    }


def _enum(db: sqlite3.Connection, table: str) -> dict[int, str]:
    if table not in _tables(db):
        return {}
    return {
        int(row[0]): str(row[2] or row[1])
        for row in db.execute(f"SELECT id, name, label FROM {_quote(table)}")
    }


def _process_identity(
    row: sqlite3.Row, processes: dict[int, tuple[int, str]]
) -> tuple[int | None, int | None, str | None]:
    global_pid = _optional_integer(row, "globalPid")
    if global_pid is None:
        global_tid = _optional_integer(row, "globalTid", "threadId")
        if global_tid is not None and global_tid >= 0:
            global_pid = (global_tid >> _PROCESS_SHIFT) << _PROCESS_SHIFT
    process = processes.get(global_pid) if global_pid is not None else None
    return (
        global_pid,
        process[0] if process is not None else None,
        process[1] if process is not None else None,
    )


def _interval(row: sqlite3.Row) -> tuple[int, int]:
    start = _integer(row, "start", "startedAt", "timestamp")
    end = _integer(row, "end", "endedAt", default=start)
    return start, max(start, end)


def _text(row: sqlite3.Row, strings: dict[int, str]) -> str:
    value = _value(row, "text")
    if value is not None:
        return str(value)
    text_id = _optional_integer(row, "textId")
    return strings.get(text_id or -1, "")


def _name(
    row: sqlite3.Row,
    strings: dict[int, str],
    *columns: str,
    fallback: str,
) -> str:
    for column in columns:
        value = _value(row, column)
        if value is None:
            continue
        if isinstance(value, int) and value in strings:
            return strings[value]
        return str(value)
    return fallback


def _integer(row: sqlite3.Row, *names: str, default: int | None = None) -> int:
    value = _value(row, *names)
    if value is None:
        if default is None:
            raise RuntimeError(f"nsys row has none of the required fields {names!r}")
        return int(default)
    return int(value)


def _optional_integer(row: sqlite3.Row, *names: str) -> int | None:
    value = _value(row, *names)
    return None if value is None else int(value)


def _value(row: sqlite3.Row, *names: str) -> Any:
    keys = set(row.keys())
    for name in names:
        if name in keys and row[name] is not None:
            return row[name]
    return None
