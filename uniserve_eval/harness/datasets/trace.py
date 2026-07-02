from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..spec import TaskName

# Required keys on every trace row, mirroring the typed TraceRequest schema in
# crates/support/benchmarks/src/traces.rs (id/task/prompt have no serde default).
_REQUIRED_KEYS = ("id", "task", "prompt")


def trace_items(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in _REQUIRED_KEYS:
                if key not in row:
                    raise ValueError(f"trace row {line_no} is missing required field {key!r}")
            try:
                TaskName(row["task"])
            except ValueError:
                raise ValueError(
                    f"trace row {line_no} has unknown task {row['task']!r}; "
                    f"expected one of {[task.value for task in TaskName]}"
                ) from None
            if ("width" in row) != ("height" in row):
                raise ValueError(f"trace row {line_no} must provide width and height together")
            rows.append(row)
    return rows
