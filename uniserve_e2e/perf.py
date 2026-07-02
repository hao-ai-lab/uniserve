"""Perf points through the measurement harness (workload type ``perf``).

Runs one operating point via ``uniserve_e2e.harness`` against the workload's
server and gates on optional ``expect_metrics`` bounds: dotted paths into the
summary's ``metrics`` tree with ``min``/``max`` floors and ceilings. Any failed
request also fails the workload.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

from .profiles import (
    ROOT,
    load_config,
    merged_env,
    resolve_server_for_workload,
    workload_dir,
    workload_spec,
)


def _metric_at(summary: dict[str, Any], dotted: str) -> Any:
    node: Any = summary.get("metrics", {})
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(f"benchmark summary has no metric {dotted!r}")
        node = node[part]
    return node


def perf(args: argparse.Namespace) -> None:
    """Run a harness perf point and gate on metric floors/ceilings."""
    config = load_config(args.config)
    workload = workload_spec(config, args.workload)
    if workload.get("type") != "perf":
        raise SystemExit(f"workload {args.workload!r} is not type=perf")
    _, server = resolve_server_for_workload(config, workload, args.server)
    out_dir = workload_dir(config, args.workload)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / config.get("python", ".venv/bin/python")),
        "-m",
        "uniserve_e2e.harness.cli",
        "--base-url",
        f"http://{server.get('host', '127.0.0.1')}:{server['port']}",
        "--task",
        str(workload["task"]),
        "--model",
        str(server["served_model_name"]),
        "--output-dir",
        str(out_dir),
    ]
    cmd.extend(str(part) for part in workload.get("args", []))
    print("perf:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT, env=merged_env(workload))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("failed_count"):
        raise SystemExit(f"benchmark had {summary['failed_count']} failed requests")
    for dotted, bounds in dict(workload.get("expect_metrics") or {}).items():
        value = float(_metric_at(summary, dotted))
        low = bounds.get("min")
        high = bounds.get("max")
        print(f"metric {dotted} = {value:.2f} (min={low}, max={high})")
        if low is not None and value < float(low):
            raise SystemExit(f"metric {dotted} = {value:.2f} below floor {low}")
        if high is not None and value > float(high):
            raise SystemExit(f"metric {dotted} = {value:.2f} above ceiling {high}")
