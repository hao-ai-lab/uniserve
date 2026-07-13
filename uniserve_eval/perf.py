"""Perf points through the measurement harness (workload type ``perf``).

Runs one operating point via ``uniserve_eval.harness`` against the workload's
server and gates on optional ``expect_metrics`` bounds: dotted paths into the
summary's ``metrics`` tree with ``min``/``max`` floors and ceilings. Any failed
request also fails the workload.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

from .backends import build_serve_cmd, resolve_cuda_visible_devices
from .harness.cli import spec_from_harness_command
from .harness.datasets import load_benchmark_inputs
from .harness.provenance import (
    effective_environment,
    execution_provenance,
    input_path_contract,
)
from .harness.report import (
    artifact_bundle_matches,
    attach_execution_contract,
    benchmark_contract,
    canonical_digest,
    write_summary_artifacts,
)
from .profiles import (
    ROOT,
    expand_profile_value,
    load_config,
    merged_env,
    require_resolved_profile_value,
    resolve_server_for_workload,
    spec_env,
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


def resolved_perf_command(
    config: dict[str, Any],
    workload_name: str,
    server_override: str | None = None,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    workload = workload_spec(config, workload_name)
    if workload.get("type") != "perf":
        raise SystemExit(f"workload {workload_name!r} is not type=perf")
    workload = expand_profile_value(workload)
    require_resolved_profile_value(workload, context=f"workload {workload_name}")
    _, server = resolve_server_for_workload(config, workload, server_override)
    out_dir = workload_dir(config, workload_name)
    command = [
        str(ROOT / config.get("python", ".venv/bin/python")),
        "-m",
        "uniserve_eval.harness.cli",
        "--base-url",
        f"http://{server.get('host', '127.0.0.1')}:{server['port']}",
        "--task",
        str(workload["task"]),
        "--model",
        str(server["served_model_name"]),
        "--output-dir",
        str(out_dir),
    ]
    command.extend(str(part) for part in workload.get("args", []))
    return command, workload, server


def perf_profile_contract_fingerprint(
    command: list[str],
    workload: dict[str, Any],
    server: dict[str, Any],
    harness_contract: dict[str, Any],
    *,
    config: dict[str, Any],
) -> str:
    return perf_profile_contract(
        command,
        workload,
        server,
        harness_contract,
        config=config,
    )["fingerprint"]


def perf_profile_contract(
    command: list[str],
    workload: dict[str, Any],
    server: dict[str, Any],
    harness_contract: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    resolved_server = expand_profile_value(server)
    server_command = build_serve_cmd(config, resolved_server, strict_env=True)
    server_overrides = spec_env(resolved_server)
    profile_value = (
        str(resolved_server["cuda_visible_devices"])
        if resolved_server.get("cuda_visible_devices") is not None
        else None
    )
    cuda_visible_devices = resolve_cuda_visible_devices(profile_value)
    if cuda_visible_devices is not None:
        server_overrides["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    model_contract = (
        input_path_contract(str(resolved_server["model"]), cwd=ROOT)
        if resolved_server.get("model") is not None
        else None
    )
    payload = {
        "schema_version": 2,
        "workload_fingerprint": canonical_digest(workload),
        "server_profile_fingerprint": canonical_digest(resolved_server),
        "harness_contract_fingerprint": harness_contract["fingerprint"],
        "model_contract": model_contract,
        "harness_execution": execution_provenance(
            command,
            merged_env(workload),
            cwd=ROOT,
            workspace_root=ROOT,
        ),
        "server_execution": execution_provenance(
            server_command,
            effective_environment(server_overrides),
            cwd=ROOT,
            workspace_root=ROOT,
        ),
    }
    return {**payload, "fingerprint": canonical_digest(payload)}


def perf(args: argparse.Namespace) -> None:
    """Run a harness perf point and gate on metric floors/ceilings."""
    config = load_config(args.config)
    cmd, workload, server = resolved_perf_command(config, args.workload, args.server)
    out_dir = workload_dir(config, args.workload)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = spec_from_harness_command(cmd)
    inputs = load_benchmark_inputs(spec)
    harness_contract = benchmark_contract(spec, inputs.measured)
    profile_contract = perf_profile_contract(
        cmd,
        workload,
        server,
        harness_contract,
        config=config,
    )
    print("perf:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT, env=merged_env(workload))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    if not artifact_bundle_matches(out_dir, summary, harness_contract):
        raise SystemExit("benchmark did not produce the current harness artifact contract")
    if summary.get("failed_count"):
        raise SystemExit(f"benchmark had {summary['failed_count']} failed requests")
    metric_failures: list[str] = []
    for dotted, bounds in dict(workload.get("expect_metrics") or {}).items():
        value = float(_metric_at(summary, dotted))
        low = bounds.get("min")
        high = bounds.get("max")
        print(f"metric {dotted} = {value:.2f} (min={low}, max={high})")
        if low is not None and value < float(low):
            metric_failures.append(f"metric {dotted} = {value:.2f} below floor {low}")
        if high is not None and value > float(high):
            metric_failures.append(f"metric {dotted} = {value:.2f} above ceiling {high}")
        summary["artifact"]["checks"][f"profile_metric:{dotted}"] = not any(
            failure.startswith(f"metric {dotted} ") for failure in metric_failures
        )
    attach_execution_contract(summary, "profile_contract", profile_contract)
    write_summary_artifacts(out_dir, summary)
    if metric_failures:
        raise SystemExit("; ".join(metric_failures))
