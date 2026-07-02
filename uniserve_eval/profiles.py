"""Profile loading and resolution for the serving evaluation driver.

A profiles file (default ``uniserve_eval/profiles.json``) declares three
sections:

  servers   — how to launch one serving backend (UniServe ``serve_args`` or an
              explicit ``command``); specs support ``extends`` with deep-merge
              so variants stay single-source.
  workloads — what to run against a compatible server (``verify`` correctness
              gates, ``perf`` harness points, or generic ``script`` runs).
  suites    — ordered workload names for a broader pass; either a plain list or
              ``{"workloads": [...], "compare": [[baseline, candidate, ...]]}``.

Everything path-like resolves against the repository root; artifacts land under
the config's ``artifact_root`` (``servers/<name>`` and ``workloads/<name>``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "profiles.json"


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def server_spec(config: dict[str, Any], name: str, seen: set[str] | None = None) -> dict[str, Any]:
    servers = config.get("servers", {})
    if name not in servers:
        known = ", ".join(sorted(servers))
        raise SystemExit(f"unknown server {name!r}; known servers: {known}")
    raw = dict(servers[name])
    parent = raw.get("extends")
    if not parent:
        return raw
    seen = seen or set()
    if name in seen:
        raise SystemExit(f"server inheritance cycle at {name!r}")
    base = server_spec(config, str(parent), seen | {name})
    return _deep_merge(base, raw)


def workload_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    workloads = config.get("workloads", {})
    if name not in workloads:
        known = ", ".join(sorted(workloads))
        raise SystemExit(f"unknown workload {name!r}; known workloads: {known}")
    return dict(workloads[name])


def suite_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Resolve a suite to its dict form: ``{"workloads": [...], "compare": [...]}``.

    Plain-list suites remain valid and resolve to ``{"workloads": [...],
    "compare": []}``.
    """
    suites = config.get("suites", {})
    if name not in suites:
        known = ", ".join(sorted(suites))
        raise SystemExit(f"unknown suite {name!r}; known suites: {known}")
    raw = suites[name]
    if isinstance(raw, list):
        return {"workloads": list(raw), "compare": []}
    workloads = list(raw.get("workloads", []))
    compare_groups = [list(group) for group in raw.get("compare", [])]
    return {"workloads": workloads, "compare": compare_groups}


def spec_env(spec: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in dict(spec.get("env") or {}).items()}


def artifact_root(config: dict[str, Any]) -> Path:
    return ROOT / config.get("artifact_root", "e2e-artifacts/current-verify")


def server_dir(config: dict[str, Any], name: str) -> Path:
    return artifact_root(config) / "servers" / name


def workload_dir(config: dict[str, Any], name: str) -> Path:
    return artifact_root(config) / "workloads" / name


def server_pid_path(config: dict[str, Any], name: str) -> Path:
    return server_dir(config, name) / "server.pid"


def server_log_path(config: dict[str, Any], name: str) -> Path:
    return server_dir(config, name) / "server.log"


def resolve_server_for_workload(
    config: dict[str, Any], workload: dict[str, Any], override: str | None
) -> tuple[str, dict[str, Any]]:
    server_name = override or workload.get("server")
    if not server_name:
        raise SystemExit("workload has no server; pass --server")
    return str(server_name), server_spec(config, str(server_name))


def workload_env(workload: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in dict(workload.get("env") or {}).items()}


def merged_env(workload: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(workload_env(workload))
    return env
