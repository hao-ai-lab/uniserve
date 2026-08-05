"""TOML configuration for explicit evaluator points."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .harness.spec import BenchmarkSpec, MetricDefinition, MetricDirection

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "profiles.toml"
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class ServerProfile:
    name: str
    command: tuple[str, ...]
    host: str
    port: int
    environment: dict[str, str]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class SuiteProfile:
    name: str
    points: tuple[str, ...]
    max_regression: float | None


@dataclass(frozen=True)
class EvaluationConfig:
    artifact_root: Path
    servers: dict[str, ServerProfile]
    benchmarks: dict[str, BenchmarkSpec]
    suites: dict[str, SuiteProfile]

    def selected_points(self, selection: str) -> tuple[BenchmarkSpec, ...]:
        if selection in self.benchmarks:
            return (self.benchmarks[selection],)
        suite = self.suites.get(selection)
        if suite is None:
            known = ", ".join(sorted({*self.benchmarks, *self.suites}))
            raise ValueError(f"unknown benchmark or suite {selection!r}; known: {known}")
        return tuple(self.benchmarks[name] for name in suite.points)


def load_config(path: Path = DEFAULT_CONFIG) -> EvaluationConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)

    servers = {
        name: _server_profile(name, value)
        for name, value in _table(raw, "servers").items()
    }
    benchmarks = {
        name: _benchmark_spec(name, value, servers)
        for name, value in _table(raw, "benchmarks").items()
    }
    suites = {
        name: _suite_profile(name, value, benchmarks)
        for name, value in _table(raw, "suites").items()
    }
    artifact_root = Path(str(raw.get("artifact_root", "artifacts/benchmark")))
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    return EvaluationConfig(
        artifact_root=artifact_root,
        servers=servers,
        benchmarks=benchmarks,
        suites=suites,
    )


def expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_REF.sub(lambda match: os.environ.get(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {str(key): expand_environment(item) for key, item in value.items()}
    return value


def unresolved_environment(value: Any) -> tuple[str, ...]:
    names: set[str] = set()
    if isinstance(value, str):
        names.update(_ENV_REF.findall(value))
    elif isinstance(value, (list, tuple)):
        for item in value:
            names.update(unresolved_environment(item))
    elif isinstance(value, dict):
        for item in value.values():
            names.update(unresolved_environment(item))
    return tuple(sorted(names))


def require_resolved(value: Any, *, context: str) -> None:
    names = unresolved_environment(value)
    if names:
        raise ValueError(f"{context} has unresolved environment variables: {', '.join(names)}")


def server_command(server: ServerProfile, executable: Path | None = None) -> tuple[str, ...]:
    command = list(server.command)
    if executable is not None:
        command[0] = str(executable)
    return tuple(command)


def _server_profile(name: str, raw: Any) -> ServerProfile:
    value = expand_environment(_mapping(raw, f"servers.{name}"))
    allowed = {"command", "host", "port", "environment"}
    _reject_unknown(value, allowed, f"servers.{name}")
    command = value.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise ValueError(f"servers.{name}.command must be a non-empty string array")
    host = value.get("host", "127.0.0.1")
    port = value.get("port")
    environment = value.get("environment", {})
    if not isinstance(host, str) or not host:
        raise ValueError(f"servers.{name}.host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise ValueError(f"servers.{name}.port must be a valid TCP port")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in environment.items()
    ):
        raise ValueError(f"servers.{name}.environment must contain string values")
    return ServerProfile(name, tuple(command), host, port, dict(environment))


def _benchmark_spec(
    name: str,
    raw: Any,
    servers: dict[str, ServerProfile],
) -> BenchmarkSpec:
    value = expand_environment(_mapping(raw, f"benchmarks.{name}"))
    server = value.pop("server", None)
    raw_metrics = value.pop("metrics", None)
    if not isinstance(server, str) or server not in servers:
        raise ValueError(f"benchmarks.{name}.server must name a declared server")
    metrics_table = _mapping(raw_metrics, f"benchmarks.{name}.metrics")
    metrics: list[MetricDefinition] = []
    for path, direction in metrics_table.items():
        if not isinstance(direction, str) or direction not in {"higher", "lower"}:
            raise ValueError(f"benchmarks.{name}.metrics.{path} must be 'higher' or 'lower'")
        parts = tuple(str(path).split("."))
        if not parts or any(not part for part in parts):
            raise ValueError(f"benchmarks.{name} has an invalid metric path {path!r}")
        metrics.append(MetricDefinition(parts, cast(MetricDirection, direction)))
    if "cfg_interval" in value:
        interval = value["cfg_interval"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"benchmarks.{name}.cfg_interval must have two values")
        value["cfg_interval"] = (float(interval[0]), float(interval[1]))
    fields = set(BenchmarkSpec.__dataclass_fields__)
    _reject_unknown(value, fields - {"name", "metrics", "server"}, f"benchmarks.{name}")
    return BenchmarkSpec(name=name, metrics=tuple(metrics), server=server, **value)


def _suite_profile(
    name: str,
    raw: Any,
    benchmarks: dict[str, BenchmarkSpec],
) -> SuiteProfile:
    value = _mapping(raw, f"suites.{name}")
    _reject_unknown(value, {"points", "max_regression"}, f"suites.{name}")
    points = value.get("points")
    if not isinstance(points, list) or not points or not all(isinstance(item, str) for item in points):
        raise ValueError(f"suites.{name}.points must be a non-empty string array")
    if len(set(points)) != len(points):
        raise ValueError(f"suites.{name}.points must be unique")
    unknown = [point for point in points if point not in benchmarks]
    if unknown:
        raise ValueError(f"suites.{name} names unknown points: {', '.join(unknown)}")
    max_regression = value.get("max_regression")
    if max_regression is not None:
        max_regression = float(max_regression)
        if not 0 <= max_regression < 1:
            raise ValueError(f"suites.{name}.max_regression must be in [0, 1)")
    return SuiteProfile(name, tuple(points), max_regression)


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    return _mapping(value, key)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a TOML table")
    return dict(value)


def _reject_unknown(value: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")
