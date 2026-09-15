"""Loads and validates evaluator servers, benchmark points, and suites."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .datasets import get_dataset
from .tasks import get_task
from .tasks.base import BenchmarkTask
from .types import (
    BenchmarkPoint,
    ImageConfig,
    LoadConfig,
    MetricDefinition,
    MetricDirection,
    SamplingConfig,
    TaskName,
    VideoConfig,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "profiles.toml"
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_DEFAULT_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-(.*?)\}")

_ROOT_FIELDS = {
    "server",
    "task",
    "model",
    "dataset",
    "dataset_revision",
    "dataset_path",
    "tokenizer",
    "endpoint",
    "question",
    "load",
    "sampling",
    "image",
    "video",
    "metrics",
}
_LOAD_FIELDS = set(LoadConfig.__dataclass_fields__)
_SAMPLING_FIELDS = set(SamplingConfig.__dataclass_fields__)
_IMAGE_FIELDS = set(ImageConfig.__dataclass_fields__)
_VIDEO_FIELDS = set(VideoConfig.__dataclass_fields__)


@dataclass(frozen=True)
class ServerProfile:
    """Describes one benchmark server and its launch environment."""

    name: str
    command: tuple[str, ...]
    host: str
    port: int
    environment: dict[str, str]

    @property
    def base_url(self) -> str:
        """Return the server's HTTP origin."""
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class ServerLaunch:
    """Contains a resolved server command and its process context."""

    command: tuple[str, ...]
    working_directory: Path
    environment: dict[str, str]


@dataclass(frozen=True)
class SuiteProfile:
    """Names an ordered collection of benchmark points."""

    name: str
    points: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    """Contains the complete validated evaluator configuration."""

    artifact_root: Path
    servers: dict[str, ServerProfile]
    benchmarks: dict[str, BenchmarkPoint]
    suites: dict[str, SuiteProfile]

    def selected_points(self, selection: str) -> tuple[BenchmarkPoint, ...]:
        """Resolve a benchmark or suite name to its ordered points."""
        if selection in self.benchmarks:
            return (self.benchmarks[selection],)
        suite = self.suites.get(selection)
        if suite is None:
            known = ", ".join(sorted({*self.benchmarks, *self.suites}))
            raise ValueError(
                f"unknown benchmark or suite {selection!r}; known: {known}"
            )
        return tuple(self.benchmarks[name] for name in suite.points)


def load_config(path: Path = DEFAULT_CONFIG) -> EvaluationConfig:
    """Load and validate an evaluator TOML file."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)

    servers = {
        name: _server_profile(name, value)
        for name, value in _table(raw, "servers").items()
    }
    benchmarks = {
        name: _benchmark_point(name, value, servers)
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
    """Expand declared environment references recursively when values exist."""
    if isinstance(value, str):
        expanded = _ENV_DEFAULT_REF.sub(
            lambda match: os.environ.get(match.group(1), match.group(2)), value
        )
        return _ENV_REF.sub(
            lambda match: os.environ.get(match.group(1), match.group(0)),
            expanded,
        )
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): expand_environment(item) for key, item in value.items()
        }
    return value


def unresolved_environment(value: Any) -> tuple[str, ...]:
    """Return sorted environment names that remain referenced in a value."""
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
    """Reject a value that still contains environment references."""
    names = unresolved_environment(value)
    if names:
        raise ValueError(
            f"{context} has unresolved environment variables: "
            f"{', '.join(names)}"
        )


def server_launch(
    server: ServerProfile, executable: Path | None = None
) -> ServerLaunch:
    """Resolve a server profile into an executable launch description."""
    command = list(server.command)
    selected_executable = (
        executable if executable is not None else Path(command[0])
    )
    if not selected_executable.is_absolute():
        selected_executable = ROOT / selected_executable
    selected_executable = selected_executable.absolute()
    command[0] = str(selected_executable)

    working_directory = next(
        (
            parent
            for parent in selected_executable.parents
            if (parent / "uniserve_worker").is_dir()
        ),
        ROOT,
    )

    _resolve_command_path(command, "--worker-python", ROOT)
    return ServerLaunch(
        tuple(command), working_directory, dict(server.environment)
    )


def _resolve_command_path(command: list[str], option: str, base: Path) -> None:
    """Make the path argument for a command option absolute in place."""
    try:
        value_index = command.index(option) + 1
    except ValueError:
        return
    if value_index >= len(command):
        raise ValueError(f"{option} requires a path")
    path = Path(command[value_index])
    if not path.is_absolute():
        command[value_index] = str((base / path).absolute())


def _server_profile(name: str, raw: Any) -> ServerProfile:
    """Validate and construct one server profile table."""
    value = expand_environment(_mapping(raw, f"servers.{name}"))
    _reject_unknown(
        value, {"command", "host", "port", "environment"}, f"servers.{name}"
    )
    command = value.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise ValueError(
            f"servers.{name}.command must be a non-empty string array"
        )
    host = value.get("host", "127.0.0.1")
    port = value.get("port")
    environment = value.get("environment", {})
    if not isinstance(host, str) or not host:
        raise ValueError(f"servers.{name}.host must be a non-empty string")
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 0 < port < 65536
    ):
        raise ValueError(f"servers.{name}.port must be a valid TCP port")
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in environment.items()
    ):
        raise ValueError(
            f"servers.{name}.environment must contain string values"
        )
    return ServerProfile(name, tuple(command), host, port, dict(environment))


def _benchmark_point(
    name: str,
    raw: Any,
    servers: dict[str, ServerProfile],
) -> BenchmarkPoint:
    """Validate and construct one task-aware benchmark point."""
    # Resolve the benchmark's named server, task, dataset adapter, and model
    # before interpreting task-specific configuration tables.
    value = expand_environment(_mapping(raw, f"benchmarks.{name}"))
    _reject_unknown(value, _ROOT_FIELDS, f"benchmarks.{name}")
    server = value.get("server")
    if not isinstance(server, str) or server not in servers:
        raise ValueError(
            f"benchmarks.{name}.server must name a declared server"
        )
    task_name = value.get("task")
    if not isinstance(task_name, str):
        raise ValueError(f"benchmarks.{name}.task must be a string")
    task = get_task(task_name)
    dataset_name = value.get("dataset")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(
            f"benchmarks.{name}.dataset must be a non-empty string"
        )
    dataset_cls = get_dataset(dataset_name)
    model = value.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"benchmarks.{name}.model must be a non-empty string")

    # Each nested parser owns its schema, while the task and dataset adapters
    # enforce modality and source requirements that cross table boundaries.
    metrics = _metrics(value.get("metrics"), f"benchmarks.{name}")
    load = _load_config(value.get("load"), f"benchmarks.{name}.load")
    sampling = _sampling_config(
        value.get("sampling"), task, f"benchmarks.{name}.sampling"
    )
    image = _image_config(value.get("image"), task, f"benchmarks.{name}.image")
    video = _video_config(value.get("video"), f"benchmarks.{name}.video")
    endpoint = task.check_endpoint(value.get("endpoint"), f"benchmarks.{name}")
    question = task.check_question(value.get("question"), f"benchmarks.{name}")
    tokenizer = value.get("tokenizer")
    dataset_path = value.get("dataset_path")
    dataset_cls.check_point(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        context=f"benchmarks.{name}",
    )
    revision = value.get("dataset_revision")
    if revision is not None and not isinstance(revision, str):
        raise ValueError(f"benchmarks.{name}.dataset_revision must be a string")

    # Construct only after every referenced component has accepted the point.
    task.check_image(image, f"benchmarks.{name}")
    return BenchmarkPoint(
        name=name,
        server=server,
        task=TaskName(task_name),
        model=model,
        dataset=dataset_name,
        metrics=metrics,
        load=load,
        sampling=sampling,
        image=image,
        video=video,
        dataset_revision=revision,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        endpoint=endpoint,
        question=question,
    )


def _metrics(raw: Any, context: str) -> tuple[MetricDefinition, ...]:
    """Parse protected metric paths and optimization directions."""
    table = _mapping(raw, f"{context}.metrics")
    metrics: list[MetricDefinition] = []
    for path, direction in table.items():
        if not isinstance(direction, str) or direction not in {
            "higher",
            "lower",
        }:
            raise ValueError(
                f"{context}.metrics.{path} must be 'higher' or 'lower'"
            )
        parts = tuple(str(path).split("."))
        if not parts or any(not part for part in parts):
            raise ValueError(f"{context} has an invalid metric path {path!r}")
        metrics.append(
            MetricDefinition(parts, cast(MetricDirection, direction))
        )
    return tuple(metrics)


def _load_config(raw: Any, context: str) -> LoadConfig:
    """Parse request arrival and concurrency settings."""
    if raw is None:
        return LoadConfig()
    value = _mapping(raw, context)
    _reject_unknown(value, _LOAD_FIELDS, context)
    if "request_rate" in value and value["request_rate"] == "inf":
        value["request_rate"] = float("inf")
    return LoadConfig(**value)


def _sampling_config(
    raw: Any, task: type[BenchmarkTask], context: str
) -> SamplingConfig:
    """Parse sampling settings with task-specific streaming defaults."""
    value = dict(_mapping(raw, context)) if raw is not None else {}
    _reject_unknown(value, _SAMPLING_FIELDS, context)
    if "stream" not in value:
        value["stream"] = task.default_stream
    extra = value.get("extra_body", {})
    if extra and not isinstance(extra, dict):
        raise ValueError(f"{context}.extra_body must be a table")
    return SamplingConfig(**value)


def _image_config(
    raw: Any, task: type[BenchmarkTask], context: str
) -> ImageConfig:
    """Parse image-generation settings accepted by the selected task."""
    if raw is None:
        return ImageConfig()
    if not task.accepts_image:
        raise ValueError(f"{context} is not valid for this task")
    value = _mapping(raw, context)
    _reject_unknown(value, _IMAGE_FIELDS, context)
    if "cfg_interval" in value:
        interval = value["cfg_interval"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"{context}.cfg_interval must have two values")
        value["cfg_interval"] = (float(interval[0]), float(interval[1]))
    return ImageConfig(**value)


def _video_config(raw: Any, context: str) -> VideoConfig:
    """Parse synchronous video-generation settings."""
    if raw is None:
        return VideoConfig()
    value = _mapping(raw, context)
    _reject_unknown(value, _VIDEO_FIELDS, context)
    return VideoConfig(**value)


def _suite_profile(
    name: str,
    raw: Any,
    benchmarks: dict[str, BenchmarkPoint],
) -> SuiteProfile:
    """Validate an ordered suite against the declared benchmark points."""
    value = _mapping(raw, f"suites.{name}")
    _reject_unknown(value, {"points"}, f"suites.{name}")
    points = value.get("points")
    if (
        not isinstance(points, list)
        or not points
        or not all(isinstance(item, str) for item in points)
    ):
        raise ValueError(
            f"suites.{name}.points must be a non-empty string array"
        )
    if len(set(points)) != len(points):
        raise ValueError(f"suites.{name}.points must be unique")
    unknown = [point for point in points if point not in benchmarks]
    if unknown:
        raise ValueError(
            f"suites.{name} names unknown points: {', '.join(unknown)}"
        )
    return SuiteProfile(name, tuple(points))


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a named root table or an empty table when absent."""
    return _mapping(raw.get(key, {}), key)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    """Return a defensive copy of a required TOML mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a TOML table")
    return dict(value)


def _reject_unknown(
    value: dict[str, Any], allowed: set[str], context: str
) -> None:
    """Reject keys outside a table's public configuration schema."""
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")
