"""Launch adapter for :class:`WorkerProcessArgs`.

The engine writes one JSON launch descriptor per rank and passes its path on
the command line as ``--launch-descriptor``. Every tuning value lives in that
descriptor, so the launching side is the single source of defaults and this
module never restates one. Only the process identity also travels on argv,
which keeps a running worker identifiable from the process table.

``uniserve_worker.main`` calls :func:`parse_worker_args` and hands the result
to ``uniserve_worker.bootstrap.launch.run_worker``.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

from uniserve_worker.config.deployment import WorkerProcessArgs


def create_worker_cli_parser() -> argparse.ArgumentParser:
    """Build the worker launch parser over identity and the descriptor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-descriptor", required=True, type=Path)
    # Identity is repeated on argv so `ps` identifies a worker without reading
    # its descriptor; the descriptor remains the authority.
    parser.add_argument("--worker-id", default="worker")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    return parser


def read_launch_descriptor(path: Path) -> Namespace:
    """Load one launch descriptor into the namespace the config consumes.

    The descriptor's keys become the namespace's attributes unchanged; value
    validation belongs to ``WorkerProcessArgs.from_namespace``, which reads
    every key it needs directly: the engine writes every launch value, so a
    descriptor that omits one is a launch contract violation.

    Raises:
        ValueError: The file cannot be read, is not valid JSON, or is not a
            JSON object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"unreadable launch descriptor {path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid launch descriptor {path}: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("launch descriptor must be a JSON object")

    return Namespace(**payload)


def parse_worker_args(
    arguments: Sequence[str] | None = None,
) -> WorkerProcessArgs:
    """Resolve the validated worker-process configuration for this launch.

    The argv identity flags are parsed but not consulted; the descriptor is
    the authority. A ``ValueError`` from reading or validating the descriptor
    becomes ``parser.error``, which prints usage and exits with status 2;
    other exceptions, such as a ``WorkerError`` from validation, propagate.
    """
    parser = create_worker_cli_parser()
    namespace = parser.parse_args(arguments)
    try:
        return WorkerProcessArgs.from_namespace(
            read_launch_descriptor(namespace.launch_descriptor)
        )
    except ValueError as error:
        parser.error(str(error))
