"""Run and compare explicit public-protocol serving benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .backends import ManagedServer
from .harness.comparison import (
    compare_suite,
)
from .harness.comparison import (
    render_markdown as render_comparison,
)
from .harness.provenance import collect_provenance
from .harness.runner import BenchmarkRunner
from .profiles import (
    DEFAULT_CONFIG,
    load_config,
    require_resolved,
    server_launch,
)


def list_items(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.section in {"all", "servers"}:
        print("[servers]")
        for name, server in config.servers.items():
            print(f"{name}\t{server.base_url}")
    if args.section in {"all", "benchmarks"}:
        print("[benchmarks]")
        for name, point in config.benchmarks.items():
            print(f"{name}\t{point.task.value}\tserver={point.server}")
    if args.section in {"all", "suites"}:
        print("[suites]")
        for name, suite in config.suites.items():
            print(f"{name}\t{','.join(suite.points)}")


def plan(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    points = config.selected_points(args.selection)
    rendered = []
    for point in points:
        server = config.servers[point.server]
        launch = server_launch(server, args.executable)
        rendered.append(
            {
                "benchmark": point.name,
                "task": point.task.value,
                "server_command": list(launch.command),
                "server_working_directory": str(launch.working_directory),
                "base_url": server.base_url,
                "dataset": point.dataset,
                "num_prompts": point.num_prompts,
                "request_rate": "inf" if point.request_rate == float("inf") else point.request_rate,
                "max_concurrency": point.max_concurrency,
                "metrics": [metric.as_dict() for metric in point.metrics],
            }
        )
    print(json.dumps(rendered, indent=2))


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output_root = args.output_root or config.artifact_root
    points = config.selected_points(args.selection)
    failures = 0
    with _host_lock():
        for point in points:
            server = config.servers[point.server]
            launch = server_launch(server, args.executable)
            require_resolved(launch.command, context=f"server {server.name}")
            require_resolved(point.workload_dict(), context=f"benchmark {point.name}")
            point_dir = output_root / point.name
            log_path = output_root / "server-logs" / f"{point.name}.log"
            provenance = collect_provenance(
                launch.command,
                launch.working_directory,
                launch.environment,
            )
            with _environment(launch.environment):
                with ManagedServer(server, launch, log_path, timeout_s=args.launch_timeout_s):
                    result = asyncio.run(
                        BenchmarkRunner(
                            server.base_url,
                            point,
                            point_dir,
                            provenance=provenance,
                            timeout_s=args.request_timeout_s,
                        ).run()
                    )
            print(
                f"{point.name}: {'pass' if result.summary['validation']['valid'] else 'fail'} "
                f"({result.summary['ok_count']}/{result.summary['request_count']} requests)"
            )
            if result.summary["validation"]["valid"] is not True:
                failures += 1
                break
    if failures:
        raise SystemExit(2)


def compare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    points = config.selected_points(args.selection)
    suite = config.suites.get(args.selection)
    max_regression = args.max_regression
    if max_regression is None:
        max_regression = suite.max_regression if suite is not None else None
    report = compare_suite(
        args.reference_root,
        args.candidate_root,
        [point.name for point in points],
        max_regression=max_regression,
    )
    markdown = render_comparison(report)
    print(markdown, end="")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "comparison.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    if report["valid"] is not True or (max_regression is not None and report["passed"] is not True):
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uniserve-eval", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("list")
    command.add_argument(
        "section",
        nargs="?",
        choices=("all", "servers", "benchmarks", "suites"),
        default="all",
    )
    command.set_defaults(function=list_items)

    command = subparsers.add_parser("plan")
    command.add_argument("selection")
    command.add_argument("--executable", type=Path)
    command.set_defaults(function=plan)

    command = subparsers.add_parser("run")
    command.add_argument("selection")
    command.add_argument("--executable", type=Path)
    command.add_argument("--output-root", type=Path)
    command.add_argument("--launch-timeout-s", type=float, default=1800)
    command.add_argument("--request-timeout-s", type=float, default=6 * 60 * 60)
    command.set_defaults(function=run)

    command = subparsers.add_parser("compare")
    command.add_argument("selection")
    command.add_argument("--reference-root", type=Path, required=True)
    command.add_argument("--candidate-root", type=Path, required=True)
    command.add_argument("--max-regression", type=float)
    command.add_argument("--output-dir", type=Path)
    command.set_defaults(function=compare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


@contextmanager
def _host_lock() -> Iterator[None]:
    path = Path("/tmp/uniserve-eval.lock")
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another uniserve-eval process holds the host lock") from error
        yield


@contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
    previous: dict[str, str | None] = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
