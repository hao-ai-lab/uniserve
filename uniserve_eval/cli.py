"""Defines commands for planning and running HTTP serving benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config, server_launch
from .pipeline.run import run_point
from .pipeline.setup import (
    applied_environment,
    describe_launch,
    host_lock,
    prepare_launch,
)
from .server import ManagedServer


def list_items(args: argparse.Namespace) -> None:
    """Print configured servers, benchmark points, and suites."""
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
    """Print the resolved execution plan for a benchmark selection.

    Unlike `run`, planning neither takes the host lock nor rejects
    unresolved environment references, so an unset `${NAME}` appears
    verbatim in the printed command.
    """
    config = load_config(args.config)
    points = config.selected_points(args.selection)
    rendered = []
    for point in points:
        server = config.servers[point.server]
        launch = server_launch(server, args.executable, config.root)
        rendered.append(
            {
                "benchmark": point.name,
                "task": point.task.value,
                "server_command": list(launch.command),
                "server_working_directory": str(launch.working_directory),
                "base_url": server.base_url,
                "dataset": point.dataset,
                "num_prompts": point.load.num_prompts,
                # `json.dumps` would render infinity as the non-standard
                # `Infinity` token.
                "request_rate": (
                    "inf"
                    if point.load.request_rate == float("inf")
                    else point.load.request_rate
                ),
                "max_concurrency": point.load.max_concurrency,
                "workload": point.workload_dict(),
                "metrics": [metric.as_dict() for metric in point.metrics],
            }
        )
    print(json.dumps(rendered, indent=2))


def run(args: argparse.Namespace) -> None:
    """Run selected points serially and stop after the first invalid result.

    Each point launches its own server, measures against it, and stops it
    before the next point starts, even when consecutive points name the same
    server. Results go to `<output root>/<point name>`, which `run_point`
    requires to be absent or empty, and server output goes to
    `<output root>/server-logs/<point name>.log`.

    Exits with status 2 when a point completes with a failed validation. An
    exception while preparing, launching, or measuring a point propagates and
    ends the run; `ManagedServer` stops a server that was already started.
    """
    config = load_config(args.config)
    output_root = args.output_root or config.artifact_root
    points = config.selected_points(args.selection)
    failures = 0

    # The lock is held for the whole selection, so another uniserve-eval
    # process fails to acquire it instead of starting a server between points.
    with host_lock():
        for point in points:
            server = config.servers[point.server]
            launch = prepare_launch(config, point, args.executable)
            point_dir = output_root / point.name
            log_path = output_root / "server-logs" / f"{point.name}.log"
            launch_record = describe_launch(launch)

            # The launch environment also applies to this process for the
            # duration of the point and is restored afterwards; the server
            # receives it through `ManagedServer`.
            with applied_environment(launch.environment):
                with ManagedServer(
                    server,
                    launch,
                    log_path,
                    timeout_s=args.launch_timeout_s,
                ):
                    result = asyncio.run(
                        run_point(
                            server.base_url,
                            point,
                            point_dir,
                            launch=launch_record,
                            timeout_s=args.request_timeout_s,
                        )
                    )

            status = "pass" if result.summary["validation"]["valid"] else "fail"
            print(
                f"{point.name}: {status} ({result.summary['ok_count']}"
                f"/{result.summary['request_count']} requests)"
            )
            if result.summary["validation"]["valid"] is not True:
                failures += 1
                break
    if failures:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluator command-line parser."""
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

    # For `plan` and `run`, `selection` is a benchmark or suite name and
    # `--executable` replaces the server binary, the first element of the
    # profile's command. A relative `--executable` resolves against the
    # profile's `root`, whereas a relative `--output-root` is taken from the
    # current directory.
    command = subparsers.add_parser("plan")
    command.add_argument("selection")
    command.add_argument("--executable", type=Path)
    command.set_defaults(function=plan)

    command = subparsers.add_parser("run")
    command.add_argument("selection")
    command.add_argument("--executable", type=Path)
    command.add_argument("--output-root", type=Path)
    command.add_argument("--launch-timeout-s", type=float, default=1800)
    command.add_argument("--request-timeout-s", type=float)
    command.set_defaults(function=run)
    return parser


def main() -> None:
    """Parse command-line arguments and dispatch the selected command."""
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
