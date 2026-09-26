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
    describe_deployment,
    host_lock,
    prepare_deployment,
)
from .server import ManagedDeployment
from .types import BenchmarkPoint


def list_items(args: argparse.Namespace) -> None:
    """Print configured servers, benchmark points, and suites."""
    config = load_config(args.config)
    if args.section in {"all", "servers"}:
        print("[servers]")
        for name, server in config.servers.items():
            print(f"{name}\t{','.join(server.base_urls)}")
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
    verbatim in the printed command. A replicated deployment lists one
    command and origin per replica.
    """
    config = load_config(args.config)
    points = config.selected_points(args.selection)
    rendered = []
    for point in points:
        server = config.servers[point.server]
        launches = [
            server_launch(instance, args.executable, config.root)
            for instance in server.instances
        ]
        rendered.append(
            {
                "benchmark": point.name,
                "task": point.task.value,
                "server": server.name,
                "server_commands": [list(item.command) for item in launches],
                "server_working_directory": str(launches[0].working_directory),
                "base_urls": list(server.base_urls),
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

    By default each point launches its own deployment, measures against it,
    and stops it before the next point starts. With `--reuse-deployment`,
    consecutive points that name the same server share one deployment, which
    stays up until the next point names a different server; each point
    still performs its own declared warmup and priming. Results go to
    `<output root>/<point name>`, which `run_point` requires to be absent or
    empty, and server output goes to `<output root>/server-logs/<first point
    of the deployment>.log`, with a `.replica-<i>` infix per replica.

    Exits with status 2 when a point completes with a failed validation,
    immediately unless `--keep-going` asks to measure the remaining points
    first. An exception while preparing, launching, or measuring a point
    propagates and ends the run; `ManagedDeployment` stops processes already
    started.
    """
    config = load_config(args.config)
    output_root = args.output_root or config.artifact_root
    points = config.selected_points(args.selection)
    invalid = 0

    # Consecutive points form one deployment group only when reuse is
    # requested and they name the same server.
    groups: list[list[BenchmarkPoint]] = []
    for point in points:
        if (
            args.reuse_deployment
            and groups
            and groups[-1][0].server == point.server
        ):
            groups[-1].append(point)
        else:
            groups.append([point])

    # The lock is held for the whole selection, so another uniserve-eval
    # process fails to acquire it instead of starting a server between points.
    with host_lock():
        for group in groups:
            first = group[0]
            server = config.servers[first.server]
            launches = prepare_deployment(config, first, args.executable)
            for point in group[1:]:
                prepare_deployment(config, point, args.executable)
            replicated = len(launches) > 1
            processes = [
                (
                    instance,
                    launch,
                    _log_path(output_root, first, index, replicated),
                )
                for index, (instance, launch) in enumerate(launches)
            ]
            launch_record = describe_deployment(
                [launch for _, launch in launches]
            )
            launch_record["deployment"] = {
                "started_for": first.name,
                "logs": [str(path) for _, _, path in processes],
            }

            # The shared deployment environment also applies to this process
            # while the group runs and is restored afterwards; each server
            # process receives its own launch environment.
            with applied_environment(dict(server.environment)):
                with ManagedDeployment(
                    processes, timeout_s=args.launch_timeout_s
                ) as deployment:
                    for point in group:
                        exited = deployment.exited()
                        if exited:
                            raise RuntimeError(
                                "a deployment process exited before "
                                f"{point.name}; see {exited[0]}"
                            )
                        result = asyncio.run(
                            run_point(
                                server.base_urls,
                                point,
                                output_root / point.name,
                                launch=launch_record,
                                timeout_s=args.request_timeout_s,
                            )
                        )
                        valid = result.summary["validation"]["valid"]
                        print(
                            f"{point.name}: {'pass' if valid else 'fail'} "
                            f"({result.summary['ok_count']}"
                            f"/{result.summary['request_count']} requests)"
                        )
                        if valid is not True:
                            invalid += 1
                            if not args.keep_going:
                                raise SystemExit(2)
    if invalid:
        raise SystemExit(2)


def _log_path(
    output_root: Path, point: BenchmarkPoint, index: int, replicated: bool
) -> Path:
    """Return a process log path named after the deployment's first point."""
    suffix = f".replica-{index}" if replicated else ""
    return output_root / "server-logs" / f"{point.name}{suffix}.log"


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
    command.add_argument("--reuse-deployment", action="store_true")
    command.add_argument("--keep-going", action="store_true")
    command.set_defaults(function=run)
    return parser


def main() -> None:
    """Parse command-line arguments and dispatch the selected command."""
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
