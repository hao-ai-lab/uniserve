"""Profile-driven serving evaluation: launch servers and run correctness gates.

Concepts:
  server   = how to launch one serving backend topology
  workload = what to run against a compatible launched server
             (verify = correctness gates over public chat completions,
              script = a generic repo-script escape hatch)
  suite    = ordered workload names for a broader pass

Examples:
  uniserve-eval list
  uniserve-eval launch gate/server/sensenova
  uniserve-eval verify gate/sensenova/default-travel
  uniserve-eval run gate/all --manage-servers

Benchmark measurement points are a separate path: see
``scripts/run_benchmarks.py`` and ``docs/benchmark-protocol.md``.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from . import backends, verify
from .profiles import (
    DEFAULT_CONFIG,
    ROOT,
    load_config,
    merged_env,
    resolve_server_for_workload,
    server_spec,
    suite_spec,
    workload_spec,
)


def run_script_workload(args: argparse.Namespace) -> None:
    """Generic escape hatch: run ``python <script> <args>`` from the repo root."""
    config = load_config(args.config)
    workload = workload_spec(config, args.workload)
    if workload.get("type") != "script":
        raise SystemExit(f"workload {args.workload!r} is not type=script")
    script = workload.get("script")
    if not script:
        raise SystemExit(f"script workload {args.workload!r} has no script")
    cmd = [str(ROOT / config.get("python", ".venv/bin/python")), str(ROOT / script)]
    cmd.extend(str(part) for part in workload.get("args", []))
    print("script:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT, env=merged_env(workload))
    expected = workload.get("expect_file")
    if expected and not (ROOT / str(expected)).exists():
        raise SystemExit(f"expected script artifact does not exist: {expected}")


def _clean_server_for_suite(config_path: Path, server: str, grace_s: float) -> None:
    backends.clean(argparse.Namespace(config=config_path, server=server, all=False, grace_s=grace_s))


def _launch_server_for_suite(config_path: Path, server: str, timeout_s: float) -> None:
    backends.launch(
        argparse.Namespace(
            config=config_path,
            server=server,
            foreground=False,
            wait=True,
            timeout_s=timeout_s,
        )
    )


def _run_workload(args: argparse.Namespace, workload_name: str) -> None:
    workload = workload_spec(load_config(args.config), workload_name)
    sub = argparse.Namespace(config=args.config, workload=workload_name, server=args.server)
    sub.output_dir = None
    kind = workload.get("type")
    if kind == "verify":
        verify.verify(sub)
    elif kind == "script":
        run_script_workload(sub)
    else:
        raise SystemExit(f"unsupported workload type for {workload_name!r}: {kind}")


def run_suite(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    suite = suite_spec(config, args.suite)
    for workload_name in suite["workloads"]:
        workload = workload_spec(config, workload_name)
        manage_server = bool(args.manage_servers or workload.get("manage_server"))
        server_name = None
        if workload.get("type") == "verify":
            server_name, _ = resolve_server_for_workload(config, workload, args.server)
        if manage_server and server_name:
            _clean_server_for_suite(args.config, server_name, args.clean_grace_s)
            _launch_server_for_suite(args.config, server_name, args.launch_timeout_s)
            try:
                _run_workload(args, workload_name)
            finally:
                _clean_server_for_suite(args.config, server_name, args.clean_grace_s)
        else:
            _run_workload(args, workload_name)


def list_items(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    sections = [args.section] if args.section != "all" else ["servers", "workloads", "suites"]
    for section in sections:
        print(f"[{section}]")
        if section == "servers":
            for name, spec in sorted(config.get("servers", {}).items()):
                resolved = server_spec(config, name)
                if resolved.get("abstract"):
                    continue
                line = f"{name}\t{resolved.get('served_model_name', '')}\t{resolved.get('host', '')}:{resolved.get('port', '')}"
                if resolved.get("description"):
                    line += f"\n\t{resolved['description']}"
                print(line)
        elif section == "workloads":
            for name, spec in sorted(config.get("workloads", {}).items()):
                line = f"{name}\t{spec.get('type', '')}\tserver={spec.get('server', '-')}"
                if spec.get("description"):
                    line += f"\n\t{spec['description']}"
                print(line)
        elif section == "suites":
            for name in sorted(config.get("suites", {})):
                suite = suite_spec(config, name)
                print(f"{name}\t{','.join(suite['workloads'])}")
        else:
            raise SystemExit(f"unknown list section {section!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uniserve-eval", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list")
    p.add_argument("section", nargs="?", choices=["all", "servers", "workloads", "suites"], default="all")
    p.set_defaults(func=list_items)

    p = sub.add_parser("launch")
    p.add_argument("server")
    p.add_argument("--foreground", action="store_true")
    p.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--timeout-s", type=float, default=1800)
    p.set_defaults(func=backends.launch)

    p = sub.add_parser("verify")
    p.add_argument("workload")
    p.add_argument("--server", help="override workload server")
    p.add_argument("--output-dir", type=Path, help="immutable output directory for this invocation")
    p.set_defaults(func=verify.verify)

    p = sub.add_parser("script")
    p.add_argument("workload")
    p.add_argument("--server", help="accepted for CLI symmetry; script workloads own their target")
    p.set_defaults(func=run_script_workload)

    p = sub.add_parser("run")
    p.add_argument("suite")
    p.add_argument("--server", help="override every workload server")
    p.add_argument("--manage-servers", action="store_true", help="launch/clean each workload server")
    p.add_argument("--launch-timeout-s", type=float, default=1800)
    p.add_argument("--clean-grace-s", type=float, default=3)
    p.set_defaults(func=run_suite)

    p = sub.add_parser("clean")
    p.add_argument("server", nargs="?")
    p.add_argument("--all", action="store_true")
    p.add_argument("--grace-s", type=float, default=3)
    p.set_defaults(func=backends.clean)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "cmd", None) == "clean" and not args.all and not args.server:
        raise SystemExit("clean requires a server or --all")
    args.func(args)


if __name__ == "__main__":
    main()
