"""Server lifecycle: launch a serving backend, wait for readiness, clean up.

A server spec with an explicit ``command`` launches any backend (e.g. a
vLLM-Omni comparison server); the default builds a UniServe launch from
``model`` / ``served_model_name`` / ``serve_args``. Launched servers get their
own session (process group), a pid file, and a log file under the artifact
root, so ``clean`` reaps them reliably even after a killed run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .profiles import (
    ROOT,
    expand_profile_value,
    load_config,
    require_resolved_profile_value,
    server_dir,
    server_log_path,
    server_pid_path,
    server_spec,
    spec_env,
)

CUDA_VISIBLE_OVERRIDE_ENV = "UNISERVE_BENCH_CUDA_VISIBLE_DEVICES"


def resolve_cuda_visible_devices(profile_value: str | None) -> str | None:
    """Pick the CUDA_VISIBLE_DEVICES for a server launch.

    The ``UNISERVE_BENCH_CUDA_VISIBLE_DEVICES`` override wins over the
    profile's (already-expanded) ``cuda_visible_devices`` value; ``None``
    means inherit the ambient environment.
    """

    override = os.environ.get(CUDA_VISIBLE_OVERRIDE_ENV)
    if override is not None:
        return override
    return profile_value


def _repo_path(config: dict[str, Any], key: str, default: str) -> str:
    value = str(expand_profile_value(config.get(key, default)))
    return str(Path(value) if Path(value).is_absolute() else ROOT / value)


def build_serve_cmd(config: dict[str, Any], spec: dict[str, Any], *, strict_env: bool = False) -> list[str]:
    spec = expand_profile_value(spec)
    config = expand_profile_value(config)
    if strict_env:
        require_resolved_profile_value(spec, context="server spec")
        require_resolved_profile_value(config.get("python", ""), context="config python")
        require_resolved_profile_value(config.get("server_bin", ""), context="config server_bin")
    if spec.get("command"):
        return [str(part) for part in spec["command"]]
    cmd = []
    # Pin the server (and the workers it spawns, which inherit affinity) to one
    # NUMA node's full core set. Servers inherit the parent's CPU mask
    # otherwise — a runner invoked under `taskset -c 0` would silently starve
    # the host scheduler, both TP worker processes, and every CUDA driver
    # thread on a single core (measured: ~30% throughput loss under 12-way
    # interleave load).
    numa_node = spec.get("numa_node", 0)
    if numa_node is not None and shutil.which("numactl"):
        cmd.extend(
            [
                "numactl",
                f"--cpunodebind={int(numa_node)}",
                f"--membind={int(numa_node)}",
            ]
        )
    cmd.extend(
        [
            _repo_path(config, "server_bin", "target/debug/uniserve"),
            "serve",
            spec["model"],
            "--served-model-name",
            spec["served_model_name"],
            "--host",
            str(spec.get("host", "127.0.0.1")),
            "--port",
            str(spec["port"]),
            "--worker-python",
            _repo_path(config, "python", ".venv/bin/python"),
        ]
    )
    cmd.extend(str(part) for part in spec.get("serve_args", []))
    return cmd


def wait_for_port(host: str, port: int, timeout_s: float) -> None:
    """Wait until the server is inference-ready, not merely listening.

    The Rust HTTP server binds its port and starts accepting connections while
    the (tensor-parallel) workers are still loading the model; during that
    window ``/health`` returns 503 and any generate request fails with a 500.
    Poll ``/health`` for a 200 so callers never race the worker load. Fall back
    to a bare socket connect only if ``/health`` is unavailable on this backend.
    """
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    health_url = f"http://{host}:{int(port)}/health"
    saw_socket = False
    while time.time() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=1):
                saw_socket = True
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
            continue
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:  # noqa: S310 - local health probe.
                if resp.status == 200:
                    return
            last_error = RuntimeError("/health returned non-200")
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                # Model still loading; keep waiting for readiness.
                last_error = exc
            else:
                # No /health endpoint on this backend (e.g. 404): the open
                # socket is the only readiness signal available.
                return
        except OSError as exc:
            last_error = exc
        time.sleep(0.5)
    if saw_socket:
        raise SystemExit(
            f"server opened {host}:{port} but was not inference-ready within {timeout_s}s: {last_error}"
        )
    raise SystemExit(f"server did not open {host}:{port} within {timeout_s}s: {last_error}")


def launch(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    spec = server_spec(config, args.server)
    if spec.get("abstract"):
        raise SystemExit(f"server {args.server!r} is an abstract base profile and cannot be launched")
    out_dir = server_dir(config, args.server)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_serve_cmd(config, spec, strict_env=True)
    env = os.environ.copy()
    profile_value = (
        str(expand_profile_value(spec["cuda_visible_devices"]))
        if spec.get("cuda_visible_devices") is not None
        else None
    )
    cuda_visible_devices = resolve_cuda_visible_devices(profile_value)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.update(spec_env(spec))
    log_path = server_log_path(config, args.server)
    print("launch:", " ".join(cmd))
    print("log:", log_path)
    if args.foreground:
        with log_path.open("w", encoding="utf-8") as log:
            raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    server_pid_path(config, args.server).write_text(f"{proc.pid}\n", encoding="utf-8")
    print("pid:", proc.pid)
    if args.wait:
        wait_for_port(str(spec.get("host", "127.0.0.1")), int(spec["port"]), args.timeout_s)
        print("ready:", f"{spec.get('host', '127.0.0.1')}:{spec['port']}")


def clean(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    names = list(config.get("servers", {})) if args.all else [args.server]
    for name in names:
        if not name:
            continue
        if server_spec(config, name).get("abstract"):
            continue
        pid_file = server_pid_path(config, name)
        if not pid_file.exists():
            print(f"{name}: no pid file")
            continue
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        print(f"{name}: stopping pid {pid}")
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(args.grace_s)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pid_file.unlink(missing_ok=True)
