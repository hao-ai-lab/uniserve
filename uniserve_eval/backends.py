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
import signal
import socket
import subprocess
import time
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
    cmd = [
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
    cmd.extend(str(part) for part in spec.get("serve_args", []))
    return cmd


def wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=1):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
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
    if spec.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(expand_profile_value(spec["cuda_visible_devices"]))
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
