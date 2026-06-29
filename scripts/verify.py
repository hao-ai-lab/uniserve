#!/usr/bin/env python3
"""Profile-driven serving verification helper.

Concepts:
  server   = how to launch one UniServe server topology
  workload = what to run against a compatible launched server
  suite    = ordered workload names for a broader verification pass

Examples:
  scripts/verify list
  scripts/verify launch sensenova-u1-mode-a-cuda-ipc
  scripts/verify generate sensenova-travel-interleave-4x
  scripts/verify bench qwen3-sharegpt-stress
  scripts/verify clean sensenova-u1-mode-a-cuda-ipc
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib import request as urlrequest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts" / "verify_config.json"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
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


def build_serve_cmd(config: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    cmd = [
        str(ROOT / config.get("server_bin", "target/debug/uniserve")),
        "serve",
        spec["model"],
        "--served-model-name",
        spec["served_model_name"],
        "--host",
        str(spec.get("host", "127.0.0.1")),
        "--port",
        str(spec["port"]),
        "--worker-python",
        str(ROOT / config.get("python", ".venv/bin/python")),
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
    out_dir = server_dir(config, args.server)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_serve_cmd(config, spec)
    env = os.environ.copy()
    if spec.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(spec["cuda_visible_devices"])
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


def parse_sse_line(line: str, current_event: str | None) -> tuple[str | None, dict[str, Any] | None, bool]:
    line = line.strip()
    if not line:
        return current_event, None, False
    if line.startswith("event:"):
        return line.split(":", 1)[1].strip(), None, False
    if not line.startswith("data:"):
        return current_event, None, False
    data = line.split(":", 1)[1].strip()
    if data == "[DONE]":
        return current_event, None, True
    obj = json.loads(data)
    obj.setdefault("type", current_event or obj.get("event") or "unknown")
    return current_event, obj, False


def png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def resolve_server_for_workload(config: dict[str, Any], workload: dict[str, Any], override: str | None) -> tuple[str, dict[str, Any]]:
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


def generate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    workload = workload_spec(config, args.workload)
    if workload.get("type") != "generate":
        raise SystemExit(f"workload {args.workload!r} is not type=generate")
    _, server = resolve_server_for_workload(config, workload, args.server)
    payload = workload["payload"]
    out_dir = workload_dir(config, workload.get("output_dir", args.workload))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("image_*.png"):
        path.unlink()
    url = f"http://{server.get('host', '127.0.0.1')}:{server['port']}/generate"
    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    images: list[dict[str, Any]] = []
    event_counts: dict[str, int] = {}
    text_token_count = 0
    errors: list[dict[str, Any]] = []
    current_event: str | None = None
    timeout_s = float(workload.get("timeout_s", 600))
    old_env: dict[str, str | None] = {}
    for key, value in workload_env(workload).items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        resp_ctx = urlrequest.urlopen(req, timeout=timeout_s)  # noqa: S310 - local verification helper.
    finally:
        for key, old in old_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    with resp_ctx as resp:
        print("status", resp.status)
        for raw in resp:
            current_event, obj, done = parse_sse_line(raw.decode("utf-8"), current_event)
            if done:
                break
            if obj is None:
                continue
            typ = obj.get("type", "unknown")
            event_counts[typ] = event_counts.get(typ, 0) + 1
            if typ == "error":
                errors.append(obj)
                print("ERROR", obj)
                break
            if "id" in obj:
                text_token_count += 1
            if typ == "image_begin":
                print("IMAGE_BEGIN", obj)
            if typ == "image_step" and obj.get("step") in {1, 10, 20, 30, 40, 50}:
                print("IMAGE_STEP", obj.get("step"))
            image_b64 = obj.get("pixels_png_b64") or obj.get("image_png_b64") or obj.get("png_b64")
            if image_b64:
                data = base64.b64decode(image_b64)
                size = png_size(data)
                idx = len(images) + 1
                image_path = out_dir / f"image_{idx}.png"
                image_path.write_bytes(data)
                images.append({"path": str(image_path), "size": list(size)})
                print(f"IMAGE_DONE {idx} {size} {image_path}")
    summary = {
        "elapsed_s": time.time() - start,
        "image_count": len(images),
        "images": images,
        "text_token_count": text_token_count,
        "event_counts": event_counts,
        "errors": errors,
        "request": payload,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit("generate returned errors")
    expected_images = workload.get("expect_images")
    if expected_images is not None and len(images) != int(expected_images):
        raise SystemExit(f"expected {expected_images} images, got {len(images)}")
    expected_steps = workload.get("expect_image_steps")
    if expected_steps is not None:
        expected_step_events = int(expected_steps) * len(images)
        if event_counts.get("image_step", 0) != expected_step_events:
            raise SystemExit(
                f"expected {expected_step_events} image_step events "
                f"({expected_steps} per image), got {event_counts.get('image_step', 0)}"
            )
    expected_w = workload.get("expect_image_width")
    expected_h = workload.get("expect_image_height")
    if expected_w is not None and expected_h is not None:
        for image in images:
            if tuple(image["size"]) != (int(expected_w), int(expected_h)):
                raise SystemExit(f"unexpected image size: {image}")


def bench(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    workload = workload_spec(config, args.workload)
    if workload.get("type") != "bench":
        raise SystemExit(f"workload {args.workload!r} is not type=bench")
    _, server = resolve_server_for_workload(config, workload, args.server)
    out_dir = workload_dir(config, args.workload)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / workload.get("output_file", workload.get("result_filename", "bench.json"))
    cmd = [
        str(ROOT / config.get("python", ".venv/bin/python")),
        str(ROOT / workload.get("script", "benchmarks/serving/bench_serving.py")),
        "--served-model-name",
        str(server["served_model_name"]),
        "--base-url",
        f"http://{server.get('host', '127.0.0.1')}:{server['port']}",
        "--dataset-name",
        str(workload.get("dataset_name", "random")),
        "--num-prompts",
        str(workload.get("num_prompts", 64)),
        "--request-rate",
        str(workload.get("request_rate", 8)),
        "--max-tokens",
        str(workload.get("max_tokens", 256)),
        "--seed",
        str(workload.get("seed", 0)),
        "--output-file",
        str(result_path),
    ]
    if workload.get("dataset_path"):
        cmd.extend(["--dataset-path", str(workload["dataset_path"])])
    print("bench:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT, env=merged_env(workload))
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if "expect_completed" in workload and result.get("completed") != workload["expect_completed"]:
            raise SystemExit(f"expected completed={workload['expect_completed']}, got {result.get('completed')}")
        if "expect_failed" in workload and result.get("failed") != workload["expect_failed"]:
            raise SystemExit(f"expected failed={workload['expect_failed']}, got {result.get('failed')}")


def run_script_workload(args: argparse.Namespace) -> None:
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
    clean(argparse.Namespace(config=config_path, server=server, all=False, grace_s=grace_s))


def _launch_server_for_suite(config_path: Path, server: str, timeout_s: float) -> None:
    launch(
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
    if workload.get("type") == "bench":
        sub = argparse.Namespace(config=args.config, workload=workload_name, server=args.server)
        bench(sub)
    elif workload.get("type") == "generate":
        sub = argparse.Namespace(config=args.config, workload=workload_name, server=args.server)
        generate(sub)
    elif workload.get("type") == "script":
        sub = argparse.Namespace(config=args.config, workload=workload_name, server=args.server)
        run_script_workload(sub)
    else:
        raise SystemExit(f"unsupported workload type for {workload_name!r}: {workload.get('type')}")


def run_suite(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    suite = config.get("suites", {}).get(args.suite)
    if suite is None:
        known = ", ".join(sorted(config.get("suites", {})))
        raise SystemExit(f"unknown suite {args.suite!r}; known suites: {known}")
    for workload_name in suite:
        workload = workload_spec(config, workload_name)
        manage_server = bool(args.manage_servers or workload.get("manage_server"))
        server_name = None
        if workload.get("type") in {"bench", "generate"}:
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
                print(f"{name}\t{resolved.get('served_model_name', '')}\t{resolved.get('host', '')}:{resolved.get('port', '')}")
        elif section == "workloads":
            for name, spec in sorted(config.get("workloads", {}).items()):
                print(f"{name}\t{spec.get('type', '')}\tserver={spec.get('server', '-')}")
        elif section == "suites":
            for name, items in sorted(config.get("suites", {}).items()):
                print(f"{name}\t{','.join(items)}")
        else:
            raise SystemExit(f"unknown list section {section!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    p.set_defaults(func=launch)

    p = sub.add_parser("generate")
    p.add_argument("workload")
    p.add_argument("--server", help="override workload server")
    p.set_defaults(func=generate)

    p = sub.add_parser("bench")
    p.add_argument("workload")
    p.add_argument("--server", help="override workload server")
    p.set_defaults(func=bench)

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
    p.set_defaults(func=clean)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "cmd", None) == "clean" and not args.all and not args.server:
        raise SystemExit("clean requires a server or --all")
    args.func(args)


if __name__ == "__main__":
    main()
