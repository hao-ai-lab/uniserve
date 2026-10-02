from __future__ import annotations

import base64
import io
import json
import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from uniserve_eval.transport.sse import iter_sse_events


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def require_uniserve_binary() -> Path:
    configured = os.environ.get("UNISERVE_BINARY")
    path = (
        Path(configured)
        if configured
        else Path.cwd() / "target" / "debug" / "uniserve"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run `cargo build -p uniserve` first"
        )
    return path


def wait_health(
    base_url: str, process: subprocess.Popen[str], timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited early with code {process.returncode}"
            )
        try:
            response = httpx.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return
            last_error = (
                f"health returned {response.status_code}: {response.text[:200]}"
            )
        except Exception as error:  # noqa: BLE001 - readiness polling reports the last failure.
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(1.0)
    raise TimeoutError(
        f"server did not become healthy within {timeout_s}s; last error: "
        f"{last_error}"
    )


def start_server(
    args: list[str], log_path: Path, env: dict[str, str] | None = None
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.Popen(
        args,
        cwd=Path.cwd(),
        env=merged_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


@contextmanager
def server_process(
    args: list[str],
    base_url: str,
    log_path: Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> Iterator[subprocess.Popen[str]]:
    process = start_server(args, log_path, env)
    try:
        wait_health(base_url, process, timeout_s)
        yield process
    finally:
        stop_server(process)


def post_sse(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 600.0,
) -> list[dict[str, Any]]:
    with httpx.stream(
        "POST", f"{base_url}{endpoint}", json=payload, timeout=timeout_s
    ) as response:
        response.raise_for_status()
        return list(iter_sse_events(response.iter_lines()))


def t2va_request(
    model: str, prompt: str, seconds: float, seed: int
) -> dict[str, Any]:
    """A MiniMax-H3 text-to-video-and-audio request at the 16:9 canvas."""
    return {
        "model": model,
        "prompt": prompt,
        "task": "t2va",
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": seconds,
        },
        "seed": seed,
    }


def png_size_from_b64(pixels_png_b64: str) -> tuple[int, int]:
    image = Image.open(io.BytesIO(base64.b64decode(pixels_png_b64)))
    image.load()
    return image.width, image.height


def tiny_input_png_b64() -> str:
    image = Image.new("RGB", (32, 32), (32, 96, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def written_deployment(directory: Path, workers: list[dict[str, Any]]) -> Path:
    """Write the deployment configuration a serve invocation is given.

    `--workers` names a file because a deployment configuration states every
    rank's node and device and every component placed on them; a test states
    one the same way a deployment does.
    """
    path = directory / "deployment.json"
    path.write_text(json.dumps(workers, indent=2))
    return path
