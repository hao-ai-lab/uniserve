"""Correctness gates over the native SSE stream (workload type ``verify``).

POSTs the workload's declared ``payload`` to ``/generate`` and validates the
event stream: exact image counts / pixel sizes / per-image step counts, text
token floors, clean termination, and zero ``error`` events. For
understanding-mode inputs, ``input_image_synthetic`` renders a deterministic
geometric scene and injects it as ``input_image_b64`` so i2t gates run
hermetically.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import time
from typing import Any
from urllib import request as urlrequest

from .profiles import (
    load_config,
    resolve_server_for_workload,
    workload_dir,
    workload_env,
    workload_spec,
)


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


def synthetic_png_b64(seed: int, width: int, height: int) -> str:
    """Deterministic geometric test scene (mirrors the benchmark harness's)."""
    import io
    import random

    from PIL import Image, ImageDraw

    rng = random.Random(seed * 1_000_003)
    image = Image.new("RGB", (width, height), (135, 206, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, int(height * 0.65), width, height], fill=(34, 139, 34))
    draw.ellipse([int(width * 0.72), int(height * 0.08), int(width * 0.9), int(height * 0.34)], fill=(255, 215, 0))
    draw.rectangle([int(width * 0.33), int(height * 0.4), int(width * 0.65), int(height * 0.75)], fill=(178, 34, 34))
    draw.polygon(
        [(int(width * 0.31), int(height * 0.4)), (int(width * 0.49), int(height * 0.2)), (int(width * 0.67), int(height * 0.4))],
        fill=(80, 40, 20),
    )
    del rng
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def verify(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    workload = workload_spec(config, args.workload)
    if workload.get("type") != "verify":
        raise SystemExit(f"workload {args.workload!r} is not type=verify")
    _, server = resolve_server_for_workload(config, workload, args.server)
    payload = dict(workload["payload"])
    synthetic = workload.get("input_image_synthetic")
    if synthetic:
        payload["input_image_b64"] = synthetic_png_b64(
            int(synthetic.get("seed", 0)),
            int(synthetic.get("width", 960)),
            int(synthetic.get("height", 640)),
        )
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
        raise SystemExit("verify returned errors")
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
    expected_min_text = workload.get("expect_min_text_tokens")
    if expected_min_text is not None and text_token_count < int(expected_min_text):
        raise SystemExit(
            f"expected at least {expected_min_text} text tokens, got {text_token_count}"
        )
    if workload.get("expect_finished", True) and event_counts.get("finished", 0) != 1:
        raise SystemExit(f"expected one finished event, got {event_counts.get('finished', 0)}")
    expected_w = workload.get("expect_image_width")
    expected_h = workload.get("expect_image_height")
    if expected_w is not None and expected_h is not None:
        for image in images:
            if tuple(image["size"]) != (int(expected_w), int(expected_h)):
                raise SystemExit(f"unexpected image size: {image}")
