"""Correctness gates over public chat-completions generation requests."""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import time
from typing import Any
from urllib import error as urlerror
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
        return current_event, {"type": "sse_done"}, True
    obj = json.loads(data)
    obj.setdefault("type", current_event or obj.get("event") or "chat.completion.chunk")
    return current_event, obj, False


def png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def verify_image_channel_dominance(
    images: list[dict[str, Any]],
    expectation: Any,
) -> None:
    if not isinstance(expectation, list) or len(expectation) != 2:
        raise SystemExit("expect_image_channel_dominance must name two RGB channels")
    channels = {"red": 0, "green": 1, "blue": 2}
    higher, lower = (str(value).lower() for value in expectation)
    if higher not in channels or lower not in channels or higher == lower:
        raise SystemExit("expect_image_channel_dominance must name two distinct RGB channels")

    from PIL import Image, ImageStat

    for image in images:
        path = image.get("path")
        if not isinstance(path, str):
            raise SystemExit("generated image metadata is missing its artifact path")
        with Image.open(path) as loaded:
            means = ImageStat.Stat(loaded.convert("RGB")).mean
        if means[channels[higher]] <= means[channels[lower]]:
            raise SystemExit(
                f"expected image mean {higher} channel to exceed {lower}: {path}"
            )


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


def inject_synthetic_image(payload: dict[str, Any], synthetic: dict[str, Any]) -> None:
    image_b64 = synthetic_png_b64(
        int(synthetic.get("seed", 0)),
        int(synthetic.get("width", 960)),
        int(synthetic.get("height", 640)),
    )
    image_part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
    }
    messages = payload.setdefault("messages", [])
    if not messages:
        messages.append({"role": "user", "content": [image_part]})
        return
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}, image_part]
        elif isinstance(content, list):
            content.append(image_part)
        else:
            message["content"] = [image_part]
        return
    messages.append({"role": "user", "content": [image_part]})


def data_url_payload(url: str) -> str | None:
    prefix = "data:image/"
    if not url.startswith(prefix):
        return None
    marker = ";base64,"
    if marker not in url:
        return None
    return url.split(marker, 1)[1]


def image_urls_from_parts(parts: Any) -> list[str]:
    images: list[str] = []
    if not isinstance(parts, list):
        return images
    for part in parts:
        if not isinstance(part, dict):
            continue
        image_url = part.get("image_url")
        if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            images.append(str(image_url["url"]))
    return images


def image_urls_from_chunk(obj: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return urls
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict):
            urls.extend(image_urls_from_parts(delta.get("images")))
            urls.extend(image_urls_from_parts(delta.get("content")))
    return urls


def image_urls_from_response(obj: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return urls
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            urls.extend(image_urls_from_parts(message.get("images")))
            urls.extend(image_urls_from_parts(message.get("content")))
    return urls


def text_from_chunk(obj: dict[str, Any]) -> str:
    chunks: list[str] = []
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for key in ("content", "reasoning", "reasoning_content"):
            value = delta.get(key)
            if isinstance(value, str):
                chunks.append(value)
    return "".join(chunks)


def text_from_response(obj: dict[str, Any]) -> str:
    chunks: list[str] = []
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    chunks.append(str(part["text"]))
    return "".join(chunks)


def finish_count(obj: dict[str, Any]) -> int:
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return 0
    return sum(1 for choice in choices if isinstance(choice, dict) and choice.get("finish_reason"))


def usage_completion_tokens(obj: dict[str, Any]) -> int | None:
    usage = obj.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        return int(usage["completion_tokens"])
    return None


def usage_image_steps(obj: dict[str, Any]) -> int | None:
    usage = obj.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("image_steps"), int):
        return int(usage["image_steps"])
    return None


def save_image(url: str, out_dir: Any, images: list[dict[str, Any]]) -> None:
    payload = data_url_payload(url)
    if payload is None:
        raise SystemExit("chat response returned an unsupported image URL shape")
    data = base64.b64decode(payload)
    size = png_size(data)
    idx = len(images) + 1
    image_path = out_dir / f"image_{idx}.png"
    image_path.write_bytes(data)
    images.append({"path": str(image_path), "size": list(size)})
    print(f"IMAGE_DONE {idx} {size} {image_path}")


def verify(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    workload = workload_spec(config, args.workload)
    if workload.get("type") != "verify":
        raise SystemExit(f"workload {args.workload!r} is not type=verify")
    _, server = resolve_server_for_workload(config, workload, args.server)
    payload = dict(workload["payload"])
    synthetic = workload.get("input_image_synthetic")
    if synthetic:
        inject_synthetic_image(payload, dict(synthetic))
    out_dir = workload_dir(config, workload.get("output_dir", args.workload))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("image_*.png"):
        path.unlink()
    endpoint = "/v1/chat/completions"
    url = f"http://{server.get('host', '127.0.0.1')}:{server['port']}{endpoint}"
    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    images: list[dict[str, Any]] = []
    event_counts: dict[str, int] = {}
    text = ""
    completion_tokens: int | None = None
    image_steps: int | None = None
    errors: list[dict[str, Any]] = []
    finished = 0
    current_event: str | None = None
    done_seen = False
    timeout_s = float(workload.get("timeout_s", 600))
    old_env: dict[str, str | None] = {}
    for key, value in workload_env(workload).items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        try:
            resp_ctx = urlrequest.urlopen(req, timeout=timeout_s)  # noqa: S310 - local verification helper.
        except urlerror.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            message = f"verify transport error: HTTP {error.code}"
            if detail:
                message = f"{message}: {detail}"
            raise SystemExit(message) from error
    finally:
        for key, old in old_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    with resp_ctx as resp:
        print("status", resp.status)
        if resp.status >= 400:
            raise SystemExit(f"verify transport error: HTTP {resp.status}")
        if payload.get("stream"):
            for raw in resp:
                current_event, obj, done = parse_sse_line(raw.decode("utf-8"), current_event)
                if done:
                    done_seen = True
                if obj is None:
                    if done:
                        break
                    continue
                typ = str(obj.get("type", "unknown"))
                event_counts[typ] = event_counts.get(typ, 0) + 1
                if isinstance(obj.get("error"), dict):
                    errors.append(obj["error"])
                    print("ERROR", obj["error"])
                    break
                text += text_from_chunk(obj)
                finished += finish_count(obj)
                tokens = usage_completion_tokens(obj)
                if tokens is not None:
                    completion_tokens = tokens
                steps = usage_image_steps(obj)
                if steps is not None:
                    image_steps = steps
                for image_url in image_urls_from_chunk(obj):
                    save_image(image_url, out_dir, images)
                if done:
                    break
        else:
            obj = json.loads(resp.read().decode("utf-8"))
            if isinstance(obj.get("error"), dict):
                errors.append(obj["error"])
                print("ERROR", obj["error"])
            event_counts["chat.completion"] = 1
            text = text_from_response(obj)
            finished = finish_count(obj)
            completion_tokens = usage_completion_tokens(obj)
            image_steps = usage_image_steps(obj)
            for image_url in image_urls_from_response(obj):
                save_image(image_url, out_dir, images)
    text_unit_count = completion_tokens if completion_tokens is not None else len(text.split())
    summary = {
        "endpoint": endpoint,
        "elapsed_s": time.time() - start,
        "image_count": len(images),
        "image_steps": image_steps,
        "images": images,
        "text": text,
        "text_unit_count": text_unit_count,
        "event_counts": event_counts,
        "errors": errors,
        "request": payload,
        "finished_count": finished,
        "done_seen": done_seen,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit("verify returned errors")
    expected_images = workload.get("expect_images")
    if expected_images is not None and len(images) != int(expected_images):
        raise SystemExit(f"expected {expected_images} images, got {len(images)}")
    expected_image_steps = workload.get("expect_image_steps")
    if expected_image_steps is not None and image_steps != int(expected_image_steps):
        raise SystemExit(f"expected {expected_image_steps} image steps, got {image_steps}")
    expected_min_text = workload.get("expect_min_text_tokens")
    if expected_min_text is not None and text_unit_count < int(expected_min_text):
        raise SystemExit(
            f"expected at least {expected_min_text} text units, got {text_unit_count}"
        )
    if workload.get("expect_finished", True) and finished < 1:
        raise SystemExit(f"expected at least one finish reason, got {finished}")
    if payload.get("stream") and not done_seen:
        raise SystemExit("expected terminal [DONE] event")
    expected_w = workload.get("expect_image_width")
    expected_h = workload.get("expect_image_height")
    if expected_w is not None and expected_h is not None:
        for image in images:
            if tuple(image["size"]) != (int(expected_w), int(expected_h)):
                raise SystemExit(f"unexpected image size: {image}")
    expected_channel_dominance = workload.get("expect_image_channel_dominance")
    if expected_channel_dominance is not None:
        verify_image_channel_dominance(images, expected_channel_dominance)
