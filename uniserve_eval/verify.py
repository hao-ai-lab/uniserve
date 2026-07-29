"""Correctness gates over public chat-completions generation requests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from . import backends
from .harness.provenance import execution_provenance, hardware_contract, repository_state
from .profiles import (
    ROOT,
    expand_profile_value,
    load_config,
    resolve_server_for_workload,
    server_profile_definition_contract,
    spec_env,
    workload_dir,
    workload_env,
    workload_spec,
)


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_sse_line(
    line: str, current_event: str | None
) -> tuple[str | None, dict[str, Any] | None, bool]:
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


def synthetic_png_b64(seed: int, width: int, height: int) -> str:
    """Deterministic geometric test scene (mirrors the benchmark harness's)."""
    import io
    import random

    from PIL import Image, ImageDraw

    rng = random.Random(seed * 1_000_003)
    image = Image.new("RGB", (width, height), (135, 206, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, int(height * 0.65), width, height], fill=(34, 139, 34))
    draw.ellipse(
        [int(width * 0.72), int(height * 0.08), int(width * 0.9), int(height * 0.34)],
        fill=(255, 215, 0),
    )
    draw.rectangle(
        [int(width * 0.33), int(height * 0.4), int(width * 0.65), int(height * 0.75)],
        fill=(178, 34, 34),
    )
    draw.polygon(
        [
            (int(width * 0.31), int(height * 0.4)),
            (int(width * 0.49), int(height * 0.2)),
            (int(width * 0.67), int(height * 0.4)),
        ],
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
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    chunks.append(str(part["text"]))
    return "".join(chunks)


def finish_count(obj: dict[str, Any]) -> int:
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return 0
    return sum(1 for choice in choices if isinstance(choice, dict) and choice.get("finish_reason"))


def finish_reasons(obj: dict[str, Any]) -> list[str]:
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return []
    return [
        str(choice["finish_reason"])
        for choice in choices
        if isinstance(choice, dict) and choice.get("finish_reason") is not None
    ]


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


def usage_image_steps_per_image(obj: dict[str, Any]) -> list[int] | None:
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("image_steps_per_image")
    if not isinstance(value, list) or any(
        not isinstance(step, int) or isinstance(step, bool) or step < 0 for step in value
    ):
        return None
    return [int(step) for step in value]


def save_image(url: str, out_dir: Any, images: list[dict[str, Any]]) -> None:
    payload = data_url_payload(url)
    if payload is None:
        raise SystemExit("chat response returned an unsupported image URL shape")
    data = base64.b64decode(payload)
    size = png_size(data)
    from PIL import Image

    with Image.open(BytesIO(data)) as loaded:
        loaded.load()
        rgb = loaded.convert("RGB")
        rgb_sha256 = bytes_digest(rgb.tobytes())
    idx = len(images) + 1
    image_path = out_dir / f"image_{idx}.png"
    image_path.write_bytes(data)
    images.append(
        {
            "path": str(image_path),
            "size": list(size),
            "png_sha256": bytes_digest(data),
            "rgb_sha256": rgb_sha256,
            "color_representation": "RGB uint8",
        }
    )
    print(f"IMAGE_DONE {idx} {size} {image_path}")


def event_manifest_entry(obj: dict[str, Any]) -> dict[str, Any]:
    text = text_from_chunk(obj)
    return {
        "type": str(obj.get("type", "unknown")),
        "visible_text_bytes": len(text.encode("utf-8")),
        "image_count": len(image_urls_from_chunk(obj)),
        "finish_reasons": finish_reasons(obj),
        "has_usage": isinstance(obj.get("usage"), dict),
        "has_error": isinstance(obj.get("error"), dict),
    }


def verification_checks(
    workload: dict[str, Any],
    *,
    images: list[dict[str, Any]],
    image_steps_per_image: list[int] | None,
    errors: list[dict[str, Any]],
    finished_count: int,
) -> tuple[dict[str, bool], list[str], list[str]]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    failures: list[str] = []

    def check(name: str, valid: bool, failure: str) -> None:
        checks[name] = valid
        if not valid:
            failures.append(failure)

    check("error_free", not errors, "verify returned errors")
    expected_images = workload.get("warn_image_count")
    if expected_images is not None and len(images) != int(expected_images):
        warnings.append(f"expected {expected_images} images, got {len(images)}")
    expected_image_steps = workload.get("expect_image_steps_per_image")
    steps_valid = (
        expected_image_steps is None
        or not images
        or (
            image_steps_per_image is not None
            and len(image_steps_per_image) == len(images)
            and all(step == int(expected_image_steps) for step in image_steps_per_image)
        )
    )
    check(
        "image_steps_per_image",
        steps_valid,
        (
            f"expected every decoded image to report {expected_image_steps} image steps, "
            f"got {image_steps_per_image}"
        ),
    )
    expect_finished = workload.get("expect_finished", True)
    check(
        "finish_reason",
        not expect_finished or finished_count >= 1,
        f"expected at least one finish reason, got {finished_count}",
    )
    expected_width = workload.get("expect_image_width")
    expected_height = workload.get("expect_image_height")
    dimensions_valid = (
        expected_width is None
        or expected_height is None
        or all(
            tuple(image.get("size", ())) == (int(expected_width), int(expected_height))
            for image in images
        )
    )
    check(
        "image_dimensions",
        dimensions_valid,
        f"expected every image to be {expected_width}x{expected_height}",
    )
    return checks, warnings, failures


def verification_provenance(
    config: dict[str, Any],
    *,
    config_path: Path,
    workload_name: str,
    workload: dict[str, Any],
    server_name: str,
    server: dict[str, Any],
) -> dict[str, Any]:
    source_state = repository_state(ROOT)
    server_command = backends.build_serve_cmd(config, server, strict_env=True)
    server_environment = os.environ.copy()
    profile_value = (
        str(expand_profile_value(server["cuda_visible_devices"]))
        if server.get("cuda_visible_devices") is not None
        else None
    )
    cuda_visible_devices = backends.resolve_cuda_visible_devices(profile_value)
    if cuda_visible_devices is not None:
        server_environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    server_environment.update(spec_env(server))
    verifier_environment = os.environ.copy()
    verifier_environment.update(workload_env(workload))
    profile_payload = {
        "schema_version": 1,
        "config_sha256": bytes_digest(config_path.read_bytes()),
        "workload": workload_name,
        "workload_sha256": canonical_digest(workload),
        "server": server_name,
        "server_profile": server_profile_definition_contract(config, server_name),
    }
    return {
        "schema_version": 1,
        "source_state": source_state,
        "profile_contract": {
            **profile_payload,
            "fingerprint": canonical_digest(profile_payload),
        },
        "hardware": hardware_contract(),
        "server_execution": execution_provenance(
            server_command,
            server_environment,
            cwd=ROOT,
            workspace_root=ROOT,
        ),
        "verifier_execution": execution_provenance(
            [sys.executable, *sys.argv],
            verifier_environment,
            cwd=ROOT,
            workspace_root=ROOT,
        ),
    }


def verify(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    workload = workload_spec(config, args.workload)
    if workload.get("type") != "verify":
        raise SystemExit(f"workload {args.workload!r} is not type=verify")
    server_name, server = resolve_server_for_workload(config, workload, args.server)
    provenance = verification_provenance(
        config,
        config_path=Path(args.config).resolve(),
        workload_name=args.workload,
        workload=workload,
        server_name=server_name,
        server=server,
    )
    payload = dict(workload["payload"])
    synthetic = workload.get("input_image_synthetic")
    if synthetic:
        inject_synthetic_image(payload, dict(synthetic))
    requested_output = getattr(args, "output_dir", None)
    out_dir = (
        Path(requested_output).resolve()
        if requested_output is not None
        else workload_dir(config, workload.get("output_dir", args.workload))
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    endpoint = "/v1/chat/completions"
    url = f"http://{server.get('host', '127.0.0.1')}:{server['port']}{endpoint}"
    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    images: list[dict[str, Any]] = []
    event_counts: dict[str, int] = {}
    text = ""
    completion_tokens: int | None = None
    image_steps: int | None = None
    image_steps_per_image: list[int] | None = None
    errors: list[dict[str, Any]] = []
    finished = 0
    current_event: str | None = None
    done_seen = False
    event_manifest: list[dict[str, Any]] = []
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
                event_manifest.append(event_manifest_entry(obj))
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
                per_image_steps = usage_image_steps_per_image(obj)
                if per_image_steps is not None:
                    image_steps_per_image = per_image_steps
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
            image_steps_per_image = usage_image_steps_per_image(obj)
            for image_url in image_urls_from_response(obj):
                save_image(image_url, out_dir, images)
            event_manifest.append(
                {
                    "type": "chat.completion",
                    "visible_text_bytes": len(text.encode("utf-8")),
                    "image_count": len(images),
                    "finish_reasons": finish_reasons(obj),
                    "has_usage": isinstance(obj.get("usage"), dict),
                    "has_error": isinstance(obj.get("error"), dict),
                }
            )
    text_unit_count = completion_tokens if completion_tokens is not None else len(text.split())
    semantic_checks, warnings, failures = verification_checks(
        workload,
        images=images,
        image_steps_per_image=image_steps_per_image,
        errors=errors,
        finished_count=finished,
    )
    artifact_checks = semantic_checks
    artifact_valid = all(artifact_checks.values())
    summary = {
        "endpoint": endpoint,
        "elapsed_s": time.perf_counter() - start,
        "image_count": len(images),
        "image_steps": image_steps,
        "image_steps_per_image": image_steps_per_image,
        "images": images,
        "text": text,
        "visible_text_sha256": bytes_digest(text.encode("utf-8")),
        "text_unit_count": text_unit_count,
        "request_sha256": canonical_digest(payload),
        "event_manifest": event_manifest,
        "event_manifest_sha256": canonical_digest(event_manifest),
        "event_counts": event_counts,
        "errors": errors,
        "request": payload,
        "finished_count": finished,
        "done_seen": done_seen,
        "warnings": warnings,
        "artifact": {
            "schema_version": 1,
            "valid": artifact_valid,
            "valid_marker": "verify-valid-v1" if artifact_valid else None,
            "checks": artifact_checks,
            "provenance": provenance,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for warning in warnings:
        print(f"WARNING {warning}", file=sys.stderr)
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(failures[0])
