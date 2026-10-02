"""Native video APIs, ending only when all encoded media bytes arrive.

A request's payload is the canonical MiniMax-H3 body (``task``,
``conditions``, ``target``, ``seed`` and any stated schedule). UniServe and
SGLang take it as is, reading condition media from its ``file://`` URIs;
vLLM-Omni and FastVideo take the same work in their own fields, vLLM-Omni
with the media uploaded as request parts and FastVideo with plain local
paths. No decoding, hashing or filesystem work belongs in these logical
requests: the media bytes were read when the request was built. The caller
owns the overall deadline and retains bodies for later inspection.

A server's own timings of a request (its inference time, named stage
durations and peak device memory) are recorded as it reports them: in the
``X-Inference-Time-S``, ``X-Stage-Durations`` and ``X-Peak-Memory-MB``
headers of a synchronous response, or the ``inference_time_s``,
``stage_durations`` and ``peak_memory_mb`` fields of a completed job.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from ..types import RequestRecord, TaskRequest

# Every MiniMax-H3 output runs at 24 frames per second.
_FPS = 24

# Response headers in which a synchronous video route reports its timings.
_TIMING_HEADERS = (
    "x-inference-time-s",
    "x-stage-durations",
    "x-peak-memory-mb",
)

# vLLM-Omni and FastVideo present a request's references grouped by media
# type in this order, whatever order they arrive in.
_TYPE_ORDER = {"image": 0, "video": 1, "video_audio": 1, "audio": 2}


async def receive_video(
    client: httpx.AsyncClient,
    base_url: str,
    request: TaskRequest,
    record: RequestRecord,
) -> None:
    """Submit the native payload, poll as needed, and collect original bytes."""
    kwargs = native_request(request)
    url = base_url.rstrip("/") + request.endpoint

    if request.endpoint.endswith("/sync"):
        await _body(client, "POST", url, record, **kwargs)
        return

    response = await client.post(url, **kwargs)
    record.note_http(response.status_code)
    if _failed_status(response, record):
        return
    job = response.json()
    job_id = quote(str(job["id"]), safe="")
    poll_url = f"{url}/{job_id}"
    while job.get("status") != "completed":
        if job.get("status") in {"failed", "cancelled"}:
            record.close_now()
            record.mark_failure("video_job_failed", str(job.get("error", job)))
            return
        await asyncio.sleep(request.poll_interval_s)
        response = await client.get(poll_url)
        if _failed_status(response, record):
            return
        job = response.json()
    _record_timings(
        record,
        job.get("inference_time_s"),
        job.get("stage_durations"),
        job.get("peak_memory_mb"),
    )

    # SGLang may publish a storage URL instead of a local content route.
    media_url = urljoin(base_url + "/", job.get("url") or f"{poll_url}/content")
    await _body(client, "GET", media_url, record)


def native_request(request: TaskRequest) -> dict[str, Any]:
    """Return the ``httpx`` body arguments of a request on its backend.

    The canonical body states the schedule in sigma points including the
    clean endpoint; a baseline's request always states it (``VideoConfig``
    requires the fields each baseline takes). ``request.video_shape`` carries
    the canvas and frame count the target resolves to.

    Raises:
        ValueError: The backend cannot be sent the request's conditions as
            the same work: in their order, or with a field it lacks.
    """
    backend = request.video_backend or "uniserve"
    payload = dict(request.payload)
    if backend == "uniserve":
        return {"json": payload}
    if backend == "sglang":
        # Native H3 admission takes the canonical body and counts sigma
        # points. Configured serving options, such as x264_preset, join it;
        # the configuration cannot restate the work fields.
        return {"json": {**payload, **request.video_extra_params}}

    shape = request.video_shape
    if shape is None:
        raise ValueError(f"a {backend} video request needs its video shape")
    conditions = payload["conditions"]
    _check_type_order(backend, conditions)
    common = {
        "model": payload["model"],
        "prompt": payload["prompt"],
        "seed": payload["seed"],
    }
    if backend == "vllm-omni":
        return _vllm_omni_request(request, payload, conditions, common)
    if backend == "fastvideo":
        return _fastvideo_request(request, payload, conditions, common)
    raise ValueError(f"unknown video backend {backend!r}")


def _vllm_omni_request(
    request: TaskRequest,
    payload: dict[str, Any],
    conditions: list[dict[str, Any]],
    common: dict[str, Any],
) -> dict[str, Any]:
    """Return vLLM-Omni's multipart body for the canonical request.

    Omni counts schedule intervals (denoiser forwards), not points. Its
    native duration field preserves fractional seconds without the generic
    OpenAI form field's integer restriction. Condition media go up as
    repeated ``input_references`` parts, which the server classifies by
    media type; keyframes take their frame indices in ``extra_params``, and
    a video's soundtrack conditions the request whenever the file has one.
    """
    task = payload["task"]
    keyframes = [
        condition["frame_index"]
        for condition in conditions
        if condition["role"] == "keyframe"
    ]
    if keyframes and task != "fl2va":
        raise ValueError("vLLM-Omni takes keyframes only for fl2va")
    extra = {
        "task": task,
        "duration": float(payload["target"]["duration_seconds"]),
        "audio_flow_shift": float(payload["audio_flow_shift"]),
    }
    if keyframes:
        extra["frame_indices"] = keyframes
    starts = [
        float(condition.get("start_time_seconds", 0.0))
        for condition in conditions
        if condition["type"] in {"video", "video_audio"}
    ]
    if any(starts):
        # One offset per video, in the videos' order.
        extra["start_time_seconds"] = starts

    shape = request.video_shape
    assert shape is not None
    fields = {
        **common,
        "width": shape.width,
        "height": shape.height,
        "aspect_ratio": payload["target"]["aspect_ratio"],
        "fps": _FPS,
        "num_inference_steps": int(payload["num_inference_steps"]) - 1,
        "guidance_scale": 1.0,
        "flow_shift": float(payload["flow_shift"]),
        # Configured serving options, such as preencode_mp4; the
        # configuration cannot restate the work fields.
        "extra_params": json.dumps({**extra, **request.video_extra_params}),
    }
    parts: list[tuple[str, tuple[Any, ...]]] = [
        (key, (None, str(value))) for key, value in fields.items()
    ]
    parts.extend(
        ("input_references", (Path(media.path).name, media.data, media.mime))
        for media in request.condition_media
    )
    return {"files": parts}


def _fastvideo_request(
    request: TaskRequest,
    payload: dict[str, Any],
    conditions: list[dict[str, Any]],
    common: dict[str, Any],
) -> dict[str, Any]:
    """Return FastVideo's JSON body for the canonical request.

    FastVideo takes the shifts from the checkpoint and counts the schedule
    in sigma points, or in fused-block forwards for a parallel decoding
    student. It validates an already aligned causal-VAE frame count, so the
    frames carry the work and integral requested seconds are kept beside
    them. References name local files by path in its typed reference
    fields; it has no keyframe or start-offset fields.
    """
    points = int(payload["num_inference_steps"])
    shape = request.video_shape
    assert shape is not None
    body: dict[str, Any] = {
        **common,
        "size": f"{shape.width}x{shape.height}",
        "fps": _FPS,
        "num_inference_steps": (
            points - 1 if request.video_parallel_decoding else points
        ),
        "num_frames": shape.frames,
    }
    seconds = float(payload["target"]["duration_seconds"])
    if seconds.is_integer():
        # The generic seconds field accepts integers only; the aligned
        # frame count alone carries a fractional duration.
        body["seconds"] = seconds
    if payload["task"] == "t2va":
        return {"json": body}

    references: dict[str, list[dict[str, str]]] = {}
    fields = {
        "image": ("image_reference", "image_url"),
        "video": ("video_reference", "video_url"),
        "video_audio": ("video_reference", "video_url"),
        "audio": ("audio_reference", "audio_url"),
    }
    for condition, media in zip(
        conditions, request.condition_media, strict=True
    ):
        if condition["role"] != "reference":
            raise ValueError("FastVideo takes reference conditions only")
        if condition.get("start_time_seconds"):
            raise ValueError("FastVideo has no reference start offset")
        field, key = fields[condition["type"]]
        references.setdefault(field, []).append({key: media.path})
    return {"json": {**body, "task": payload["task"], **references}}


def _check_type_order(backend: str, conditions: list[dict[str, Any]]) -> None:
    """Require conditions in the order a type-grouping backend presents.

    Raises:
        ValueError: A later condition has an earlier media type, so the
            backend would present the conditions in another order.
    """
    order = [_TYPE_ORDER[condition["type"]] for condition in conditions]
    if order != sorted(order):
        raise ValueError(
            f"{backend} presents images, then videos, then audio; the "
            "request's conditions must already be in that order"
        )


def _record_timings(
    record: RequestRecord,
    inference_s: Any,
    stages: Any,
    peak_memory_mib: Any,
) -> None:
    """Record the server-reported timings that parse as finite numbers.

    A server reports these as it defines them; a malformed or absent value
    is left unset rather than failing a request that returned its media.
    """
    record.server_inference_s = _finite(inference_s)
    record.server_peak_memory_mib = _finite(peak_memory_mib)
    if isinstance(stages, str):
        try:
            stages = json.loads(stages)
        except json.JSONDecodeError:
            stages = None
    if isinstance(stages, Mapping):
        record.server_stage_s = {
            str(name): value
            for name, value in (
                (name, _finite(seconds)) for name, seconds in stages.items()
            )
            if value is not None
        }


def _finite(value: Any) -> float | None:
    """Return a value as a finite float, or ``None`` when it is not one."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _failed_status(response: httpx.Response, record: RequestRecord) -> bool:
    if response.status_code < 400:
        return False
    record.status_code = response.status_code
    if record.final_event_time is None:
        record.close_now()
    record.mark_failure(
        f"transport_status_{response.status_code}", response.text[:500]
    )
    return True


async def _body(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    record: RequestRecord,
    **kwargs,
) -> None:
    async with client.stream(
        method, url, follow_redirects=True, **kwargs
    ) as response:
        record.note_http(response.status_code)
        record.video_mime = response.headers.get("content-type", "")
        # Encoded bodies are buffered in host memory; no media work delays the
        # next slot. Keep even invalid bodies for post-run failure artifacts.
        record.video_body = await response.aread()
        record.close_now()
        if not _failed_status(response, record):
            record.mark_success()
            # A synchronous route reports its timings in headers, any of
            # which a server may omit; a job's content download carries
            # none, and its job fields stand.
            if any(name in response.headers for name in _TIMING_HEADERS):
                _record_timings(
                    record,
                    response.headers.get("x-inference-time-s"),
                    response.headers.get("x-stage-durations"),
                    response.headers.get("x-peak-memory-mb"),
                )
