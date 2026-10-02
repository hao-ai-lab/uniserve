"""Native video APIs, ending only when all encoded media bytes arrive.

A request's payload is the canonical MiniMax-H3 body (``task``,
``conditions``, ``target``, ``seed`` and any stated schedule). UniServe and
SGLang take it as is; vLLM-Omni and FastVideo take the same work in their own
fields. No decoding, hashing or filesystem work belongs in these logical
requests. The caller owns the overall deadline and retains bodies for later
inspection.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from ..types import RequestRecord, TaskRequest

# Every MiniMax-H3 output runs at 24 frames per second.
_FPS = 24


async def receive_video(
    client: httpx.AsyncClient,
    base_url: str,
    request: TaskRequest,
    record: RequestRecord,
) -> None:
    """Submit the native payload, poll as needed, and collect original bytes."""
    kwargs = _native_request(request)
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

    # SGLang may publish a storage URL instead of a local content route.
    media_url = urljoin(base_url + "/", job.get("url") or f"{poll_url}/content")
    await _body(client, "GET", media_url, record)


def _native_request(request: TaskRequest) -> dict[str, Any]:
    """Return the ``httpx`` body arguments of a request on its backend.

    The canonical body states the schedule in sigma points including the
    clean endpoint; a baseline's request always states it (``VideoConfig``
    requires the fields each baseline takes). ``request.video_shape`` carries
    the canvas and frame count the target resolves to.
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
    seconds = float(payload["target"]["duration_seconds"])
    common = {
        "model": payload["model"],
        "prompt": payload["prompt"],
        "seed": payload["seed"],
    }
    if backend == "vllm-omni":
        # Omni counts schedule intervals (denoiser forwards), not points. Its
        # native duration field also preserves fractional seconds without the
        # generic OpenAI form field's integer restriction.
        fields = {
            **common,
            "width": shape.width,
            "height": shape.height,
            "aspect_ratio": payload["target"]["aspect_ratio"],
            "fps": _FPS,
            "num_inference_steps": int(payload["num_inference_steps"]) - 1,
            "guidance_scale": 1.0,
            "flow_shift": float(payload["flow_shift"]),
            "extra_params": json.dumps(
                {
                    "task": payload["task"],
                    "duration": seconds,
                    "audio_flow_shift": float(payload["audio_flow_shift"]),
                    # Configured serving options, such as preencode_mp4; the
                    # configuration cannot restate the fields above.
                    **request.video_extra_params,
                }
            ),
        }
        # vLLM-Omni's native video endpoint consumes multipart form fields.
        return {
            "files": {key: (None, str(value)) for key, value in fields.items()}
        }
    if backend == "fastvideo":
        # FastVideo counts sigma points and takes the shifts from the
        # checkpoint. It validates an already aligned causal-VAE frame count,
        # so the frames carry the work and the requested seconds are kept
        # beside them.
        body = {
            **common,
            "size": f"{shape.width}x{shape.height}",
            "fps": _FPS,
            "num_inference_steps": int(payload["num_inference_steps"]),
            "num_frames": shape.frames,
        }
        if seconds.is_integer():
            # The generic seconds field accepts integers only; the aligned
            # frame count alone carries a fractional duration.
            body["seconds"] = seconds
        return {"json": body}
    raise ValueError(f"unknown video backend {backend!r}")


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
