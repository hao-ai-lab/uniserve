"""The native video API, ending only when all encoded media bytes arrive.

A request's payload is the canonical MiniMax-H3 body (``task``,
``conditions``, ``target`` and ``seed``), which UniServe takes as JSON,
reading condition media from its ``file://`` URIs. ``/v1/videos/sync``
answers with the MP4; ``/v1/videos`` answers with a job, which is polled
until it completes and whose MP4 is then read from its content route. No
decoding, hashing or filesystem work belongs in these logical requests. The
caller owns the overall deadline and retains bodies for later inspection.

The server's own timings of a request (its inference time and named stage
durations) are recorded as it reports them: in the ``X-Inference-Time-S``
and ``X-Stage-Durations`` headers of a synchronous response, or the
``inference_time_s`` and ``stage_durations`` fields of a completed job.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import aiohttp

from ..types import RequestRecord, TaskRequest

# Response headers in which the synchronous video route reports timings.
_TIMING_HEADERS = ("x-inference-time-s", "x-stage-durations")


async def receive_video(
    client: aiohttp.ClientSession,
    base_url: str,
    request: TaskRequest,
    record: RequestRecord,
) -> None:
    """Submit the payload, poll a job as needed, and collect original bytes."""
    url = base_url.rstrip("/") + request.endpoint

    if request.endpoint.endswith("/sync"):
        await _body(client, "POST", url, record, json=request.payload)
        return

    async with client.post(url, json=request.payload) as response:
        record.note_http(response.status)
        if await _failed_status(response, record):
            return
        job = await response.json()
    job_id = quote(str(job["id"]), safe="")
    poll_url = f"{url}/{job_id}"
    while job.get("status") != "completed":
        if job.get("status") in {"failed", "cancelled"}:
            record.close_now()
            record.mark_failure("video_job_failed", str(job.get("error", job)))
            return
        await asyncio.sleep(request.poll_interval_s)
        async with client.get(poll_url) as response:
            if await _failed_status(response, record):
                return
            job = await response.json()
    _record_timings(
        record, job.get("inference_time_s"), job.get("stage_durations")
    )
    await _body(client, "GET", f"{poll_url}/content", record)


def _record_timings(
    record: RequestRecord, inference_s: Any, stages: Any
) -> None:
    """Record the server-reported timings that parse as finite numbers.

    The server reports these as it defines them; a malformed or absent value
    is left unset rather than failing a request that returned its media.
    """
    record.server_inference_s = _finite(inference_s)
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


async def _failed_status(
    response: aiohttp.ClientResponse, record: RequestRecord
) -> bool:
    if response.status < 400:
        return False
    record.status_code = response.status
    if record.final_event_time is None:
        record.close_now()
    record.mark_failure(
        f"transport_status_{response.status}",
        (await response.text(errors="replace"))[:500],
    )
    return True


async def _body(
    client: aiohttp.ClientSession,
    method: str,
    url: str,
    record: RequestRecord,
    **kwargs,
) -> None:
    async with client.request(
        method, url, allow_redirects=True, **kwargs
    ) as response:
        record.note_http(response.status)
        record.video_mime = response.headers.get("content-type", "")
        # Encoded bodies are buffered in host memory; no media work delays the
        # next slot. Keep even invalid bodies for post-run failure artifacts.
        record.video_body = await response.read()
        record.close_now()
        if not await _failed_status(response, record):
            record.mark_success()
            # The synchronous route reports its timings in headers, which it
            # may omit; a job's content download carries none, and its job
            # fields stand.
            if any(name in response.headers for name in _TIMING_HEADERS):
                _record_timings(
                    record,
                    response.headers.get("x-inference-time-s"),
                    response.headers.get("x-stage-durations"),
                )
