"""Native video APIs, ending only when all encoded media bytes arrive.

No decoding, hashing or filesystem work belongs in these logical requests.
The caller owns the overall deadline and retains bodies for later inspection.
"""

from __future__ import annotations

import asyncio
import json
from urllib.parse import quote, urljoin

import httpx

from ..types import RequestRecord, TaskRequest


async def receive_video(
    client: httpx.AsyncClient,
    base_url: str,
    request: TaskRequest,
    record: RequestRecord,
) -> None:
    """Submit the native payload, poll as needed, and collect original bytes."""
    backend = request.video_backend or "uniserve"
    payload = dict(request.payload)
    if backend == "fastvideo":
        # FastVideo counts sigma points: nine points yield eight forwards.
        payload.update(size="1344x768", fps=24, num_inference_steps=9)
    if backend == "vllm-omni":
        seconds = float(payload.pop("seconds"))
        # Omni's pinned V2 ladder counts intervals (forwards), not points.
        # Its native duration field also preserves fractional seconds without
        # the generic OpenAI form field's integer restriction.
        payload.update(
            width=1344,
            height=768,
            aspect_ratio="16:9",
            fps=24,
            num_inference_steps=8,
            guidance_scale=1.0,
            flow_shift=10.0,
            extra_params=json.dumps(
                {"task": "t2va", "duration": seconds, "audio_flow_shift": 3.0}
            ),
        )
        # vLLM-Omni's native video endpoint consumes multipart form fields.
        fields = {key: (None, str(value)) for key, value in payload.items()}
        kwargs = {"files": fields}
    else:
        if backend == "fastvideo":
            # FastVideo validates an already aligned causal-VAE frame count.
            # Preserve requested seconds separately from the actual work.
            frames = round(float(payload["seconds"]) * 24)
            payload["num_frames"] = frames + (5 - frames) % 17
            if not float(payload["seconds"]).is_integer():
                # The generic seconds field accepts integers only; the
                # aligned native frame count carries fractional durations.
                payload.pop("seconds")
        elif backend == "sglang":
            # Native H3 admission rejects fps/num_frames and derives both
            # modalities from this canonical target, not generic seconds.
            seconds = float(payload.pop("seconds"))
            payload.update(
                task="t2va",
                conditions=[],
                target={
                    "short_edge": 768,
                    "aspect_ratio": "16:9",
                    "duration_seconds": seconds,
                },
                num_inference_steps=9,
                flow_shift=10.0,
                audio_flow_shift=3.0,
            )
        kwargs = {"json": payload}
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
