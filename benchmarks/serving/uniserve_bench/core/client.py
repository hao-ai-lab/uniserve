"""Async transport: send one request and produce a :class:`RequestRecord`.

Three wire shapes are supported, selected by ``TaskRequest.kind``:

* ``openai_chat`` -- OpenAI ``/v1/chat/completions`` SSE (LLM serving). TTFT/ITL
  are measured per content chunk; ``output_len`` comes from ``usage`` when the
  server emits it (``stream_options.include_usage``), else the requested length.
* ``native_generate`` -- UniServe native ``/generate`` SSE (i2i + interleave).
  Text tokens drive TTFT/ITL; ``image_begin``/``image_step``/``image_done`` drive
  the image metrics; ``finished`` provides server-reported token/image counts.
* ``images_generations`` -- OpenAI-style ``/v1/images/generations`` (t2i), a
  single non-streaming JSON ``{"data": [{"b64_json", ...}]}`` response.

Receive timestamps are stamped per SSE event by ``sse.aiter_sse_events`` so
TTFT/ITL reflect arrival time even though we collect the stream into a list.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from ..metrics.common import RequestRecord
from ..response_classifier import (
    classify_json_image_response,
    classify_native_events,
    classify_openai_events,
)
from ..sse import aiter_sse_events
from ..tasks.base import TaskRequest


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    request: TaskRequest,
    request_id: str,
    *,
    task: str,
    prompt_len: int = 0,
    output_len_fallback: int = 0,
) -> RequestRecord:
    payload = {key: value for key, value in request.payload.items() if value is not None}
    url = base_url.rstrip("/") + request.endpoint
    record = RequestRecord(request_id=request_id, task=task)
    record.start_time = time.perf_counter()
    try:
        if request.kind == "images_generations":
            await _send_images(client, url, payload, record)
        elif request.kind == "openai_chat":
            await _send_sse(
                client, url, payload, record, protocol="openai",
                prompt_len=prompt_len, output_len_fallback=output_len_fallback,
            )
        else:
            await _send_sse(
                client, url, payload, record, protocol="native",
                prompt_len=prompt_len, output_len_fallback=output_len_fallback,
            )
    except Exception as error:  # noqa: BLE001 - benchmarks emit structured failures.
        record.latency = time.perf_counter() - record.start_time
        record.success = False
        record.classifier = "harness_or_transport_failure"
        record.error = f"{type(error).__name__}: {error}"
    return record


async def _send_images(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
) -> None:
    response = await client.post(url, json=payload)
    record.latency = time.perf_counter() - record.start_time
    record.status_code = response.status_code
    try:
        data = response.json()
    except Exception:  # noqa: BLE001 - non-JSON body is a protocol failure.
        record.success = False
        record.classifier = f"transport_status_{response.status_code}"
        record.error = response.text[:500]
        return
    body_ok, classifier = classify_json_image_response(data)
    transport_ok = response.status_code < 400
    if not transport_ok and classifier == "ok":
        classifier = f"transport_status_{response.status_code}"
    record.success = body_ok and transport_ok
    record.classifier = classifier
    images = data.get("data") if isinstance(data, dict) else None
    count = len(images) if isinstance(images, list) else 0
    if record.success:
        record.images = max(1, count)
        # Non-streaming: every returned image shares the request E2E latency.
        record.image_latencies = [record.latency] * record.images


async def _send_sse(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
    *,
    protocol: str,
    prompt_len: int,
    output_len_fallback: int,
) -> None:
    async with client.stream("POST", url, json=payload) as response:
        record.status_code = response.status_code
        if response.status_code != 200:
            body = await response.aread()
            record.latency = time.perf_counter() - record.start_time
            record.success = False
            record.classifier = f"transport_status_{response.status_code}"
            record.error = body.decode("utf-8", errors="replace")[:500]
            return
        events = await aiter_sse_events(
            response.aiter_lines(), stamp_time=True, on_parse_error="record"
        )
    record.latency = _last_event_time(events, record.start_time) - record.start_time
    if protocol == "openai":
        _parse_openai(events, record, output_len_fallback=output_len_fallback, prompt_len=prompt_len)
    else:
        _parse_native(events, record, prompt_len_fallback=prompt_len)


def _parse_openai(
    events: list[dict[str, Any]],
    record: RequestRecord,
    *,
    output_len_fallback: int,
    prompt_len: int,
) -> None:
    ok, classifier = classify_openai_events(events)
    record.success = ok
    record.classifier = classifier
    record.prompt_len = prompt_len

    content_times: list[float] = []
    output_len = output_len_fallback
    prompt_tokens: int | None = None
    for event in events:
        usage = event.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("completion_tokens"), int):
                output_len = int(usage["completion_tokens"])
            if isinstance(usage.get("prompt_tokens"), int):
                prompt_tokens = int(usage["prompt_tokens"])
        content = _openai_delta_text(event)
        if content:
            record.text_chunks.append(content)
            record.generated_text += content
            timestamp = event.get("_client_t")
            if timestamp is not None:
                content_times.append(float(timestamp))

    if prompt_tokens is not None:
        record.prompt_len = prompt_tokens
    record.output_len = output_len
    if content_times:
        record.ttft = content_times[0] - record.start_time
        record.itl = [content_times[i] - content_times[i - 1] for i in range(1, len(content_times))]


def _parse_native(
    events: list[dict[str, Any]],
    record: RequestRecord,
    *,
    prompt_len_fallback: int,
) -> None:
    ok, classifier = classify_native_events(events)
    record.success = ok
    record.classifier = classifier
    record.prompt_len = prompt_len_fallback

    text_times: list[float] = []
    itl: list[float] = []
    last_text_time: float | None = None
    image_since_last_text = False
    first_image_begin_t: float | None = None
    image_begins: dict[Any, tuple[float | None, int | None]] = {}
    image_done_events: list[dict[str, Any]] = []
    finished: dict[str, Any] | None = None

    for event in events:
        kind = event.get("type")
        timestamp = event.get("_client_t")
        if kind == "text":
            record.generated_text += event.get("text", "")
            if timestamp is not None:
                timestamp = float(timestamp)
                if last_text_time is None:
                    record.ttft = timestamp - record.start_time
                elif not image_since_last_text:
                    # Skip the inter-token gap that straddles an image so the
                    # image generation time does not inflate text ITL.
                    itl.append(timestamp - last_text_time)
                last_text_time = timestamp
                image_since_last_text = False
                text_times.append(timestamp)
        elif kind == "image_begin":
            image_since_last_text = True
            if timestamp is not None and first_image_begin_t is None:
                first_image_begin_t = float(timestamp)
            steps = event.get("steps")
            image_begins[event.get("image_id")] = (
                float(timestamp) if timestamp is not None else None,
                int(steps) if steps is not None else None,
            )
        elif kind == "image_done":
            image_done_events.append(event)
        elif kind == "finished":
            finished = event

    record.itl = itl

    for event in image_done_events:
        image_id = event.get("image_id")
        timestamp = event.get("_client_t")
        begin_t, steps = image_begins.get(image_id, (None, None))
        if timestamp is not None:
            record.image_latencies.append(float(timestamp) - record.start_time)
            if begin_t is not None:
                record.image_gen_seconds.append(float(timestamp) - begin_t)
        if steps is not None:
            record.image_steps.append(steps)
    if first_image_begin_t is not None:
        record.first_image_latency = first_image_begin_t - record.start_time

    images = len(image_done_events)
    output_len = len(text_times)
    if finished is not None:
        if isinstance(finished.get("completion_tokens"), int):
            output_len = int(finished["completion_tokens"])
        if isinstance(finished.get("prompt_tokens"), int):
            record.prompt_len = int(finished["prompt_tokens"])
        if isinstance(finished.get("images"), int):
            images = int(finished["images"])
    record.output_len = output_len
    record.images = images


def _last_event_time(events: list[dict[str, Any]], default_start: float) -> float:
    times = [float(event["_client_t"]) for event in events if event.get("_client_t") is not None]
    return max(times) if times else time.perf_counter()


def _openai_delta_text(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return ""
    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
            reasoning = delta.get("reasoning")
            if isinstance(reasoning, str):
                parts.append(reasoning)
            reasoning_content = delta.get("reasoning_content")
            if isinstance(reasoning_content, str):
                parts.append(reasoning_content)
        text = choice.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)
