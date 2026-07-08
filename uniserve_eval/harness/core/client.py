"""Async transport: send one request and produce a :class:`RequestRecord`.

Three wire shapes are supported, selected by ``TaskRequest.kind``:

* ``openai_chat`` -- OpenAI ``/v1/chat/completions`` SSE (LLM serving and chat interleave). TTFT/ITL are measured per text chunk, ``delta.images`` drives image counts and image latency, and ``output_len`` comes from ``usage`` when the server emits it (``stream_options.include_usage``), else the requested length.
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
from ..sse import TERMINAL_EVENT_TYPES, aiter_sse_events, aiter_sse_events_from_text
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
    scheduled_time: float | None = None,
) -> RequestRecord:
    payload = {key: value for key, value in request.payload.items() if value is not None}
    url = base_url.rstrip("/") + request.endpoint
    record = RequestRecord(request_id=request_id, task=task)
    record.scheduled_time = scheduled_time
    record.start_time = time.perf_counter()
    try:
        if request.kind == "images_generations":
            await _send_images(client, url, payload, record)
        elif request.kind == "openai_chat_json":
            await _send_chat_json(client, url, payload, record)
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
    record.http_response_time = time.perf_counter()
    record.latency = time.perf_counter() - record.start_time
    record.final_event_time = record.start_time + record.latency
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


async def _send_chat_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
) -> None:
    """One non-streamed chat completion (diffusion-pipeline chat backends)."""
    response = await client.post(url, json=payload)
    record.http_response_time = time.perf_counter()
    record.latency = time.perf_counter() - record.start_time
    record.final_event_time = record.start_time + record.latency
    record.status_code = response.status_code
    try:
        data = response.json()
    except Exception:  # noqa: BLE001 - non-JSON body is a protocol failure.
        record.success = False
        record.classifier = f"transport_status_{response.status_code}"
        record.error = response.text[:500]
        return
    choices = data.get("choices") if isinstance(data, dict) else None
    content = ""
    if isinstance(choices, list) and choices:
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        finish_reason = choice0.get("finish_reason")
        if isinstance(finish_reason, str):
            record.finish_reason = finish_reason
        stop_reason = choice0.get("stop_reason")
        if isinstance(stop_reason, str):
            record.stop_reason = stop_reason
        message = choice0.get("message")
        raw = (message or {}).get("content")
        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, list):
            content = "".join(
                part.get("text", "")
                for part in raw
                if isinstance(part, dict) and part.get("type") == "text"
            )
    record.generated_text = content
    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        if isinstance(usage.get("completion_tokens"), int):
            record.output_len = int(usage["completion_tokens"])
        if isinstance(usage.get("prompt_tokens"), int):
            record.prompt_len = int(usage["prompt_tokens"])
    transport_ok = response.status_code < 400
    record.success = transport_ok and bool(content)
    record.classifier = "ok" if record.success else (
        f"transport_status_{response.status_code}" if not transport_ok else "empty_completion"
    )
    # Non-streaming: the full completion shares the request E2E latency.
    record.ttft = record.latency
    record.first_text_time = record.final_event_time


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
        record.http_response_time = time.perf_counter()
        record.status_code = response.status_code
        if response.status_code != 200:
            body = await response.aread()
            record.latency = time.perf_counter() - record.start_time
            record.final_event_time = record.start_time + record.latency
            record.success = False
            record.classifier = f"transport_status_{response.status_code}"
            record.error = body.decode("utf-8", errors="replace")[:500]
            return
        if protocol == "native":
            events = await aiter_sse_events_from_text(
                response.aiter_text(),
                stamp_time=True,
                on_parse_error="record",
                stop_on=TERMINAL_EVENT_TYPES,
            )
        else:
            events = await aiter_sse_events(
                response.aiter_lines(),
                stamp_time=True,
                on_parse_error="record",
            )
    last_event_time = _last_event_time(events, record.start_time)
    record.final_event_time = last_event_time
    record.latency = last_event_time - record.start_time
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

    itl: list[float] = []
    last_text_time: float | None = None
    image_since_last_text = False
    output_len = output_len_fallback
    prompt_tokens: int | None = None
    for event in events:
        choices = event.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                finish_reason = choice.get("finish_reason")
                if isinstance(finish_reason, str):
                    record.finish_reason = finish_reason
                stop_reason = choice.get("stop_reason")
                if isinstance(stop_reason, str):
                    record.stop_reason = stop_reason
        usage = event.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("completion_tokens"), int):
                output_len = int(usage["completion_tokens"])
            if isinstance(usage.get("prompt_tokens"), int):
                prompt_tokens = int(usage["prompt_tokens"])
        content = _openai_delta_text(event)
        images = _openai_delta_images(event)
        timestamp = event.get("_client_t")
        if content:
            record.text_chunks.append(content)
            record.generated_text += content
            if timestamp is not None:
                timestamp_f = float(timestamp)
                if last_text_time is None:
                    record.ttft = timestamp_f - record.start_time
                    record.first_text_time = timestamp_f
                elif not image_since_last_text:
                    itl.append(timestamp_f - last_text_time)
                last_text_time = timestamp_f
                image_since_last_text = False
        if images:
            record.images += len(images)
            image_since_last_text = True
            if timestamp is not None:
                timestamp_f = float(timestamp)
                if record.first_image_latency is None:
                    record.first_image_latency = timestamp_f - record.start_time
                    record.first_image_done_time = timestamp_f
                record.image_latencies.extend([timestamp_f - record.start_time] * len(images))

    if prompt_tokens is not None:
        record.prompt_len = prompt_tokens
    record.output_len = output_len
    record.itl = itl


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
    image_step_counts: dict[Any, int] = {}
    image_done_events: list[dict[str, Any]] = []
    finished: dict[str, Any] | None = None

    for event in events:
        kind = event.get("type")
        timestamp = event.get("_client_t")
        if kind == "scheduled":
            queued_at = event.get("queued_at")
            scheduled_at = event.get("scheduled_at")
            if isinstance(queued_at, (int, float)):
                record.server_queued_at = float(queued_at)
            if isinstance(scheduled_at, (int, float)):
                record.server_scheduled_at = float(scheduled_at)
        elif kind == "text":
            record.generated_text += event.get("text", "")
            if timestamp is not None:
                timestamp = float(timestamp)
                if last_text_time is None:
                    record.ttft = timestamp - record.start_time
                    record.first_text_time = timestamp
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
                record.first_image_begin_time = first_image_begin_t
            steps = event.get("steps")
            image_begins[event.get("image_id")] = (
                float(timestamp) if timestamp is not None else None,
                int(steps) if steps is not None else None,
            )
        elif kind == "image_step":
            image_since_last_text = True
            image_id = event.get("image_id")
            image_step_counts[image_id] = image_step_counts.get(image_id, 0) + 1
        elif kind == "image_done":
            image_since_last_text = True
            image_done_events.append(event)
        elif kind == "finished":
            finished = event

    record.itl = itl

    for event in image_done_events:
        image_id = event.get("image_id")
        timestamp = event.get("_client_t")
        begin_t, steps = image_begins.get(image_id, (None, None))
        if timestamp is not None:
            done_t = float(timestamp)
            record.image_latencies.append(done_t - record.start_time)
            if record.first_image_done_time is None:
                record.first_image_done_time = done_t
            if begin_t is not None:
                record.image_gen_seconds.append(done_t - begin_t)
            record.image_spans.append(
                {
                    "image_id": image_id,
                    "begin_ms": (begin_t - record.start_time) * 1000.0
                    if begin_t is not None
                    else None,
                    "done_ms": (done_t - record.start_time) * 1000.0,
                    "generation_ms": (done_t - begin_t) * 1000.0
                    if begin_t is not None
                    else None,
                    "steps": steps,
                    "step_events": image_step_counts.get(image_id, 0),
                }
            )
        if steps is not None:
            record.image_steps.append(steps)
    if first_image_begin_t is not None:
        record.first_image_latency = first_image_begin_t - record.start_time

    images = len(image_done_events)
    output_len = len(text_times)
    if finished is not None:
        reason = finished.get("reason")
        if isinstance(reason, str):
            record.finish_reason = reason
        stop_reason = finished.get("stop_reason")
        if isinstance(stop_reason, str):
            record.stop_reason = stop_reason
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


def _openai_delta_images(event: dict[str, Any]) -> list[dict[str, Any]]:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return []
    images: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        delta_images = delta.get("images")
        if isinstance(delta_images, list):
            images.extend(part for part in delta_images if isinstance(part, dict))
        content = delta.get("content")
        if isinstance(content, list):
            images.extend(
                part
                for part in content
                if isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part)
            )
    return images
