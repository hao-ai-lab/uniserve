"""HTTP transport for chat completions and image generation."""

from __future__ import annotations

import time
from typing import Any, TypeGuard

import httpx

from ..types import IMAGES_GENERATIONS, RequestRecord, TaskRequest
from .classify import (
    classify_json_image_response,
    classify_openai_events,
    openai_delta_images,
    openai_delta_text,
    openai_message_images,
    openai_message_text,
)
from .images import ImageOutputError, decode_openai_image_parts
from .sse import aiter_sse_events


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
    record.requested_output_len = int(output_len_fallback)
    record.scheduled_time = scheduled_time
    record.endpoint = request.endpoint
    record.start_time = time.perf_counter()
    try:
        if request.endpoint == IMAGES_GENERATIONS:
            await _send_images(client, url, payload, record)
        elif request.stream:
            await _send_sse(
                client,
                url,
                payload,
                record,
                prompt_len=prompt_len,
                output_len_fallback=output_len_fallback,
            )
        else:
            await _send_chat_json(
                client,
                url,
                payload,
                record,
                prompt_len=prompt_len,
                output_len_fallback=output_len_fallback,
            )
    except Exception as error:  # noqa: BLE001 - benchmarks emit structured failures.
        record.latency = time.perf_counter() - record.start_time
        record.success = False
        record.classifier = "transport_failure"
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
    except Exception:
        record.success = False
        record.classifier = f"transport_status_{response.status_code}"
        record.error = response.text[:500]
        return
    body_ok, classifier = classify_json_image_response(data)
    transport_ok = response.status_code < 400
    if not transport_ok and classifier == "ok":
        classifier = f"transport_status_{response.status_code}"
    images = data.get("data") if isinstance(data, dict) else None
    image_error: str | None = None
    if body_ok and isinstance(images, list):
        if not all(isinstance(image, dict) for image in images):
            image_error = "protocol_invalid_image_part"
        else:
            image_error = _decode_record_images(images, record)
    record.success = body_ok and transport_ok and image_error is None
    record.classifier = image_error or classifier
    if image_error is not None:
        record.error = image_error
    if record.success:
        record.image_latencies = [record.latency] * record.images


async def _send_chat_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
    *,
    prompt_len: int,
    output_len_fallback: int,
) -> None:
    response = await client.post(url, json=payload)
    record.http_response_time = time.perf_counter()
    record.latency = time.perf_counter() - record.start_time
    record.final_event_time = record.start_time + record.latency
    record.status_code = response.status_code
    try:
        data = response.json()
    except Exception:
        record.success = False
        record.classifier = f"transport_status_{response.status_code}"
        record.error = response.text[:500]
        return
    _parse_chat_json(
        data,
        record,
        status_code=response.status_code,
        prompt_len=prompt_len,
        output_len_fallback=output_len_fallback,
    )


def _parse_chat_json(
    data: Any,
    record: RequestRecord,
    *,
    status_code: int,
    prompt_len: int,
    output_len_fallback: int,
) -> None:
    choices = data.get("choices") if isinstance(data, dict) else None
    content = ""
    images: list[dict[str, Any]] = []
    if isinstance(choices, list) and choices:
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        finish_reason = choice0.get("finish_reason")
        if isinstance(finish_reason, str):
            record.finish_reason = finish_reason
        stop_reason = choice0.get("stop_reason")
        if isinstance(stop_reason, str):
            record.stop_reason = stop_reason
        message = choice0.get("message")
        content = openai_message_text(message)
        images = openai_message_images(message)
    record.generated_text = content
    image_error = _decode_record_images(images, record)
    if images and image_error is None:
        record.image_latencies = [record.latency] * record.images
    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        if isinstance(usage.get("completion_tokens"), int):
            record.output_len = int(usage["completion_tokens"])
            record.output_len_source = "server_usage"
        if isinstance(usage.get("prompt_tokens"), int):
            record.prompt_len = int(usage["prompt_tokens"])
            record.prompt_len_source = "server_usage"
        image_steps = _image_steps_per_image(usage)
        if image_steps is not None:
            record.image_steps = image_steps
    if record.output_len_source != "server_usage":
        record.output_len = output_len_fallback
    if record.prompt_len_source != "server_usage":
        record.prompt_len = prompt_len
    if isinstance(data, dict):
        _capture_cached_prompt_tokens(record, data)
    transport_ok = status_code < 400
    record.success = transport_ok and image_error is None and bool(content or record.decoded_images)
    record.classifier = (
        "ok"
        if record.success
        else (
            f"transport_status_{status_code}"
            if not transport_ok
            else image_error or "empty_completion"
        )
    )
    if image_error is not None:
        record.error = image_error
    record.token_timing_available = False


async def _send_sse(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
    *,
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
        if _is_json_content_type(response.headers.get("content-type", "")):
            await response.aread()
            record.latency = time.perf_counter() - record.start_time
            record.final_event_time = record.start_time + record.latency
            record.success = False
            record.classifier = "protocol_expected_sse"
            record.error = "stream request received a JSON response"
            return
        events = await aiter_sse_events(
            response.aiter_lines(),
            stamp_time=True,
            on_parse_error="record",
        )
    last_event_time = _last_event_time(events)
    record.final_event_time = last_event_time
    record.latency = last_event_time - record.start_time
    _parse_openai(events, record, output_len_fallback=output_len_fallback, prompt_len=prompt_len)


def _is_json_content_type(content_type: str) -> bool:
    media_type = content_type.partition(";")[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _parse_openai(
    events: list[Any],
    record: RequestRecord,
    *,
    output_len_fallback: int,
    prompt_len: int,
) -> None:
    ok, classifier = classify_openai_events(events)
    record.success = ok
    record.classifier = classifier
    record.prompt_len = prompt_len
    record.token_timing_available = False
    if any(not isinstance(event, dict) for event in events):
        return

    itl: list[float] = []
    last_text_time: float | None = None
    image_since_last_text = False
    output_len = output_len_fallback
    prompt_tokens: int | None = None
    completion_tokens_from_usage = False
    image_parts: list[dict[str, Any]] = []
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
                completion_tokens_from_usage = True
            if isinstance(usage.get("prompt_tokens"), int):
                prompt_tokens = int(usage["prompt_tokens"])
            image_steps = _image_steps_per_image(usage)
            if image_steps is not None:
                record.image_steps = image_steps
        _capture_cached_prompt_tokens(record, event)
        content = openai_delta_text(event)
        images = openai_delta_images(event)
        timestamp = event.get("_client_t")
        if content:
            record.token_timing_available = True
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
            image_parts.extend(images)
            image_since_last_text = True
            if timestamp is not None:
                timestamp_f = float(timestamp)
                if record.first_image_latency is None:
                    record.first_image_latency = timestamp_f - record.start_time
                    record.first_image_done_time = timestamp_f
                record.image_latencies.extend([timestamp_f - record.start_time] * len(images))

    if prompt_tokens is not None:
        record.prompt_len = prompt_tokens
        record.prompt_len_source = "server_usage"
    record.output_len = output_len
    if completion_tokens_from_usage:
        record.output_len_source = "server_usage"
    record.itl = itl
    image_error = _decode_record_images(image_parts, record)
    if image_error is not None:
        record.success = False
        record.classifier = image_error
        record.error = image_error


def _decode_record_images(parts: list[dict[str, Any]], record: RequestRecord) -> str | None:
    try:
        decoded = decode_openai_image_parts(parts)
    except ImageOutputError as error:
        return error.classifier
    record.decoded_images = decoded
    record.images = len(decoded)
    return None


def _capture_cached_prompt_tokens(record: RequestRecord, payload: dict[str, Any]) -> None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return
    cached = details.get("cached_tokens")
    if _is_token_count(cached):
        record.cached_prompt_tokens = int(cached)
        record.cached_prompt_tokens_source = "openai_usage_prompt_tokens_details"


def _image_steps_per_image(usage: dict[str, Any]) -> list[int] | None:
    value = usage.get("image_steps_per_image")
    if not isinstance(value, list) or any(not _is_token_count(step) for step in value):
        return None
    return [int(step) for step in value]


def _is_token_count(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _last_event_time(events: list[Any]) -> float:
    times = [
        float(event["_client_t"])
        for event in events
        if isinstance(event, dict) and event.get("_client_t") is not None
    ]
    return max(times) if times else time.perf_counter()
