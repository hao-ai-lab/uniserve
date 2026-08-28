"""Issue benchmark HTTP requests and record their outputs."""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..types import IMAGES_GENERATIONS, VIDEOS_SYNC, RequestRecord, TaskRequest
from .images import ImageOutputError, decode_openai_image_parts
from .openai import OpenAIChat
from .sse import aiter_sse_events
from .video import VideoOutputError, inspect_video_bytes


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
    record.begin(
        endpoint=request.endpoint,
        scheduled_time=scheduled_time,
        requested_output_len=int(output_len_fallback),
    )
    try:
        if request.endpoint == VIDEOS_SYNC:
            await _send_video(client, url, payload, record)
        elif request.endpoint == IMAGES_GENERATIONS:
            await _send_images(client, url, payload, record)
        elif request.stream:
            await _send_chat_stream(client, url, payload, record, prompt_len, output_len_fallback)
        else:
            await _send_chat(client, url, payload, record, prompt_len, output_len_fallback)
    except Exception as error:  # noqa: BLE001 - benchmarks emit structured failures.
        record.mark_transport_exception(error)
    return record


async def _send_images(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
) -> None:
    response = await client.post(url, json=payload)
    record.note_http(response.status_code)
    record.close_now()
    try:
        data = response.json()
    except Exception:
        record.mark_failure(f"transport_status_{response.status_code}", response.text[:500])
        return
    body_ok, classifier = _classify_images(data)
    transport_ok = response.status_code < 400
    if not transport_ok and classifier == "ok":
        classifier = f"transport_status_{response.status_code}"
    images = data.get("data") if isinstance(data, dict) else None
    image_error: str | None = None
    if body_ok and isinstance(images, list):
        if not all(isinstance(image, dict) for image in images):
            image_error = "invalid_image_part"
            record.mark_failure(image_error, image_error)
        else:
            image_error = _attach_images(record, images, assign_json_latency=False)
    if image_error is None and body_ok and transport_ok:
        record.mark_success()
        record.attach_images(record.decoded_images, assign_json_latency=True)
    elif image_error is None:
        record.mark_failure(classifier)


async def _send_chat(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
    prompt_len: int,
    output_len_fallback: int,
) -> None:
    response = await client.post(url, json=payload)
    record.note_http(response.status_code)
    record.close_now()
    try:
        data = response.json()
    except Exception:
        record.mark_failure(f"transport_status_{response.status_code}", response.text[:500])
        return
    choices = data.get("choices") if isinstance(data, dict) else None
    content = ""
    images: list[dict[str, Any]] = []
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        record.apply_choice_metadata(choice)
        message = choice.get("message")
        content = OpenAIChat.message_text(message if isinstance(message, dict) else None)
        images = OpenAIChat.message_images(message if isinstance(message, dict) else None)
    record.generated_text = content
    record.token_timing_available = False
    image_error = _attach_images(record, images, assign_json_latency=True)
    if isinstance(data, dict):
        usage = data.get("usage")
        if isinstance(usage, dict):
            record.apply_usage(usage)
        record.apply_cached_prompt_tokens(data)
    record.apply_token_fallbacks(prompt_len=prompt_len, output_len_fallback=output_len_fallback)
    if image_error is not None:
        return
    if response.status_code < 400 and bool(content or record.decoded_images):
        record.mark_success()
    else:
        record.mark_failure(
            f"transport_status_{response.status_code}"
            if response.status_code >= 400
            else "empty_completion"
        )


async def _send_chat_stream(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
    prompt_len: int,
    output_len_fallback: int,
) -> None:
    async with client.stream("POST", url, json=payload) as response:
        record.note_http(response.status_code)
        if response.status_code != 200:
            body = await response.aread()
            record.close_now()
            record.mark_failure(
                f"transport_status_{response.status_code}",
                body.decode("utf-8", errors="replace")[:500],
            )
            return
        if _is_json_content_type(response.headers.get("content-type", "")):
            await response.aread()
            record.close_now()
            record.mark_failure("response_expected_sse", "stream request received a JSON response")
            return
        events = await aiter_sse_events(
            response.aiter_lines(),
            stamp_time=True,
            on_parse_error="record",
        )
    record.close_at(_last_event_time(events))
    ok, classifier = OpenAIChat.classify_events(events)
    if ok:
        record.mark_success()
    else:
        record.mark_failure(classifier)
    record.prompt_len = prompt_len
    record.output_len = output_len_fallback
    record.token_timing_available = False
    if any(not isinstance(event, dict) for event in events):
        return

    last_text_time: float | None = None
    image_since_last_text = False
    image_parts: list[dict[str, Any]] = []
    for event in events:
        choices = event.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    record.apply_choice_metadata(choice)
        usage = event.get("usage")
        if isinstance(usage, dict):
            record.apply_usage(usage)
        record.apply_cached_prompt_tokens(event)
        content = OpenAIChat.delta_text(event)
        images = OpenAIChat.delta_images(event)
        timestamp = event.get("_client_t")
        timestamp_f = float(timestamp) if timestamp is not None else None
        if content:
            record.add_text(
                content,
                timestamp_f,
                last_text_time=last_text_time,
                count_itl=not image_since_last_text,
            )
            if timestamp_f is not None:
                last_text_time = timestamp_f
                image_since_last_text = False
        if images:
            image_parts.extend(images)
            image_since_last_text = True
            record.add_image_arrival(len(images), timestamp_f)
    _attach_images(record, image_parts)


async def _send_video(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
) -> None:
    async with client.stream("POST", url, json=payload) as response:
        record.note_http(response.status_code)
        body = await response.aread()
        record.close_now()
        if response.status_code >= 400:
            record.mark_failure(
                f"transport_status_{response.status_code}",
                body.decode("utf-8", errors="replace")[:500],
            )
            return
        try:
            record.decoded_video = inspect_video_bytes(
                body, declared_mime=response.headers.get("content-type", "")
            )
        except VideoOutputError as error:
            record.mark_failure(error.classifier, str(error))
            return
        record.mark_success()


def _classify_images(payload: Any) -> tuple[bool, str]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        return False, "empty_image_data"
    first = data[0]
    if not isinstance(first, dict) or not first.get("b64_json"):
        return False, "missing_image_payload"
    return True, "ok"


def _attach_images(
    record: RequestRecord,
    parts: list[dict[str, Any]],
    *,
    assign_json_latency: bool = False,
) -> str | None:
    try:
        decoded = decode_openai_image_parts(parts)
    except ImageOutputError as error:
        record.mark_failure(error.classifier, error.classifier)
        return error.classifier
    record.attach_images(decoded, assign_json_latency=assign_json_latency)
    return None


def _is_json_content_type(content_type: str) -> bool:
    media_type = content_type.partition(";")[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _last_event_time(events: list[Any]) -> float:
    times = [
        float(event["_client_t"])
        for event in events
        if isinstance(event, dict) and event.get("_client_t") is not None
    ]
    return max(times) if times else time.perf_counter()
