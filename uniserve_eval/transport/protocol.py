"""Protocol transports selected by TaskRequest endpoint and stream."""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..types import IMAGES_GENERATIONS, RequestRecord, TaskRequest
from .images import ImageOutputError, decode_openai_image_parts
from .openai import OpenAIChat
from .sse import aiter_sse_events


class ProtocolTransport:
    async def send(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        record: RequestRecord,
        *,
        prompt_len: int = 0,
        output_len_fallback: int = 0,
    ) -> None:
        raise NotImplementedError

    def attach_decoded(
        self,
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


class ImagesGenerationsTransport(ProtocolTransport):
    @staticmethod
    def classify(payload: dict[str, Any]) -> tuple[bool, str]:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return False, "protocol_empty_image_data"
        first = data[0]
        if not isinstance(first, dict) or not first.get("b64_json"):
            return False, "protocol_missing_image_payload"
        return True, "ok"

    async def send(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        record: RequestRecord,
        *,
        prompt_len: int = 0,
        output_len_fallback: int = 0,
    ) -> None:
        response = await client.post(url, json=payload)
        record.note_http(response.status_code)
        record.close_now()
        try:
            data = response.json()
        except Exception:
            record.mark_failure(
                f"transport_status_{response.status_code}",
                response.text[:500],
            )
            return
        body_ok, classifier = self.classify(data)
        transport_ok = response.status_code < 400
        if not transport_ok and classifier == "ok":
            classifier = f"transport_status_{response.status_code}"
        images = data.get("data") if isinstance(data, dict) else None
        image_error: str | None = None
        if body_ok and isinstance(images, list):
            if not all(isinstance(image, dict) for image in images):
                image_error = "protocol_invalid_image_part"
                record.mark_failure(image_error, image_error)
            else:
                image_error = self.attach_decoded(record, images, assign_json_latency=False)
        if image_error is None and body_ok and transport_ok:
            record.mark_success()
            record.attach_images(record.decoded_images, assign_json_latency=True)
        elif image_error is None:
            record.mark_failure(classifier)


class ChatJsonTransport(ProtocolTransport):
    async def send(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        record: RequestRecord,
        *,
        prompt_len: int = 0,
        output_len_fallback: int = 0,
    ) -> None:
        response = await client.post(url, json=payload)
        record.note_http(response.status_code)
        record.close_now()
        try:
            data = response.json()
        except Exception:
            record.mark_failure(
                f"transport_status_{response.status_code}",
                response.text[:500],
            )
            return
        self._apply(data, record, status_code=response.status_code, prompt_len=prompt_len, output_len_fallback=output_len_fallback)

    def _apply(
        self,
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
            record.apply_choice_metadata(choice0)
            message = choice0.get("message")
            content = OpenAIChat.message_text(message if isinstance(message, dict) else None)
            images = OpenAIChat.message_images(message if isinstance(message, dict) else None)
        record.generated_text = content
        record.token_timing_available = False
        image_error = self.attach_decoded(record, images, assign_json_latency=True)
        if isinstance(data, dict):
            usage = data.get("usage")
            if isinstance(usage, dict):
                record.apply_usage(usage)
            record.apply_cached_prompt_tokens(data)
        record.apply_token_fallbacks(prompt_len=prompt_len, output_len_fallback=output_len_fallback)
        transport_ok = status_code < 400
        if image_error is not None:
            return
        if transport_ok and bool(content or record.decoded_images):
            record.mark_success()
            return
        record.mark_failure(
            f"transport_status_{status_code}" if not transport_ok else "empty_completion"
        )


class ChatSseTransport(ProtocolTransport):
    async def send(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        record: RequestRecord,
        *,
        prompt_len: int = 0,
        output_len_fallback: int = 0,
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
                record.mark_failure(
                    "protocol_expected_sse",
                    "stream request received a JSON response",
                )
                return
            events = await aiter_sse_events(
                response.aiter_lines(),
                stamp_time=True,
                on_parse_error="record",
            )
        record.close_at(_last_event_time(events))
        self._apply(events, record, prompt_len=prompt_len, output_len_fallback=output_len_fallback)

    def _apply(
        self,
        events: list[Any],
        record: RequestRecord,
        *,
        prompt_len: int,
        output_len_fallback: int,
    ) -> None:
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

        self.attach_decoded(record, image_parts)


def transport_for(request: TaskRequest) -> ProtocolTransport:
    if request.endpoint == IMAGES_GENERATIONS:
        return ImagesGenerationsTransport()
    if request.stream:
        return ChatSseTransport()
    return ChatJsonTransport()


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
