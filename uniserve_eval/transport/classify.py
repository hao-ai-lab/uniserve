from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def classify_openai_events(events: Sequence[Any]) -> tuple[bool, str]:
    if not events:
        return False, "protocol_empty_response"
    if any(not isinstance(event, dict) for event in events):
        return False, "protocol_invalid_event"
    if any(event.get("type") == "parse_error" for event in events):
        return False, "protocol_invalid_event"
    if any(isinstance(event.get("error"), dict) for event in events):
        return False, "model_error"
    if not any(event.get("type") == "sse_done" for event in events) and not any(
        _has_finish_reason(event) for event in events
    ):
        return False, "protocol_missing_terminal"
    if not any(openai_delta_text(event) or openai_delta_images(event) for event in events):
        return False, "protocol_empty_output"
    return True, "ok"


def classify_json_image_response(payload: dict[str, Any]) -> tuple[bool, str]:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return False, "protocol_empty_image_data"
    first = data[0]
    if not isinstance(first, dict) or not first.get("b64_json"):
        return False, "protocol_missing_image_payload"
    return True, "ok"


def openai_message_text(message: dict[str, Any] | None) -> str:
    message = message or {}
    parts: list[str] = []
    reasoning = _reasoning_text(message)
    if reasoning:
        parts.append(reasoning)
    raw = message.get("content")
    if isinstance(raw, str):
        parts.append(raw)
    elif isinstance(raw, list):
        parts.extend(
            part.get("text", "")
            for part in raw
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return "".join(parts)


def openai_message_images(message: dict[str, Any] | None) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    if not isinstance(message, dict):
        return images
    direct = message.get("images")
    if isinstance(direct, list):
        images.extend(part for part in direct if isinstance(part, dict))
    content = message.get("content")
    if isinstance(content, list):
        images.extend(
            part for part in content if isinstance(part, dict) and _is_image_part(part)
        )
    return images


def openai_delta_text(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict):
            reasoning = _reasoning_text(delta)
            if reasoning:
                parts.append(reasoning)
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
        text = choice.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def openai_delta_images(event: dict[str, Any]) -> list[dict[str, Any]]:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
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
                part for part in content if isinstance(part, dict) and _is_image_part(part)
            )
    return images


def _has_finish_reason(event: dict[str, Any]) -> bool:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(choice, dict) and choice.get("finish_reason") is not None for choice in choices
    )


def _is_image_part(part: dict[str, Any]) -> bool:
    return part.get("type") == "image_url" or "image_url" in part or "b64_json" in part


def _reasoning_text(message: dict[str, Any]) -> str:
    value = message.get("reasoning_content")
    if not isinstance(value, str):
        value = message.get("reasoning")
    return value if isinstance(value, str) else ""
