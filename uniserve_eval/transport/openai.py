"""OpenAI chat completion value extraction and event classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class OpenAIChat:
    @staticmethod
    def classify_events(events: Sequence[Any]) -> tuple[bool, str]:
        if not events:
            return False, "response_empty_response"
        if any(not isinstance(event, dict) for event in events):
            return False, "response_invalid_event"
        if any(event.get("type") == "parse_error" for event in events):
            return False, "response_invalid_event"
        if any(isinstance(event.get("error"), dict) for event in events):
            return False, "model_error"
        if not any(event.get("type") == "sse_done" for event in events) and not any(
            OpenAIChat._has_finish_reason(event) for event in events
        ):
            return False, "response_missing_terminal"
        if not any(
            OpenAIChat.delta_text(event) or OpenAIChat.delta_images(event) for event in events
        ):
            return False, "response_empty_output"
        return True, "ok"

    @staticmethod
    def message_text(message: dict[str, Any] | None) -> str:
        message = message or {}
        parts: list[str] = []
        reasoning = OpenAIChat._reasoning_text(message)
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

    @staticmethod
    def message_images(message: dict[str, Any] | None) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        if not isinstance(message, dict):
            return images
        direct = message.get("images")
        if isinstance(direct, list):
            images.extend(part for part in direct if isinstance(part, dict))
        content = message.get("content")
        if isinstance(content, list):
            images.extend(
                part for part in content if isinstance(part, dict) and OpenAIChat._is_image_part(part)
            )
        return images

    @staticmethod
    def delta_text(event: dict[str, Any]) -> str:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        parts: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                reasoning = OpenAIChat._reasoning_text(delta)
                if reasoning:
                    parts.append(reasoning)
                content = delta.get("content")
                if isinstance(content, str):
                    parts.append(content)
            text = choice.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def delta_images(event: dict[str, Any]) -> list[dict[str, Any]]:
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
                    part
                    for part in content
                    if isinstance(part, dict) and OpenAIChat._is_image_part(part)
                )
        return images

    @staticmethod
    def _has_finish_reason(event: dict[str, Any]) -> bool:
        choices = event.get("choices")
        if not isinstance(choices, list):
            return False
        return any(
            isinstance(choice, dict) and choice.get("finish_reason") is not None
            for choice in choices
        )

    @staticmethod
    def _is_image_part(part: dict[str, Any]) -> bool:
        return part.get("type") == "image_url" or "image_url" in part or "b64_json" in part

    @staticmethod
    def _reasoning_text(message: dict[str, Any]) -> str:
        value = message.get("reasoning_content")
        if not isinstance(value, str):
            value = message.get("reasoning")
        return value if isinstance(value, str) else ""
