from __future__ import annotations

from typing import Any

# --- Native SSE event vocabulary (single source of truth) ---------------------
#
# The canonical event "type" strings and the terminal-event set are produced by
# the production emitter in `crates/frontend/native-api/src/events.rs`
# (`event_json` for the type strings, `is_terminal` for which are terminal).
# These constants mirror that vocabulary so the benchmark classifier here and
# the Rust benchmark classifier in
# `crates/support/benchmarks/src/native_events.rs` agree on exactly one
# contract. If the production vocabulary changes (e.g. a type is renamed), both
# copies must be updated together; the parity test in native_events.rs and the
# assertions below guard the most load-bearing strings.
NATIVE_EVENT_TYPES = frozenset(
    {
        "scheduled",
        "text",
        "logprobs",
        "image_begin",
        "image_step",
        "image_done",
        "finished",
        "rejected",
        "error",
    }
)

# Terminal events (mirror of native-api `is_terminal`): a native stream is
# terminated by exactly one of these. Only "finished" is a success terminal;
# "rejected" and "error" are failure terminals.
NATIVE_FINISHED = "finished"
NATIVE_REJECTED = "rejected"
NATIVE_ERROR = "error"
NATIVE_TERMINAL_TYPES = frozenset({NATIVE_FINISHED, NATIVE_REJECTED, NATIVE_ERROR})


def classify_native_events(events: list[dict[str, Any]]) -> tuple[bool, str]:
    if not events:
        return False, "protocol_empty_response"
    if any(event.get("type") == NATIVE_ERROR for event in events):
        return False, "model_error"
    if any(event.get("type") == NATIVE_REJECTED for event in events):
        return False, "unsupported_contract"
    finished = sum(1 for event in events if event.get("type") == NATIVE_FINISHED)
    if finished == 0:
        return False, "protocol_missing_terminal"
    # Production emits exactly one terminal `finished` per stream; more than one
    # is a protocol violation. This matches the Rust classifier's `finished == 1`
    # rule in crates/support/benchmarks/src/native_events.rs.
    if finished > 1:
        return False, "protocol_duplicate_terminal"
    return True, "ok"


def classify_openai_events(events: list[dict[str, Any]]) -> tuple[bool, str]:
    if not events:
        return False, "protocol_empty_response"
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
    """Text content of one non-streamed chat.completion message."""
    raw = (message or {}).get("content")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(
            part.get("text", "")
            for part in raw
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def openai_message_images(message: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Image parts of one non-streamed chat.completion message.

    Images arrive either under ``message.images`` (UniServe native chat,
    LightLLM V2) or as image-typed content parts.
    """
    images: list[dict[str, Any]] = []
    if not isinstance(message, dict):
        return images
    direct = message.get("images")
    if isinstance(direct, list):
        images.extend(part for part in direct if isinstance(part, dict))
    content = message.get("content")
    if isinstance(content, list):
        images.extend(
            part
            for part in content
            if isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part)
        )
    return images


def _has_finish_reason(event: dict[str, Any]) -> bool:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return False
    return any(isinstance(choice, dict) and choice.get("finish_reason") is not None for choice in choices)


def openai_delta_text(event: dict[str, Any]) -> str:
    """Concatenated text of one streamed chat chunk (content + reasoning)."""
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
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


def openai_delta_images(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Image parts of one streamed chat chunk (delta.images or content parts)."""
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
                if isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part)
            )
    return images
