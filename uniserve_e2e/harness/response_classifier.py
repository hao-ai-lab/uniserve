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
    if not any(_openai_delta_text(event) for event in events):
        return False, "protocol_empty_text"
    return True, "ok"


def classify_json_image_response(payload: dict[str, Any]) -> tuple[bool, str]:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return False, "protocol_empty_image_data"
    first = data[0]
    if not isinstance(first, dict) or not first.get("b64_json"):
        return False, "protocol_missing_image_payload"
    return True, "ok"


def _has_finish_reason(event: dict[str, Any]) -> bool:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return False
    return any(isinstance(choice, dict) and choice.get("finish_reason") is not None for choice in choices)


def _openai_delta_text(event: dict[str, Any]) -> str:
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
        text = choice.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)
