"""Server-Sent Events framing for chat completions streams."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterable, Iterable, Iterator
from json import JSONDecodeError
from typing import Any, Callable, Literal

ParseErrorPolicy = Literal["raise", "record"]
_INCOMPLETE = object()


def _decode_event(
    data: str,
    received: float,
    *,
    stamp_time: bool,
    on_parse_error: ParseErrorPolicy,
) -> dict[str, Any] | Any:
    if data == "[DONE]":
        return {"type": "sse_done", "_client_t": received} if stamp_time else {"type": "sse_done"}
    try:
        event = json.loads(data)
    except JSONDecodeError as error:
        if on_parse_error == "raise":
            raise
        return {
            "type": "parse_error",
            "_client_t": received,
            "error": f"invalid SSE JSON payload: {data!r} ({error})",
        }
    if stamp_time and isinstance(event, dict):
        event["_client_t"] = received
    return event


def _try_decode_complete_event(
    data: str,
    received: float,
    *,
    stamp_time: bool,
) -> Any:
    if data == "[DONE]":
        return {"type": "sse_done", "_client_t": received} if stamp_time else {"type": "sse_done"}
    try:
        event = json.loads(data)
    except JSONDecodeError:
        return _INCOMPLETE
    if stamp_time and isinstance(event, dict):
        event["_client_t"] = received
    return event


def iter_sse_events(
    lines: Iterable[str],
    *,
    stamp_time: bool = False,
    on_parse_error: ParseErrorPolicy = "raise",
    stop_on: Callable[[Any], bool] | frozenset[str] | None = None,
) -> Iterator[Any]:
    stop = _make_stop(stop_on)
    data_lines: list[str] = []

    def flush() -> Iterator[Any]:
        nonlocal data_lines
        if not data_lines:
            return
        data = "\n".join(data_lines)
        data_lines = []
        yield _decode_event(
            data,
            time.perf_counter(),
            stamp_time=stamp_time,
            on_parse_error=on_parse_error,
        )

    for line in lines:
        if line == "":
            for event in flush():
                yield event
                if stop(event):
                    return
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:")
        if data.startswith(" "):
            data = data[1:]
        data_lines.append(data)
        if stop_on is not None:
            event = _try_decode_complete_event(
                "\n".join(data_lines),
                time.perf_counter(),
                stamp_time=stamp_time,
            )
            if event is not _INCOMPLETE and stop(event):
                data_lines = []
                yield event
                return
    for event in flush():
        yield event
        if stop(event):
            return


async def aiter_sse_events(
    lines: AsyncIterable[str],
    *,
    stamp_time: bool = False,
    on_parse_error: ParseErrorPolicy = "raise",
    stop_on: Callable[[Any], bool] | frozenset[str] | None = None,
) -> list[Any]:
    stop = _make_stop(stop_on)
    events: list[Any] = []
    data_lines: list[str] = []

    def flush() -> bool:
        nonlocal data_lines
        if not data_lines:
            return False
        data = "\n".join(data_lines)
        data_lines = []
        event = _decode_event(
            data,
            time.perf_counter(),
            stamp_time=stamp_time,
            on_parse_error=on_parse_error,
        )
        events.append(event)
        return stop(event)

    async for line in lines:
        if line == "":
            if flush():
                return events
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:")
        if data.startswith(" "):
            data = data[1:]
        data_lines.append(data)
        if stop_on is not None:
            event = _try_decode_complete_event(
                "\n".join(data_lines),
                time.perf_counter(),
                stamp_time=stamp_time,
            )
            if event is not _INCOMPLETE and stop(event):
                data_lines = []
                events.append(event)
                return events
    flush()
    return events


def _make_stop(
    stop_on: Callable[[Any], bool] | frozenset[str] | None,
) -> Callable[[Any], bool]:
    if stop_on is None:
        return lambda _event: False
    if callable(stop_on):
        return stop_on
    terminal = stop_on

    def stop(event: Any) -> bool:
        return isinstance(event, dict) and event.get("type") in terminal

    return stop
