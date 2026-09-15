"""Frames and decodes Server-Sent Events from chat completion streams."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterable, Callable, Iterable, Iterator
from json import JSONDecodeError
from typing import Any, Literal

ParseErrorPolicy = Literal["raise", "record"]
_INCOMPLETE = object()


class SseParser:
    """Incrementally assembles data lines into decoded SSE events."""

    def __init__(
        self,
        *,
        stamp_time: bool = False,
        on_parse_error: ParseErrorPolicy = "raise",
        stop_on: Callable[[Any], bool] | frozenset[str] | None = None,
    ) -> None:
        """Configure timestamps, parse-error handling, and terminal detection."""  # noqa: E501
        self.stamp_time = stamp_time
        self.on_parse_error = on_parse_error
        self.stop = _make_stop(stop_on)
        self._probe_complete = stop_on is not None
        self._data_lines: list[str] = []

    def feed(self, line: str) -> tuple[Any | None, bool]:
        """Consume one framing line and return any completed event and stop state."""  # noqa: E501
        if line == "":
            event = self._flush()
            return event, event is not None and self.stop(event)
        if line.startswith(":"):
            return None, False
        if not line.startswith("data:"):
            return None, False
        data = line.removeprefix("data:")
        if data.startswith(" "):
            data = data[1:]
        self._data_lines.append(data)
        if not self._probe_complete:
            return None, False
        event = _try_decode_complete_event(
            "\n".join(self._data_lines),
            time.perf_counter(),
            stamp_time=self.stamp_time,
        )
        if event is _INCOMPLETE or not self.stop(event):
            return None, False
        self._data_lines = []
        return event, True

    def finish(self) -> tuple[Any | None, bool]:
        """Flush a final unterminated event at end of stream."""
        event = self._flush()
        return event, event is not None and self.stop(event)

    def _flush(self) -> Any | None:
        """Decode and clear the currently buffered data lines."""
        if not self._data_lines:
            return None
        data = "\n".join(self._data_lines)
        self._data_lines = []
        return _decode_event(
            data,
            time.perf_counter(),
            stamp_time=self.stamp_time,
            on_parse_error=self.on_parse_error,
        )


def _decode_event(
    data: str,
    received: float,
    *,
    stamp_time: bool,
    on_parse_error: ParseErrorPolicy,
) -> dict[str, Any] | Any:
    """Decode one complete payload under the configured error policy."""
    if data == "[DONE]":
        return (
            {"type": "sse_done", "_client_t": received}
            if stamp_time
            else {"type": "sse_done"}
        )
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
    """Probe buffered data without treating incomplete JSON as an error."""
    if data == "[DONE]":
        return (
            {"type": "sse_done", "_client_t": received}
            if stamp_time
            else {"type": "sse_done"}
        )
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
    """Yield decoded events from a synchronous line iterable."""
    parser = SseParser(
        stamp_time=stamp_time, on_parse_error=on_parse_error, stop_on=stop_on
    )
    for line in lines:
        event, done = parser.feed(line)
        if event is not None:
            yield event
            if done:
                return
    event, _done = parser.finish()
    if event is not None:
        yield event


async def aiter_sse_events(
    lines: AsyncIterable[str],
    *,
    stamp_time: bool = False,
    on_parse_error: ParseErrorPolicy = "raise",
    stop_on: Callable[[Any], bool] | frozenset[str] | None = None,
) -> list[Any]:
    """Collect decoded events from an asynchronous line iterable."""
    parser = SseParser(
        stamp_time=stamp_time, on_parse_error=on_parse_error, stop_on=stop_on
    )
    events: list[Any] = []
    async for line in lines:
        event, done = parser.feed(line)
        if event is not None:
            events.append(event)
            if done:
                return events
    event, _done = parser.finish()
    if event is not None:
        events.append(event)
    return events


def _make_stop(
    stop_on: Callable[[Any], bool] | frozenset[str] | None,
) -> Callable[[Any], bool]:
    """Normalize a callback or event-type set into a stop predicate."""
    if stop_on is None:
        return lambda _event: False
    if callable(stop_on):
        return stop_on
    terminal = stop_on

    def stop(event: Any) -> bool:
        """Match dictionary events against configured terminal types."""
        return isinstance(event, dict) and event.get("type") in terminal

    return stop
