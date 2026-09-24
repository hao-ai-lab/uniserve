"""Frames and decodes Server-Sent Events from chat completion streams.

The parser implements a subset of SSE framing: only ``data:`` fields are
read, comment lines and every other field are ignored, consecutive data lines
are joined with newlines, and a blank line dispatches the buffered event.
Each dispatched payload is decoded as JSON, except the ``[DONE]`` sentinel,
which becomes ``{"type": "sse_done"}``.

With ``stamp_time`` enabled, decoded dict events carry a ``_client_t`` key
holding the ``time.perf_counter()`` reading at which the event completed. The
streaming chat transport in ``uniserve_eval.transport.client`` derives TTFT,
inter-token latency, image arrival latency, and request close time from these
stamps, and ``OpenAIChat.classify_events`` interprets the ``sse_done`` and
``parse_error`` marker events produced here.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterable, Callable, Iterable, Iterator
from json import JSONDecodeError
from typing import Any, Literal

# "raise" propagates JSONDecodeError to the caller; "record" replaces the
# payload with a {"type": "parse_error", ...} event so the stream continues.
ParseErrorPolicy = Literal["raise", "record"]

# Returned by the early-completion probe when buffered data is not yet valid
# JSON; distinct from None, which is itself a valid decoded JSON value.
_INCOMPLETE = object()


class SseParser:
    """Incrementally assembles data lines into decoded SSE events.

    One parser holds the pending data lines of a single stream. Callers feed
    lines in arrival order and call ``finish`` once when the stream ends
    without a stop event.

    Args:
        stamp_time: Add ``_client_t`` to each decoded dict event.
        on_parse_error: Policy for payloads that are not valid JSON.
        stop_on: A predicate over decoded events, or a set of ``type``
            values that mark a terminal dict event. ``None`` never stops.
    """

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
        # Without a stop condition, events complete only at a blank line or
        # at `finish`; the per-line JSON probe in `feed` is skipped.
        self._probe_complete = stop_on is not None
        self._data_lines: list[str] = []

    def feed(self, line: str) -> tuple[Any | None, bool]:
        """Consume one framing line and return any completed event and stop state.

        A blank line dispatches the buffered data. When a stop condition is
        configured, the buffered data is also probed as complete JSON after
        each data line, so a terminal event is returned as soon as its data
        arrives, without waiting for the dispatching blank line. A probe that
        decodes a non-terminal event leaves the buffer for the normal
        dispatch.

        Args:
            line: One stream line without its line terminator.

        Returns:
            The decoded event, or ``None`` when this line completed no event,
            and whether that event satisfies the stop condition.

        Raises:
            json.JSONDecodeError: A dispatched payload is not valid JSON and
                the parse-error policy is ``"raise"``.
        """  # noqa: E501
        if line == "":
            event = self._flush()
            return event, event is not None and self.stop(event)
        if line.startswith(":"):
            return None, False
        if not line.startswith("data:"):
            return None, False

        # SSE strips a single leading space from the field value, if present.
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
        # The terminal event is consumed here, so its trailing blank line
        # later finds an empty buffer and dispatches nothing.
        self._data_lines = []
        return event, True

    def finish(self) -> tuple[Any | None, bool]:
        """Flush a final event whose dispatching blank line never arrived.

        Returns and raises as ``feed`` does for a blank line.
        """
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
    """Decode one complete payload under the configured error policy.

    Returns the ``sse_done`` marker for ``[DONE]``, the decoded JSON value
    otherwise, or a ``parse_error`` event under the ``"record"`` policy. With
    ``stamp_time`` set, the ``sse_done`` marker and decoded dict values carry
    ``_client_t`` and other JSON values do not; ``parse_error`` events carry
    it regardless of ``stamp_time``.
    """
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
    """Probe buffered data without treating incomplete JSON as an error.

    Returns ``_INCOMPLETE`` when the data does not yet decode; otherwise
    decodes exactly as ``_decode_event`` does.
    """
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
    """Yield decoded events from a synchronous line iterable.

    Iteration of ``lines`` stops right after the first event that satisfies
    ``stop_on``; otherwise ``lines`` is drained and any unterminated trailing
    event is flushed. Arguments are as for ``SseParser``, and a parse error
    under the ``"raise"`` policy propagates from the generator.
    """
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
    """Collect decoded events from an asynchronous line iterable.

    Events are returned as a list, but each ``_client_t`` stamp is taken as
    its line is read from ``lines``, not when the list is returned. Stop,
    flush, and parse-error behavior match ``iter_sse_events``.
    """
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
