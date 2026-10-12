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
from collections.abc import AsyncIterable, Iterable, Iterator
from json import JSONDecodeError
from typing import Any, Literal

# "raise" propagates JSONDecodeError to the caller; "record" replaces the
# payload with a {"type": "parse_error", ...} event so the stream continues.
ParseErrorPolicy = Literal["raise", "record"]


class SseParser:
    """Incrementally assembles data lines into decoded SSE events.

    One parser holds the pending data lines of a single stream. Callers feed
    lines in arrival order and call ``finish`` once when the stream ends.

    Args:
        stamp_time: Add ``_client_t`` to each decoded dict event.
        on_parse_error: Policy for payloads that are not valid JSON.
    """

    def __init__(
        self,
        *,
        stamp_time: bool = False,
        on_parse_error: ParseErrorPolicy = "raise",
    ) -> None:
        """Configure timestamps and parse-error handling."""
        self.stamp_time = stamp_time
        self.on_parse_error = on_parse_error
        self._data_lines: list[str] = []

    def feed(self, line: str) -> Any | None:
        """Consume one framing line and return the event it completes, if any.

        A blank line dispatches the buffered data; every other line completes
        no event.

        Args:
            line: One stream line without its line terminator.

        Returns:
            The decoded event, or ``None`` when this line completed no event.

        Raises:
            json.JSONDecodeError: A dispatched payload is not valid JSON and
                the parse-error policy is ``"raise"``.
        """
        if line == "":
            return self._flush()
        if not line.startswith("data:"):
            # Comment lines and every field other than `data` are ignored.
            return None

        # SSE strips a single leading space from the field value, if present.
        data = line.removeprefix("data:")
        if data.startswith(" "):
            data = data[1:]
        self._data_lines.append(data)
        return None

    def finish(self) -> Any | None:
        """Flush a final event whose dispatching blank line never arrived.

        Returns and raises as ``feed`` does for a blank line.
        """
        return self._flush()

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


def iter_sse_events(
    lines: Iterable[str],
    *,
    stamp_time: bool = False,
    on_parse_error: ParseErrorPolicy = "raise",
) -> Iterator[Any]:
    """Yield decoded events from a synchronous line iterable.

    ``lines`` is drained and any unterminated trailing event is flushed.
    Arguments are as for ``SseParser``, and a parse error under the
    ``"raise"`` policy propagates from the generator.
    """
    parser = SseParser(stamp_time=stamp_time, on_parse_error=on_parse_error)
    for line in lines:
        event = parser.feed(line)
        if event is not None:
            yield event
    event = parser.finish()
    if event is not None:
        yield event


async def aiter_sse_events(
    lines: AsyncIterable[str],
    *,
    stamp_time: bool = False,
    on_parse_error: ParseErrorPolicy = "raise",
) -> list[Any]:
    """Collect decoded events from an asynchronous line iterable.

    Events are returned as a list, but each ``_client_t`` stamp is taken as
    its line is read from ``lines``, not when the list is returned. Flush
    and parse-error behavior match ``iter_sse_events``.
    """
    parser = SseParser(stamp_time=stamp_time, on_parse_error=on_parse_error)
    events: list[Any] = []
    async for line in lines:
        event = parser.feed(line)
        if event is not None:
            events.append(event)
    event = parser.finish()
    if event is not None:
        events.append(event)
    return events
