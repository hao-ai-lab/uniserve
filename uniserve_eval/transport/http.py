"""Shares the aiohttp request primitives of the endpoint transports.

Every benchmark request runs on one `aiohttp.ClientSession` that
`run_point` opens with an unbounded connector, the client shape of the
SGLang and vLLM serving benchmarks. JSON endpoints read the complete body
before the record closes; streamed endpoints read the body as it arrives and
split it into lines of any length.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import aiohttp

from ..types import RequestRecord


@dataclass(frozen=True)
class ResponseBody:
    """Holds a complete response's status, raw body, and content type."""

    status: int
    data: bytes
    content_type: str

    @property
    def text(self) -> str:
        """Return the body decoded as UTF-8, replacing invalid bytes."""
        return self.data.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """Decode the body as JSON.

        Raises:
            ValueError: The body is not valid JSON or not UTF-8.
        """
        return json.loads(self.data)


async def post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
) -> ResponseBody:
    """POST a JSON payload and read its complete response into `record`.

    The HTTP status is recorded when the response headers arrive and the
    record closes once the whole body has been read, before the caller
    parses it.
    """
    async with session.post(url, json=payload) as response:
        record.note_http(response.status)
        data = await response.read()
        content_type = response.headers.get("content-type", "")
    record.close_now()
    return ResponseBody(response.status, data, content_type)


async def response_lines(
    content: aiohttp.StreamReader,
) -> AsyncIterator[str]:
    """Yield a streamed body's lines, without terminators, as they arrive.

    Lines split at each line feed, with a preceding carriage return removed,
    and may be of any length: an event carrying an encoded image can exceed
    any fixed line buffer. A final unterminated line is yielded at the end
    of the body. Splitting at an ASCII line feed never cuts a UTF-8
    sequence, so each line decodes on its own.
    """
    buffer = bytearray()
    scanned = 0
    async for chunk in content.iter_any():
        buffer.extend(chunk)
        while (newline := buffer.find(b"\n", scanned)) >= 0:
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            scanned = 0
            yield line.removesuffix(b"\r").decode("utf-8", errors="replace")
        scanned = len(buffer)
    if buffer:
        yield (
            bytes(buffer).removesuffix(b"\r").decode("utf-8", errors="replace")
        )
