from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from tests.python.fixtures.http_stub import Handler, stub_server
from uniserve_eval.transport.client import send_request
from uniserve_eval.types import (
    CHAT_COMPLETIONS,
    IMAGES_GENERATIONS,
    RequestRecord,
    TaskRequest,
)

pytestmark = pytest.mark.unit


def _send(
    handler: Handler,
    request: TaskRequest,
    *,
    task: str,
    output_len_fallback: int = 0,
) -> RequestRecord:
    """Send one request to a loopback server that answers with `handler`."""

    async def run() -> RequestRecord:
        async with stub_server(handler) as base_url:
            async with aiohttp.ClientSession() as session:
                return await send_request(
                    session,
                    base_url,
                    request,
                    "req-1",
                    task=task,
                    output_len_fallback=output_len_fallback,
                )

    return asyncio.run(run())


def _event_stream(body: str) -> Handler:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(text=body, content_type="text/event-stream")

    return handler


def test_image_generation_error_body_is_classified_by_status() -> None:
    # UniServe's OpenAI-compatible error body for a failed request.
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "error": {
                    "message": "engine failed",
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
            status=500,
        )

    record = _send(
        handler,
        TaskRequest(IMAGES_GENERATIONS, {"model": "m"}, stream=False),
        task="t2i",
    )

    assert record.success is False
    assert record.status_code == 500
    assert record.classifier == "transport_status_500"
    assert record.images == 0


def test_stream_request_rejects_a_json_response() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    record = _send(
        handler,
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
        task="text",
    )

    assert record.success is False
    assert record.classifier == "response_expected_sse"


def test_chat_json_records_visible_text_and_server_usage() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "caption"},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        )

    record = _send(
        handler,
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=False),
        task="i2t",
    )

    assert record.success is True
    assert record.generated_text == "caption"
    assert record.record_dict()["generated_text"] == "caption"
    assert record.prompt_len_source == "server_usage"
    assert record.output_len_source == "server_usage"
    assert record.token_timing_available is False


def test_stream_records_reasoning_content_as_public_text() -> None:
    body = "".join(
        (
            'data: {"choices":[{"delta":{"reasoning_content":"reason"},'
            '"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":4,'
            '"completion_tokens":1}}\n\n',
            "data: [DONE]\n\n",
        )
    )

    record = _send(
        _event_stream(body),
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
        task="text",
        output_len_fallback=1,
    )

    assert record.success is True
    assert record.classifier == "ok"
    assert record.generated_text == "reason"
    assert record.record_dict()["generated_text"] == "reason"
    assert record.token_timing_available is True
    assert record.prompt_len_source == "server_usage"
    assert record.output_len_source == "server_usage"


def test_stream_event_longer_than_any_line_buffer_is_read_whole() -> None:
    # One committed 256-token canvas, or an encoded image, arrives as a single
    # data line; CRLF framing is accepted as well.
    block = "x" * 300_000
    body = (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": block}}]})
        + "\r\n\r\n"
        + 'data: {"choices":[{"delta":{},"finish_reason":"length"}],'
        '"usage":{"prompt_tokens":4,"completion_tokens":256}}\r\n\r\n'
        + "data: [DONE]\r\n\r\n"
    )

    record = _send(
        _event_stream(body),
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
        task="text",
        output_len_fallback=256,
    )

    assert record.success is True
    assert record.generated_text == block
    assert record.output_len == 256
    assert record.finish_reason == "length"


def test_chat_json_records_reasoning_content_as_public_text() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "reasoning_content": "reason",
                            "content": None,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }
        )

    record = _send(
        handler,
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=False),
        task="i2t",
    )

    assert record.success is True
    assert record.generated_text == "reason"
    assert record.token_timing_available is False


def test_stream_rejects_non_object_events() -> None:
    record = _send(
        _event_stream('data: ["invalid"]\n\ndata: [DONE]\n\n'),
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
        task="text",
    )

    assert record.success is False
    assert record.classifier == "response_invalid_event"
    assert record.token_timing_available is False
