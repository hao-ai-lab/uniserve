from __future__ import annotations

import asyncio

import httpx
import pytest

from uniserve_eval.transport.client import send_request
from uniserve_eval.types import CHAT_COMPLETIONS, TaskRequest

pytestmark = pytest.mark.unit


def test_stream_request_rejects_a_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            headers={"content-type": "application/json"},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            record = await send_request(
                client,
                "http://server",
                TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
                "req-1",
                task="text",
            )
        assert record.success is False
        assert record.classifier == "protocol_expected_sse"

    asyncio.run(run())


def test_chat_json_records_visible_text_and_server_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "caption"},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            record = await send_request(
                client,
                "http://server",
                TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=False),
                "req-1",
                task="i2t",
            )
        assert record.success is True
        assert record.generated_text == "caption"
        assert record.prompt_len_source == "server_usage"
        assert record.output_len_source == "server_usage"
        assert record.token_timing_available is False

    asyncio.run(run())


def test_stream_records_reasoning_content_as_public_text() -> None:
    body = "".join(
        (
            'data: {"choices":[{"delta":{"reasoning_content":"reason"},"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1}}\n\n',
            "data: [DONE]\n\n",
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            record = await send_request(
                client,
                "http://server",
                TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
                "req-1",
                task="text",
                output_len_fallback=1,
            )
        assert record.success is True
        assert record.classifier == "ok"
        assert record.generated_text == "reason"
        assert record.token_timing_available is True
        assert record.prompt_len_source == "server_usage"
        assert record.output_len_source == "server_usage"

    asyncio.run(run())


def test_chat_json_records_reasoning_content_as_public_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"reasoning_content": "reason", "content": None},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            record = await send_request(
                client,
                "http://server",
                TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=False),
                "req-1",
                task="i2t",
            )
        assert record.success is True
        assert record.generated_text == "reason"
        assert record.token_timing_available is False

    asyncio.run(run())


def test_stream_rejects_non_object_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='data: ["invalid"]\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            record = await send_request(
                client,
                "http://server",
                TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
                "req-1",
                task="text",
            )
        assert record.success is False
        assert record.classifier == "protocol_invalid_event"
        assert record.token_timing_available is False

    asyncio.run(run())
