"""Serves canned HTTP responses on a loopback port for transport tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from aiohttp import web
from aiohttp.test_utils import TestServer

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@asynccontextmanager
async def stub_server(handler: Handler) -> AsyncIterator[str]:
    """Answer every request with `handler`; yield the server's base URL."""
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("")).rstrip("/")
    finally:
        await server.close()
