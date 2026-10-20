"""Issues benchmark video requests and records their observable outputs.

``send_request`` is the single entry point the benchmark runner
(``uniserve_eval.pipeline.run``) uses per request. It sends the request
through the native video transport and folds everything observable into one
``RequestRecord``: HTTP status, success and a stable failure classifier,
client timing, and the encoded media. Video bodies remain encoded until the
runner inspects them after measurement.

All timestamps are ``time.perf_counter`` values on the same clock as
``RequestRecord.start_time``. Requests run on the caller's
``aiohttp.ClientSession``.
"""

from __future__ import annotations

import asyncio

import aiohttp

from ..types import RequestRecord, TaskRequest
from .native_video import receive_video


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    request: TaskRequest,
    request_id: str,
    *,
    task: str,
    prompt_len: int = 0,
    scheduled_time: float | None = None,
    timeout_s: float = 6 * 60 * 60,
) -> RequestRecord:
    """Dispatch one video request and wait for its complete media body.

    ``prompt_len`` is the row's dataset-side prompt token count, recorded
    with the request.

    Returns:
        The closed record. HTTP status failures carry their own classifiers.
        Expiration of the overall logical deadline becomes
        ``request_deadline``; other raised ``Exception`` instances,
        including aiohttp connection errors, become ``transport_failure``.
        Exceptions outside ``Exception``, such as ``asyncio.CancelledError``,
        propagate.
    """
    record = RequestRecord(request_id=request_id, task=task)
    record.begin(endpoint=request.endpoint, scheduled_time=scheduled_time)
    record.base_url = base_url
    record.prompt_len = prompt_len
    try:
        record.requested_seconds = float(
            request.payload["target"]["duration_seconds"]
        )
        record.video_shape = request.video_shape
        # One deadline covers dispatch, polling and the complete media body.
        async with asyncio.timeout(timeout_s):
            await receive_video(session, base_url, request, record)
    except TimeoutError:
        record.close_at(record.start_time + timeout_s)
        record.mark_failure(
            "request_deadline", f"logical request exceeded {timeout_s}s"
        )
    except Exception as error:  # noqa: BLE001 - benchmarks emit structured failures.
        record.mark_transport_exception(error)
    return record
