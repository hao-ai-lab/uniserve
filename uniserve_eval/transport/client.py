"""HTTP transport for chat completions and image generation."""

from __future__ import annotations

import httpx

from ..types import RequestRecord, TaskRequest
from .protocol import transport_for


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    request: TaskRequest,
    request_id: str,
    *,
    task: str,
    prompt_len: int = 0,
    output_len_fallback: int = 0,
    scheduled_time: float | None = None,
) -> RequestRecord:
    payload = {key: value for key, value in request.payload.items() if value is not None}
    url = base_url.rstrip("/") + request.endpoint
    record = RequestRecord(request_id=request_id, task=task)
    record.begin(
        endpoint=request.endpoint,
        scheduled_time=scheduled_time,
        requested_output_len=int(output_len_fallback),
    )
    try:
        await transport_for(request).send(
            client,
            url,
            payload,
            record,
            prompt_len=prompt_len,
            output_len_fallback=output_len_fallback,
        )
    except Exception as error:  # noqa: BLE001 - benchmarks emit structured failures.
        record.mark_transport_exception(error)
    return record
