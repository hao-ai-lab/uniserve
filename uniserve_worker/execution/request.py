"""Numerical progress values and the native request lifecycle interface.

Request and RequestPool own admission, dependencies, accepted progress, and
retirement in Rust. Numerical routines exchange immutable progress values and
retain their tensor-backed diffusion state through the request object.
"""

from __future__ import annotations

from dataclasses import dataclass

from uniserve_worker._uniserve_ipc import Request, RequestPool, RequestProgress
from uniserve_worker.protocol.call import CallStatus
from uniserve_worker.protocol.identity import CallId, RequestKey

__all__ = ["Request", "RequestPool", "RequestProgress", "RequestResult"]


@dataclass(frozen=True, slots=True)
class RequestResult:
    """An observed call status and the progress accepted by its output owner.

    Direct callers apply these values with `RequestPool.apply_result`;
    serving accepts native completion updates in the executor.
    """

    request_key: RequestKey
    call_id: CallId
    status: CallStatus
    progress: RequestProgress | None
