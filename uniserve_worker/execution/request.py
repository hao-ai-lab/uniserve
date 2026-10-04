"""Numerical progress values and the native request lifecycle interface.

Request and RequestPool own admission, dependencies, accepted progress, and
retirement in Rust. Numerical routines exchange immutable progress values and
retain their tensor-backed diffusion state through the request object.
"""

from __future__ import annotations

from dataclasses import dataclass

from uniserve_worker._uniserve_ipc import Request, RequestPool
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.call import CallStatus
from uniserve_worker.protocol.identity import CallId, RequestKey

__all__ = ["Request", "RequestPool", "RequestProgress", "RequestResult"]


@dataclass(frozen=True, slots=True)
class RequestProgress:
    """Accepted or projected coordinates for a state-consuming call.

    KV lengths count tokens, and `flow_step` counts the completed solver
    steps of a diffusion trajectory. Construction rejects negative
    coordinates and a visible KV length outside `[0, kv_computed_len]`.
    """

    logical_position: int = 0
    rng_counter: int = 0
    flow_step: int = 0
    kv_visible_len: int = 0
    kv_computed_len: int = 0
    prompt_logits_ready: bool = False

    def __post_init__(self) -> None:
        if (
            self.logical_position < 0
            or self.rng_counter < 0
            or self.flow_step < 0
        ):
            raise invalid_descriptor(
                "request execution coordinates are negative"
            )
        if not 0 <= self.kv_visible_len <= self.kv_computed_len:
            raise invalid_descriptor("request KV extents are not contained")


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
