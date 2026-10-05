"""Call keys and device continuation policy shared by numerical consumers."""

from __future__ import annotations

from uniserve_worker.protocol.call import Call, ForwardMode
from uniserve_worker.protocol.identity import CallIdentity


def call_identity(call: Call) -> CallIdentity:
    """Return the call's ``(request_key, call_id)`` identity.

    The request key carries the request epoch, so calls of different epochs
    of one request id never share an identity.
    """
    return call.request_key, call.call_id


def device_gated(call: Call) -> bool:
    """Whether a call's completion predicate gates it on the device.

    A canvas step may be queued behind the step before it, predicated on
    that step's completion. The worker keeps the same decision in the
    slot's continuation flag and runs the step as a no-op on the device
    once an earlier step stopped its block (``CanvasRunner.step``), so the
    predicate is never read on the host and the step does not wait for it.
    """
    return call.kind is ForwardMode.TOKEN_DENOISING and call.canvas is not None
