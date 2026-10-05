"""Call keys shared by numerical consumers."""

from __future__ import annotations

from uniserve_worker.protocol.call import Call
from uniserve_worker.protocol.identity import CallIdentity


def call_identity(call: Call) -> CallIdentity:
    """Return the call's ``(request_key, call_id)`` identity.

    The request key carries the request epoch, so calls of different epochs
    of one request id never share an identity.
    """
    return call.request_key, call.call_id
