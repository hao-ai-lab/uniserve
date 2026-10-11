"""A worker that refuses a request reports why in the call's completion.

The engine returns the stated reason to the client as an invalid-request
rejection, so a refusal's completion carries it on the wire.
"""

import pytest

from uniserve_worker.errors import ErrorCode
from uniserve_worker.protocol.call import CallStatus
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.output import (
    CallOutput,
    FinishFlags,
    TimingCounters,
)

pytestmark = pytest.mark.unit

REASON = "media input exceeds the worker's condition capacity"


def _completion(error_code, error_message):
    return CallOutput(
        request_key=RequestKey(engine_id=0, request_id=7, request_epoch=1),
        call_id=CallId(3, 0),
        status=CallStatus.ERROR,
        error_code=error_code,
        error_message=error_message,
        timing_counters=TimingCounters(),
        kv_visible_len=0,
        kv_computed_len=0,
        num_completed_steps=0,
        committed_tokens=(),
        finish_flags=FinishFlags(),
    )


def test_a_refusal_carries_its_reason_across_the_wire():
    refused = _completion(ErrorCode.INVALID_REQUEST, REASON)
    mapping = refused.to_mapping()
    assert mapping["error_code"] is ErrorCode.INVALID_REQUEST
    assert mapping["error_message"] == REASON
