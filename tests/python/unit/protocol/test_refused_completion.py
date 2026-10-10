"""A worker that refuses a request reports why in the call's completion.

The engine returns the stated reason to the client as an invalid-request
rejection, so a refusal's completion carries it on the wire and no other
completion does.
"""

import pytest

from uniserve_worker.errors import WorkerError
from uniserve_worker.protocol.call import CallStatus, ErrorCode, MediaCall
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.output import (
    FinishFlags,
    RequestOutput,
    TimingCounters,
)

pytestmark = pytest.mark.unit

REASON = "media input exceeds the worker's condition capacity"


def _completion(error_code, error_message):
    return RequestOutput(
        request_key=RequestKey(engine_id=0, request_id=7, request_epoch=1),
        call_id=CallId(3, 0),
        status=CallStatus.ERROR,
        product_generations=(),
        error_code=error_code,
        error_message=error_message,
        timing_counters=TimingCounters(),
        kind=MediaCall.VISION_ENCODING,
        position=0,
        kv_visible_len=0,
        kv_computed_len=0,
        num_completed_steps=0,
        committed_tokens=(),
        finish_flags=FinishFlags(),
    )


def test_a_refusal_carries_its_reason_across_the_wire():
    refused = _completion(ErrorCode.INVALID_REQUEST, REASON)
    refused.validate()
    mapping = refused.to_mapping()
    assert mapping["error_code"] == "invalid_request"
    assert RequestOutput.from_mapping(mapping) == refused


@pytest.mark.parametrize(
    ("error_code", "error_message"),
    [
        (ErrorCode.INVALID_REQUEST, None),
        (ErrorCode.COMPUTE_ERROR, REASON),
    ],
)
def test_only_a_refusal_states_a_reason(error_code, error_message):
    with pytest.raises(WorkerError, match="states why"):
        _completion(error_code, error_message).validate()
