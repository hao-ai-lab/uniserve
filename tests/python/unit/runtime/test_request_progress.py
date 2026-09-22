"""Explicit request results preserve progress and request-epoch ownership."""

from dataclasses import replace

import pytest

from uniserve_worker.execution.request import (
    RequestPool,
    RequestProgress,
    RequestResult,
)
from uniserve_worker.protocol.batch import NewRequest
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    CallStatus,
    ImageParams,
    MediaCall,
)
from uniserve_worker.protocol.identity import CallId, RequestKey

pytestmark = pytest.mark.unit


def _call(key: RequestKey, batch_id: int) -> Call:
    return Call(
        request_key=key,
        call_id=CallId(batch_id, 0),
        coordinates=CallCoordinates(),
        kind=MediaCall.DENOISING,
        bounds=Bounds(),
    )


def test_late_result_cannot_regress_progress_or_retire_pending_work():
    key = RequestKey(1, 7, 1)
    pool = RequestPool(1)
    try:
        pool.start(NewRequest(key, 1, image=ImageParams(height=16, width=16)))
        first, second = _call(key, 1), _call(key, 2)
        pool.add_pending((first, second))
        pool.finish(key)
        assert not pool.retirement_ready(key)

        accepted = RequestProgress(flow_step=2)
        pool.apply_result(
            RequestResult(key, second.call_id, CallStatus.OK, accepted)
        )
        assert pool.get(key.request_id).accepted_progress == accepted
        assert not pool.retirement_ready(key)

        pool.apply_result(
            RequestResult(
                key, first.call_id, CallStatus.OK, RequestProgress(flow_step=1)
            )
        )
        assert pool.get(key.request_id).accepted_progress == accepted
        assert pool.retirement_ready(key)
        pool.retire(key.request_id)
    finally:
        pool.close()


def test_cancelled_epoch_cannot_change_a_reused_request_slot():
    key = RequestKey(1, 7, 1)
    admission = NewRequest(key, 1, image=ImageParams(height=16, width=16))
    pool = RequestPool(1)
    try:
        pool.start(admission)
        cancelled = _call(key, 1)
        pool.add_pending((cancelled,))
        pool.cancel_calls((cancelled,))
        assert pool.get(key.request_id).closed
        assert pool.retirement_ready(key)
        pool.retire(key.request_id)

        successor = RequestKey(1, 7, 2)
        pool.start(replace(admission, request_key=successor))
        pool.apply_result(
            RequestResult(
                key,
                cancelled.call_id,
                CallStatus.ERROR,
                RequestProgress(flow_step=3),
            )
        )
        pool.cancel_calls((cancelled,))
        current = pool.get(successor.request_id)
        assert not current.closed
        assert current.accepted_progress == RequestProgress()
        assert pool.bind_calls((_call(successor, 2),), (1,)) == (current,)
    finally:
        pool.close()
