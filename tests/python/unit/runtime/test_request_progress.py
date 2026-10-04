"""Explicit request results preserve progress and request-epoch ownership."""

from dataclasses import replace

import pytest

from uniserve_worker.errors import WorkerError
from uniserve_worker.execution.request import (
    RequestPool,
    RequestProgress,
    RequestResult,
)
from uniserve_worker.protocol.batch import DiffusionParams, NewRequest
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    CallStatus,
    ImageParams,
    MediaCall,
    TransferMode,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.video import VideoAdmission, VideoTask

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


def test_conflicting_pending_calls_do_not_partially_commit():
    key = RequestKey(1, 7, 1)
    pool = RequestPool(1)
    try:
        pool.start(NewRequest(key, 1, image=ImageParams(height=16, width=16)))
        first = _call(key, 1)
        with pytest.raises(RuntimeError, match="repeats an executing call"):
            pool.add_pending((first, first))
        assert pool.retirement_ready(key)

        pool.add_pending((first,))
        second = _call(key, 2)
        with pytest.raises(RuntimeError, match="repeats an executing call"):
            pool.add_pending((second, first))
        pool.apply_result(
            RequestResult(key, first.call_id, CallStatus.OK, None)
        )
        assert pool.retirement_ready(key)
    finally:
        pool.close()


@pytest.mark.parametrize("commit_order", ((1, 2), (2, 1)))
def test_predecessor_follows_committed_state_and_skips_independent_media(
    commit_order,
):
    key = RequestKey(1, 7, 1)
    pool = RequestPool(1)
    try:
        pool.start(
            NewRequest(
                key,
                1,
                diffusion=DiffusionParams(1, 1, 3, 0, 16, 16),
                prompt_token_ids=(1,),
                video=VideoAdmission(VideoTask.T2VA, (1,)),
            )
        )
        first, second = (_call(key, batch) for batch in commit_order)
        third = _call(key, 3)
        copy = replace(third, kind=TransferMode.TENSOR)
        assert pool.predecessors((first,)) == {first.call_id: CallId(0, 0)}

        pool.add_pending((first, second))
        assert pool.predecessors((third,)) == {third.call_id: second.call_id}
        assert pool.predecessors((copy,)) == {copy.call_id: None}
        pool.apply_result(
            RequestResult(key, second.call_id, CallStatus.PREDICATED, None)
        )
        assert pool.predecessors((third,)) == {third.call_id: first.call_id}
        pool.apply_result(
            RequestResult(key, first.call_id, CallStatus.OK, None)
        )
        assert pool.predecessors((third,)) == {third.call_id: first.call_id}
    finally:
        pool.close()


def test_admission_preserves_live_slot_and_reuses_only_retired_state():
    key = RequestKey(1, 7, 1)
    admission = NewRequest(key, 1, image=ImageParams(height=16, width=16))
    successor = replace(admission, request_key=RequestKey(1, 8, 2))
    pool = RequestPool(1)
    try:
        assert pool.start(admission) == 1
        first = _call(key, 1)
        pool.add_pending((first,))
        progress = RequestProgress(flow_step=1, rng_counter=11)
        pool.apply_result(
            RequestResult(key, first.call_id, CallStatus.OK, progress)
        )
        assert pool.start(admission) is None
        assert pool.get(key.request_id).accepted_progress == progress
        assert pool.get(key.request_id).rng_counter == 11
        assert pool.has_open_requests()

        pool.finish(key)
        assert not pool.has_open_requests()
        with pytest.raises(WorkerError, match="occupied"):
            pool.start(successor)
        pool.retire(key.request_id)
        assert pool.start(successor) == 1
        assert pool.request_ids() == (successor.request_key.request_id,)
        assert pool.get(successor.request_key.request_id).accepted_progress == (
            RequestProgress()
        )
    finally:
        pool.close()
