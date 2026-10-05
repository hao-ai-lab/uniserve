"""Explicit request results preserve progress and request-epoch ownership."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import replace
from threading import Event

import pytest
import torch

from uniserve.media import image
from uniserve.model import ConditionRole
from uniserve_worker.errors import WorkerError
from uniserve_worker.execution.diffusion_state import DiffusionState, SlotLadder
from uniserve_worker.execution.host import HostLane
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
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.protocol.video import (
    AudioClip,
    ConditionVision,
    ImageFit,
    MediaLocator,
    VideoAdmission,
    VideoClip,
    VideoCondition,
    VideoTask,
)

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

        initial = pool.get(key.request_id).accepted_progress
        accepted = RequestProgress(flow_step=2)
        pool.apply_result(
            RequestResult(key, second.call_id, CallStatus.OK, accepted)
        )
        assert pool.get(key.request_id).accepted_progress == accepted
        assert initial == RequestProgress()
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


def test_video_admission_retains_all_condition_tracks():
    canvas = image.Config(32, 32)
    source = MediaLocator("condition-media", 64)
    audio = AudioClip(32000, 0, 32000, 32000)
    conditions = (
        VideoCondition(
            ConditionRole.REFERENCE,
            source,
            image=ImageFit(canvas, 0, 0, canvas),
            video=None,
            audio=None,
            vision=ConditionVision((1, 2, 2), 1, ()),
            latent_units=(4,),
            audio_rows=0,
        ),
        VideoCondition(
            ConditionRole.REFERENCE,
            source,
            image=None,
            video=VideoClip(canvas, 0, 2, 1),
            audio=audio,
            vision=ConditionVision((1, 2, 2), 1, (0,)),
            latent_units=(4,),
            audio_rows=1,
        ),
        VideoCondition(
            ConditionRole.REFERENCE,
            source,
            image=None,
            video=None,
            audio=audio,
            vision=None,
            latent_units=(),
            audio_rows=1,
        ),
    )
    video = VideoAdmission(VideoTask.REF2VA, (1,), conditions)
    admission = NewRequest(
        RequestKey(1, 7, 1),
        1,
        diffusion=DiffusionParams(1, 1, 3, 0, 16, 16),
        prompt_token_ids=(1,),
        video=video,
    )
    pool = RequestPool(1)
    try:
        assert pool.start(admission) == 1
        decoded = NewRequest.from_mapping(admission.to_mapping())
        assert pool.start(decoded) is None
        assert pool.get(7).admission.video == video
        # An admission retry must not silently replace the soundtrack that
        # this epoch's numerical consumers already retained.
        changed = replace(conditions[1], audio=replace(audio, start_sample=1))
        replacement = replace(
            video, conditions=(conditions[0], changed, conditions[2])
        )
        with pytest.raises(WorkerError, match="conflicts"):
            pool.start(replace(admission, video=replacement))
    finally:
        pool.close()


def test_diffusion_close_drains_a_failed_host_write():
    release = Event()
    destination = torch.zeros(1)

    def stage():
        if not release.wait(5):
            raise TimeoutError("host staging was not released")
        destination.fill_(7)
        raise RuntimeError("host staging failed after writing")

    lane = HostLane(max_inflight=1, workers=1)
    try:
        staging = lane.reserve().submit(stage)
        state = DiffusionState(
            size=(), schedules={}, slot=SlotLadder(staging=staging)
        )
        with ThreadPoolExecutor(max_workers=1) as tasks:
            closing = tasks.submit(state.close)
            try:
                with pytest.raises(TimeoutError):
                    closing.result(timeout=0.05)
            finally:
                release.set()

            closing.result(timeout=5)
        assert destination.item() == 7
        with pytest.raises(RuntimeError, match="host staging failed"):
            staging.result()
    finally:
        release.set()
        lane.close()


def test_diffusion_close_retires_cancelled_host_staging():
    destination = torch.zeros(1)
    lane = HostLane(max_inflight=1, workers=1)
    try:
        staging = lane.reserve().configure(lambda: destination.fill_(7))
        staging.cancel()
        state = DiffusionState(
            size=(), schedules={}, slot=SlotLadder(staging=staging)
        )
        state.close()
        assert destination.item() == 0
    finally:
        lane.close()
