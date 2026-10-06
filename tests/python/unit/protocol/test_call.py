"""Native calls preserve numerical parameters across IPC and processes."""

import pickle

import pytest

from uniserve_worker.errors import WorkerError
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    ForwardMode,
    SamplingState,
)
from uniserve_worker.protocol.identity import CallId, RequestKey

pytestmark = pytest.mark.unit


def test_call_reconstruction_and_replacement_preserve_numerical_inputs():
    call = Call(
        RequestKey(1, 2, 3),
        CallId(7, 0),
        CallCoordinates(),
        ForwardMode.PREFILL,
        Bounds(max_tokens=2),
        input_token_ids=(3, 7),
        sampling_state=SamplingState(allowed_token_ids=()),
    )
    call.validate()
    for restored in (
        Call.from_mapping(call.to_mapping()),
        pickle.loads(pickle.dumps(call)),
    ):
        assert restored == call
        assert restored.input_token_ids == (3, 7)
        assert restored.sampling_state.allowed_token_ids == ()
        assert restored.advances_state

    successor = call.replace(
        call_id=CallId(8, 0), input_token_ids=(9,), sampling_state=None
    )
    assert successor.call_id == CallId(8, 0)
    assert successor.input_token_ids == (9,)
    assert successor.sampling_state is None
    assert successor.bounds == call.bounds
    assert call.input_token_ids == (3, 7)


def test_call_validation_rejects_visible_kv_beyond_initialized_tokens():
    call = Call(
        RequestKey(1, 2, 3),
        CallId(7, 0),
        CallCoordinates(kv_visible_len=3, kv_computed_len=2),
        ForwardMode.PREFILL,
        Bounds(max_tokens=1),
        input_token_ids=(3,),
    )
    with pytest.raises(WorkerError, match="visible KV"):
        call.validate()
    with pytest.raises(WorkerError, match="visible KV"):
        Call.from_mapping(call.to_mapping())
