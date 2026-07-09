"""Behavioral tests for the runtime resource ledger and the text tensor stager.

Covers two public seams:

* :class:`~uniserve_worker.runtime.resources.ResourceRuntime` -- the per-class /
  per-request acquire/release ledger and its pressure snapshot. Pure
  counts/units, no tensors, so these run on the unit tier.
* :class:`~uniserve_worker.runtime.tensor_staging.TextTensorStager` -- the H2D
  staging ring. The ring cursor and host-buffer reuse/grow semantics are
  observable on the CPU path (a CPU ``.to('cpu')`` is a no-op, so the staged
  tensor's ``data_ptr`` reflects the reused slot buffer); the genuinely
  device-specific CUDA-buffer-reuse variant is gated to the GPU tier.
"""
from __future__ import annotations

import pytest
import torch

from uniserve_worker.contracts.batches import TextBatch
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.runtime.resources import VALID_CLASSES, ResourceRuntime
from uniserve_worker.runtime.tensor_staging import TextTensorStager

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# ResourceRuntime: residency ledger                                           #
# --------------------------------------------------------------------------- #


def test_acquire_accumulates_per_class_per_request_used_and_residency():
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 10})

    rt.acquire("kv_block", req_id=1, units=3)
    rt.acquire("kv_block", req_id=2, units=2)

    assert rt.used("kv_block") == 5
    assert rt.total_active() == 5
    assert sorted(rt.residency(), key=lambda r: r["request_id"]) == [
        {"class": "kv_block", "request_id": 1, "units": 3},
        {"class": "kv_block", "request_id": 2, "units": 2},
    ]


def test_acquire_same_request_twice_sums_into_one_residency_entry():
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 10})

    rt.acquire("kv_block", req_id=1, units=2)
    rt.acquire("kv_block", req_id=1, units=4)

    assert rt.used("kv_block") == 6
    assert rt.residency() == [{"class": "kv_block", "request_id": 1, "units": 6}]


def test_release_class_frees_only_that_class_and_returns_freed_units():
    rt = ResourceRuntime(["kv_block", "scratch"], totals={"kv_block": 10, "scratch": 8})
    rt.acquire("kv_block", req_id=7, units=3)
    rt.acquire("scratch", req_id=7, units=2)

    freed = rt.release_class("kv_block", req_id=7)

    assert freed == 3
    assert rt.used("kv_block") == 0
    # The request's residency in the *other* class is untouched.
    assert rt.used("scratch") == 2
    assert rt.total_active() == 2


def test_release_request_drops_all_classes_for_that_request_only():
    rt = ResourceRuntime(["kv_block", "scratch"], totals={"kv_block": 10, "scratch": 8})
    rt.acquire("kv_block", req_id=7, units=3)
    rt.acquire("scratch", req_id=7, units=2)
    rt.acquire("kv_block", req_id=9, units=4)

    freed = rt.release_request(req_id=7)

    assert freed == 5  # 3 kv_block + 2 scratch for req 7
    assert rt.used("kv_block") == 4  # req 9 untouched
    assert rt.used("scratch") == 0
    assert rt.residency() == [{"class": "kv_block", "request_id": 9, "units": 4}]


def test_total_active_reaches_zero_only_after_every_request_releases():
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 10})
    rt.acquire("kv_block", req_id=1, units=2)
    rt.acquire("kv_block", req_id=2, units=3)

    rt.release_request(req_id=1)
    assert rt.total_active() == 3, "still resident while req 2 holds units"

    rt.release_request(req_id=2)
    assert rt.total_active() == 0


def test_release_of_absent_request_frees_zero_and_leaves_ledger_unchanged():
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 10})
    rt.acquire("kv_block", req_id=1, units=2)

    assert rt.release_class("kv_block", req_id=999) == 0
    assert rt.release_request(req_id=999) == 0
    assert rt.used("kv_block") == 2


def test_acquire_nonpositive_units_is_a_noop():
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 10})
    rt.acquire("kv_block", req_id=1, units=4)

    rt.acquire("kv_block", req_id=1, units=0)
    rt.acquire("kv_block", req_id=2, units=-5)

    assert rt.used("kv_block") == 4
    assert rt.residency() == [{"class": "kv_block", "request_id": 1, "units": 4}]


# --------------------------------------------------------------------------- #
# ResourceRuntime: pressure snapshot                                          #
# --------------------------------------------------------------------------- #


def test_pressure_reports_free_as_total_minus_used_per_class():
    rt = ResourceRuntime(["kv_block", "scratch"], totals={"kv_block": 10, "scratch": 4})
    rt.acquire("kv_block", req_id=1, units=3)

    by_class = {row["class"]: row for row in rt.pressure()}

    assert by_class["kv_block"] == {
        "class": "kv_block",
        "total": 10,
        "used": 3,
        "evictable": 0,
        "free": 7,
    }
    assert by_class["scratch"] == {
        "class": "scratch",
        "total": 4,
        "used": 0,
        "evictable": 0,
        "free": 4,
    }


def test_pressure_free_is_clamped_to_zero_when_class_has_zero_capacity():
    # A class declared but missing an explicit total defaults to zero capacity.
    rt = ResourceRuntime(["kv_block"], totals={})

    (row,) = rt.pressure()

    assert row == {"class": "kv_block", "total": 0, "used": 0, "evictable": 0, "free": 0}


# --------------------------------------------------------------------------- #
# ResourceRuntime: lease / capability violations                             #
# --------------------------------------------------------------------------- #


def test_acquire_beyond_total_raises_lease_violation_and_preserves_residency():
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 5})
    rt.acquire("kv_block", req_id=1, units=4)

    with pytest.raises(WorkerError) as excinfo:
        rt.acquire("kv_block", req_id=2, units=2)

    err = excinfo.value
    assert err.code == ErrorCode.RESOURCE_LEASE_VIOLATION
    assert err.req_id == 2
    assert err.details == {"class": "kv_block", "used": 4, "requested": 2, "total": 5}
    # The over-cap request acquired nothing; prior residency is unchanged.
    assert rt.used("kv_block") == 4
    assert rt.residency() == [{"class": "kv_block", "request_id": 1, "units": 4}]


def test_acquire_exactly_at_total_is_allowed():
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 5})
    rt.acquire("kv_block", req_id=1, units=4)

    rt.acquire("kv_block", req_id=2, units=1)

    assert rt.used("kv_block") == 5


def test_acquire_on_undeclared_class_raises_capability_mismatch():
    # ``encoder_output`` is a VALID_CLASS but was not declared for this runtime,
    # so it is not managed here and acquiring against it is a capability mismatch.
    assert "encoder_output" in VALID_CLASSES
    rt = ResourceRuntime(["kv_block"], totals={"kv_block": 5})

    with pytest.raises(WorkerError) as excinfo:
        rt.acquire("encoder_output", req_id=1, units=1)

    assert excinfo.value.code == ErrorCode.CAPABILITY_MISMATCH


def test_unknown_resource_class_is_filtered_at_construction_and_not_acquirable():
    rt = ResourceRuntime(["kv_block", "not_a_real_class"], totals={"kv_block": 5})

    # The bogus class is dropped (not raised) at construction.
    assert rt.classes == ["kv_block"]
    with pytest.raises(WorkerError) as excinfo:
        rt.acquire("not_a_real_class", req_id=1, units=1)
    assert excinfo.value.code == ErrorCode.CAPABILITY_MISMATCH


# --------------------------------------------------------------------------- #
# TextTensorStager: ring cursor                                               #
# --------------------------------------------------------------------------- #


def test_next_slot_cycles_through_ring_then_wraps_to_the_first_slot():
    stager = TextTensorStager(ring_depth=3)

    slots = [stager.next_slot() for _ in range(4)]

    # The first three slots are distinct backing dicts...
    assert slots[0].buffers is not slots[1].buffers
    assert slots[1].buffers is not slots[2].buffers
    assert slots[0].buffers is not slots[2].buffers
    # ...and the fourth wraps back to the first.
    assert slots[3].buffers is slots[0].buffers


def test_ring_depth_is_clamped_to_at_least_one():
    stager = TextTensorStager(ring_depth=0)

    assert stager.ring_depth == 1
    # With a single slot every next_slot() returns the same backing buffers.
    assert stager.next_slot().buffers is stager.next_slot().buffers


# --------------------------------------------------------------------------- #
# TextTensorStager: host buffer reuse / grow (slot API)                       #
# --------------------------------------------------------------------------- #


def test_long_buffer_reuses_backing_storage_within_capacity():
    slot = TextTensorStager(ring_depth=1).next_slot()

    first = slot.long_buffer("ids", 8, pin=False)
    base_ptr = first.data_ptr()
    smaller = slot.long_buffer("ids", 5, pin=False)
    same = slot.long_buffer("ids", 8, pin=False)

    assert smaller.data_ptr() == base_ptr
    assert smaller.numel() == 5
    assert same.data_ptr() == base_ptr
    assert first.dtype == torch.long


def test_long_buffer_allocates_fresh_storage_when_growing_past_capacity():
    slot = TextTensorStager(ring_depth=1).next_slot()

    first = slot.long_buffer("ids", 8, pin=False)
    base_ptr = first.data_ptr()
    grown = slot.long_buffer("ids", 20, pin=False)

    assert grown.data_ptr() != base_ptr
    assert grown.numel() == 20


def test_int_buffer_is_int32_and_reuses_within_capacity():
    slot = TextTensorStager(ring_depth=1).next_slot()

    first = slot.int_buffer("lens", 6, pin=False)
    base_ptr = first.data_ptr()
    reused = slot.int_buffer("lens", 4, pin=False)

    assert first.dtype == torch.int32
    assert reused.dtype == torch.int32
    assert reused.data_ptr() == base_ptr


def test_bool_buffer_is_bool_and_reuses_within_capacity():
    slot = TextTensorStager(ring_depth=1).next_slot()

    first = slot.bool_buffer("mask", 6, pin=False)
    base_ptr = first.data_ptr()
    reused = slot.bool_buffer("mask", 4, pin=False)

    assert first.dtype == torch.bool
    assert reused.dtype == torch.bool
    assert reused.data_ptr() == base_ptr


def test_distinct_buffer_names_get_independent_storage():
    slot = TextTensorStager(ring_depth=1).next_slot()

    a = slot.long_buffer("input_ids", 8, pin=False)
    b = slot.long_buffer("positions", 8, pin=False)

    assert a.data_ptr() != b.data_ptr()


# --------------------------------------------------------------------------- #
# TextTensorStager: staged ForwardBatch contract (CPU path)                   #
# --------------------------------------------------------------------------- #


def _text_batch(ops):
    return TextBatch.from_ops(ForwardMode.EXTEND, tuple(ops))


def test_stage_text_builds_the_text_core_index_tensors():
    text = _text_batch(
        [
            {"req_id": 1, "token_ids": [10, 11, 12], "pos_range": [0, 3]},
            {"req_id": 2, "token_ids": [20, 21], "pos_range": [5, 7]},
        ]
    )
    slot = TextTensorStager(ring_depth=2).next_slot()

    fb = TextTensorStager(ring_depth=2).stage_text(text, "cpu", stage_slot=slot)

    assert fb.forward_mode is ForwardMode.EXTEND
    assert fb.req_ids == (1, 2)
    assert fb.input_ids.tolist() == [10, 11, 12, 20, 21]
    assert fb.positions.tolist() == [0, 1, 2, 5, 6]
    assert fb.extend_seq_lens.tolist() == [3, 2]
    assert fb.extend_start_loc.tolist() == [0, 3]
    assert fb.extend_prefix_lens.tolist() == [0, 5]
    assert fb.seq_lens.tolist() == [3, 7]
    assert fb.last_token_indices.tolist() == [2, 4]
    assert fb.num_token_non_padded == 5
    assert fb.padded_num_tokens == 5
    assert fb.has_padding is False


def test_stage_text_pads_token_tail_with_zeros_when_padded_num_tokens_given():
    text = _text_batch(
        [
            {"req_id": 1, "token_ids": [10, 11, 12], "pos_range": [0, 3]},
            {"req_id": 2, "token_ids": [20, 21], "pos_range": [5, 7]},
        ]
    )
    stager = TextTensorStager(ring_depth=2)

    fb = stager.stage_text(text, "cpu", stage_slot=stager.next_slot(), padded_num_tokens=8)

    assert fb.input_ids.tolist() == [10, 11, 12, 20, 21, 0, 0, 0]
    assert fb.positions.tolist() == [0, 1, 2, 5, 6, 0, 0, 0]
    assert fb.num_token_non_padded == 5
    assert fb.padded_num_tokens == 8
    assert fb.has_padding is True


def test_stage_text_reuses_slot_buffers_across_calls_on_cpu_path():
    # On the CPU path the staged tensor aliases the reused host slot buffer, so a
    # second stage of the same shape into the same slot keeps the same storage
    # while overwriting it with the new payload.
    stager = TextTensorStager(ring_depth=2)
    slot = stager.next_slot()
    first = stager.stage_text(
        _text_batch([{"req_id": 1, "token_ids": [10, 11, 12], "pos_range": [0, 3]}]),
        "cpu",
        stage_slot=slot,
    )
    in_ptr = first.input_ids.data_ptr()
    pos_ptr = first.positions.data_ptr()

    second = stager.stage_text(
        _text_batch([{"req_id": 2, "token_ids": [40, 41, 42], "pos_range": [2, 5]}]),
        "cpu",
        stage_slot=slot,
    )

    assert second.input_ids.data_ptr() == in_ptr
    assert second.positions.data_ptr() == pos_ptr
    assert second.input_ids.tolist() == [40, 41, 42]
    assert second.positions.tolist() == [2, 3, 4]


def test_stage_text_grows_slot_buffer_to_fresh_storage_for_a_larger_batch():
    stager = TextTensorStager(ring_depth=2)
    slot = stager.next_slot()
    small = stager.stage_text(
        _text_batch([{"req_id": 1, "token_ids": [10, 11], "pos_range": [0, 2]}]),
        "cpu",
        stage_slot=slot,
    )
    small_ptr = small.input_ids.data_ptr()

    big = stager.stage_text(
        _text_batch(
            [{"req_id": 1, "token_ids": list(range(64)), "pos_range": [0, 64]}]
        ),
        "cpu",
        stage_slot=slot,
    )

    assert big.input_ids.data_ptr() != small_ptr
    assert big.input_ids.numel() == 64
    assert big.input_ids.tolist()[:3] == [0, 1, 2]


def test_stage_text_rejects_padded_num_tokens_below_real_token_count():
    text = _text_batch([{"req_id": 1, "token_ids": [10, 11, 12], "pos_range": [0, 3]}])
    stager = TextTensorStager(ring_depth=1)

    with pytest.raises(WorkerError) as excinfo:
        stager.stage_text(text, "cpu", stage_slot=stager.next_slot(), padded_num_tokens=2)

    assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_stage_text_rejects_op_with_zero_tokens():
    text = _text_batch([{"req_id": 1, "token_ids": [], "pos_range": [0, 0]}])
    stager = TextTensorStager(ring_depth=1)

    with pytest.raises(WorkerError) as excinfo:
        stager.stage_text(text, "cpu", stage_slot=stager.next_slot())

    assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR


# --------------------------------------------------------------------------- #
# TextTensorStager: CUDA buffer reuse (GPU tier)                              #
# --------------------------------------------------------------------------- #


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
def test_stage_text_reuses_cuda_device_buffer_within_capacity():
    device = "cuda:0"
    text = _text_batch([{"req_id": 1, "token_ids": [10, 11, 12], "pos_range": [0, 3]}])
    stager = TextTensorStager(ring_depth=1)
    slot = stager.next_slot()

    first = stager.stage_text(text, device, stage_slot=slot)
    base_ptr = first.input_ids.data_ptr()
    second = stager.stage_text(text, device, stage_slot=slot)

    assert first.input_ids.device.type == "cuda"
    assert second.input_ids.data_ptr() == base_ptr
    assert second.input_ids.cpu().tolist() == [10, 11, 12]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
def test_stage_text_grows_cuda_device_buffer_to_fresh_pointer():
    device = "cuda:0"
    stager = TextTensorStager(ring_depth=1)
    slot = stager.next_slot()
    small = stager.stage_text(
        _text_batch([{"req_id": 1, "token_ids": [10, 11], "pos_range": [0, 2]}]),
        device,
        stage_slot=slot,
    )
    small_ptr = small.input_ids.data_ptr()

    big = stager.stage_text(
        _text_batch([{"req_id": 1, "token_ids": list(range(64)), "pos_range": [0, 64]}]),
        device,
        stage_slot=slot,
    )

    assert big.input_ids.data_ptr() != small_ptr
    assert big.input_ids.numel() == 64
    assert big.input_ids.cpu().tolist()[:3] == [0, 1, 2]
