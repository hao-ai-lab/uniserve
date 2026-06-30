"""INV-OUTPUT-ROUNDTRIP: typed forward outputs serialize to wire dicts that
satisfy the matching per-op-kind result schema, and the low-level wire-field
validators reject the malformed shapes they are meant to guard.

Each ``ForwardOutput`` value object (``contracts.outputs``) is built, serialized
via ``to_seq_result()``, then validated against the per-op-kind schema published
in ``contracts.op_kinds`` through ``validate_seq_result(seq_result, op, index)``.
A passing validation means the serialized dict is wire-legal for that op kind.
"""
from __future__ import annotations

import pytest

from uniserve_worker.contracts.caps import CONTROL_KINDS
from uniserve_worker.contracts.op_kinds import (
    COMMIT_GEN,
    COMMIT_WRITEBACK,
    DECODE_UND,
    DENOISE_GEN,
    PREFILL_UND,
    TARGET_VERIFY_UND,
    VAE_ENCODE,
    VIT_ENCODE,
    validate_seq_result,
    wire_int_field,
    wire_str_list,
)
from uniserve_worker.contracts.outputs import (
    CommitOutput,
    DenoiseOutput,
    EncodeOutput,
    FrameOutput,
    TextTokenOutput,
)
from uniserve_worker.foundation.errors import ErrorCode, WorkerError

pytestmark = pytest.mark.contract


# --- INV-OUTPUT-ROUNDTRIP: per-op-kind output -> wire dict -> schema validation ---


@pytest.mark.parametrize("kind", [PREFILL_UND, DECODE_UND, TARGET_VERIFY_UND])
def test_text_token_output_minimal_roundtrips_for_text_kinds(kind):
    """A bare sampled token serializes to the required-only text result dict."""
    output = TextTokenOutput(req_id=11, sampled_token_id=42)

    seq_result = output.to_seq_result()

    assert seq_result == {"req_id": 11, "sampled_token_id": 42}
    # Validates against every text op kind without raising.
    validate_seq_result(seq_result, {"req_id": 11, "kind": kind}, 0)


def test_text_token_output_with_logprobs_roundtrips_and_validates():
    """sampled_logprob and top_logprobs are carried and pass the text schema."""
    output = TextTokenOutput(
        req_id=7,
        sampled_token_id=42,
        sampled_logprob=-0.5,
        top_logprobs=[(7, -0.1), (8, -0.2)],
        num_accepted_tokens=2,
    )

    seq_result = output.to_seq_result()

    assert seq_result == {
        "req_id": 7,
        "sampled_token_id": 42,
        "sampled_logprob": -0.5,
        "top_logprobs": [(7, -0.1), (8, -0.2)],
        "num_accepted_tokens": 2,
    }
    validate_seq_result(seq_result, {"req_id": 7, "kind": DECODE_UND}, 0)


def test_text_token_output_omits_absent_optional_fields():
    """None optionals are dropped from the wire dict (only req_id + required)."""
    output = TextTokenOutput(req_id=3, sampled_token_id=1)

    seq_result = output.to_seq_result()

    assert "sampled_logprob" not in seq_result
    assert "top_logprobs" not in seq_result
    assert "num_accepted_tokens" not in seq_result


def test_denoise_output_roundtrips_and_validates():
    """denoise_done + num_steps_done satisfy the denoise_gen schema."""
    output = DenoiseOutput(req_id=2, denoise_done=True, num_steps_done=3)

    seq_result = output.to_seq_result()

    assert seq_result == {"req_id": 2, "denoise_done": True, "num_steps_done": 3}
    validate_seq_result(seq_result, {"req_id": 2, "kind": DENOISE_GEN}, 0)


@pytest.mark.parametrize("kind", [COMMIT_GEN, COMMIT_WRITEBACK])
def test_commit_output_full_roundtrips_with_locator(kind):
    """A commit result with image + sampled token + locator passes the commit schema."""
    output = CommitOutput(
        req_id=3,
        image_png_b64="iVBORw0KGgo=",
        image_hw=(512, 256),
        sampled_token_id=9,
        num_tokens=4,
        locator="kv://0/0",
    )

    seq_result = output.to_seq_result()

    # image_hw is declared a list-tuple field: serialized as a JSON-style list.
    assert seq_result["image_hw"] == [512, 256]
    assert isinstance(seq_result["image_hw"], list)
    assert seq_result["locator"] == "kv://0/0"
    assert seq_result["sampled_token_id"] == 9
    validate_seq_result(seq_result, {"req_id": 3, "kind": kind}, 0)


def test_commit_output_minimal_drops_all_optionals():
    """An all-None commit output serializes to just req_id and still validates."""
    output = CommitOutput(req_id=3)

    seq_result = output.to_seq_result()

    assert seq_result == {"req_id": 3}
    validate_seq_result(seq_result, {"req_id": 3, "kind": COMMIT_GEN}, 0)


@pytest.mark.parametrize("kind", [VAE_ENCODE, VIT_ENCODE])
def test_encode_output_roundtrips_and_validates(kind):
    """encoder_handle is required; num_tokens (a tail field) is carried optionally."""
    output = EncodeOutput(req_id=4, encoder_handle=77, num_tokens=10)

    seq_result = output.to_seq_result()

    assert seq_result == {"req_id": 4, "encoder_handle": 77, "num_tokens": 10}
    validate_seq_result(seq_result, {"req_id": 4, "kind": kind}, 0)


def test_encode_output_with_image_hw_serializes_tuple_as_list():
    """The optional image_hw tuple is wire-serialized as a list and validates."""
    output = EncodeOutput(req_id=4, encoder_handle=5, image_hw=(8, 8))

    seq_result = output.to_seq_result()

    assert seq_result["image_hw"] == [8, 8]
    assert isinstance(seq_result["image_hw"], list)
    validate_seq_result(seq_result, {"req_id": 4, "kind": VIT_ENCODE}, 0)


def test_frame_output_roundtrips_and_validates():
    """A PostProcess encode_frame result carries cumulative num_tokens."""
    output = FrameOutput(req_id=5, num_tokens=12)

    seq_result = output.to_seq_result()

    assert seq_result == {"req_id": 5, "num_tokens": 12}
    validate_seq_result(seq_result, {"req_id": 5, "kind": "encode_frame"}, 0)


def test_seq_result_req_id_must_match_op_req_id():
    """A serialized result whose req_id disagrees with the op is rejected."""
    seq_result = TextTokenOutput(req_id=2, sampled_token_id=1).to_seq_result()

    with pytest.raises(WorkerError) as exc:
        validate_seq_result(seq_result, {"req_id": 1, "kind": PREFILL_UND}, 0)

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR


# --- wire_int: rejects bool and non-int, enforces minimums ---


@pytest.mark.parametrize("value", [True, False])
def test_wire_int_rejects_bool(value):
    """Booleans are not accepted as integers even though bool subclasses int."""
    with pytest.raises(WorkerError) as exc:
        wire_int_field(value, "field")

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert exc.value.message == "field must be an integer"


@pytest.mark.parametrize("value", ["5", 1.0, None, [1]])
def test_wire_int_rejects_non_int(value):
    """Strings, floats, None, and lists are rejected as non-integers."""
    with pytest.raises(WorkerError) as exc:
        wire_int_field(value, "field")

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert exc.value.message == "field must be an integer"


def test_wire_int_rejects_value_below_minimum():
    """A value below the inclusive lower bound is rejected."""
    with pytest.raises(WorkerError) as exc:
        wire_int_field(0, "field", minimum=1)

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert exc.value.message == "field must be >= 1"


def test_wire_int_accepts_value_at_minimum_boundary():
    """The minimum is inclusive: a value equal to it is returned unchanged."""
    assert wire_int_field(1, "field", minimum=1) == 1


def test_wire_int_returns_value_when_valid():
    """A plain non-negative int passes and is returned as-is."""
    assert wire_int_field(0, "field") == 0
    assert wire_int_field(99, "field") == 99


# --- wire_str_list: rejects bare string, unknown items, and duplicates ---


def test_wire_str_list_rejects_bare_string():
    """A bare string is not a list of strings even though it is a Sequence."""
    with pytest.raises(WorkerError) as exc:
        wire_str_list("copy_blocks", "controls", CONTROL_KINDS)

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert exc.value.message == "controls must be a list of strings"


def test_wire_str_list_rejects_unknown_item():
    """An item outside the allowed set is rejected with its index and value."""
    with pytest.raises(WorkerError) as exc:
        wire_str_list(["copy_blocks", "not_a_control"], "controls", CONTROL_KINDS)

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert exc.value.message == "controls[1] has unknown value 'not_a_control'"


def test_wire_str_list_rejects_duplicate_item():
    """A duplicated allowed item is rejected (no repeats permitted)."""
    with pytest.raises(WorkerError) as exc:
        wire_str_list(["sleep", "sleep"], "controls", CONTROL_KINDS)

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert exc.value.message == "controls contains duplicate value 'sleep'"


def test_wire_str_list_rejects_non_string_item():
    """A non-string element is rejected before the allowed-set check."""
    with pytest.raises(WorkerError) as exc:
        wire_str_list([123], "controls", CONTROL_KINDS)

    assert exc.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert exc.value.message == "controls[0] must be a string"


def test_wire_str_list_accepts_distinct_allowed_items_as_tuple():
    """Distinct allowed items pass and are returned as a tuple in input order."""
    result = wire_str_list(["wake_up", "sleep"], "controls", CONTROL_KINDS)

    assert result == ("wake_up", "sleep")
    assert isinstance(result, tuple)
