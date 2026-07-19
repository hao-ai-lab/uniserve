"""Contract tests for ForwardStats wire projection and ForwardContext lifecycle.

These pin the small, public behavioral surface of the per-forward statistics
accumulator and the per-forward context var:

* :meth:`ForwardStats.to_wire` converts nanosecond timers to microseconds by
  integer floor division and copies non-timer counters out verbatim;
* the flat forwarding properties on :class:`ForwardStats` round-trip onto the
  nested grouped accumulators;
* :func:`component_timer_start` / :func:`record_component_elapsed` no-op when
  ``stats`` is ``None``;
* :func:`get_forward_context` returns a default empty context when unset;
* :func:`use_forward_context` restores the prior context on normal exit and
  also when the body raises.
"""
from __future__ import annotations

import pytest

from uniserve_worker.contracts import ForwardStats
from uniserve_worker.contracts.forward_context import (
    ForwardContext,
    component_timer_start,
    get_forward_context,
    record_component_elapsed,
    use_forward_context,
)

pytestmark = [pytest.mark.unit]


# --------------------------------------------------------------------------
# ForwardStats.to_wire: ns -> us floor division
# --------------------------------------------------------------------------


def test_to_wire_attention_ns_floor_divided_to_us():
    stats = ForwardStats()
    # 2_500 ns floor-divides to 2 us (not 3, not 2.5).
    stats.attention.ns = 2_500
    stats.attention.launches = 4

    wire = stats.to_wire()

    assert wire["attention_us"] == 2
    assert wire["attention_launches"] == 4


def test_to_wire_attention_ns_below_one_us_floors_to_zero():
    stats = ForwardStats()
    stats.attention.ns = 999

    wire = stats.to_wire()

    assert wire["attention_us"] == 0


def test_to_wire_operator_ns_floor_divided_to_us():
    stats = ForwardStats()
    stats.operators.ns = 7_999
    stats.operators.launches = 3

    wire = stats.to_wire()

    assert wire["operator_us"] == 7
    assert wire["operator_launches"] == 3


def test_to_wire_mode_us_maps_each_mode_ns_by_floor_division():
    stats = ForwardStats()
    stats.mode.mode_ns = {"decode": 3_400, "prefill": 10_999}

    wire = stats.to_wire()

    assert wire["mode_us"] == {"decode": 3, "prefill": 10}


def test_to_wire_component_us_maps_each_component_ns_by_floor_division():
    stats = ForwardStats()
    stats.mode.component_ns = {"attn": 1_000, "mlp": 1_500}

    wire = stats.to_wire()

    assert wire["component_us"] == {"attn": 1, "mlp": 1}


def test_to_wire_copies_non_timer_counts_verbatim():
    stats = ForwardStats()
    stats.mode.mode_counts = {"decode": 5}
    stats.mode.mode_tokens = {"decode": 40}
    stats.attention.backend_counts = {"flashinfer": 2}
    stats.cuda_graph.replays = 9
    stats.spec_verify.accepted_tokens = 11

    wire = stats.to_wire()

    assert wire["mode_counts"] == {"decode": 5}
    assert wire["mode_tokens"] == {"decode": 40}
    assert wire["attention_backend_counts"] == {"flashinfer": 2}
    assert wire["cuda_graph_replays"] == 9
    assert wire["spec_verify_accepted_tokens"] == 11


def test_to_wire_dict_fields_are_independent_copies():
    stats = ForwardStats()
    stats.mode.mode_counts = {"decode": 1}

    wire = stats.to_wire()
    wire["mode_counts"]["decode"] = 999

    # Mutating the projection must not bleed back into the accumulator.
    assert stats.mode.mode_counts == {"decode": 1}


def test_to_wire_default_stats_is_all_zero_and_empty():
    wire = ForwardStats().to_wire()

    assert wire["attention_us"] == 0
    assert wire["attention_launches"] == 0
    assert wire["operator_us"] == 0
    assert wire["mode_us"] == {}
    assert wire["component_us"] == {}
    assert wire["mode_counts"] == {}
    assert wire["cuda_graph_captures"] == 0
    assert wire["spec_verify_rows"] == 0


# --------------------------------------------------------------------------
# Flat forwarding properties round-trip onto nested accumulators
# --------------------------------------------------------------------------


def test_flat_scalar_property_writes_through_to_nested_accumulator():
    stats = ForwardStats()

    stats.attention_launches = 7

    assert stats.attention.launches == 7
    assert stats.attention_launches == 7


def test_flat_dict_property_writes_through_to_nested_accumulator():
    stats = ForwardStats()

    stats.cuda_graph_runtime_mode_counts = {"decode": 3}

    assert stats.cuda_graph.runtime_mode_counts == {"decode": 3}
    assert stats.cuda_graph_runtime_mode_counts == {"decode": 3}


def test_nested_write_is_visible_through_flat_property():
    stats = ForwardStats()

    stats.text_relay.token_relay_hits = 12

    assert stats.text_decode_token_relay_hits == 12


def test_flat_property_renames_text_decode_relay_fields():
    # The flat surface renames text_relay.token_relay_hits to
    # text_decode_token_relay_hits; assert that mapping is wired correctly.
    stats = ForwardStats()

    stats.text_decode_position_relay_misses = 4

    assert stats.text_relay.position_relay_misses == 4


# --------------------------------------------------------------------------
# record helpers accumulate; component timing no-ops when stats is None
# --------------------------------------------------------------------------


def test_record_attention_launch_accumulates_launches_ns_and_backend():
    stats = ForwardStats()

    stats.record_attention_launch("flashinfer", 1_000)
    stats.record_attention_launch("flashinfer", 500)

    assert stats.attention.launches == 2
    assert stats.attention.ns == 1_500
    assert stats.attention.backend_counts == {"flashinfer": 2}


def test_record_mode_shape_accumulates_ops_and_tokens():
    stats = ForwardStats()

    stats.record_mode_shape("decode", ops=2, tokens=10)
    stats.record_mode_shape("decode", ops=3, tokens=5)

    assert stats.mode.mode_counts == {"decode": 5}
    assert stats.mode.mode_tokens == {"decode": 15}


def test_component_timer_start_returns_positive_stamp_when_stats_present():
    stats = ForwardStats()

    stamp = component_timer_start(stats)

    assert isinstance(stamp, int)
    assert stamp > 0


def test_timing_none_guard_returns_disabled_sentinel_and_records_nothing():
    # With no ForwardStats being collected, the timer reports the disabled
    # sentinel (0) and recording returns None as a safe no-op — so callers never
    # have to branch on whether stats collection is active for this forward.
    assert component_timer_start(None) == 0
    assert record_component_elapsed(None, "attn", start_ns=123) is None


def test_record_component_elapsed_accumulates_into_component_ns():
    stats = ForwardStats()
    # start_ns=0 means elapsed == perf_counter_ns(), which is strictly positive.
    record_component_elapsed(stats, "attn", start_ns=0)

    assert stats.mode.component_ns["attn"] > 0


def test_context_component_timer_start_noops_when_stats_unset():
    # ForwardContext.stats defaults to None; the bound timer mirrors the free fn.
    assert ForwardContext().component_timer_start() == 0


def test_context_record_component_elapsed_writes_into_owned_stats():
    stats = ForwardStats()
    ctx = ForwardContext(stats=stats)

    ctx.record_component_elapsed("mlp", start_ns=0)

    assert stats.mode.component_ns["mlp"] > 0


# --------------------------------------------------------------------------
# Forward context var: default, restore on exit, restore on exception
# --------------------------------------------------------------------------


def test_get_forward_context_returns_default_empty_context_when_unset():
    ctx = get_forward_context()

    assert isinstance(ctx, ForwardContext)
    assert ctx.attention_backend is None
    assert ctx.attention_plan is None
    assert ctx.graph_binding is None
    assert ctx.kv_pool is None
    assert ctx.stats is None


def test_use_forward_context_publishes_context_inside_body():
    inner = ForwardContext(attention_preference="flashinfer")

    with use_forward_context(inner) as published:
        assert published is inner
        assert get_forward_context() is inner


def test_use_forward_context_restores_prior_token_on_normal_exit():
    # Default (unset) before; must be restored to the default after.
    before = get_forward_context()
    assert before.attention_preference is None

    with use_forward_context(ForwardContext(attention_preference="inner")):
        assert get_forward_context().attention_preference == "inner"

    assert get_forward_context().attention_preference is None


def test_use_forward_context_restores_prior_token_when_body_raises():
    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with use_forward_context(ForwardContext(attention_preference="inner")):
            assert get_forward_context().attention_preference == "inner"
            raise _Boom()

    # try/finally semantics: the context must be restored despite the exception.
    assert get_forward_context().attention_preference is None


def test_use_forward_context_restores_outer_context_when_nested():
    outer = ForwardContext(attention_preference="outer")
    inner = ForwardContext(attention_preference="inner")

    with use_forward_context(outer):
        assert get_forward_context() is outer
        with use_forward_context(inner):
            assert get_forward_context() is inner
        # Exiting the inner scope restores the outer context, not the default.
        assert get_forward_context() is outer

    assert get_forward_context().attention_preference is None
