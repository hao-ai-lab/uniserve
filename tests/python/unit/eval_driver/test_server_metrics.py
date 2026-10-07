"""Measured-window telemetry: server counters and selected-GPU utilization."""

from __future__ import annotations

from typing import Any

import pytest

from uniserve_eval.load.gpu import GpuStorageSampler
from uniserve_eval.server_metrics import counter_deltas, window_summary

pytestmark = pytest.mark.unit

_BEFORE = """# TYPE uniserve:scheduler_domain_time_us counter
uniserve:scheduler_domain_time_us_total{domain="prefill",phase="device"} 1000
uniserve:scheduler_domain_time_us_total{domain="prefill",phase="queue"} 50
uniserve:request_prefill_time_seconds_sum{model_name="m"} 1.5
uniserve:request_prefill_time_seconds_count{model_name="m"} 3
uniserve:kv_cache_usage_perc{model_name="m"} 0.25
http_requests_total{path="/v1/systemone",reason="a b"} 4
# EOF
"""

_AFTER = """# TYPE uniserve:scheduler_domain_time_us counter
uniserve:scheduler_domain_time_us_total{domain="prefill",phase="device"} 1600
uniserve:scheduler_domain_time_us_total{domain="prefill",phase="queue"} 50
uniserve:scheduler_domain_time_us_total{domain="denoise",phase="device"} 40 17
uniserve:request_prefill_time_seconds_sum{model_name="m"} 2.5
uniserve:request_prefill_time_seconds_count{model_name="m"} 5
uniserve:kv_cache_usage_perc{model_name="m"} 0.75
http_requests_total{path="/v1/systemone",reason="a b"} 9
# EOF
"""


def test_counter_deltas_cover_counters_that_advanced_in_the_window() -> None:
    # The gauge and the unchanged queue counter are left out; a series that
    # first appears after the window opened started from zero, and its
    # trailing timestamp is not its value.
    assert counter_deltas(_BEFORE, _AFTER) == {
        'http_requests_total{path="/v1/systemone",reason="a b"}': 5.0,
        'uniserve:request_prefill_time_seconds_count{model_name="m"}': 2.0,
        'uniserve:request_prefill_time_seconds_sum{model_name="m"}': 1.0,
        'uniserve:scheduler_domain_time_us_total{domain="denoise",'
        'phase="device"}': 40.0,
        'uniserve:scheduler_domain_time_us_total{domain="prefill",'
        'phase="device"}': 600.0,
    }


def test_window_without_both_scrapes_is_unavailable() -> None:
    assert window_summary(None, _AFTER) == {
        "available": False,
        "counter_deltas": {},
    }


def _sample(time: float, gpu0: int, gpu1: int) -> dict[str, Any]:
    return {
        "time": time,
        "gpus": [
            {"index": 0, "memory_used_mib": 1, "utilization_gpu_pct": gpu0},
            {"index": 1, "memory_used_mib": 1, "utilization_gpu_pct": gpu1},
        ],
    }


def test_window_utilization_reports_selected_gpus_inside_the_window() -> None:
    sampler = GpuStorageSampler()
    # A warmup sample before the window, three inside it, one after it; GPU
    # 1 belongs to another job.
    sampler.sample_records.extend(
        [
            _sample(9.0, 100, 100),
            _sample(10.0, 60, 100),
            _sample(10.5, 80, 100),
            _sample(11.0, 100, 100),
            _sample(12.0, 0, 100),
        ]
    )

    summary = sampler.window_utilization(10.0, 11.0, (0,))

    assert summary == {
        "interval_s": 0.5,
        "gpus": {"0": {"samples": 3, "mean_pct": 80.0, "p50_pct": 80.0}},
    }
