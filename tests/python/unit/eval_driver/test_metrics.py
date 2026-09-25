"""Summary metrics follow the observed arrival times of streamed output."""

from __future__ import annotations

import pytest

from uniserve_eval.metrics import summarize
from uniserve_eval.types import RequestRecord

pytestmark = pytest.mark.unit


def test_peak_token_rate_keeps_the_gap_spanning_an_image() -> None:
    # An interleaved stream: two text events, an image 1.5 s after the send,
    # then three text events. The gap spanning the image is excluded from
    # ITL, but the tokens after it still arrived more than a second later.
    record = RequestRecord(
        request_id="row", task="interleave", success=True, latency=2.9
    )
    record.add_text("a", 0.1, last_text_time=None, count_itl=True)
    record.add_text("b", 0.2, last_text_time=0.1, count_itl=True)
    record.add_image_arrival(1, 1.5)
    record.add_text("c", 2.6, last_text_time=0.2, count_itl=False)
    record.add_text("d", 2.7, last_text_time=2.6, count_itl=True)
    record.add_text("e", 2.8, last_text_time=2.7, count_itl=True)

    summary = summarize([record], 3.0)

    # Seconds [0, 1) and [2, 3) hold two and three text events.
    assert summary["max_output_tokens_per_s"] == 3.0
