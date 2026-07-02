"""Parity tests for the Family A (LLM serving) stream summarizer.

The harness is self-contained -- it does NOT import SGLang. Instead these tests
pin ``summarize_stream`` to the exact metric *formulas* SGLang uses by
re-deriving every headline metric independently with NumPy primitives:

* TTFT/TPOT/ITL/E2E percentiles via ``np.percentile`` (x1000 => ms);
* ``TPOT = (latency - ttft) / (output_len - 1)`` per request (server output_len),
  not the mean of ITLs;
* throughput = totals / wall-clock ``dur_s``; ``concurrency = sum(e2e)/dur``;
* peak ``max_output_tokens_per_s`` / ``max_concurrent_requests`` via 1s bucketing.

If the formulas ever drift from SGLang's definitions these re-derivations break.
"""
from __future__ import annotations

import numpy as np
import pytest

from uniserve_e2e.harness.metrics import summarize_stream
from uniserve_e2e.harness.metrics.common import RequestRecord

pytestmark = [pytest.mark.unit]

DUR_S = 10.0


class _StubTokenizer:
    """Whitespace tokenizer for the retokenized-output cross-check."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


def _records() -> list[RequestRecord]:
    specs = [
        # start, ttft, itl, output_len, prompt_len, text
        (0.0, 0.10, [0.02, 0.03], 3, 10, "alpha beta gamma"),
        (0.0, 0.20, [0.04, 0.05, 0.06], 4, 20, "a b c d"),
        (0.0, 0.05, [0.01], 2, 5, "x y"),
        (0.0, 0.30, [], 1, 8, "solo"),
    ]
    records: list[RequestRecord] = []
    for idx, (start, ttft, itl, output_len, prompt_len, text) in enumerate(specs):
        latency = ttft + sum(itl)
        records.append(
            RequestRecord(
                request_id=f"r{idx}",
                task="text",
                success=True,
                classifier="ok",
                start_time=start,
                latency=latency,
                ttft=ttft,
                itl=list(itl),
                prompt_len=prompt_len,
                output_len=output_len,
                generated_text=text,
                text_chunks=text.split(),
            )
        )
    return records


def test_stream_formulas_match_numpy() -> None:
    records = _records()
    summary = summarize_stream(records, DUR_S, tokenizer=_StubTokenizer())

    ttfts = [r.ttft for r in records]
    tpots = [(r.latency - r.ttft) / (r.output_len - 1) for r in records if r.output_len > 1]
    itls = [value for r in records for value in r.itl]
    e2e = [r.latency for r in records]
    total_output = sum(r.output_len for r in records)
    total_input = sum(r.prompt_len for r in records)

    assert summary["completed"] == len(records)
    assert summary["mean_ttft_ms"] == pytest.approx(float(np.mean(ttfts)) * 1000)
    assert summary["p90_ttft_ms"] == pytest.approx(float(np.percentile(ttfts, 90)) * 1000)
    assert summary["p99_ttft_ms"] == pytest.approx(float(np.percentile(ttfts, 99)) * 1000)
    assert summary["median_tpot_ms"] == pytest.approx(float(np.percentile(tpots, 50)) * 1000)
    assert summary["p90_tpot_ms"] == pytest.approx(float(np.percentile(tpots, 90)) * 1000)
    assert summary["median_itl_ms"] == pytest.approx(float(np.percentile(itls, 50)) * 1000)
    assert summary["p95_itl_ms"] == pytest.approx(float(np.percentile(itls, 95)) * 1000)
    assert summary["max_itl_ms"] == pytest.approx(float(np.max(itls)) * 1000)
    assert summary["p95_e2e_latency_ms"] == pytest.approx(float(np.percentile(e2e, 95)) * 1000)
    assert summary["output_throughput"] == pytest.approx(total_output / DUR_S)
    assert summary["input_throughput"] == pytest.approx(total_input / DUR_S)
    assert summary["total_throughput"] == pytest.approx((total_input + total_output) / DUR_S)
    assert summary["request_throughput"] == pytest.approx(len(records) / DUR_S)
    assert summary["concurrency"] == pytest.approx(float(np.sum(e2e)) / DUR_S)
    # Retokenized cross-check (whitespace tokenizer => token == word count).
    assert summary["total_output_tokens_retokenized"] == sum(len(r.generated_text.split()) for r in records)


def test_stream_peak_bucketing_is_deterministic() -> None:
    # Two identical requests, both active only in second-bucket 0, each emitting
    # 3 tokens in bucket 0 => peak concurrency 2, peak tokens/s 6.
    records = [
        RequestRecord(
            request_id=f"p{idx}",
            task="text",
            success=True,
            classifier="ok",
            start_time=0.0,
            latency=0.30,
            ttft=0.10,
            itl=[0.10, 0.10],
            prompt_len=4,
            output_len=3,
            generated_text="a b c",
        )
        for idx in range(2)
    ]
    summary = summarize_stream(records, DUR_S)
    assert summary["max_concurrent_requests"] == 2
    assert summary["max_output_tokens_per_s"] == pytest.approx(6.0)
