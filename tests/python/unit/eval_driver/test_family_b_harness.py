"""Family B (image-speed) benchmark harness behaviors.

These cover the self-contained, server-free pieces of the UniServe serving
benchmark harness that Family B (t2i / i2i) relies on:

* ``summarize_image`` aggregate fields for the non-streaming image cases;
* the response classifiers (OpenAI SSE / JSON image) routing
  sample payloads to the right ``(ok, classifier)`` verdict;
* ``iter_sse_events`` framing across record/line boundaries, including an event
  whose JSON payload is split across two ``data:`` lines that flush as one event;
* dataset loaders (``trace`` JSONL, UEval local JSONL, ShareGPT inline JSON)
  returning the documented row shapes from tiny inline fixtures;
* ``build_summary`` emitting the documented top-level schema;
* the CLI mapping a pass/fail run outcome to exit code 0 / 2.

The Family A stream summarizer parity is covered by ``test_stream_parity.py`` and
is not duplicated here.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import uniserve_eval.harness.core.client as client_module
import uniserve_eval.harness.datasets as datasets_module
from uniserve_eval.harness import cli
from uniserve_eval.harness.core.client import _parse_openai
from uniserve_eval.harness.datasets import (
    load_benchmark_inputs,
    load_dataset_rows,
    load_sharegpt,
    load_ueval,
    trace_items,
)
from uniserve_eval.harness.image_outputs import (
    image_output_contract,
    image_output_mismatch,
    inspect_image_bytes,
)
from uniserve_eval.harness.metrics import summarize_image
from uniserve_eval.harness.metrics.common import RequestRecord
from uniserve_eval.harness.report import (
    benchmark_contract,
    build_summary,
)
from uniserve_eval.harness.response_classifier import (
    classify_json_image_response,
    classify_openai_events,
)
from uniserve_eval.harness.runner import RunResult
from uniserve_eval.harness.spec import BenchmarkSpec, TaskName
from uniserve_eval.harness.sse import (
    aiter_sse_events,
    aiter_sse_events_from_text,
    iter_sse_events,
)
from uniserve_eval.harness.tasks.i2i import I2ITask
from uniserve_eval.harness.tasks.i2t import I2TTask
from uniserve_eval.harness.tasks.interleave import InterleaveTask
from uniserve_eval.harness.tasks.t2i import T2ITask
from uniserve_eval.harness.tasks.text import TextTask

pytestmark = [pytest.mark.unit]
TERMINAL_TYPES = frozenset({"finished"})


class _WhitespaceTokenizer:
    """Deterministic tokenizer: token count == whitespace word count."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


def _png_bytes(width: int = 2, height: int = 3) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), (17, 31, 47)).save(output, format="PNG")
    return output.getvalue()


def _png_data_url(width: int = 2, height: int = 3) -> str:
    encoded = base64.b64encode(_png_bytes(width, height)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _public_commit(event_seq: int, modality: str, committed_at: float) -> dict[str, object]:
    return {
        "event_seq": event_seq,
        "modality": modality,
        "committed_at": committed_at,
        "semantic_root": {
            "producer_op_id": event_seq,
            "point_index": event_seq,
            "semantic_digest": f"{event_seq:064x}",
        },
    }


def _successful_image_record(
    request_id: str = "a", *, width: int = 2, height: int = 3
) -> RequestRecord:
    image = inspect_image_bytes(_png_bytes(width, height))
    return RequestRecord(
        request_id=request_id,
        task="t2i",
        success=True,
        latency=1.0,
        images=1,
        classifier="ok",
        image_output_mode="required",
        requested_image_count=1,
        requested_image_width=width,
        requested_image_height=height,
        decoded_images=[image],
    )


def test_streamed_multimodal_image_count_is_an_upper_bound() -> None:
    contract = image_output_contract(
        {
            "modalities": ["text", "image"],
            "image_config": {"num_images": 4},
        },
        request_kind="openai_chat",
        task="interleave",
    )
    images = [inspect_image_bytes(_png_bytes()) for _ in range(2)]

    assert contract.mode == "optional"
    assert contract.count_is_cap is True
    assert image_output_mismatch([], contract) is None
    assert image_output_mismatch(images, contract) is None
    assert image_output_mismatch(images * 3, contract) == "protocol_image_count_mismatch"


def test_image_generation_count_is_exact() -> None:
    contract = image_output_contract(
        {"n": 4},
        request_kind="images_generations",
        task="t2i",
    )
    images = [inspect_image_bytes(_png_bytes()) for _ in range(2)]

    assert contract.mode == "required"
    assert contract.count_is_cap is False
    assert image_output_mismatch([], contract) == "protocol_missing_decoded_image"
    assert image_output_mismatch(images, contract) == "protocol_image_count_mismatch"


# --- summarize_image (Family B aggregate fields) ------------------------------


def test_summarize_image_nonstreaming_aggregates() -> None:
    # Two successful t2i requests (no per-image stream events) at 2s and 4s E2E,
    # one failed request that must be excluded entirely. dur_s = 10s.
    records = [
        RequestRecord(request_id="a", task="t2i", success=True, latency=2.0, images=1),
        RequestRecord(request_id="b", task="t2i", success=True, latency=4.0, images=1),
        RequestRecord(request_id="c", task="t2i", success=False, latency=99.0, images=1),
    ]

    summary = summarize_image(records, dur_s=10.0)

    assert summary["completed_requests"] == 2
    assert summary["completed_images"] == 2
    assert summary["request_throughput"] == pytest.approx(2 / 10.0)
    assert summary["images_per_second"] == pytest.approx(2 / 10.0)
    assert summary["images_per_minute"] == pytest.approx(60.0 * 2 / 10.0)
    lat = summary["image_latency_ms"]
    # Non-streaming: per-image latency is the request E2E (seconds * 1000).
    assert lat["count"] == 2
    assert lat["mean"] == pytest.approx(float(np.mean([2.0, 4.0])) * 1000)
    assert lat["min"] == pytest.approx(2000.0)
    assert lat["max"] == pytest.approx(4000.0)
    assert lat["p50"] == pytest.approx(float(np.percentile([2.0, 4.0], 50)) * 1000)
    # Stream-only extras must be absent for the non-streaming case.
    assert "time_to_first_image_ms" not in summary
    assert "image_generation_ms" not in summary
    assert "steps_per_second" not in summary


def test_summarize_image_i2i_stream_extras() -> None:
    # One request whose stream exposed image events: first image at
    # 1s, generation 2s, 10 diffusion steps -> steps/s = 5.
    record = RequestRecord(
        request_id="x",
        task="i2i",
        success=True,
        latency=5.0,
        images=1,
        image_latencies=[5.0],
        first_image_latency=1.0,
        image_gen_seconds=[2.0],
        image_steps=[10],
    )

    summary = summarize_image([record], dur_s=10.0)

    assert summary["completed_images"] == 1
    assert summary["image_latency_ms"]["p50"] == pytest.approx(5000.0)
    assert summary["time_to_first_image_ms"]["p50"] == pytest.approx(1000.0)
    assert summary["image_generation_ms"]["p50"] == pytest.approx(2000.0)
    assert summary["steps_per_second"]["mean"] == pytest.approx(10 / 2.0)


# --- response classifiers -----------------------------------------------------


def test_classify_openai_events_routes_payloads() -> None:
    streamed = [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"finish_reason": "stop", "delta": {}}]},
    ]
    assert classify_openai_events(streamed) == (True, "ok")
    # The OpenAI [DONE] sentinel (rendered by the SSE reader as sse_done) is a
    # valid terminal too.
    done_terminal = [{"choices": [{"delta": {"content": "hi"}}]}, {"type": "sse_done"}]
    assert classify_openai_events(done_terminal) == (True, "ok")
    assert classify_openai_events([]) == (False, "protocol_empty_response")
    assert classify_openai_events([{"error": {"message": "boom"}}]) == (False, "model_error")
    assert classify_openai_events([{"choices": [{"delta": {"content": "hi"}}]}]) == (
        False,
        "protocol_missing_terminal",
    )
    assert classify_openai_events([{"choices": [{"finish_reason": "stop", "delta": {}}]}]) == (
        False,
        "protocol_empty_output",
    )


def test_classify_openai_events_counts_reasoning_content_as_text() -> None:
    events = [
        {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"type": "sse_done"},
    ]

    assert classify_openai_events(events) == (True, "ok")


def test_classify_openai_events_counts_delta_images_as_output() -> None:
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"type": "sse_done"},
    ]

    assert classify_openai_events(events) == (True, "ok")


def test_classify_json_image_response_routes_payloads() -> None:
    assert classify_json_image_response({"data": [{"b64_json": "AAAA"}]}) == (True, "ok")
    assert classify_json_image_response({}) == (False, "protocol_empty_image_data")
    assert classify_json_image_response({"data": []}) == (False, "protocol_empty_image_data")
    assert classify_json_image_response({"data": [{"url": "http://x"}]}) == (
        False,
        "protocol_missing_image_payload",
    )


def test_openai_parser_counts_delta_images_without_charging_text_itl() -> None:
    record = RequestRecord(request_id="openai-default", task="default")
    record.start_time = 10.0
    events = [
        {
            "choices": [{"delta": {"content": "a"}}],
            "public_commit": _public_commit(1, "text", 1.0),
            "_client_t": 11.0,
        },
        {
            "choices": [
                {
                    "delta": {
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {"url": _png_data_url()},
                            }
                        ]
                    }
                }
            ],
            "public_commit": _public_commit(2, "image", 3.0),
            "_client_t": 13.0,
        },
        {
            "choices": [{"delta": {"content": "b"}}],
            "public_commit": _public_commit(3, "text", 4.0),
            "_client_t": 14.0,
        },
        {
            "choices": [{"delta": {"content": "c"}}],
            "public_commit": _public_commit(4, "text", 4.25),
            "_client_t": 14.25,
        },
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 5},
                "image_steps_per_image": [50],
            }
        },
        {"type": "sse_done"},
    ]

    _parse_openai(events, record, output_len_fallback=0, prompt_len=5)

    assert record.success is True
    assert record.generated_text == "abc"
    assert record.ttft == pytest.approx(1.0)
    assert record.itl == pytest.approx([0.25])
    assert record.images == 1
    assert record.first_image_latency == pytest.approx(3.0)
    assert record.image_latencies == pytest.approx([3.0])
    assert record.image_steps == [50]
    assert record.prompt_len == 7
    assert record.output_len == 3
    assert record.prompt_len_source == "server_usage"
    assert record.output_len_source == "server_usage"
    assert record.cached_prompt_tokens == 5
    assert record.cached_prompt_tokens_source == "openai_usage_prompt_tokens_details"
    assert record.output_modalities == ["text", "image", "text"]
    assert record.modality_events == [
        {
            "modalities": ["text"],
            "client_time": 11.0,
            "text_bytes": 1,
            "image_count": 0,
            "public_commit": _public_commit(1, "text", 1.0),
        },
        {
            "modalities": ["image"],
            "client_time": 13.0,
            "text_bytes": 0,
            "image_count": 1,
            "public_commit": _public_commit(2, "image", 3.0),
        },
        {
            "modalities": ["text"],
            "client_time": 14.0,
            "text_bytes": 1,
            "image_count": 0,
            "public_commit": _public_commit(3, "text", 4.0),
        },
        {
            "modalities": ["text"],
            "client_time": 14.25,
            "text_bytes": 1,
            "image_count": 0,
            "public_commit": _public_commit(4, "text", 4.25),
        },
    ]
    assert record.finish_reason == "stop"
    persisted = record.record_dict()
    assert persisted["generated_text_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert [event["client_offset_ms"] for event in persisted["modality_events"]] == pytest.approx(
        [1000.0, 3000.0, 4000.0, 4250.0]
    )


def test_interleave_summary_requires_visible_text_image_transition() -> None:
    spec = BenchmarkSpec(
        task=TaskName.INTERLEAVE,
        model="SenseNova-U1",
        num_prompts=1,
        max_tokens=256,
        max_images=1,
        width=2,
        height=3,
        steps=50,
        denoise_updates=50,
        ignore_eos=False,
        acceptance_min_images_per_success=1.0,
    )
    record = RequestRecord(
        request_id="interleave-1",
        task="interleave",
        success=True,
        classifier="ok",
        latency=4.0,
        ttft=0.5,
        itl=[0.1],
        token_timing_available=True,
        prompt_len=16,
        output_len=3,
        prompt_len_source="server_usage",
        output_len_source="server_usage",
        generated_text="intro caption",
        image_output_mode="optional",
        images=1,
        image_latencies=[3.0],
        output_modalities=["text", "image", "text"],
        modality_events=[
            {
                "modalities": ["text"],
                "client_time": 0.5,
                "text_bytes": 5,
                "image_count": 0,
                "public_commit": _public_commit(1, "text", 10.0),
            },
            {
                "modalities": ["text"],
                "client_time": 0.75,
                "text_bytes": 8,
                "image_count": 0,
                "public_commit": _public_commit(2, "text", 10.2),
            },
            {
                "modalities": ["image"],
                "client_time": 3.0,
                "text_bytes": 0,
                "image_count": 1,
                "public_commit": _public_commit(3, "image", 12.0),
            },
            {
                "modalities": ["text"],
                "client_time": 3.5,
                "text_bytes": 4,
                "image_count": 0,
                "public_commit": _public_commit(4, "text", 12.4),
            },
            {
                "modalities": ["text"],
                "client_time": 3.7,
                "text_bytes": 3,
                "image_count": 0,
                "public_commit": _public_commit(5, "text", 12.6),
            },
        ],
        decoded_images=[inspect_image_bytes(_png_bytes())],
    )
    contract = benchmark_contract(spec, [{"id": "interleave-1"}])

    summary = build_summary(spec, "http://x", [record], dur_s=4.0, contract=contract)

    assert summary["artifact"]["generation_conformance"]["valid"] is True
    assert summary["artifact"]["checks"]["interleave_latency_conformance"] is True
    assert summary["metrics"]["ttft_ms"]["count"] == 1
    assert summary["metrics"]["tpot_ms"]["count"] == 1
    assert summary["metrics"]["images"]["image_latency_ms"]["count"] == 1
    timing = summary["metrics"]["modality_interleave"]["transition_timing"]
    assert timing["valid"] is True
    assert timing["timestamp_coverage"] == 1.0
    assert timing["public_commit_coverage"] == 1.0
    assert timing["transition_sample_coverage"] == 1.0
    assert timing["request_signatures"] == {"interleave-1": "text->image->text"}
    assert timing["transition_latency_ms"]["count"] == 2
    assert timing["transition_latency_ms"]["mean"] == pytest.approx(1375.0)
    assert timing["text_to_image_transition_latency_ms"]["mean"] == pytest.approx(2250.0)
    assert timing["image_to_text_transition_latency_ms"]["mean"] == pytest.approx(500.0)
    assert timing["server_transition_latency_ms"]["count"] == 2
    assert timing["server_text_to_image_transition_latency_ms"]["mean"] == pytest.approx(1800.0)
    assert timing["server_image_to_text_transition_latency_ms"]["mean"] == pytest.approx(400.0)
    assert timing["client_delivery_transition_delta_ms"]["mean"] == pytest.approx(275.0)
    correlations = timing["boundary_correlations"]["interleave-1"]
    assert [(item["source_event_seq"], item["destination_event_seq"]) for item in correlations] == [
        (2, 3),
        (3, 4),
    ]
    digest = timing["latency_definition_digest"]
    assert isinstance(digest, str) and len(digest) == 64

    missing_text = dataclasses.replace(
        record,
        generated_text="",
        output_modalities=["image"],
        modality_events=[
            {
                "modalities": ["image"],
                "client_time": 3.0,
                "text_bytes": 0,
                "image_count": 1,
                "public_commit": _public_commit(3, "image", 12.0),
            }
        ],
    )
    invalid_summary = build_summary(
        spec,
        "http://x",
        [missing_text],
        dur_s=4.0,
        contract=contract,
    )
    conformance = invalid_summary["artifact"]["generation_conformance"]
    assert conformance["valid"] is False
    assert conformance["mismatched_request_ids"] == ["interleave-1"]


def test_interleave_summary_counts_text_only_response_as_warned_success() -> None:
    spec = BenchmarkSpec(
        task=TaskName.INTERLEAVE,
        model="SenseNova-U1",
        num_prompts=2,
        max_tokens=256,
        max_images=1,
        width=2,
        height=3,
        steps=50,
        denoise_updates=50,
        ignore_eos=False,
        acceptance_min_success=2,
    )
    image = inspect_image_bytes(_png_bytes())
    multimodal = RequestRecord(
        request_id="interleave-1",
        task="interleave",
        success=True,
        classifier="ok",
        latency=3.0,
        ttft=0.5,
        token_timing_available=True,
        prompt_len=16,
        output_len=3,
        prompt_len_source="server_usage",
        output_len_source="server_usage",
        generated_text="intro caption",
        image_output_mode="optional",
        requested_image_count=1,
        requested_image_count_is_cap=True,
        requested_image_width=2,
        requested_image_height=3,
        images=1,
        image_latencies=[2.0],
        output_modalities=["text", "image"],
        modality_events=[
            {
                "modalities": ["text"],
                "client_time": 0.5,
                "text_bytes": 5,
                "image_count": 0,
                "public_commit": _public_commit(1, "text", 10.0),
            },
            {
                "modalities": ["image"],
                "client_time": 2.0,
                "text_bytes": 0,
                "image_count": 1,
                "public_commit": _public_commit(2, "image", 11.0),
            },
        ],
        decoded_images=[image],
    )
    text_only = RequestRecord(
        request_id="interleave-2",
        task="interleave",
        success=True,
        classifier="ok",
        warnings=["no_generated_image"],
        latency=2.0,
        ttft=0.25,
        token_timing_available=True,
        prompt_len=12,
        output_len=3,
        prompt_len_source="server_usage",
        output_len_source="server_usage",
        generated_text="textual steps",
        image_output_mode="optional",
        requested_image_count=1,
        requested_image_count_is_cap=True,
        requested_image_width=2,
        requested_image_height=3,
        output_modalities=["text"],
        modality_events=[
            {
                "modalities": ["text"],
                "client_time": 0.25,
                "text_bytes": 13,
                "image_count": 0,
                "public_commit": _public_commit(3, "text", 12.0),
            }
        ],
    )
    contract = benchmark_contract(spec, [{"id": "interleave-1"}, {"id": "interleave-2"}])

    summary = build_summary(
        spec,
        "http://x",
        [multimodal, text_only],
        dur_s=3.0,
        contract=contract,
    )

    assert summary["ok_count"] == 2
    assert summary["failed_count"] == 0
    assert summary["warnings"] == {
        "request_count": 1,
        "total_count": 1,
        "counts": {"no_generated_image": 1},
    }
    assert summary["artifact"]["warnings"] == summary["warnings"]
    assert summary["artifact"]["valid"] is True
    assert summary["metrics"]["ttft_ms"]["count"] == 2
    assert summary["metrics"]["tpot_ms"]["count"] == 2
    timing = summary["metrics"]["modality_interleave"]["transition_timing"]
    assert timing["request_count"] == 2
    assert timing["complete_request_count"] == 2
    assert timing["request_signatures"] == {
        "interleave-1": "text->image",
        "interleave-2": "text",
    }
    assert timing["transition_latency_ms"]["count"] == 1


@pytest.mark.parametrize(
    "events",
    [
        [
            {
                "modalities": ["text", "image"],
                "client_time": 1.0,
                "text_bytes": 5,
                "image_count": 1,
            }
        ],
        [
            {
                "modalities": ["text"],
                "client_time": None,
                "text_bytes": 5,
                "image_count": 0,
            },
            {
                "modalities": ["image"],
                "client_time": 2.0,
                "text_bytes": 0,
                "image_count": 1,
            },
        ],
    ],
)
def test_interleave_timing_requires_unambiguous_complete_event_timestamps(
    events: list[dict[str, object]],
) -> None:
    spec = BenchmarkSpec(
        task=TaskName.INTERLEAVE,
        model="SenseNova-U1",
        num_prompts=1,
        max_tokens=256,
        max_images=1,
        width=2,
        height=3,
        steps=50,
        denoise_updates=50,
        ignore_eos=False,
    )
    record = RequestRecord(
        request_id="interleave-1",
        task="interleave",
        success=True,
        classifier="ok",
        latency=3.0,
        ttft=0.5,
        token_timing_available=True,
        prompt_len=16,
        output_len=3,
        prompt_len_source="server_usage",
        output_len_source="server_usage",
        generated_text="intro",
        image_output_mode="optional",
        images=1,
        image_latencies=[2.0],
        output_modalities=["text", "image"],
        modality_events=events,
        decoded_images=[inspect_image_bytes(_png_bytes())],
    )
    contract = benchmark_contract(spec, [{"id": "interleave-1"}])

    summary = build_summary(spec, "http://x", [record], dur_s=3.0, contract=contract)

    assert summary["artifact"]["generation_conformance"]["valid"] is True
    assert summary["artifact"]["checks"]["interleave_latency_conformance"] is False
    assert summary["artifact"]["valid"] is False


@pytest.mark.parametrize(
    ("events", "attribution_status", "warning"),
    [
        (
            [
                {
                    "modalities": ["text"],
                    "client_time": 1.0,
                    "text_bytes": 5,
                    "image_count": 0,
                },
                {
                    "modalities": ["image"],
                    "client_time": 2.0,
                    "text_bytes": 0,
                    "image_count": 1,
                },
            ],
            "unavailable",
            "server_public_commit_unavailable",
        ),
        (
            [
                {
                    "modalities": ["text"],
                    "client_time": 1.0,
                    "text_bytes": 5,
                    "image_count": 0,
                    "public_commit": _public_commit(1, "text", 1.0),
                },
                {
                    "modalities": ["image"],
                    "client_time": 2.0,
                    "text_bytes": 0,
                    "image_count": 1,
                },
            ],
            "partial",
            "server_public_commit_partial",
        ),
    ],
)
def test_interleave_timing_uses_complete_client_boundaries(
    events: list[dict[str, object]], attribution_status: str, warning: str
) -> None:
    spec = BenchmarkSpec(
        task=TaskName.INTERLEAVE,
        model="SenseNova-U1",
        num_prompts=1,
        max_tokens=256,
        max_images=1,
        width=2,
        height=3,
        steps=50,
        denoise_updates=50,
        ignore_eos=False,
    )
    record = RequestRecord(
        request_id="interleave-1",
        task="interleave",
        success=True,
        classifier="ok",
        latency=3.0,
        ttft=0.5,
        token_timing_available=True,
        prompt_len=16,
        output_len=3,
        prompt_len_source="server_usage",
        output_len_source="server_usage",
        generated_text="intro",
        image_output_mode="optional",
        images=1,
        image_latencies=[2.0],
        output_modalities=["text", "image"],
        modality_events=events,
        decoded_images=[inspect_image_bytes(_png_bytes())],
    )
    contract = benchmark_contract(spec, [{"id": "interleave-1"}])

    summary = build_summary(spec, "http://x", [record], dur_s=3.0, contract=contract)

    assert summary["artifact"]["checks"]["interleave_latency_conformance"] is True
    assert summary["artifact"]["valid"] is True
    assert summary["warnings"] == {
        "request_count": 1,
        "total_count": 1,
        "counts": {warning: 1},
    }
    timing = summary["metrics"]["modality_interleave"]["transition_timing"]
    assert timing["server_attribution"]["status"] == attribution_status
    assert timing["transition_latency_ms"]["mean"] == pytest.approx(1000.0)


def test_interleave_dataset_serves_ueval_prompts_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        datasets_module,
        "load_ueval",
        lambda *_args, **_kwargs: [
            {
                "id": "ueval-000000",
                "task": "interleave",
                "prompt": "How to draw a cartoon cat? Show each step visually and textually.",
            }
        ],
    )
    spec = BenchmarkSpec(
        task=TaskName.INTERLEAVE,
        model="SenseNova-U1",
        num_prompts=1,
    )

    assert spec.dataset == "ueval"
    assert load_dataset_rows(spec) == [
        {
            "id": "ueval-000000",
            "task": "interleave",
            "prompt": "How to draw a cartoon cat? Show each step visually and textually.",
        }
    ]


def test_interleave_task_carries_fixed_image_generation_controls() -> None:
    request = InterleaveTask(
        BenchmarkSpec(
            task=TaskName.INTERLEAVE,
            model="SenseNova-U1",
            max_tokens=256,
            max_images=1,
            width=2048,
            height=1152,
            steps=50,
            image_think=False,
            image_t_eps=0.02,
        )
    ).build_request({"prompt": "write, illustrate, and conclude"})

    assert request.payload["modalities"] == ["text", "image"]
    assert request.payload["image_config"] == {
        "num_images": 1,
        "width": 2048,
        "height": 1152,
        "steps": 50,
        "seed": 42,
    }
    assert "think" not in request.payload
    assert "t_eps" not in request.payload


def test_openai_parser_captures_sglang_cached_token_breakdown_when_usage_omits_it() -> None:
    record = RequestRecord(request_id="cache", task="text", start_time=10.0)
    events = [
        {"choices": [{"delta": {"content": "a"}}], "_client_t": 11.0},
        {
            "choices": [],
            "sglext": {
                "cached_tokens_details": {
                    "device": 4,
                    "host": 2,
                    "storage": 1,
                    "storage_backend": "file",
                }
            },
        },
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"usage": {"prompt_tokens": 9, "completion_tokens": 1}},
        {"type": "sse_done"},
    ]

    _parse_openai(events, record, output_len_fallback=0, prompt_len=0)

    assert record.success is True
    assert record.cached_prompt_tokens == 7
    assert record.cached_prompt_tokens_source == "sglang_sglext_cached_tokens_details"
    assert record.record_dict()["cached_prompt_tokens"] == 7


def test_interleave_task_omits_image_cap_unless_explicit() -> None:
    uncapped = InterleaveTask(
        BenchmarkSpec(
            task=TaskName.INTERLEAVE,
            model="SenseNova-U1",
            max_tokens=8192,
            width=2048,
            height=1152,
            steps=50,
        )
    ).build_request({"prompt": "show each step visually and textually"})

    assert uncapped.payload["image_config"] == {
        "width": 2048,
        "height": 1152,
        "steps": 50,
        "seed": 42,
    }

    capped = InterleaveTask(
        BenchmarkSpec(
            task=TaskName.INTERLEAVE,
            model="SenseNova-U1",
            max_tokens=8192,
            max_images=8,
            width=2048,
            height=1152,
            steps=50,
        )
    ).build_request({"prompt": "show each step visually and textually"})

    assert capped.payload["image_config"]["num_images"] == 8


def test_interleave_task_can_emit_openai_chat_wire() -> None:
    request = InterleaveTask(
        BenchmarkSpec(
            task=TaskName.INTERLEAVE,
            model="SenseNova-U1",
            max_tokens=8192,
            max_images=4,
            width=2048,
            height=1152,
            steps=50,
            wire="openai_chat",
        )
    ).build_request({"prompt": "show each step visually and textually"})

    assert request.endpoint == "/v1/chat/completions"
    assert request.kind == "openai_chat"
    assert request.payload["modalities"] == ["text", "image"]
    assert request.payload["stream"] is True
    assert request.payload["stream_options"] == {"include_usage": True}
    assert request.payload["max_completion_tokens"] == 8192
    assert request.payload["messages"] == [
        {"role": "user", "content": "show each step visually and textually"}
    ]
    assert request.payload["image_config"] == {
        "num_images": 4,
        "width": 2048,
        "height": 1152,
        "steps": 50,
        "seed": 42,
    }


def test_spec_rejects_unsupported_wire_for_task() -> None:
    with pytest.raises(ValueError, match="does not support wire"):
        BenchmarkSpec(task=TaskName.TEXT, model="m", wire="native")


def test_i2t_task_streams_openai_chat_wire() -> None:
    request = I2TTask(
        BenchmarkSpec(
            task=TaskName.I2T,
            model="SenseNova-U1",
            max_tokens=256,
            wire="openai_chat",
        )
    ).build_request({"prompt": "Describe this image.", "input_image_b64": "QUJD"})

    assert request.endpoint == "/v1/chat/completions"
    assert request.kind == "openai_chat"
    assert request.payload["stream"] is True
    assert request.payload["stream_options"] == {"include_usage": True}
    assert request.payload["modalities"] == ["text"]
    assert request.payload["max_completion_tokens"] == 256
    assert "extra_args" not in request.payload
    parts = request.payload["messages"][0]["content"]
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_i2t_task_openai_chat_json_wire_is_not_streamed() -> None:
    request = I2TTask(
        BenchmarkSpec(
            task=TaskName.I2T,
            model="SenseNova-U1",
            max_tokens=256,
            wire="openai_chat_json",
        )
    ).build_request({"prompt": "Describe this image.", "input_image_b64": "QUJD"})

    assert request.endpoint == "/v1/chat/completions"
    assert request.kind == "openai_chat_json"
    assert "stream" not in request.payload
    assert request.payload["max_completion_tokens"] == 256
    assert "extra_args" not in request.payload


def test_i2t_vllm_omni_request_schema_receives_extra_args() -> None:
    request = I2TTask(
        BenchmarkSpec(
            task=TaskName.I2T,
            model="SenseNova-U1",
            max_tokens=256,
            wire="openai_chat_json",
            request_schema="vllm_omni",
        )
    ).build_request({"prompt": "Describe this image.", "input_image_b64": "QUJD"})

    assert request.payload["extra_args"]["max_tokens"] == 256


def test_i2t_task_preserves_the_input_image_mime_type() -> None:
    request = I2TTask(
        BenchmarkSpec(task=TaskName.I2T, model="M", wire="openai_chat")
    ).build_request(
        {
            "prompt": "Describe this image.",
            "input_image_b64": "QUJD",
            "input_image_mime": "image/jpeg",
        }
    )

    parts = request.payload["messages"][0]["content"]
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_nonstreaming_i2t_does_not_fabricate_token_timing() -> None:
    spec = BenchmarkSpec(
        task=TaskName.I2T,
        model="M",
        num_prompts=1,
        wire="openai_chat_json",
    )
    record = RequestRecord(
        request_id="a",
        task="i2t",
        success=True,
        classifier="ok",
        latency=2.0,
        output_len=32,
        token_timing_available=False,
    )

    summary = build_summary(spec, "http://server:1", [record], dur_s=2.0)

    assert summary["metrics"]["mean_e2e_latency_ms"] == pytest.approx(2000.0)
    assert summary["metrics"]["token_timing_available"] is False
    assert summary["metrics"]["p50_ttft_ms"] is None
    assert summary["metrics"]["p50_tpot_ms"] is None
    assert summary["metrics"]["p50_itl_ms"] is None
    assert record.record_dict()["ttft_ms"] is None
    assert record.record_dict()["tpot_ms"] is None


@pytest.mark.parametrize(
    ("task_class", "task", "wire", "item"),
    [
        (InterleaveTask, TaskName.INTERLEAVE, "openai_chat", {"prompt": "p"}),
        (
            I2ITask,
            TaskName.I2I,
            "openai_chat_json",
            {"prompt": "p", "input_image_b64": "QUJD"},
        ),
        (
            I2TTask,
            TaskName.I2T,
            "openai_chat",
            {"prompt": "p", "input_image_b64": "QUJD"},
        ),
        (T2ITask, TaskName.T2I, "openai_chat_json", {"prompt": "p"}),
        (TextTask, TaskName.TEXT, "openai_chat", {"prompt": "p"}),
    ],
)
def test_chat_task_builders_preserve_declared_sampling_contract(
    task_class, task: TaskName, wire: str, item: dict
) -> None:
    request = task_class(
        BenchmarkSpec(
            task=task,
            model="M",
            wire=wire,
            temperature=0.35,
            top_p=0.82,
            top_k=1 if task is TaskName.TEXT else None,
            min_p=0.0 if task is TaskName.TEXT else None,
            repetition_penalty=1.0 if task is TaskName.TEXT else None,
            frequency_penalty=0.0 if task is TaskName.TEXT else None,
            presence_penalty=0.0 if task is TaskName.TEXT else None,
            sampling_seed=42 if task is TaskName.TEXT else None,
            chat_template_kwargs={"enable_thinking": True} if task is TaskName.TEXT else {},
            ignore_eos=False,
        )
    ).build_request(item)

    assert request.payload["temperature"] == 0.35
    assert request.payload["top_p"] == 0.82
    assert request.payload["ignore_eos"] is False
    if task is TaskName.TEXT:
        assert request.payload["top_k"] == 1
        assert request.payload["min_p"] == 0.0
        assert request.payload["repetition_penalty"] == 1.0
        assert request.payload["frequency_penalty"] == 0.0
        assert request.payload["presence_penalty"] == 0.0
        assert request.payload["seed"] == 42
        assert "chat_template_kwargs" not in request.payload


def test_sglang_request_schema_receives_template_kwargs() -> None:
    request = TextTask(
        BenchmarkSpec(
            task=TaskName.TEXT,
            model="M",
            chat_template_kwargs={"enable_thinking": True},
            request_schema="sglang",
        )
    ).build_request({"prompt": "p"})

    assert request.payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_interleave_task_emits_declared_sampling_seed() -> None:
    request = InterleaveTask(
        BenchmarkSpec(
            task=TaskName.INTERLEAVE,
            model="M",
            sampling_seed=42,
        )
    ).build_request({"prompt": "p"})

    assert request.payload["seed"] == 42


def test_chat_json_counts_message_images() -> None:
    record = RequestRecord(request_id="x", task="t2i")
    record.start_time = 0.0

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "images": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _png_data_url()},
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 0,
                    "image_steps_per_image": [50],
                },
            }

    class _Client:
        @staticmethod
        async def post(url, json):
            return _Resp()

    asyncio.run(
        client_module._send_chat_json(_Client(), "http://x/v1/chat/completions", {}, record)
    )
    assert record.success
    assert record.classifier == "ok"
    assert record.images == 1
    assert len(record.image_latencies) == 1
    assert record.image_steps == [50]


# --- SSE framing --------------------------------------------------------------


def test_iter_sse_events_frames_multiple_records() -> None:
    lines = [
        'data: {"type": "text", "value": "a"}',
        "",
        'data: {"type": "finished"}',
        "",
    ]

    events = list(iter_sse_events(lines))

    assert events == [{"type": "text", "value": "a"}, {"type": "finished"}]


def test_iter_sse_events_joins_event_split_across_two_data_lines() -> None:
    # A single SSE record whose JSON payload arrives as two ``data:`` lines
    # (e.g. one transport chunk carries the head, the next carries the tail).
    # Multi-line ``data:`` blocks are joined with "\n" and decoded as one event;
    # the blank line flushes them as exactly one record.
    lines = [
        'data: {"type":',
        'data:  "finished"}',
        "",
    ]

    events = list(iter_sse_events(lines))

    assert events == [{"type": "finished"}]


def test_iter_sse_events_strips_optional_leading_space_consistently() -> None:
    no_space = list(iter_sse_events(['data:{"a":1}', ""]))
    one_space = list(iter_sse_events(['data: {"a":1}', ""]))

    assert no_space == [{"a": 1}]
    assert no_space == one_space


def test_iter_sse_events_ignores_comments_and_non_data_lines() -> None:
    lines = [": keep-alive", "event: message", 'data: {"k": 1}', ""]

    events = list(iter_sse_events(lines))

    assert events == [{"k": 1}]


def test_iter_sse_events_flushes_trailing_record_without_blank_line() -> None:
    events = list(iter_sse_events(['data: {"k": 2}']))

    assert events == [{"k": 2}]


def test_iter_sse_events_done_sentinel_becomes_synthetic_event() -> None:
    events = list(iter_sse_events(["data: [DONE]", ""]))

    assert events == [{"type": "sse_done"}]


def test_iter_sse_events_stop_on_terminal_halts_after_first_terminal() -> None:
    lines = [
        'data: {"type": "text"}',
        "",
        'data: {"type": "finished"}',
        "",
        'data: {"type": "text", "after": true}',
        "",
    ]

    events = list(iter_sse_events(lines, stop_on=TERMINAL_TYPES))

    assert events == [{"type": "text"}, {"type": "finished"}]


def test_iter_sse_events_stop_on_terminal_without_trailing_blank_line() -> None:
    lines = [
        'data: {"type": "text"}',
        "",
        'data: {"type": "finished"}',
        'data: {"type": "text", "after": true}',
        "",
    ]

    events = list(iter_sse_events(lines, stop_on=TERMINAL_TYPES))

    assert events == [{"type": "text"}, {"type": "finished"}]


def test_aiter_sse_events_stop_on_terminal_without_waiting_for_eof() -> None:
    async def lines() -> object:
        yield 'data: {"type": "text"}'
        yield ""
        yield 'data: {"type": "finished"}'
        await asyncio.sleep(60.0)
        yield 'data: {"type": "text", "after": true}'

    async def run_once() -> list[dict[str, object]]:
        return await asyncio.wait_for(
            aiter_sse_events(lines(), stop_on=TERMINAL_TYPES),
            timeout=0.25,
        )

    events = asyncio.run(run_once())

    assert events == [{"type": "text"}, {"type": "finished"}]


def test_aiter_sse_events_from_text_stops_on_terminal_without_newline_or_eof() -> None:
    async def chunks() -> object:
        yield 'data: {"type": "text"}\n\n'
        yield 'data: {"type": "fin'
        yield 'ished"}'
        await asyncio.sleep(60.0)
        yield '\n\ndata: {"type": "text", "after": true}\n\n'

    async def run_once() -> list[dict[str, object]]:
        return await asyncio.wait_for(
            aiter_sse_events_from_text(chunks(), stop_on=TERMINAL_TYPES),
            timeout=0.25,
        )

    events = asyncio.run(run_once())

    assert events == [{"type": "text"}, {"type": "finished"}]


def test_iter_sse_events_records_malformed_payload_when_policy_is_record() -> None:
    events = list(iter_sse_events(["data: not json", ""], on_parse_error="record"))

    assert len(events) == 1
    assert events[0]["type"] == "parse_error"
    assert "invalid SSE JSON payload" in events[0]["error"]


def test_iter_sse_events_raises_on_malformed_payload_by_default() -> None:
    with pytest.raises(json.JSONDecodeError):
        list(iter_sse_events(["data: not json", ""], on_parse_error="raise"))


# --- dataset loaders (tiny inline fixtures) -----------------------------------


def test_trace_items_loads_rows_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"id": "1", "task": "text", "prompt": "hi"})
        + "\n\n"
        + json.dumps({"id": "2", "task": "t2i", "prompt": "draw", "width": 64, "height": 64})
        + "\n",
        encoding="utf-8",
    )

    rows = trace_items(path)

    assert [row["id"] for row in rows] == ["1", "2"]
    assert rows[1]["task"] == "t2i"
    assert rows[1]["width"] == 64


def test_load_dataset_rows_trace_caps_to_num_prompts(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": str(i), "task": "text", "prompt": f"p{i}"}) for i in range(5))
        + "\n",
        encoding="utf-8",
    )
    spec = BenchmarkSpec(
        task=TaskName.TEXT, model="M", dataset="trace", dataset_path=str(path), num_prompts=3
    )

    rows = load_dataset_rows(spec)

    assert [row["id"] for row in rows] == ["0", "1", "2"]


def test_benchmark_inputs_require_the_declared_request_count(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"id": "only", "task": "t2i", "prompt": "draw"}) + "\n",
        encoding="utf-8",
    )
    spec = BenchmarkSpec(
        task=TaskName.T2I,
        model="M",
        dataset="trace",
        dataset_path=str(path),
        num_prompts=2,
    )

    with pytest.raises(ValueError, match="requires exactly 2"):
        load_benchmark_inputs(spec)


def test_trace_items_rejects_row_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "1", "task": "text"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field 'prompt'"):
        trace_items(path)


def test_trace_items_rejects_unknown_task(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": "1", "task": "bogus", "prompt": "x"}) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unknown task"):
        trace_items(path)


def test_trace_items_rejects_width_without_height(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": "1", "task": "t2i", "prompt": "x", "width": 64}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="width and height together"):
        trace_items(path)


def test_load_ueval_local_jsonl_shapes_rows(tmp_path: Path) -> None:
    path = tmp_path / "ueval.jsonl"
    path.write_text(
        "\n".join(json.dumps({"prompt": f"ueval prompt {i}"}) for i in range(4)) + "\n",
        encoding="utf-8",
    )

    rows = load_ueval(str(path), num_requests=2, seed=7)

    assert len(rows) == 2
    assert all(row["task"] == "interleave" for row in rows)
    assert [row["id"] for row in rows] == ["ueval-000000", "ueval-000001"]
    # All sampled prompts come from the source set (sampling, not invention).
    source = {f"ueval prompt {i}" for i in range(4)}
    assert {row["prompt"] for row in rows} <= source


def test_load_ueval_sampling_is_seed_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "ueval.jsonl"
    path.write_text(
        "\n".join(json.dumps({"prompt": f"ueval prompt {i}"}) for i in range(6)) + "\n",
        encoding="utf-8",
    )

    first = load_ueval(str(path), num_requests=3, seed=7)
    second = load_ueval(str(path), num_requests=3, seed=7)

    assert [row["prompt"] for row in first] == [row["prompt"] for row in second]


def test_load_sharegpt_inline_filters_and_shapes_rows(tmp_path: Path) -> None:
    # Whitespace tokenizer => token count == word count. Row 3 has < 2 turns and
    # must be dropped; the two valid conversations survive with computed lengths.
    dataset = [
        {"conversations": [{"value": "hello there friend"}, {"value": "yes indeed ok"}]},
        {"conversations": [{"value": "short one"}, {"value": "a b c d e"}]},
        {"conversations": [{"value": "only one turn"}]},
    ]
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    rows = load_sharegpt(str(path), num_requests=10, tokenizer=_WhitespaceTokenizer(), seed=42)

    assert len(rows) == 2
    assert {row["task"] for row in rows} == {"text"}
    assert [row["id"] for row in rows] == ["sharegpt-000000", "sharegpt-000001"]
    by_prompt = {row["prompt"]: row for row in rows}
    assert by_prompt["hello there friend"]["prompt_len"] == 3
    assert by_prompt["hello there friend"]["output_len"] == 3
    assert by_prompt["short one"]["prompt_len"] == 2
    assert by_prompt["short one"]["output_len"] == 5


def test_load_sharegpt_download_pins_declared_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"value": "hello there friend"},
                        {"value": "yes indeed ok"},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_hf_hub_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(dataset_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    rows = load_sharegpt(
        "",
        num_requests=1,
        tokenizer=_WhitespaceTokenizer(),
        seed=42,
        revision="pinned-revision",
    )

    assert len(rows) == 1
    assert calls == [
        {
            "repo_id": "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "filename": "ShareGPT_V3_unfiltered_cleaned_split.json",
            "repo_type": "dataset",
            "revision": "pinned-revision",
        }
    ]


def test_load_dataset_rows_sharegpt_requires_tokenizer() -> None:
    spec = BenchmarkSpec(task=TaskName.TEXT, model="M", dataset="sharegpt")

    with pytest.raises(ValueError, match="requires a tokenizer"):
        load_dataset_rows(spec, tokenizer=None)


# --- build_summary schema -----------------------------------------------------


def test_build_summary_emits_documented_schema_for_image_task() -> None:
    spec = BenchmarkSpec(task=TaskName.T2I, model="M", num_prompts=2, request_rate=float("inf"))
    records = [
        RequestRecord(
            request_id="a", task="t2i", success=True, latency=2.0, images=1, classifier="ok"
        ),
        RequestRecord(
            request_id="b",
            task="t2i",
            success=False,
            latency=1.0,
            images=1,
            classifier="model_error",
        ),
    ]

    summary = build_summary(spec, "http://server:1", records, dur_s=10.0)

    assert set(summary) == {
        "harness_status",
        "task",
        "dataset",
        "endpoint",
        "wire",
        "model",
        "base_url",
        "server_info",
        "spec",
        "load",
        "elapsed_s",
        "request_count",
        "ok_count",
        "failed_count",
        "classifiers",
        "warnings",
        "metric_family",
        "metrics",
        "artifact",
    }
    assert summary["harness_status"] == "completed"
    assert summary["task"] == "t2i"
    assert summary["dataset"] == "mjhq"
    assert summary["base_url"] == "http://server:1"
    assert summary["metric_family"] == "image"
    assert summary["request_count"] == 2
    assert summary["ok_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["classifiers"] == {"ok": 1, "model_error": 1}
    assert summary["warnings"] == {"request_count": 0, "total_count": 0, "counts": {}}
    assert summary["elapsed_s"] == pytest.approx(10.0)
    assert summary["load"]["mode"] == "saturation"
    assert summary["load"]["request_rate"] == "inf"
    assert summary["metrics"]["completed_images"] == 1
    assert summary["artifact"]["valid"] is False
    assert summary["artifact"]["valid_marker"] is None


def test_build_summary_reports_observed_endpoint_for_single_wire() -> None:
    spec = BenchmarkSpec(task=TaskName.INTERLEAVE, model="M", num_prompts=1)
    records = [
        RequestRecord(
            request_id="a",
            task="default",
            success=True,
            latency=1.0,
            classifier="ok",
            endpoint="/v1/chat/completions",
        )
    ]

    summary = build_summary(spec, "http://x", records, dur_s=1.0)

    assert summary["endpoint"] == "/v1/chat/completions"
    assert summary["spec"]["endpoint"] == "/v1/chat/completions"


def test_text_artifact_requires_server_reported_exact_generation_work() -> None:
    spec = BenchmarkSpec(task=TaskName.TEXT, model="M", num_prompts=1)
    contract = benchmark_contract(spec, [{"id": "a"}])
    exact = RequestRecord(
        request_id="a",
        task="text",
        success=True,
        classifier="ok",
        finish_reason="length",
        prompt_len=7,
        output_len=11,
        requested_output_len=11,
        prompt_len_source="server_usage",
        output_len_source="server_usage",
        generated_text="answer",
    )
    summary = build_summary(spec, "http://x", [exact], dur_s=1.0, contract=contract)

    assert summary["artifact"]["checks"]["generation_conformance"] is True
    assert summary["artifact"]["generation_conformance"]["checked_requests"] == 1

    fallback = RequestRecord(**{**exact.__dict__, "output_len_source": "requested_fallback"})
    invalid = build_summary(spec, "http://x", [fallback], dur_s=1.0, contract=contract)
    assert invalid["artifact"]["checks"]["generation_conformance"] is False
    assert invalid["artifact"]["generation_conformance"]["mismatched_request_ids"] == ["a"]


def test_fixed_work_i2t_artifact_requires_server_reported_exact_generation_work() -> None:
    spec = BenchmarkSpec(
        task=TaskName.I2T,
        model="M",
        dataset="image-dir",
        dataset_path="unused",
        num_prompts=1,
        max_tokens=256,
        ignore_eos=True,
    )
    contract = benchmark_contract(spec, [{"id": "a"}])
    exact = RequestRecord(
        request_id="a",
        task="i2t",
        success=True,
        classifier="ok",
        finish_reason="length",
        prompt_len=17,
        output_len=256,
        requested_output_len=256,
        prompt_len_source="server_usage",
        output_len_source="server_usage",
        generated_text="answer",
    )

    summary = build_summary(spec, "http://x", [exact], dur_s=1.0, contract=contract)
    assert summary["artifact"]["generation_conformance"]["valid"] is True

    early_stop = RequestRecord(**{**exact.__dict__, "finish_reason": "stop", "output_len": 19})
    invalid = build_summary(spec, "http://x", [early_stop], dur_s=1.0, contract=contract)
    assert invalid["artifact"]["generation_conformance"]["valid"] is False
    assert invalid["artifact"]["generation_conformance"]["mismatched_request_ids"] == ["a"]


def test_natural_eos_i2t_does_not_require_fixed_output_length() -> None:
    spec = BenchmarkSpec(
        task=TaskName.I2T,
        model="M",
        dataset="image-dir",
        dataset_path="unused",
        num_prompts=1,
        max_tokens=256,
        ignore_eos=False,
    )
    contract = benchmark_contract(spec, [{"id": "a"}])
    natural_stop = RequestRecord(
        request_id="a",
        task="i2t",
        success=True,
        classifier="ok",
        finish_reason="stop",
        prompt_len=17,
        output_len=19,
        requested_output_len=256,
        prompt_len_source="server_usage",
        output_len_source="server_usage",
        generated_text="answer",
    )

    summary = build_summary(
        spec,
        "http://x",
        [natural_stop],
        dur_s=1.0,
        contract=contract,
    )
    assert summary["artifact"]["generation_conformance"]["valid"] is True
    assert summary["artifact"]["generation_conformance"]["policy"] == "successful_response"


def test_build_summary_selects_stream_family_for_text_task() -> None:
    spec = BenchmarkSpec(task=TaskName.TEXT, model="M", num_prompts=1, request_rate=float("inf"))
    records = [
        RequestRecord(
            request_id="a",
            task="text",
            success=True,
            latency=0.3,
            ttft=0.1,
            itl=[0.1, 0.1],
            output_len=3,
            prompt_len=4,
            classifier="ok",
            generated_text="a b c",
        )
    ]

    summary = build_summary(spec, "http://x", records, dur_s=10.0)

    assert summary["metric_family"] == "stream"
    assert summary["metrics"]["completed"] == 1


def test_build_summary_reports_stream_timing_attribution() -> None:
    spec = BenchmarkSpec(task=TaskName.TEXT, model="M", num_prompts=1, request_rate=1)
    records = [
        RequestRecord(
            request_id="a",
            task="text",
            success=True,
            classifier="ok",
            scheduled_time=0.9,
            start_time=1.0,
            http_response_time=1.2,
            first_text_time=1.5,
            latency=1.0,
            ttft=0.5,
            output_len=2,
            prompt_len=4,
            server_queued_at=100.0,
            server_scheduled_at=100.3,
        )
    ]

    summary = build_summary(spec, "http://x", records, dur_s=1.0)

    timing = summary["metrics"]["timing_attribution"]
    assert timing["client_dispatch_wait_ms"]["p50"] == pytest.approx(100.0)
    assert timing["http_response_ms"]["p50"] == pytest.approx(200.0)
    assert timing["stream_first_text_wait_ms"]["p50"] == pytest.approx(300.0)
    assert timing["server_queue_wait_ms"]["p50"] == pytest.approx(300.0)
    assert timing["ttft_residual_after_server_queue_ms"]["p50"] == pytest.approx(200.0)


# --- CLI exit-code mapping ----------------------------------------------------


class _StubRunner:
    """Stands in for the network-bound BenchmarkRunner at the CLI's public seam.

    Produces a real summary (via the production ``build_summary``) from the
    injected records so the CLI exercises its genuine pass/fail decision logic.
    """

    records: list[RequestRecord] = []

    def __init__(self, base_url: str, spec: BenchmarkSpec, output_dir) -> None:
        self.spec = spec
        self.output_dir = Path(output_dir)

    async def run(self) -> RunResult:
        rows = [{"id": record.request_id} for record in type(self).records]
        summary = build_summary(
            self.spec,
            "http://stub",
            type(self).records,
            dur_s=5.0,
            contract=benchmark_contract(self.spec, rows),
        )
        return RunResult(summary=summary, output_dir=self.output_dir)


def _run_cli_with_stub(monkeypatch, tmp_path, records: list[RequestRecord]) -> int:
    class Runner(_StubRunner):
        pass

    Runner.records = records
    monkeypatch.setattr(cli, "BenchmarkRunner", Runner)
    return cli.main(
        [
            "--base-url",
            "http://stub",
            "--task",
            "t2i",
            "--model",
            "M",
            "--output-dir",
            str(tmp_path),
            "--num-prompts",
            "1",
            "--warmup-requests",
            "0",
        ]
    )


def test_cli_returns_zero_when_all_requests_succeed(monkeypatch, tmp_path: Path) -> None:
    records = [_successful_image_record()]

    exit_code = _run_cli_with_stub(monkeypatch, tmp_path, records)

    assert exit_code == 0


def test_cli_returns_two_when_a_request_fails(monkeypatch, tmp_path: Path) -> None:
    records = [
        RequestRecord(
            request_id="a",
            task="t2i",
            success=False,
            latency=1.0,
            images=1,
            classifier="model_error",
        )
    ]

    exit_code = _run_cli_with_stub(monkeypatch, tmp_path, records)

    assert exit_code == 2
