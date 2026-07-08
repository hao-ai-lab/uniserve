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
import json
from pathlib import Path

import numpy as np
import pytest

import uniserve_eval.harness.core.client as client_module
from uniserve_eval.harness import cli
from uniserve_eval.harness.core.client import _parse_openai
from uniserve_eval.harness.datasets import (
    load_dataset_rows,
    load_sharegpt,
    load_ueval,
    trace_items,
)
from uniserve_eval.harness.metrics import summarize_image
from uniserve_eval.harness.metrics.common import RequestRecord
from uniserve_eval.harness.report import build_summary
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
from uniserve_eval.harness.tasks.default import DefaultTask
from uniserve_eval.harness.tasks.i2t import I2TTask
from uniserve_eval.harness.tasks.t2i import T2ITask

pytestmark = [pytest.mark.unit]
TERMINAL_TYPES = frozenset({"finished"})


class _WhitespaceTokenizer:
    """Deterministic tokenizer: token count == whitespace word count."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


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
        {"choices": [{"delta": {"images": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}}]},
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
        {"choices": [{"delta": {"content": "a"}}], "_client_t": 11.0},
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
            ],
            "_client_t": 13.0,
        },
        {"choices": [{"delta": {"content": "b"}}], "_client_t": 14.0},
        {"choices": [{"delta": {"content": "c"}}], "_client_t": 14.25},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
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
    assert record.prompt_len == 7
    assert record.output_len == 3
    assert record.finish_reason == "stop"


def test_default_task_omits_image_cap_unless_explicit() -> None:
    uncapped = DefaultTask(
        BenchmarkSpec(
            task=TaskName.DEFAULT,
            model="SenseNova-U1",
            max_tokens=8192,
            width=2048,
            height=1152,
            steps=50,
        )
    ).build_request({"prompt": "show each step visually and textually"})

    assert uncapped.payload["image_config"] == {"width": 2048, "height": 1152, "steps": 50}

    capped = DefaultTask(
        BenchmarkSpec(
            task=TaskName.DEFAULT,
            model="SenseNova-U1",
            max_tokens=8192,
            max_images=8,
            width=2048,
            height=1152,
            steps=50,
        )
    ).build_request({"prompt": "show each step visually and textually"})

    assert capped.payload["image_config"]["num_images"] == 8


def test_default_task_can_emit_openai_chat_wire() -> None:
    request = DefaultTask(
        BenchmarkSpec(
            task=TaskName.DEFAULT,
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
    assert request.payload["messages"] == [{"role": "user", "content": "show each step visually and textually"}]
    assert request.payload["image_config"] == {"num_images": 4, "width": 2048, "height": 1152, "steps": 50}


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


def test_t2i_task_can_emit_image_only_chat_wire() -> None:
    request = T2ITask(
        BenchmarkSpec(
            task=TaskName.T2I,
            model="BAGEL",
            width=1024,
            height=1024,
            steps=50,
            wire="openai_chat_json",
        )
    ).build_request({"prompt": "a red bicycle"})

    assert request.endpoint == "/v1/chat/completions"
    assert request.kind == "openai_chat_json"
    assert request.payload["modalities"] == ["image"]
    assert request.payload["image_config"] == {
        "width": 1024,
        "height": 1024,
        "steps": 50,
        "seed": 42,
    }


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
                                    "image_url": {"url": "data:image/png;base64,QUJD"},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0},
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


# --- SSE framing --------------------------------------------------------------


def test_iter_sse_events_frames_multiple_records() -> None:
    lines = [
        "data: {\"type\": \"text\", \"value\": \"a\"}",
        "",
        "data: {\"type\": \"finished\"}",
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
        "data: {\"type\":",
        "data:  \"finished\"}",
        "",
    ]

    events = list(iter_sse_events(lines))

    assert events == [{"type": "finished"}]


def test_iter_sse_events_strips_optional_leading_space_consistently() -> None:
    no_space = list(iter_sse_events(["data:{\"a\":1}", ""]))
    one_space = list(iter_sse_events(["data: {\"a\":1}", ""]))

    assert no_space == [{"a": 1}]
    assert no_space == one_space


def test_iter_sse_events_ignores_comments_and_non_data_lines() -> None:
    lines = [": keep-alive", "event: message", "data: {\"k\": 1}", ""]

    events = list(iter_sse_events(lines))

    assert events == [{"k": 1}]


def test_iter_sse_events_flushes_trailing_record_without_blank_line() -> None:
    events = list(iter_sse_events(["data: {\"k\": 2}"]))

    assert events == [{"k": 2}]


def test_iter_sse_events_done_sentinel_becomes_synthetic_event() -> None:
    events = list(iter_sse_events(["data: [DONE]", ""]))

    assert events == [{"type": "sse_done"}]


def test_iter_sse_events_stop_on_terminal_halts_after_first_terminal() -> None:
    lines = [
        "data: {\"type\": \"text\"}",
        "",
        "data: {\"type\": \"finished\"}",
        "",
        "data: {\"type\": \"text\", \"after\": true}",
        "",
    ]

    events = list(iter_sse_events(lines, stop_on=TERMINAL_TYPES))

    assert events == [{"type": "text"}, {"type": "finished"}]


def test_iter_sse_events_stop_on_terminal_without_trailing_blank_line() -> None:
    lines = [
        "data: {\"type\": \"text\"}",
        "",
        "data: {\"type\": \"finished\"}",
        "data: {\"type\": \"text\", \"after\": true}",
        "",
    ]

    events = list(iter_sse_events(lines, stop_on=TERMINAL_TYPES))

    assert events == [{"type": "text"}, {"type": "finished"}]


def test_aiter_sse_events_stop_on_terminal_without_waiting_for_eof() -> None:
    async def lines() -> object:
        yield "data: {\"type\": \"text\"}"
        yield ""
        yield "data: {\"type\": \"finished\"}"
        await asyncio.sleep(60.0)
        yield "data: {\"type\": \"text\", \"after\": true}"

    async def run_once() -> list[dict[str, object]]:
        return await asyncio.wait_for(
            aiter_sse_events(lines(), stop_on=TERMINAL_TYPES),
            timeout=0.25,
        )

    events = asyncio.run(run_once())

    assert events == [{"type": "text"}, {"type": "finished"}]


def test_aiter_sse_events_from_text_stops_on_terminal_without_newline_or_eof() -> None:
    async def chunks() -> object:
        yield "data: {\"type\": \"text\"}\n\n"
        yield "data: {\"type\": \"fin"
        yield "ished\"}"
        await asyncio.sleep(60.0)
        yield "\n\ndata: {\"type\": \"text\", \"after\": true}\n\n"

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
        "\n".join(
            json.dumps({"id": str(i), "task": "text", "prompt": f"p{i}"}) for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    spec = BenchmarkSpec(
        task=TaskName.TEXT, model="M", dataset="trace", dataset_path=str(path), num_prompts=3
    )

    rows = load_dataset_rows(spec)

    assert [row["id"] for row in rows] == ["0", "1", "2"]


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
    assert all(row["task"] == "default" for row in rows)
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


def test_load_dataset_rows_sharegpt_requires_tokenizer() -> None:
    spec = BenchmarkSpec(task=TaskName.TEXT, model="M", dataset="sharegpt")

    with pytest.raises(ValueError, match="requires a tokenizer"):
        load_dataset_rows(spec, tokenizer=None)


# --- build_summary schema -----------------------------------------------------


def test_build_summary_emits_documented_schema_for_image_task() -> None:
    spec = BenchmarkSpec(
        task=TaskName.T2I, model="M", num_prompts=2, request_rate=float("inf")
    )
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
        "metric_family",
        "metrics",
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
    assert summary["elapsed_s"] == pytest.approx(10.0)
    assert summary["load"]["mode"] == "saturation"
    assert summary["load"]["request_rate"] == "inf"
    assert summary["metrics"]["completed_images"] == 1


def test_build_summary_reports_observed_endpoint_for_single_wire() -> None:
    spec = BenchmarkSpec(task=TaskName.DEFAULT, model="M", num_prompts=1)
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


def test_build_summary_selects_stream_family_for_text_task() -> None:
    spec = BenchmarkSpec(
        task=TaskName.TEXT, model="M", num_prompts=1, request_rate=float("inf")
    )
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
        summary = build_summary(self.spec, "http://stub", type(self).records, dur_s=5.0)
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
    records = [
        RequestRecord(
            request_id="a", task="t2i", success=True, latency=1.0, images=1, classifier="ok"
        )
    ]

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
