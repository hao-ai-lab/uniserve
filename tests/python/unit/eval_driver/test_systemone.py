"""Decision-readout points: corpus rows, requests, answers, and rates."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from tests.python.fixtures.http_stub import stub_server
from uniserve_eval.datasets.systemone import SystemOneDataset
from uniserve_eval.metrics import summarize
from uniserve_eval.pipeline.run import run_point
from uniserve_eval.tasks.systemone import SystemOneTask
from uniserve_eval.transport.client import send_request
from uniserve_eval.types import (
    DJEV_EVALUATE,
    SYSTEMONE,
    BenchmarkPoint,
    Example,
    LoadConfig,
    MetricDefinition,
    RequestRecord,
    TaskName,
)

pytestmark = pytest.mark.unit

# A NanoJev corpus row: request content plus evaluation-only fields.
_CORPUS_ROW = {
    "id": "home:0",
    "state": "Home log: user wants the office fan set to off.",
    "questions": {
        "device": {
            "type": "choice",
            "instructions": "Select the device named in the request.",
            "criteria": {"unit_0": "bedroom speaker", "unit_1": "office fan"},
        },
        "execute": {"type": "boolean", "instructions": "Is execution allowed?"},
        "risk": {
            "type": "score",
            "instructions": "Classify the request.",
            "criteria": ["Authorized", "Not authorized"],
        },
    },
    "gold": {"device": "unit_1", "execute": True, "risk": 0},
    "teacher": {"native_probs": {"execute": {"true": 0.96}}},
    "metadata": {"source": "programmatic"},
}


def _point(
    *,
    endpoint: str = SYSTEMONE,
    dataset_path: str | None = None,
    num_prompts: int = 1,
) -> BenchmarkPoint:
    return BenchmarkPoint(
        name="readout",
        server="server",
        task=TaskName.SYSTEMONE,
        model="diffusiongemma",
        dataset="systemone",
        dataset_path=dataset_path,
        endpoint=endpoint,
        metrics=(MetricDefinition(("questions_per_second",), "higher"),),
        load=LoadConfig(num_prompts=num_prompts, warmup_requests=1),
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_session_trace_keeps_order_and_session_identity(tmp_path: Path) -> None:
    dataset = tmp_path / "trace.jsonl"
    raw = [
        {**_CORPUS_ROW, "id": f"{bot}:{step}", "session_id": bot}
        for step in range(3)
        for bot in ("a", "b")
    ]
    _write_rows(dataset, raw)
    rows = SystemOneDataset(
        _point(dataset_path=str(dataset), num_prompts=6)
    ).load()

    assert [row.id for row in rows] == [row["id"] for row in raw]
    assert [row.session_id for row in rows] == [
        row["session_id"] for row in raw
    ]
    task = SystemOneTask(_point())
    assert all(
        "session_id" not in task.build_request(row).payload for row in rows
    )


@pytest.mark.parametrize("session", ["", 0, [], None])
def test_incomplete_or_invalid_session_trace_fails(
    tmp_path: Path, session
) -> None:
    dataset = tmp_path / "trace.jsonl"
    _write_rows(
        dataset,
        [
            {**_CORPUS_ROW, "id": "a:0", "session_id": "a"},
            {**_CORPUS_ROW, "id": "b:0", "session_id": session},
        ],
    )
    with pytest.raises(ValueError, match="session_id"):
        SystemOneDataset(
            _point(dataset_path=str(dataset), num_prompts=2)
        ).load()


def test_corpus_rows_carry_only_the_request_in_the_official_types(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rows.jsonl"
    _write_rows(
        dataset, [{**_CORPUS_ROW, "images": ["data:image/png;base64,AA"]}]
    )

    (row,) = SystemOneDataset(_point(dataset_path=str(dataset))).load()

    assert row.state == _CORPUS_ROW["state"]
    assert row.images == ["data:image/png;base64,AA"]
    assert list(row.questions or {}) == ["device", "execute", "risk"]
    assert (row.questions or {})["execute"] == {
        "type": "noul",
        "instructions": "Is execution allowed?",
    }
    assert "gold" not in json.dumps(row.as_dict())
    assert "teacher" not in json.dumps(row.as_dict())


def test_corpus_order_is_seeded_and_the_selection_is_a_prefix(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rows.jsonl"
    _write_rows(
        dataset,
        [{**_CORPUS_ROW, "id": f"row-{index}"} for index in range(20)],
    )

    first = SystemOneDataset(
        _point(dataset_path=str(dataset), num_prompts=20)
    ).load()
    again = SystemOneDataset(
        _point(dataset_path=str(dataset), num_prompts=20)
    ).load()
    prefix = SystemOneDataset(
        _point(dataset_path=str(dataset), num_prompts=5)
    ).load()

    ids = [row.id for row in first]
    assert ids == [row.id for row in again]
    assert sorted(ids) == sorted(f"row-{index}" for index in range(20))
    assert ids != [f"row-{index}" for index in range(20)]
    assert [row.id for row in prefix] == ids[:5]


def test_unknown_question_type_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "rows.jsonl"
    row = json.loads(json.dumps(_CORPUS_ROW))
    row["questions"]["risk"]["type"] = "ranking"
    _write_rows(dataset, [row])

    with pytest.raises(ValueError, match="unsupported type 'ranking'"):
        SystemOneDataset(_point(dataset_path=str(dataset))).load()


def _example() -> Example:
    return Example(
        id="home:0",
        prompt="",
        state="state text",
        questions={
            "execute": {"type": "noul", "instructions": "Allowed?"},
            "device": {
                "type": "choice",
                "criteria": {"unit_0": "fan", "unit_1": None},
            },
        },
        images=["data:image/jpeg;base64,AA"],
    )


def test_systemone_request_follows_the_official_schema() -> None:
    request = SystemOneTask(_point()).build_request(_example())

    assert request.endpoint == SYSTEMONE
    assert request.stream is False
    assert request.payload == {
        "model": "diffusiongemma",
        "state": "state text",
        "questions": _example().questions,
        "x_images": ["data:image/jpeg;base64,AA"],
    }


def test_djev_request_carries_one_state_in_its_schema() -> None:
    request = SystemOneTask(_point(endpoint=DJEV_EVALUATE)).build_request(
        _example()
    )

    assert request.endpoint == DJEV_EVALUATE
    assert request.payload == {
        "states": [
            {
                "id": "home:0",
                "state": "state text",
                "questions": {
                    "execute": {"type": "boolean", "instructions": "Allowed?"},
                    "device": {
                        "type": "choice",
                        "criteria": {"unit_0": "fan", "unit_1": None},
                    },
                },
                "images": ["data:image/jpeg;base64,AA"],
            }
        ]
    }


def _send(endpoint: str, body: dict[str, Any]) -> RequestRecord:
    """Send the example's readout to a server that answers with `body`."""

    async def handler(request: web.Request) -> web.Response:
        assert request.path == endpoint
        return web.json_response(body)

    async def run() -> RequestRecord:
        async with stub_server(handler) as base_url:
            async with aiohttp.ClientSession() as session:
                task = SystemOneTask(_point(endpoint=endpoint))
                request = task.build_request(_example())
                return await send_request(
                    session, base_url, request, "home:0", task="systemone"
                )

    return asyncio.run(run())


_ANSWERS = {
    "execute": {"type": "noul", "noul": 0.9},
    "device": {
        "type": "choice",
        "choice": "unit_0",
        "probabilities": {"unit_0": 0.8, "unit_1": 0.2},
        "confidence": 0.6,
    },
}


def test_systemone_answers_and_usage_are_recorded() -> None:
    record = _send(
        SYSTEMONE,
        {
            "model": "diffusiongemma",
            "answers": _ANSWERS,
            "usage": {"input_tokens": 612, "output_tokens": 0},
        },
    )

    assert record.success is True
    assert record.answers == _ANSWERS
    assert (record.decision_states, record.decision_questions) == (1, 2)
    assert (record.prompt_len, record.prompt_len_source) == (
        612,
        "server_usage",
    )
    assert record.output_len == 0
    assert record.token_timing_available is False


def test_unanswered_question_fails_the_readout() -> None:
    record = _send(
        SYSTEMONE,
        {
            "model": "diffusiongemma",
            "answers": {"execute": _ANSWERS["execute"]},
            "usage": {"input_tokens": 612, "output_tokens": 0},
        },
    )

    assert record.success is False
    assert record.classifier == "incomplete_answers"


@pytest.mark.parametrize("cached", [None, 0, 128])
def test_djev_answers_are_keyed_by_question_with_state_prompt_tokens(
    cached,
) -> None:
    record = _send(
        DJEV_EVALUATE,
        {
            "model": "djev",
            "states": [
                {
                    "id": "home:0",
                    "answers": _ANSWERS,
                    "prompt_tokens": 598,
                    **(
                        {"cached_prompt_tokens": cached}
                        if cached is not None
                        else {}
                    ),
                }
            ],
        },
    )

    assert record.success is True
    assert record.answers == _ANSWERS
    assert (record.prompt_len, record.prompt_len_source) == (
        598,
        "server_usage",
    )
    assert record.decision_questions == 2
    assert record.cached_prompt_tokens == cached
    assert record.cached_prompt_tokens_source == (
        "djev_states_cached_prompt_tokens"
        if cached is not None
        else "unavailable"
    )


def test_readout_rates_count_answered_states_and_questions() -> None:
    records = [
        RequestRecord(
            request_id=f"row-{index}",
            task="systemone",
            success=True,
            latency=0.1,
            decision_states=1,
            decision_questions=3,
        )
        for index in range(4)
    ]
    records.append(
        RequestRecord(request_id="failed", task="systemone", success=False)
    )

    summary = summarize(records, 2.0)

    assert summary["request_throughput"] == 2.0
    assert summary["states_per_second"] == 2.0
    assert summary["questions_per_second"] == 6.0
    assert summary["completed_questions"] == 12


class _ReadoutServer(BaseHTTPRequestHandler):
    """Answers System One readouts and counts them in `/metrics`."""

    readouts = 0

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler method name.
        body = (
            "# TYPE uniserve:scheduler_domain_time_us counter\n"
            'uniserve:scheduler_domain_time_us_total{domain="token_denoising",'
            f'phase="device"}} {125 * type(self).readouts}\n'
            f"uniserve:num_requests_running {type(self).readouts}\n"
            "# EOF\n"
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/openmetrics-text")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name.
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        type(self).readouts += 1
        answers = {
            question_id: {"type": "noul", "noul": 0.5}
            for question_id in request["questions"]
        }
        body = json.dumps(
            {
                "model": request["model"],
                "answers": answers,
                "usage": {"input_tokens": 300, "output_tokens": 0},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def test_readout_point_records_window_counters_and_rates(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rows.jsonl"
    _write_rows(
        dataset, [{**_CORPUS_ROW, "id": f"row-{index}"} for index in range(3)]
    )
    point = _point(dataset_path=str(dataset), num_prompts=3)
    _ReadoutServer.readouts = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReadoutServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "result"
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        result = asyncio.run(run_point([base_url], point, output))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    summary = result.summary
    assert summary["validation"]["valid"] is True
    assert summary["metrics"]["completed_questions"] == 9
    # The warmup readout precedes the window, so the counters advanced by
    # exactly the three measured readouts; the gauge is not a counter.
    assert summary["server_metrics"][base_url] == {
        "available": True,
        "counter_deltas": {
            'uniserve:scheduler_domain_time_us_total{domain="token_denoising",'
            'phase="device"}': 375.0
        },
    }
    snapshots = json.loads((output / "server_metrics.json").read_text())
    assert base_url in snapshots["before"]
    assert base_url in snapshots["after"]
    records = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    assert [record["decision_questions"] for record in records] == [3, 3, 3]
