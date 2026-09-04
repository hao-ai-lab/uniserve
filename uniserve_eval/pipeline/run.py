"""Executes one benchmark point and writes its complete result bundle."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np

from ..artifacts import ArtifactWriter
from ..datasets import load_examples
from ..load import GpuMemorySampler, WarmupFailure, run_load
from ..nsys import NsysCapture
from ..tasks import get_task
from ..transport import send_request
from ..types import BenchmarkPoint, Example, RequestRecord, RunResult, selected_rows_identity
from .report import build_summary, render_markdown


async def run_point(
    base_url: str,
    point: BenchmarkPoint,
    output_dir: str | Path,
    *,
    launch: dict[str, Any] | None = None,
    timeout_s: float = 6 * 60 * 60.0,
    measurement: NsysCapture | None = None,
) -> RunResult:
    """Run dataset loading, warmup, measured load, validation, and persistence."""

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"result directory is not empty: {output_path}")

    writer = ArtifactWriter(output_path)
    started_at = time.time()
    launch_record = launch or {}
    selection: dict[str, Any] | None = None
    warmup_records: list[RequestRecord] = []
    records: list[RequestRecord] = []
    sampler: GpuMemorySampler | None = None
    duration = 0.0

    # Empty streams and the preparing record make partial failures inspectable with
    # the same artifact names as completed points.
    _write_empty_streams(writer)
    writer.write_json(
        "run.json",
        _run_state(
            "preparing",
            point,
            started_at=started_at,
            launch=launch_record,
        ),
    )

    try:
        # Dataset selection and request construction are fixed before the measured
        # window opens, and their identity is persisted with the running state.
        task = get_task(point.task)(point)
        rows, tokenizer = load_examples(point)
        selection = selected_rows_identity(rows)
        writer.write_json(
            "run.json",
            _run_state(
                "running",
                point,
                started_at=started_at,
                launch=launch_record,
                selected_rows=selection,
            ),
        )

        np.random.seed(point.load.seed)
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:

            async def submit(example: Example, scheduled: float | None) -> RequestRecord:
                """Build and send one task request with its timing fallbacks."""

                request = task.build_request(example)
                output_len_fallback = int(
                    example.output_len
                    if example.output_len is not None
                    else point.sampling.max_tokens or 0
                )
                return await send_request(
                    client,
                    base_url.rstrip("/"),
                    request,
                    request_id=example.id,
                    task=point.task.value,
                    prompt_len=int(example.prompt_len or 0),
                    output_len_fallback=output_len_fallback,
                    scheduled_time=scheduled,
                )

            sampler = GpuMemorySampler()
            sampler.start()

            try:
                load_result = await run_load(
                    rows,
                    request_rate=point.load.request_rate,
                    max_concurrency=point.load.max_concurrency,
                    submit=submit,
                    warmup_requests=point.load.warmup_requests,
                    measurement=measurement,
                )
            except WarmupFailure as error:
                warmup_records = cast(list[RequestRecord], list(error.outputs))
                raise
            finally:
                sampler.stop()

            # Only measured outputs feed metrics; warmup records remain a separate
            # diagnostic stream.
            warmup_records = cast(list[RequestRecord], list(load_result.warmup_outputs))
            records = cast(list[RequestRecord], list(load_result.outputs))
            duration = load_result.duration_s
            server_version = await _fetch_server_version(client, base_url)

        summary = build_summary(
            point,
            base_url.rstrip("/"),
            records,
            duration,
            task=task,
            selected_rows=selection,
            tokenizer=tokenizer,
            server_version=server_version,
            launch=launch_record,
        )

        # The completed lifecycle record is written after every result artifact so
        # it acts as the bundle's commit marker.
        if sampler.summary() is not None:
            summary["gpu_memory"] = sampler.summary()
        _write_records(writer, "warmup_requests.jsonl", warmup_records)
        _write_records(writer, "requests.jsonl", records)
        writer.write_jsonl("gpu_samples.jsonl", list(sampler.sample_records))
        writer.write_json("summary.json", summary)
        (output_path / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
        writer.write_json(
            "run.json",
            _run_state(
                "completed",
                point,
                started_at=started_at,
                completed_at=time.time(),
                valid=summary["validation"]["valid"],
                launch=launch_record,
                selected_rows=selection,
            ),
        )
        return RunResult(summary, output_path)
    except BaseException as error:
        # Failure artifacts preserve every record collected before the exception.
        if sampler is not None:
            sampler.stop()
        _write_records(writer, "warmup_requests.jsonl", warmup_records)
        _write_records(writer, "requests.jsonl", records)
        writer.write_jsonl(
            "gpu_samples.jsonl",
            list(sampler.sample_records) if sampler is not None else [],
        )
        failure = _run_state(
            "failed",
            point,
            started_at=started_at,
            completed_at=time.time(),
            valid=False,
            launch=launch_record,
            selected_rows=selection,
        )
        failure["error"] = {"type": type(error).__name__, "message": str(error)}
        if sampler is not None and sampler.summary() is not None:
            failure["gpu_memory"] = sampler.summary()
        writer.write_json("run.json", failure)
        raise


def _write_empty_streams(writer: ArtifactWriter) -> None:
    """Create durable empty record streams before fallible benchmark work."""

    writer.write_jsonl("warmup_requests.jsonl", [])
    writer.write_jsonl("requests.jsonl", [])
    writer.write_jsonl("gpu_samples.jsonl", [])


def _write_records(
    writer: ArtifactWriter,
    name: str,
    records: list[RequestRecord],
) -> None:
    """Persist decoded media samples and their request records."""

    for record in records:
        for image in record.decoded_images:
            writer.write_image_sample(image)
        if record.decoded_video is not None:
            writer.write_video_sample(record.decoded_video)
    writer.write_jsonl(name, [record.record_dict() for record in records])


def _run_state(
    status: str,
    point: BenchmarkPoint,
    *,
    started_at: float,
    launch: dict[str, Any],
    selected_rows: dict[str, Any] | None = None,
    completed_at: float | None = None,
    valid: bool | None = None,
) -> dict[str, Any]:
    """Build the lifecycle record for the benchmark's current state."""

    state: dict[str, Any] = {
        "status": status,
        "benchmark": point.name,
        "workload": point.workload_dict(),
        "started_at": started_at,
        "launch": launch,
    }
    if selected_rows is not None:
        state["selected_rows"] = selected_rows
    if completed_at is not None:
        state["completed_at"] = completed_at
    if valid is not None:
        state["valid"] = valid
    return state


async def _fetch_server_version(client: httpx.AsyncClient, base_url: str) -> dict[str, Any] | None:
    """Fetch optional server provenance without affecting benchmark completion."""

    try:
        response = await client.get(base_url.rstrip("/") + "/version", timeout=15.0)
        if response.status_code == 200 and isinstance(payload := response.json(), dict):
            return payload
    except Exception:
        return None
    return None
