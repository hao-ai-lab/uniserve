"""Executes one benchmark point and writes its complete result bundle.

`cli.run` calls `run_point` once per point while the point's server is up.
The bundle in the output directory contains:

- `run.json`: the lifecycle record, rewritten as the status moves from
  `preparing` to `running` (once dataset rows are selected) and then to
  `completed` or `failed`; a failure before row selection goes directly from
  `preparing` to `failed`.
- `warmup_requests.jsonl` and `requests.jsonl`: warmup and measured request
  records; only measured records feed metrics and validation. When the
  measured window fails or is interrupted, `requests.jsonl` holds the
  measured requests that finished before it, in completion order, and
  `run.json` is `failed`.
- `gpu_samples.jsonl`: `GpuStorageSampler` telemetry snapshots.
- `samples/`: decoded image and video outputs referenced by the records.
- `summary.json` and `summary.md`: the validated summary, written only on the
  completion path.

`run.json` is written last on both the completion and failure paths, so a
bundle is complete only when its `run.json` status is `completed`. If an
artifact write on the failure path raises, `run.json` keeps its earlier
status.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np

from ..artifacts import ArtifactWriter
from ..datasets import load_examples
from ..load import GpuStorageSampler, WarmupFailure, run_load
from ..tasks import get_task
from ..transport import send_request
from ..types import (
    BenchmarkPoint,
    Example,
    RequestRecord,
    RunResult,
    selected_rows_identity,
)
from .report import build_summary, render_markdown


async def run_point(
    base_url: str,
    point: BenchmarkPoint,
    output_dir: str | Path,
    *,
    launch: dict[str, Any] | None = None,
    timeout_s: float = 6 * 60 * 60.0,
) -> RunResult:
    """Run dataset loading, warmup, measured load, validation, and persistence.

    Args:
        base_url: The server origin, with or without a trailing slash.
        point: The resolved benchmark point to execute.
        output_dir: The bundle directory; it must be absent or empty.
        launch: The `describe_launch` provenance record, if any.
        timeout_s: The httpx timeout in seconds, applied to each connect,
            read, write, and pool wait of a benchmark request; the
            `/version` provenance fetch uses its own shorter timeout.

    Returns:
        The summary and bundle directory. A point whose validation fails
        still returns normally, with `validation.valid` false.

    Raises:
        FileExistsError: If `output_dir` is not empty; nothing is written.
        BaseException: Any error from task lookup, dataset loading, the load
            run (including `WarmupFailure` and cancellation), summarizing, or
            artifact writing is re-raised after the records collected so far
            and the failed `run.json` are written, provided the failure
            path's own artifact writes succeed.
    """  # noqa: E501
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"result directory is not empty: {output_path}")

    writer = ArtifactWriter(output_path)
    started_at = time.time()
    launch_record = launch or {}
    selection: dict[str, Any] | None = None
    warmup_records: list[RequestRecord] = []
    records: list[RequestRecord] = []
    sampler: GpuStorageSampler | None = None
    duration = 0.0

    # Empty streams and the preparing record make partial failures
    # inspectable with the same artifact names as completed points.
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
        # The task adapter and dataset rows are fixed before the measured
        # window opens, and the rows' identity is persisted with the running
        # state.
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

        # `get_request` in `uniserve_eval.load.arrival` draws exponential
        # inter-arrival gaps from the global NumPy generator, so seeding it
        # makes a finite-rate arrival schedule reproducible.
        np.random.seed(point.load.seed)

        # The connection pool is unbounded, so the only client-side limit on
        # in-flight requests is `run_load`'s optional `max_concurrency`
        # semaphore.
        limits = httpx.Limits(
            max_connections=None, max_keepalive_connections=None
        )
        async with httpx.AsyncClient(
            timeout=timeout_s, limits=limits
        ) as client:

            async def submit(
                example: Example, scheduled: float | None
            ) -> RequestRecord:
                """Build and send one task request with its token fallbacks.

                `scheduled` is None for warmup requests and the measured
                arrival time from `time.perf_counter` otherwise.
                """
                request = task.build_request(example)

                # The fallback becomes the record's `requested_output_len`;
                # it is 0 when neither the row nor `sampling.max_tokens`
                # declares an output limit.
                output_len_fallback = int(
                    example.output_len
                    if example.output_len is not None
                    else point.sampling.max_tokens or 0
                )
                record = await send_request(
                    client,
                    base_url.rstrip("/"),
                    request,
                    request_id=example.id,
                    task=point.task.value,
                    prompt_len=int(example.prompt_len or 0),
                    output_len_fallback=output_len_fallback,
                    scheduled_time=scheduled,
                )

                # Measured records are kept as they finish, so a load that
                # raises or is cancelled still leaves them for the failure
                # bundle. `run_load` cancels every outstanding submission
                # before it raises, so none is added afterwards.
                if scheduled is not None:
                    records.append(record)
                return record

            # Sampling spans warmup, the settle pause, and the measured
            # window.
            sampler = GpuStorageSampler()
            sampler.start()

            try:
                load_result = await run_load(
                    rows,
                    request_rate=point.load.request_rate,
                    max_concurrency=point.load.max_concurrency,
                    submit=submit,
                    warmup_requests=point.load.warmup_requests,
                )
            except WarmupFailure as error:
                # `WarmupFailure` is the only `run_load` exception that
                # carries outputs; keeping them lets the failure bundle
                # persist the warmup records.
                warmup_records = cast(list[RequestRecord], list(error.outputs))
                raise
            finally:
                sampler.stop()

            # Only measured outputs feed metrics; warmup records remain a
            # separate diagnostic stream. The completed load's outputs, in
            # submission order, replace the completion-order records that
            # `submit` collected.
            warmup_records = cast(
                list[RequestRecord], list(load_result.warmup_outputs)
            )
            records = cast(list[RequestRecord], list(load_result.outputs))
            duration = load_result.duration_s

            # Provenance is fetched after the measured window closes.
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

        if sampler.summary() is not None:
            summary["gpu_memory"] = sampler.summary()

        # The completed lifecycle record is written after every result
        # artifact so it acts as the bundle's commit marker.
        _write_records(writer, "warmup_requests.jsonl", warmup_records)
        _write_records(writer, "requests.jsonl", records)
        writer.write_jsonl("gpu_samples.jsonl", list(sampler.sample_records))
        writer.write_json("summary.json", summary)
        (output_path / "summary.md").write_text(
            render_markdown(summary), encoding="utf-8"
        )
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
        # The streams hold warmup records from a `WarmupFailure`, the
        # measured requests that finished before a failure or cancellation
        # inside the measured window, or both complete streams when the
        # failure came after `run_load` returned. `BaseException` also
        # covers cancellation and KeyboardInterrupt. `stop` is idempotent,
        # so a second call after the `finally` above is safe.
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
    """Persist decoded media samples and their request records.

    Samples are written under `samples/` before the JSONL stream that
    references them.

    Raises:
        ValueError: From `ArtifactWriter`, including `ImageOutputError` and
            `VideoOutputError`, when sample bytes fail inspection, do not
            match their recorded metadata, or collide with a different sample.
    """
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
    """Build the lifecycle record for the benchmark's current state.

    Optional fields are included only when provided, so a `preparing` record
    carries no row identity and only terminal records carry `completed_at`
    and `valid`.
    """
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


async def _fetch_server_version(
    client: httpx.AsyncClient, base_url: str
) -> dict[str, Any] | None:
    """Fetch optional server provenance without affecting benchmark completion.

    UniServe serves `GET /version`; any JSON object response is accepted.

    Returns:
        The response object, or None on a non-200 status, a body that is not
        a JSON object, or any `Exception` raised by the request or decoding.
    """  # noqa: E501
    # The short per-call timeout overrides the client's request timeout.
    try:
        response = await client.get(
            base_url.rstrip("/") + "/version", timeout=15.0
        )
        if response.status_code == 200 and isinstance(
            payload := response.json(), dict
        ):
            return payload
    except Exception:
        return None
    return None
