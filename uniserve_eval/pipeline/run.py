"""Executes one benchmark point and writes its complete result bundle.

`cli.run` calls `run_point` once per point while the point's server is up.
The bundle in the output directory contains:

- `run.json`: the lifecycle record, rewritten as the status moves from
  `preparing` to `running` (once dataset rows are selected) and then to
  `completed` or `failed`; a failure before row selection goes directly from
  `preparing` to `failed`.
- `warmup_requests.jsonl`, `priming_requests.jsonl`, and `requests.jsonl`:
  excluded warmup, excluded queue priming, and measured request
  records; only measured records feed metrics and validation. When the
  measured window fails or is interrupted, `requests.jsonl` holds the
  measured requests that finished before it, in completion order, and
  `run.json` is `failed`.
- `gpu_samples.jsonl`: `GpuStorageSampler` telemetry snapshots.
- `samples/`: image outputs and original video response bodies, including
  invalid bodies. A configured `video.media_dir` holds measured videos.
- `media_index.jsonl` and `media_validity.jsonl`: original-output references,
  checksums, request identities, timing records, and validity checks.
- `summary.json` and `summary.md`: the validated summary, written only on the
  completion path.

`run.json` is written last on both the completion and failure paths, so a
bundle is complete only when its `run.json` status is `completed`. The failed
`run.json` is written even when the failure path cannot persist the record
streams; it then carries that persistence error as `artifact_error`.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import resource
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np

from ..artifacts import ArtifactWriter
from ..datasets import load_examples
from ..load import GpuStorageSampler, WarmupFailure, run_load
from ..tasks import get_task
from ..tasks.video import VideoTask
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
    base_urls: Sequence[str],
    point: BenchmarkPoint,
    output_dir: str | Path,
    *,
    launch: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> RunResult:
    """Run dataset loading, warmup, measured load, validation, and persistence.

    Args:
        base_urls: The deployment's server origins, with or without a
            trailing slash. With several replicas, `ReplicaRouter` sends
            each request to one of them.
        point: The resolved benchmark point to execute.
        output_dir: The bundle directory; it must be absent or empty.
        launch: The `describe_deployment` provenance record, if any.
        timeout_s: Override the profile's logical video request deadline and
            HTTP operation timeout. Provenance uses a separate short timeout.

    Returns:
        The summary and bundle directory. A point whose validation fails
        still returns normally, with `validation.valid` false.

    Raises:
        FileExistsError: If `output_dir` is not empty; nothing is written.
        BaseException: Any error from task lookup, dataset loading, the load
            run (including `WarmupFailure` and cancellation), summarizing, or
            artifact writing is re-raised after the records collected so far
            and the failed `run.json` are written. An error while persisting
            the records there is recorded in `run.json` rather than raised;
            only a failure to write `run.json` itself replaces the original
            error.
    """  # noqa: E501
    output_path = Path(output_dir)
    timeout_s = point.load.request_timeout_s if timeout_s is None else timeout_s
    router = ReplicaRouter([url.rstrip("/") for url in base_urls])
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"result directory is not empty: {output_path}")

    writer = ArtifactWriter(output_path)
    started_at = time.time()
    launch_record = launch or {}
    selection: dict[str, Any] | None = None
    warmup_records: list[RequestRecord] = []
    priming_records: list[RequestRecord] = []
    records: list[RequestRecord] = []
    sampler: GpuStorageSampler | None = None
    duration = 0.0
    phase = "measured"

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
        if len({row.id for row in rows}) != len(rows):
            raise ValueError("request IDs must be unique within a manifest")
        warmup_rows = _excluded_rows(point, point.load.warmup_manifest)
        priming_rows = _excluded_rows(point, point.load.priming_manifest)
        # Frozen manifests can carry thousands of token IDs per row. Copying
        # those provenance fields belongs before dispatch, outside every slot.
        all_rows = [*rows, *(warmup_rows or []), *(priming_rows or [])]
        examples = {id(row): row.as_dict() for row in all_rows}
        video_requests = (
            {id(row): task.build_request(row) for row in all_rows}
            if isinstance(task, VideoTask)
            else {}
        )
        selection = selected_rows_identity(rows)
        writer.write_jsonl("examples.jsonl", [row.as_dict() for row in rows])
        writer.write_json(
            "client.json",
            {
                "request_timeout_s": timeout_s,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "connection_limits": None,
                "keepalive_expiry_s": None,
                "media_buffer": "host_memory",
                "warmup_rows": selected_rows_identity(warmup_rows or []),
                "priming_rows": selected_rows_identity(priming_rows or []),
            },
        )
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
            max_connections=None,
            max_keepalive_connections=None,
            keepalive_expiry=None,
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
                request = video_requests.get(id(example))
                if request is None:
                    request = task.build_request(example)

                # The fallback becomes the record's `requested_output_len`;
                # it is 0 when neither the row nor `sampling.max_tokens`
                # declares an output limit.
                output_len_fallback = int(
                    example.output_len
                    if example.output_len is not None
                    else point.sampling.max_tokens or 0
                )
                # Routing happens inside the submission slot, so a request
                # waits on its chosen replica like on any queueing server.
                with router.route() as base_url:
                    record = await send_request(
                        client,
                        base_url,
                        request,
                        request_id=example.id,
                        task=point.task.value,
                        prompt_len=int(example.prompt_len or 0),
                        output_len_fallback=output_len_fallback,
                        scheduled_time=scheduled,
                        timeout_s=timeout_s,
                    )
                record.example = examples[id(example)]

                # Measured records are kept as they finish, so a load that
                # raises or is cancelled still leaves them for the failure
                # bundle. `run_load` cancels every outstanding submission
                # before it raises, so none is added afterwards.
                if scheduled is None:
                    warmup_records.append(record)
                elif phase == "priming":
                    priming_records.append(record)
                else:
                    records.append(record)
                return record

            # Sampling spans excluded requests and the measured window.
            sampler = GpuStorageSampler()
            sampler.start()

            def inspect(records: list[RequestRecord]) -> None:
                if isinstance(task, VideoTask):
                    for record in records:
                        task.inspect_output(record)

            try:
                if priming_rows:
                    phase = "priming"
                    # Shape warmup precedes queue priming. Its validation is
                    # outside all slots, and priming has no subsequent pause.
                    primed = await run_load(
                        priming_rows,
                        request_rate=float("inf"),
                        max_concurrency=point.load.max_concurrency,
                        submit=submit,
                        warmup_requests=point.load.warmup_requests,
                        warmup_rows=warmup_rows,
                        inspect_warmup=inspect,
                    )
                    warmup_records = list(primed.warmup_outputs)
                    priming_records = list(primed.outputs)
                    phase = "measured"
                    if not all(record.success for record in priming_records):
                        raise WarmupFailure(priming_records)
                    # Priming is excluded from metrics. Inspection runs after
                    # measurement to preserve the primed queues and caches.
                load_result = await run_load(
                    rows,
                    request_rate=point.load.request_rate,
                    max_concurrency=point.load.max_concurrency,
                    submit=submit,
                    warmup_requests=0
                    if priming_rows
                    else point.load.warmup_requests,
                    warmup_rows=[] if priming_rows else warmup_rows,
                    inspect_warmup=inspect,
                )
            finally:
                sampler.stop()

            # Only measured outputs feed metrics; warmup records remain a
            # separate diagnostic stream. The completed load's outputs, in
            # submission order, replace the completion-order records that
            # `submit` collected.
            if not priming_rows:
                warmup_records = list(load_result.warmup_outputs)
            records = cast(list[RequestRecord], list(load_result.outputs))
            duration = load_result.duration_s
            inspect(records)
            inspect(priming_records)

            # Provenance is fetched after the measured window closes.
            server_version = await _fetch_server_version(
                client, router.base_urls[0]
            )

        summary = build_summary(
            point,
            router.base_urls,
            records,
            duration,
            task=task,
            selected_rows=selection,
            tokenizer=tokenizer,
            server_version=server_version,
            launch=launch_record,
        )
        if priming_records:
            summary["validation"]["checks"]["priming_valid"] = all(
                record.success for record in priming_records
            )
            summary["validation"]["valid"] = all(
                summary["validation"]["checks"].values()
            )

        if sampler.summary() is not None:
            summary["gpu_memory"] = sampler.summary()
        summary["client"] = {
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "retained_body_bytes": sum(
                len(record.video_body or b"")
                for record in [*warmup_records, *priming_records, *records]
            ),
        }

        # The completed lifecycle record is written after every result
        # artifact so it acts as the bundle's commit marker.
        _write_records(writer, "warmup_requests.jsonl", warmup_records)
        _write_records(writer, "priming_requests.jsonl", priming_records)
        _write_records(writer, "requests.jsonl", records, point.video.media_dir)
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

        # Persisting the streams fails again when the original error came
        # from persisting them, e.g. a media sample that cannot be written
        # or fails its checks. That second error is recorded instead of
        # replacing the original one, so the failed state is still written
        # and the original error, including a cancellation, propagates.
        try:
            writer.write_jsonl(
                "gpu_samples.jsonl",
                list(sampler.sample_records) if sampler is not None else [],
            )
            _write_records(writer, "warmup_requests.jsonl", warmup_records)
            _write_records(writer, "priming_requests.jsonl", priming_records)
            _write_records(
                writer, "requests.jsonl", records, point.video.media_dir
            )
        except Exception as artifact_error:
            failure["artifact_error"] = {
                "type": type(artifact_error).__name__,
                "message": str(artifact_error),
            }
        writer.write_json("run.json", failure)
        raise


class ReplicaRouter:
    """Route each request to the replica with the fewest in-flight requests.

    This is client-side dispatch over independent replica endpoints, the
    least-outstanding-requests policy of common HTTP load balancers. Ties
    resolve in rotation starting after the previous choice, so an idle
    deployment receives requests round-robin. A single origin always wins.
    """

    def __init__(self, base_urls: Sequence[str]) -> None:
        """Track the in-flight count of each origin, in the given order."""
        if not base_urls:
            raise ValueError("a deployment requires at least one origin")
        self.base_urls = tuple(base_urls)
        self.active = [0] * len(self.base_urls)
        self.last = len(self.base_urls) - 1

    @contextlib.contextmanager
    def route(self) -> Iterator[str]:
        """Yield the chosen origin and count it busy until the block exits."""
        count = len(self.base_urls)
        order = [(self.last + offset) % count for offset in range(1, count + 1)]
        index = min(order, key=lambda candidate: self.active[candidate])
        self.last = index
        self.active[index] += 1
        try:
            yield self.base_urls[index]
        finally:
            self.active[index] -= 1


def _write_empty_streams(writer: ArtifactWriter) -> None:
    """Create durable empty record streams before fallible benchmark work."""
    writer.write_jsonl("warmup_requests.jsonl", [])
    writer.write_jsonl("priming_requests.jsonl", [])
    writer.write_jsonl("requests.jsonl", [])
    writer.write_jsonl("gpu_samples.jsonl", [])


def _write_records(
    writer: ArtifactWriter,
    name: str,
    records: list[RequestRecord],
    media_dir: str | None = None,
) -> None:
    """Persist original media and its request records after measurement.

    Samples are written under `samples/` before the JSONL stream that
    references them.

    Raises:
        ValueError: Invalid image metadata or unsafe media request IDs.
        FileExistsError: Different media already occupies the requested path.
    """
    for record in records:
        for image in record.decoded_images:
            writer.write_image_sample(image)
        if record.video_body is not None:
            checksum = hashlib.sha256(record.video_body).hexdigest()
            if media_dir is not None:
                if Path(
                    record.request_id
                ).name != record.request_id or record.request_id in {".", ".."}:
                    raise ValueError(
                        "media request ID must be a filename component"
                    )
                path = Path(media_dir) / f"{record.request_id}.mp4"
            else:
                path = writer.samples_dir / f"{checksum}.mp4"
            writer.write_original_video(path, record.video_body)
            record.original_output = {
                "path": str(path),
                "sha256": checksum,
                "byte_size": len(record.video_body),
                "mime": record.video_mime,
            }
    writer.write_jsonl(name, [record.record_dict() for record in records])
    if name == "requests.jsonl":
        writer.write_jsonl(
            "media_index.jsonl",
            [
                {
                    "request_id": record.request_id,
                    "success": record.success,
                    "classifier": record.classifier,
                    "original_output": record.original_output,
                    "media_checks": record.media_checks,
                    "example": record.example,
                    "run": str((writer.output_dir / "run.json").resolve()),
                    "timing_records": str(
                        (writer.output_dir / "requests.jsonl").resolve()
                    ),
                }
                for record in records
            ],
        )
        writer.write_jsonl(
            "media_validity.jsonl",
            [
                {
                    "request_id": record.request_id,
                    "valid": record.success,
                    "classifier": record.classifier,
                    "checks": record.media_checks,
                    "video": record.decoded_video.metadata_dict()
                    if record.decoded_video
                    else None,
                }
                for record in records
                if record.task == "video"
            ],
        )


def _excluded_rows(
    point: BenchmarkPoint, path: str | None
) -> list[Example] | None:
    """Load all rows of an explicit excluded manifest in its frozen order."""
    if path is None:
        return None
    count = sum(
        bool(line.strip()) for line in Path(path).read_text().splitlines()
    )
    excluded = replace(
        point,
        dataset="jsonl",
        dataset_path=path,
        load=replace(point.load, num_prompts=count),
    )
    rows, _ = load_examples(excluded)
    return rows


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
