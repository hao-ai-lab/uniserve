"""Single-operating-point benchmark runner.

Loads the dataset, drives the warmup + timed region via the sglang-mirrored
arrival engine, summarizes with the task's metric family, and writes
``run.json`` / ``requests.jsonl`` / ``summary.json`` / ``summary.md``.

One ``run()`` is exactly one operating point (one rate, one concurrency); sweeps
are the CLI's job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .artifacts import ArtifactWriter
from .core.arrival import run_load
from .core.client import send_request
from .core.gpu_sampler import GpuMemorySampler
from .datasets import load_benchmark_inputs
from .metrics.common import RequestRecord
from .report import (
    attach_execution_contract,
    benchmark_contract,
    build_summary,
    image_sample_collection_contract,
    record_collection_contract,
    render_markdown,
    spec_to_dict,
    write_summary_artifacts,
)
from .spec import BenchmarkSpec
from .tasks import TASKS


@dataclass
class RunResult:
    summary: dict[str, Any]
    output_dir: Path


class BenchmarkRunner:
    def __init__(
        self,
        base_url: str,
        spec: BenchmarkSpec,
        output_dir: str | Path,
        timeout_s: float = 6 * 60 * 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.spec = spec
        self.writer = ArtifactWriter(output_dir)
        self.timeout_s = timeout_s
        self.task = TASKS[spec.task.value](spec)

    async def run(self) -> RunResult:
        for commit_marker in ("summary.json", "artifact_manifest.json", "summary.md"):
            (self.writer.output_dir / commit_marker).unlink(missing_ok=True)
        self.writer.clear_samples()
        inputs = load_benchmark_inputs(self.spec)
        rows = inputs.measured
        tokenizer = inputs.tokenizer
        contract = benchmark_contract(self.spec, rows)

        started_at = time.time()
        self.writer.write_json(
            "run.json",
            {
                "harness_status": "running",
                "spec": spec_to_dict(self.spec),
                "base_url": self.base_url,
                "items": len(rows),
                "started_at": started_at,
            },
        )
        self.writer.write_jsonl("requests.jsonl", [])
        self.writer.write_jsonl("gpu_samples.jsonl", [])

        # Deterministic Poisson arrivals (matches sglang's np.random.seed(seed)).
        np.random.seed(self.spec.seed)

        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        async with httpx.AsyncClient(timeout=self.timeout_s, limits=limits) as client:
            async def submit(row: dict[str, Any]) -> RequestRecord:
                return await self._submit(client, row)

            async def warmup_submit(row: dict[str, Any]) -> RequestRecord:
                record = await self._submit(client, row)
                # A warmup that reached the server and terminated cleanly did
                # its job even when a reasoning model produced no visible output.
                if (
                    not record.success
                    and record.status_code == 200
                    and record.classifier == "protocol_empty_output"
                ):
                    record.success = True
                    record.classifier = "warmup_empty_output_ok"
                return record

            sampler = GpuMemorySampler() if self.spec.sample_gpu_memory else None
            if sampler is not None:
                sampler.start()
            try:
                records, dur_s = await run_load(
                    rows,
                    request_rate=self.spec.request_rate,
                    max_concurrency=self.spec.max_concurrency,
                    submit=submit,
                    warmup_submit=warmup_submit,
                    warmup_requests=self.spec.warmup_requests,
                    warmup_rows=inputs.warmup,
                )
            finally:
                if sampler is not None:
                    sampler.stop()
            server_info = await self._fetch_server_info(client)

        summary = build_summary(
            self.spec,
            self.base_url,
            records,
            dur_s,
            tokenizer=tokenizer,
            server_info=server_info,
            contract=contract,
        )
        gpu_samples: list[dict[str, Any]] = []
        if sampler is not None:
            summary["gpu_memory"] = sampler.summary()
            gpu_samples = list(sampler.sample_records)

        for record in records:
            for image in record.decoded_images:
                self.writer.write_image_sample(image)
        request_records = [record.record_dict() for record in records]
        attach_execution_contract(
            summary,
            "request_records",
            record_collection_contract(request_records),
        )
        attach_execution_contract(
            summary,
            "gpu_samples",
            record_collection_contract(gpu_samples),
        )
        attach_execution_contract(
            summary,
            "image_samples",
            image_sample_collection_contract(request_records),
        )
        self.writer.write_jsonl("requests.jsonl", request_records)
        self.writer.write_jsonl("gpu_samples.jsonl", gpu_samples)
        self.writer.write_json(
            "run.json",
            {
                "harness_status": "completed",
                "artifact_valid": summary["artifact"]["valid"],
                "spec": spec_to_dict(self.spec),
                "base_url": self.base_url,
                "items": len(rows),
                "started_at": started_at,
                "completed_at": time.time(),
            },
        )
        (self.writer.output_dir / "summary.md").write_text(
            render_markdown(summary), encoding="utf-8"
        )
        write_summary_artifacts(self.writer.output_dir, summary)
        return RunResult(summary=summary, output_dir=self.writer.output_dir)

    async def _submit(self, client: httpx.AsyncClient, row: dict[str, Any]) -> RequestRecord:
        request = self.task.build_request(row)
        output_len_fallback = int(row.get("output_len") or self.spec.max_tokens or 0)
        return await send_request(
            client,
            self.base_url,
            request,
            request_id=str(row.get("id") or f"request-{time.time_ns()}"),
            task=request.semantic_task or self.spec.task.value,
            prompt_len=int(row.get("prompt_len") or 0),
            output_len_fallback=output_len_fallback,
            scheduled_time=(
                float(row["_harness_scheduled_time"])
                if row.get("_harness_scheduled_time") is not None
                else None
            ),
        )

    async def _fetch_server_info(self, client: httpx.AsyncClient) -> dict[str, Any] | None:
        for endpoint in ("/server_info", "/get_server_info", "/model_info", "/version"):
            try:
                response = await client.get(self.base_url + endpoint, timeout=15.0)
                if response.status_code != 200:
                    continue
                payload = response.json()
                if isinstance(payload, dict):
                    return {"source_endpoint": endpoint, "payload": payload}
            except Exception:  # noqa: BLE001 - try the next standard inspection endpoint.
                continue
        return None
