"""Execute one benchmark point and write its raw result bundle."""

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
from .report import build_summary, render_markdown, selected_rows_identity
from .spec import BenchmarkSpec
from .tasks import TASKS


@dataclass(frozen=True)
class RunResult:
    summary: dict[str, Any]
    output_dir: Path


class BenchmarkRunner:
    def __init__(
        self,
        base_url: str,
        spec: BenchmarkSpec,
        output_dir: str | Path,
        *,
        provenance: dict[str, Any] | None = None,
        timeout_s: float = 6 * 60 * 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.spec = spec
        self.output_dir = Path(output_dir)
        self.provenance = provenance or {}
        self.timeout_s = timeout_s
        self.task = TASKS[spec.task.value](spec)

    async def run(self) -> RunResult:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"result directory is not empty: {self.output_dir}")
        inputs = load_benchmark_inputs(self.spec)
        rows = inputs.measured
        selection = selected_rows_identity(rows)
        writer = ArtifactWriter(self.output_dir)
        started_at = time.time()
        writer.write_json(
            "run.json",
            {
                "status": "running",
                "benchmark": self.spec.name,
                "workload": self.spec.workload_dict(),
                "selected_rows": selection,
                "started_at": started_at,
                "provenance": self.provenance,
            },
        )
        writer.write_jsonl("requests.jsonl", [])
        writer.write_jsonl("gpu_samples.jsonl", [])

        np.random.seed(self.spec.seed)
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        async with httpx.AsyncClient(timeout=self.timeout_s, limits=limits) as client:

            async def submit(row: dict[str, Any]) -> RequestRecord:
                return await self._submit(client, row)

            sampler = GpuMemorySampler() if self.spec.sample_gpu_memory else None
            if sampler is not None:
                sampler.start()
            try:
                records, duration = await run_load(
                    rows,
                    request_rate=self.spec.request_rate,
                    max_concurrency=self.spec.max_concurrency,
                    submit=submit,
                    warmup_submit=submit,
                    warmup_requests=self.spec.warmup_requests,
                    warmup_rows=inputs.warmup,
                )
            finally:
                if sampler is not None:
                    sampler.stop()
            server_version = await self._fetch_server_version(client)

        summary = build_summary(
            self.spec,
            self.base_url,
            records,
            duration,
            task=self.task,
            selected_rows=selection,
            tokenizer=inputs.tokenizer,
            server_version=server_version,
            provenance=self.provenance,
        )
        gpu_samples = list(sampler.sample_records) if sampler is not None else []
        if sampler is not None:
            summary["gpu_memory"] = sampler.summary()
        for record in records:
            for image in record.decoded_images:
                writer.write_image_sample(image)
        writer.write_jsonl("requests.jsonl", [record.record_dict() for record in records])
        writer.write_jsonl("gpu_samples.jsonl", gpu_samples)
        writer.write_json("summary.json", summary)
        (self.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
        writer.write_json(
            "run.json",
            {
                "status": "completed",
                "benchmark": self.spec.name,
                "workload": self.spec.workload_dict(),
                "selected_rows": selection,
                "started_at": started_at,
                "completed_at": time.time(),
                "valid": summary["validation"]["valid"],
                "provenance": self.provenance,
            },
        )
        return RunResult(summary, self.output_dir)

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

    async def _fetch_server_version(self, client: httpx.AsyncClient) -> dict[str, Any] | None:
        try:
            response = await client.get(self.base_url + "/version", timeout=15.0)
            if response.status_code == 200 and isinstance(payload := response.json(), dict):
                return payload
        except Exception:
            return None
        return None
