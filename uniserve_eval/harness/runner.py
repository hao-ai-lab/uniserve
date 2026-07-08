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
from .datasets import load_dataset_rows
from .metrics.common import RequestRecord
from .report import build_summary, render_markdown, spec_to_dict
from .spec import BenchmarkSpec, TaskName
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
        self._tokenizer: Any = None

    async def run(self) -> RunResult:
        tokenizer = self._load_tokenizer()
        rows = load_dataset_rows(self.spec, tokenizer=tokenizer)
        if self.spec.num_prompts and len(rows) > self.spec.num_prompts:
            rows = rows[: self.spec.num_prompts]

        self.writer.write_json(
            "run.json",
            {
                "harness_status": "running",
                "spec": spec_to_dict(self.spec),
                "base_url": self.base_url,
                "items": len(rows),
                "started_at": time.time(),
            },
        )
        (self.writer.output_dir / "requests.jsonl").write_text("", encoding="utf-8")

        # Deterministic Poisson arrivals (matches sglang's np.random.seed(seed)).
        np.random.seed(self.spec.seed)

        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        async with httpx.AsyncClient(timeout=self.timeout_s, limits=limits) as client:

            async def submit(row: dict[str, Any]) -> RequestRecord:
                return await self._submit(client, row)

            async def warmup_submit(row: dict[str, Any]) -> RequestRecord:
                warm = {**row, "output_len": 32, "max_tokens": 32}
                record = await self._submit(client, warm)
                # A warmup that reached the server and terminated cleanly did
                # its job even when the capped token budget produced no visible
                # output — e.g. a reasoning model whose hidden <think> stream
                # consumes all 32 tokens before any content/image is emitted.
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
        )
        if sampler is not None:
            summary["gpu_memory"] = sampler.summary()

        self.writer.write_json("summary.json", summary)
        for record in records:
            self.writer.append_jsonl("requests.jsonl", record.record_dict())
        (self.writer.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
        return RunResult(summary=summary, output_dir=self.writer.output_dir)

    async def _submit(self, client: httpx.AsyncClient, row: dict[str, Any]) -> RequestRecord:
        request = self.task.build_request(row)
        output_len_fallback = int(row.get("output_len") or self.spec.max_tokens or 0)
        return await send_request(
            client,
            self.base_url,
            request,
            request_id=str(row.get("id") or f"request-{time.time_ns()}"),
            task=self.spec.task.value,
            prompt_len=int(row.get("prompt_len") or 0),
            output_len_fallback=output_len_fallback,
        )

    def _load_tokenizer(self) -> Any | None:
        if self.spec.task != TaskName.TEXT:
            return None
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer

        tokenizer_id = self.spec.tokenizer or self.spec.model
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        return self._tokenizer

    async def _fetch_server_info(self, client: httpx.AsyncClient) -> dict[str, Any] | None:
        try:
            response = await client.get(self.base_url + "/version", timeout=5.0)
            if response.status_code == 200:
                return response.json()
        except Exception:  # noqa: BLE001 - server_info is best-effort context only.
            return None
        return None
