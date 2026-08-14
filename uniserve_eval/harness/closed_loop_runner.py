"""Steady-state closed-loop benchmark execution.

Unlike the arrival-process runner, this runner maintains a target number of
in-flight requests throughout the measured window.  Guard requests absorb the
ramp, warmup, and drain boundaries so the formal sample is neither a prefill
burst nor a falling-concurrency tail.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import httpx

from .artifacts import ArtifactWriter
from .core.arrival import ClosedLoopResult, run_closed_loop
from .core.client import send_request
from .datasets import load_benchmark_inputs
from .metrics.common import RequestRecord
from .report import build_summary, render_markdown, selected_rows_identity
from .runner import RunResult
from .spec import BenchmarkSpec
from .tasks import TASKS


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Load-control parameters that are independent of the workload."""

    concurrency: int
    guard_prompts: int = 96
    ramp_interval_s: float = 0.25
    warmup_completions: int | None = None
    minimum_mean_occupancy_fraction: float = 0.99
    minimum_target_occupancy_fraction: float = 0.99

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("closed-loop concurrency must be positive")
        if self.guard_prompts < self.concurrency:
            raise ValueError("guard prompts must cover every concurrency slot")
        if not 0.0 <= self.minimum_mean_occupancy_fraction <= 1.0:
            raise ValueError("mean occupancy threshold must be in [0, 1]")
        if not 0.0 <= self.minimum_target_occupancy_fraction <= 1.0:
            raise ValueError("target occupancy threshold must be in [0, 1]")


class ClosedLoopBenchmarkRunner:
    """Run one workload point at measured steady-state concurrency."""

    def __init__(
        self,
        base_url: str,
        spec: BenchmarkSpec,
        output_dir: str | Path,
        load: ClosedLoopConfig,
        *,
        provenance: dict[str, Any] | None = None,
        timeout_s: float = 6 * 60 * 60.0,
    ) -> None:
        if spec.num_prompts < 1:
            raise ValueError("closed-loop measurement requires formal prompts")
        self.base_url = base_url.rstrip("/")
        self.spec = spec
        self.output_dir = Path(output_dir)
        self.load = load
        self.provenance = provenance or {}
        self.timeout_s = timeout_s
        self.task = TASKS[spec.task.value](spec)

    async def run(self) -> RunResult:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"result directory is not empty: {self.output_dir}")

        source_spec = replace(
            self.spec,
            num_prompts=self.load.guard_prompts + self.spec.num_prompts,
        )
        inputs = load_benchmark_inputs(source_spec)
        guard_rows = inputs.measured[: self.load.guard_prompts]
        measured_rows = inputs.measured[self.load.guard_prompts :]
        guard_selection = selected_rows_identity(guard_rows)
        measured_selection = selected_rows_identity(measured_rows)
        load_control = self._load_control_dict()
        writer = ArtifactWriter(self.output_dir)
        started_at = time.time()
        writer.write_json(
            "run.json",
            {
                "status": "running",
                "benchmark": self.spec.name,
                "workload": self.spec.workload_dict(),
                "load_control": load_control,
                "selected_rows": measured_selection,
                "guard_rows": guard_selection,
                "started_at": started_at,
                "provenance": self.provenance,
            },
        )
        writer.write_jsonl("requests.jsonl", [])
        writer.write_jsonl("guard_requests.jsonl", [])
        writer.write_jsonl("gpu_samples.jsonl", [])

        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        async with httpx.AsyncClient(timeout=self.timeout_s, limits=limits) as client:

            async def submit(row: dict[str, Any]) -> RequestRecord:
                record = await self._submit(client, row)
                if row.get("_harness_stage") == "guard":
                    # Guard responses are validation evidence, not output samples.
                    # Releasing their decoded bytes bounds harness memory at high C.
                    record.decoded_images.clear()
                return record

            result = await run_closed_loop(
                measured_rows,
                guard_rows,
                concurrency=self.load.concurrency,
                submit=submit,
                ramp_interval_s=self.load.ramp_interval_s,
                warmup_completions=self.load.warmup_completions,
            )
            server_version = await self._fetch_server_version(client)

        records = list(result.outputs)
        guard_records = list(result.guard_outputs)
        summary = build_summary(
            self.spec,
            self.base_url,
            records,
            result.duration_s,
            task=self.task,
            selected_rows=measured_selection,
            tokenizer=inputs.tokenizer,
            server_version=server_version,
            provenance=self.provenance,
        )
        summary["load_control"] = load_control
        summary["occupancy"] = self._occupancy_dict(result)
        summary["guard_rows"] = guard_selection
        self._merge_load_validation(summary, result, guard_records)

        for record in records:
            for image in record.decoded_images:
                writer.write_image_sample(image)
        writer.write_jsonl("requests.jsonl", [record.record_dict() for record in records])
        writer.write_jsonl(
            "guard_requests.jsonl",
            [record.record_dict() for record in guard_records],
        )
        writer.write_json("summary.json", summary)
        (self.output_dir / "summary.md").write_text(
            render_markdown(summary),
            encoding="utf-8",
        )
        writer.write_json(
            "run.json",
            {
                "status": "completed",
                "benchmark": self.spec.name,
                "workload": self.spec.workload_dict(),
                "load_control": load_control,
                "selected_rows": measured_selection,
                "guard_rows": guard_selection,
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

    def _load_control_dict(self) -> dict[str, Any]:
        return {
            **asdict(self.load),
            "model": "closed_loop_replacement",
            "arrival_rate": None,
            "warmup_completions": (
                self.load.warmup_completions or self.load.concurrency
            ),
            "formal_prompts": self.spec.num_prompts,
        }

    @staticmethod
    def _occupancy_dict(result: ClosedLoopResult) -> dict[str, Any]:
        return {
            "target": result.target_concurrency,
            "maximum": result.maximum_inflight,
            "minimum": result.minimum_inflight,
            "mean": result.mean_inflight,
            "mean_fraction": result.mean_inflight / result.target_concurrency,
            "target_fraction": result.target_occupancy_fraction,
            "warmup_completions": result.warmup_completions,
            "measured_requests": result.measured_requests,
        }

    def _merge_load_validation(
        self,
        summary: dict[str, Any],
        result: ClosedLoopResult,
        guard_records: list[RequestRecord],
    ) -> None:
        validation = summary["validation"]
        checks = validation["checks"]
        checks.update(
            {
                "closed_loop_target_reached": (
                    result.maximum_inflight == self.load.concurrency
                ),
                "closed_loop_warmup_complete": (
                    result.warmup_completions
                    >= (self.load.warmup_completions or self.load.concurrency)
                ),
                "closed_loop_mean_occupancy": (
                    result.mean_inflight
                    >= self.load.minimum_mean_occupancy_fraction * self.load.concurrency
                ),
                "closed_loop_target_occupancy": (
                    result.target_occupancy_fraction
                    >= self.load.minimum_target_occupancy_fraction
                ),
                "guard_requests_succeeded": (
                    bool(guard_records) and all(record.success for record in guard_records)
                ),
            }
        )
        validation["statistics"].update(
            {
                "guard_request_count": len(guard_records),
                "successful_guard_requests": sum(record.success for record in guard_records),
            }
        )
        validation["valid"] = bool(checks) and all(checks.values())
