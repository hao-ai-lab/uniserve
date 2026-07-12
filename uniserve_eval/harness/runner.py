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
    plan_summary,
    record_collection_contract,
    render_markdown,
    spec_to_dict,
    write_summary_artifacts,
)
from .spec import BenchmarkSpec
from .tasks import TASKS
from .tasks.base import TaskRequest


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
        rows, tokenizer = load_benchmark_inputs(self.spec)
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
            plan_evidence = await self._collect_plan_evidence(client, rows)

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
            plan_evidence=plan_evidence,
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
            task=self.spec.task.value,
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

    async def _collect_plan_evidence(
        self,
        client: httpx.AsyncClient,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        policy = self.spec.plan_evidence_policy
        if policy == "declared_contract":
            return {"source": policy, "plan": plan_summary(self.spec)}
        if not rows:
            return {
                "source": policy,
                "plan": None,
                "error": "the dataset produced no request for plan inspection",
            }
        request = self.task.build_request(rows[0])
        if policy == "reference_protocol":
            return {
                "source": policy,
                "plan": plan_summary(self.spec),
                "request": reference_request_summary(request),
            }
        plan_endpoints = {
            "/v1/chat/completions": "/v1/chat/completions/plan",
            "/v1/images/generations": "/v1/images/generations/plan",
        }
        endpoint = plan_endpoints.get(request.endpoint)
        if endpoint is None:
            return {
                "source": "runtime_inspection",
                "plan": None,
                "error": f"no runtime plan endpoint is defined for {request.endpoint}",
            }
        try:
            response = await client.post(
                self.base_url + endpoint,
                json=request.payload,
                headers={"x-request-id": "plan-inspection"},
                timeout=30.0,
            )
            response.raise_for_status()
            plan = response.json()
            if not isinstance(plan, dict) or not isinstance(plan.get("profile_id"), str):
                raise ValueError("runtime plan response is not a PlanInspection object")
            return {
                "source": "runtime_inspection",
                "endpoint": endpoint,
                "plan": plan,
                "request": reference_request_summary(request),
            }
        except Exception as error:  # noqa: BLE001 - the artifact records inspection failure.
            return {
                "source": "runtime_inspection",
                "endpoint": endpoint,
                "plan": None,
                "error": str(error),
            }


def reference_request_summary(request: TaskRequest) -> dict[str, Any]:
    """Return a prompt-free semantic summary derived from one emitted request."""
    payload = request.payload
    image = payload.get("image_config")
    image_config = image if isinstance(image, dict) else {}
    width = image_config.get("width")
    height = image_config.get("height")
    size = payload.get("size")
    if (width is None or height is None) and isinstance(size, str) and "x" in size:
        width_text, height_text = size.lower().split("x", maxsplit=1)
        if width_text.isdigit() and height_text.isdigit():
            width, height = int(width_text), int(height_text)
    steps = image_config.get("steps", payload.get("steps"))
    alternate_steps = payload.get("num_inference_steps")
    steps_consistent = alternate_steps is None or steps is None or alternate_steps == steps
    messages = payload.get("messages")
    return {
        "endpoint": request.endpoint,
        "kind": request.kind,
        "model": payload.get("model"),
        "stream": payload.get("stream", False),
        "modalities": payload.get("modalities"),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "input_image_count": _count_input_images(messages),
        "generation": {
            "max_tokens": payload.get("max_completion_tokens", payload.get("max_tokens")),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "top_k": payload.get("top_k"),
            "min_p": payload.get("min_p"),
            "repetition_penalty": payload.get("repetition_penalty"),
            "frequency_penalty": payload.get("frequency_penalty"),
            "presence_penalty": payload.get("presence_penalty"),
            "seed": payload.get("seed"),
            "chat_template_kwargs": payload.get("chat_template_kwargs") or {},
            "ignore_eos": payload.get("ignore_eos"),
            "structured_outputs": payload.get("structured_outputs"),
            "response_format": payload.get("response_format"),
        },
        "image": {
            "width": width,
            "height": height,
            "steps": steps,
            "steps_consistent": steps_consistent,
            "max_images": image_config.get("num_images", payload.get("n")),
            "seed": image_config.get("seed", payload.get("seed")),
            "guidance_scale": image_config.get("guidance_scale", payload.get("guidance_scale")),
            "image_guidance_scale": image_config.get(
                "image_guidance_scale", payload.get("image_guidance_scale")
            ),
            "cfg_norm": image_config.get("cfg_norm", payload.get("cfg_norm")),
            "cfg_interval": image_config.get("cfg_interval", payload.get("cfg_interval")),
            "timestep_shift": image_config.get("timestep_shift", payload.get("timestep_shift")),
            "think": image_config.get("think", payload.get("think")),
            "t_eps": image_config.get("t_eps", payload.get("t_eps")),
        },
        "adapter": "base" if payload.get("lora_request") is None else "lora",
    }


def _count_input_images(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        count += sum(
            1
            for part in content
            if isinstance(part, dict) and part.get("type") in {"image", "image_url"}
        )
    return count
