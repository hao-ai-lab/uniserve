"""Summary assembly + human-readable report for a single benchmark run."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from .artifacts import ArtifactWriter
from .metrics import RequestRecord, summarize_image, summarize_stream
from .spec import BenchmarkSpec


def spec_to_dict(spec: BenchmarkSpec) -> dict[str, Any]:
    data = asdict(spec)
    data["task"] = spec.task.value
    rate = data.get("request_rate")
    if isinstance(rate, float) and math.isinf(rate):
        data["request_rate"] = "inf"
    return cast(
        dict[str, Any],
        json.loads(
            json.dumps(
                data,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        ),
    )


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def benchmark_contract(
    spec: BenchmarkSpec,
    selected_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    selected = {
        "count": len(selected_rows),
        "sha256": canonical_digest(list(selected_rows)),
    }
    payload = {
        "schema_version": 1,
        "spec": spec_to_dict(spec),
        "selected_rows": selected,
    }
    return {**payload, "fingerprint": canonical_digest(payload)}


_PARITY_IDENTITY_FIELDS = frozenset(
    {"model", "name", "runtime_profile_id", "plan_evidence_policy"}
)


def benchmark_parity_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Normalize a harness contract to backend-independent comparison semantics."""
    spec = contract.get("spec")
    selected_rows = contract.get("selected_rows")
    if not isinstance(spec, dict) or not isinstance(selected_rows, dict):
        raise ValueError("benchmark contract has no normalized spec or selected rows")
    normalized_spec = {
        key: value for key, value in spec.items() if key not in _PARITY_IDENTITY_FIELDS
    }
    payload = {
        "schema_version": 1,
        "spec": normalized_spec,
        "selected_rows": selected_rows,
    }
    return {**payload, "fingerprint": canonical_digest(payload)}


def benchmark_contract_is_valid(
    contract: Any,
    spec: BenchmarkSpec,
    request_count: int,
) -> bool:
    if not isinstance(contract, dict):
        return False
    selected = contract.get("selected_rows")
    if not isinstance(selected, dict):
        return False
    payload = {
        "schema_version": contract.get("schema_version"),
        "spec": contract.get("spec"),
        "selected_rows": selected,
    }
    return bool(
        contract.get("schema_version") == 1
        and contract.get("spec") == spec_to_dict(spec)
        and selected.get("count") == request_count
        and isinstance(selected.get("sha256"), str)
        and len(selected["sha256"]) == 64
        and contract.get("fingerprint") == canonical_digest(payload)
    )


def canonical_summary_matches(
    summary: Any,
    expected_contract: dict[str, Any],
    *,
    profile_contract_fingerprint: str | None = None,
) -> bool:
    if not isinstance(summary, dict):
        return False
    artifact = summary.get("artifact")
    if not isinstance(artifact, dict):
        return False
    checks = artifact.get("checks")
    if not (
        artifact.get("schema_version") == 2
        and artifact.get("valid") is True
        and artifact.get("valid_marker") == "canonical-valid-v2"
        and isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values())
        and artifact.get("contract") == expected_contract
    ):
        return False
    if profile_contract_fingerprint is None:
        return True
    profile_contract = artifact.get("profile_contract")
    return bool(
        isinstance(profile_contract, dict)
        and profile_contract.get("schema_version") == 2
        and profile_contract.get("fingerprint") == profile_contract_fingerprint
    )


def record_collection_contract(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    payload = list(records)
    return {
        "schema_version": 1,
        "count": len(payload),
        "sha256": canonical_digest(payload),
    }


def canonical_artifact_bundle_matches(
    output_dir: str | Path,
    summary: Any,
    expected_contract: dict[str, Any],
    *,
    profile_contract_fingerprint: str | None = None,
) -> bool:
    """Validate every durable member of a committed benchmark point."""
    if not canonical_summary_matches(
        summary,
        expected_contract,
        profile_contract_fingerprint=profile_contract_fingerprint,
    ):
        return False
    assert isinstance(summary, dict)
    artifact = summary["artifact"]
    output_path = Path(output_dir)
    try:
        manifest = json.loads(
            (output_path / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        run = json.loads((output_path / "run.json").read_text(encoding="utf-8"))
        requests = _read_jsonl(output_path / "requests.jsonl")
        gpu_samples = _read_jsonl(output_path / "gpu_samples.jsonl")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False
    if manifest != artifact:
        return False
    if not (
        run.get("harness_status") == "completed"
        and run.get("artifact_valid") is True
        and run.get("items") == summary.get("request_count")
        and run.get("spec") == summary.get("spec")
        and run.get("base_url") == summary.get("base_url")
    ):
        return False
    return bool(
        artifact.get("request_records") == record_collection_contract(requests)
        and artifact.get("gpu_samples") == record_collection_contract(gpu_samples)
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object record")
            records.append(value)
    return records


def attach_execution_contract(
    summary: dict[str, Any],
    name: str,
    contract: dict[str, Any],
    *,
    valid: bool = True,
) -> None:
    """Attach execution evidence and recompute the canonical validity marker."""
    artifact = summary.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("benchmark summary has no artifact contract")
    checks = artifact.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("benchmark artifact has no validity checks")
    artifact[name] = contract
    checks[name] = valid
    artifact["valid"] = all(value is True for value in checks.values())
    artifact["valid_marker"] = "canonical-valid-v2" if artifact["valid"] else None


def write_summary_artifacts(output_dir: str | Path, summary: dict[str, Any]) -> None:
    """Persist the manifest first and summary last as the point's commit marker."""
    artifact = summary.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("benchmark summary has no artifact contract")
    writer = ArtifactWriter(output_dir)
    writer.write_json("artifact_manifest.json", artifact)
    writer.write_json("summary.json", summary)


def load_mode(spec: BenchmarkSpec) -> str:
    if spec.request_rate != float("inf"):
        return "open_loop_poisson"
    if spec.max_concurrency:
        return "closed_loop"
    return "saturation"


def plan_summary(spec: BenchmarkSpec) -> dict[str, Any]:
    """Sanitized semantic plan declared for one formal measurement point."""
    return {
        "runtime_profile_id": spec.runtime_profile_id,
        "protocol_adapter": spec.wire,
        "measurement_interface": spec.measurement_interface,
        "context": {
            "dataset": spec.dataset,
            "preprocessing": spec.preprocessing,
            "prompt_source": spec.dataset_path or spec.dataset,
        },
        "generation": {
            "output_constraint": spec.output_constraint,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "ignore_eos": spec.ignore_eos,
            "structured_output": spec.structured_output_policy,
        },
        "image": {
            "width": spec.width,
            "height": spec.height,
            "steps": spec.steps,
            "max_images": spec.max_images,
            "seed": spec.seed,
            "guidance_scale": spec.guidance_scale,
            "image_guidance_scale": spec.image_guidance_scale,
            "cfg_norm": spec.cfg_norm,
            "cfg_interval": list(spec.cfg_interval) if spec.cfg_interval is not None else None,
            "timestep_shift": spec.timestep_shift,
        },
        "cache": {
            "read": spec.cache_read_policy,
            "write": spec.cache_write_policy,
        },
        "adapter": spec.adapter_selection,
    }


def artifact_contract(
    spec: BenchmarkSpec,
    *,
    request_count: int,
    ok_count: int,
    failed_count: int,
    total_images: int,
    plan_evidence: dict[str, Any] | None,
    contract: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_plan_source = spec.plan_evidence_policy
    if plan_evidence is None:
        plan_evidence = {
            "source": "declared_contract",
            "plan": plan_summary(spec),
        }
    plan = plan_evidence.get("plan")
    plan_evidence_valid = (
        plan_evidence.get("source") == expected_plan_source
    )
    if plan_evidence_valid and expected_plan_source == "declared_contract":
        plan_evidence_valid = isinstance(plan, dict)
    elif plan_evidence_valid and expected_plan_source == "runtime_inspection":
        plan_evidence_valid = isinstance(plan, dict) and _runtime_plan_matches_declared_contract(
            plan, spec
        )
    elif plan_evidence_valid and expected_plan_source == "reference_protocol":
        request = plan_evidence.get("request")
        plan_evidence_valid = isinstance(request, dict) and _reference_request_matches_contract(
            request, spec
        )
    checks = {
        "declared_request_count": request_count == spec.num_prompts,
        "minimum_successful_requests": ok_count >= spec.acceptance_min_success,
        "maximum_failed_requests": failed_count <= spec.acceptance_max_failed,
        "single_measured_run": spec.measured_runs == 1,
        "minimum_images": total_images
        >= math.ceil(ok_count * spec.acceptance_min_images_per_success),
        "plan_evidence": plan_evidence_valid,
        "contract_fingerprint": benchmark_contract_is_valid(contract, spec, request_count),
    }
    valid = all(checks.values())
    return {
        "schema_version": 2,
        "valid": valid,
        "valid_marker": "canonical-valid-v2" if valid else None,
        "checks": checks,
        "acceptance": {
            "minimum_successful_requests": spec.acceptance_min_success,
            "maximum_failed_requests": spec.acceptance_max_failed,
            "minimum_images_per_success": spec.acceptance_min_images_per_success,
        },
        "measured_runs": spec.measured_runs,
        "server_topology": spec.server_topology,
        "declared_contract": plan_summary(spec),
        "plan_summary": plan_evidence.get("plan", plan_evidence.get("request")),
        "plan_evidence": plan_evidence,
        "contract": contract,
    }


def _runtime_plan_matches_declared_contract(
    plan: dict[str, Any], spec: BenchmarkSpec
) -> bool:
    expected_profile = spec.runtime_profile_id
    if expected_profile == "unspecified":
        return False
    profile_id = plan.get("profile_id")
    dialect_id = plan.get("dialect_id")
    identity_matches = (
        isinstance(profile_id, str)
        and isinstance(dialect_id, str)
        and (
            profile_id == expected_profile
            or profile_id.startswith(f"{expected_profile}:")
            or dialect_id == expected_profile
        )
    )
    generation = plan.get("generation")
    cache = plan.get("cache")
    if not identity_matches or not isinstance(generation, dict) or not isinstance(cache, dict):
        return False
    if generation.get("constraint") != spec.output_constraint:
        return False
    if spec.max_tokens is not None and generation.get("max_tokens") != spec.max_tokens:
        return False
    if not _same_float(generation.get("temperature"), spec.temperature):
        return False
    if not _same_float(generation.get("top_p"), spec.top_p):
        return False
    if generation.get("ignore_eos") is not spec.ignore_eos:
        return False
    if cache.get("read_enabled") is not (spec.cache_read_policy == "enabled"):
        return False
    if cache.get("write_enabled") is not (spec.cache_write_policy == "enabled"):
        return False
    if _adapter_name(plan.get("adapter")) != spec.adapter_selection:
        return False
    if spec.output_constraint not in {"default", "gen_only"}:
        return True
    image = generation.get("image")
    if not isinstance(image, dict):
        return False
    exact_image_fields: dict[str, int | str | None] = {
        "width": spec.width,
        "height": spec.height,
        "steps": spec.steps,
        "max_images": spec.max_images,
        "seed": spec.seed,
        "cfg_renorm_type": spec.cfg_norm,
    }
    for key, expected_exact in exact_image_fields.items():
        if expected_exact is not None and image.get(key) != expected_exact:
            return False
    float_image_fields: dict[str, float | None] = {
        "cfg_text_scale": spec.guidance_scale,
        "cfg_img_scale": spec.image_guidance_scale,
        "timestep_shift": spec.timestep_shift,
    }
    for key, expected_float in float_image_fields.items():
        if expected_float is not None and not _same_float(image.get(key), expected_float):
            return False
    if spec.cfg_interval is not None:
        actual_interval = image.get("cfg_interval")
        if not isinstance(actual_interval, (list, tuple)) or len(actual_interval) != 2:
            return False
        if not all(
            _same_float(actual, expected)
            for actual, expected in zip(actual_interval, spec.cfg_interval, strict=True)
        ):
            return False
    return True


def _reference_request_matches_contract(
    request: dict[str, Any], spec: BenchmarkSpec
) -> bool:
    if request.get("endpoint") != spec.endpoint or request.get("kind") != spec.wire:
        return False
    if request.get("model") != spec.model or request.get("adapter") != spec.adapter_selection:
        return False
    modalities = request.get("modalities")
    if spec.output_constraint == "gen_only":
        if request.get("kind") != "images_generations" and modalities != ["image"]:
            return False
    elif spec.output_constraint == "default":
        if modalities != ["text", "image"]:
            return False
    elif spec.output_constraint == "und_only" and modalities is not None and modalities != ["text"]:
        return False
    generation = request.get("generation")
    if not isinstance(generation, dict):
        return False
    if spec.max_tokens is not None and generation.get("max_tokens") != spec.max_tokens:
        return False
    for key, expected in (("temperature", spec.temperature), ("top_p", spec.top_p)):
        actual = generation.get(key)
        if actual is not None and not _same_float(actual, expected):
            return False
    actual_ignore_eos = generation.get("ignore_eos")
    if actual_ignore_eos is not None and actual_ignore_eos is not spec.ignore_eos:
        return False
    if spec.structured_output_policy == "none" and (
        generation.get("structured_outputs") is not None
        or generation.get("response_format") is not None
    ):
        return False
    if spec.task.value in {"i2i", "i2t"} and request.get("input_image_count", 0) < 1:
        return False
    if spec.output_constraint not in {"default", "gen_only"}:
        return True
    image = request.get("image")
    if not isinstance(image, dict) or image.get("steps_consistent") is not True:
        return False
    exact_fields = {
        "width": spec.width,
        "height": spec.height,
        "steps": spec.steps,
        "max_images": spec.max_images,
        "seed": spec.seed,
        "cfg_norm": spec.cfg_norm,
    }
    for key, expected_exact in exact_fields.items():
        if expected_exact is not None and image.get(key) != expected_exact:
            return False
    float_fields = {
        "guidance_scale": spec.guidance_scale,
        "image_guidance_scale": spec.image_guidance_scale,
        "timestep_shift": spec.timestep_shift,
    }
    for key, expected_float in float_fields.items():
        if expected_float is not None and not _same_float(image.get(key), expected_float):
            return False
    if spec.cfg_interval is not None:
        actual_interval = image.get("cfg_interval")
        if not isinstance(actual_interval, (list, tuple)) or len(actual_interval) != 2:
            return False
        if not all(
            _same_float(actual, expected)
            for actual, expected in zip(actual_interval, spec.cfg_interval, strict=True)
        ):
            return False
    return True


def _same_float(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6
    )


def _adapter_name(adapter: Any) -> str | None:
    if isinstance(adapter, str):
        return adapter.lower()
    if isinstance(adapter, dict) and len(adapter) == 1:
        return next(iter(adapter)).lower()
    return None


def build_summary(
    spec: BenchmarkSpec,
    base_url: str,
    records: list[RequestRecord],
    dur_s: float,
    *,
    tokenizer: Any | None = None,
    server_info: dict[str, Any] | None = None,
    plan_evidence: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ok = [r for r in records if r.success]
    classifiers: dict[str, int] = {}
    for record in records:
        classifiers[record.classifier] = classifiers.get(record.classifier, 0) + 1

    if spec.is_stream_task:
        family = "stream"
        metrics = summarize_stream(records, dur_s, tokenizer=tokenizer)
    else:
        family = "image"
        metrics = summarize_image(records, dur_s)
    endpoint = _observed_endpoint(records) or spec.endpoint

    summary: dict[str, Any] = {
        "harness_status": "completed",
        "task": spec.task.value,
        "dataset": spec.dataset,
        "endpoint": endpoint,
        "wire": spec.wire,
        "model": spec.model,
        "base_url": base_url,
        "server_info": server_info,
        "spec": spec_to_dict(spec),
        "load": {
            "mode": load_mode(spec),
            "request_rate": "inf" if spec.request_rate == float("inf") else spec.request_rate,
            "max_concurrency": spec.max_concurrency,
            "warmup_requests": spec.warmup_requests,
            "num_prompts": spec.num_prompts,
            "seed": spec.seed,
        },
        "elapsed_s": dur_s,
        "request_count": len(records),
        "ok_count": len(ok),
        "failed_count": len(records) - len(ok),
        "classifiers": classifiers,
        "metric_family": family,
        "metrics": metrics,
    }
    summary["artifact"] = artifact_contract(
        spec,
        request_count=summary["request_count"],
        ok_count=summary["ok_count"],
        failed_count=summary["failed_count"],
        total_images=(
            int(metrics.get("completed_images", 0))
            if family == "image"
            else int(metrics.get("images", {}).get("total_images", 0))
        ),
        plan_evidence=plan_evidence,
        contract=contract,
    )
    return summary


def _observed_endpoint(records: list[RequestRecord]) -> str | None:
    endpoints = {record.endpoint for record in records if record.endpoint}
    return next(iter(endpoints)) if len(endpoints) == 1 else None


def render_markdown(summary: dict[str, Any]) -> str:
    load = summary["load"]
    lines = [
        f"# {summary['task']} benchmark ({summary['dataset']})",
        "",
        f"- model: `{summary['model']}`  endpoint: `{summary['endpoint']}`",
        f"- load: {load['mode']}  rate={load['request_rate']}  "
        f"max_concurrency={load['max_concurrency']}  num_prompts={load['num_prompts']}",
        f"- requests: {summary['ok_count']}/{summary['request_count']} ok  "
        f"elapsed={summary['elapsed_s']:.2f}s",
        "",
    ]
    metrics = summary["metrics"]
    if summary["metric_family"] == "stream":
        lines += _stream_markdown(metrics)
    else:
        lines += _image_markdown(metrics)
    return "\n".join(lines) + "\n"


def _row(label: str, *values: Any) -> str:
    cells = " | ".join(_fmt(value) for value in values)
    return f"| {label} | {cells} |"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "" if value is None else str(value)


def _stream_markdown(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "| metric | p50 | p90 | p95 | p99 | mean |",
        "|---|---|---|---|---|---|",
        _row("TTFT (ms)", metrics["p50_ttft_ms"], metrics["p90_ttft_ms"], metrics["p95_ttft_ms"], metrics["p99_ttft_ms"], metrics["mean_ttft_ms"]),
        _row("TPOT (ms)", metrics["p50_tpot_ms"], metrics["p90_tpot_ms"], metrics["p95_tpot_ms"], metrics["p99_tpot_ms"], metrics["mean_tpot_ms"]),
        _row("ITL (ms)", metrics["p50_itl_ms"], metrics["p90_itl_ms"], metrics["p95_itl_ms"], metrics["p99_itl_ms"], metrics["mean_itl_ms"]),
        _row("E2E (ms)", metrics["p50_e2e_latency_ms"], metrics["p90_e2e_latency_ms"], metrics["p95_e2e_latency_ms"], metrics["p99_e2e_latency_ms"], metrics["mean_e2e_latency_ms"]),
        "",
        f"- output throughput: {_fmt(metrics['output_throughput'])} tok/s",
        f"- request throughput: {_fmt(metrics['request_throughput'])} req/s",
        f"- total throughput: {_fmt(metrics['total_throughput'])} tok/s",
        f"- concurrency: {_fmt(metrics['concurrency'])}  "
        f"peak tok/s: {_fmt(metrics['max_output_tokens_per_s'])}  "
        f"peak concurrent: {metrics['max_concurrent_requests']}",
    ]
    if "images" in metrics:
        img = metrics["images"]["image_latency_ms"]
        lines += [
            "",
            f"- default-task images: {metrics['images']['total_images']} total, "
            f"{_fmt(metrics['images']['images_per_second'])} img/s, "
            f"image E2E p50/p99 = {_fmt(img['p50'])}/{_fmt(img['p99'])} ms",
        ]
    return lines


def _image_markdown(metrics: dict[str, Any]) -> list[str]:
    lat = metrics["image_latency_ms"]
    lines = [
        "| metric | p50 | p90 | p95 | p99 | mean |",
        "|---|---|---|---|---|---|",
        _row("image latency (ms)", lat["p50"], lat["p90"], lat["p95"], lat["p99"], lat["mean"]),
        "",
        f"- images/s: {_fmt(metrics['images_per_second'])}  "
        f"images/min: {_fmt(metrics['images_per_minute'])}",
        f"- request throughput: {_fmt(metrics['request_throughput'])} req/s",
        f"- completed: {metrics['completed_requests']} requests, {metrics['completed_images']} images",
    ]
    if "time_to_first_image_ms" in metrics:
        ttfi = metrics["time_to_first_image_ms"]
        lines.append(f"- time-to-first-image p50/p99 = {_fmt(ttfi['p50'])}/{_fmt(ttfi['p99'])} ms")
    if "steps_per_second" in metrics:
        sps = metrics["steps_per_second"]
        lines.append(f"- steps/s p50 = {_fmt(sps['p50'])}")
    return lines
