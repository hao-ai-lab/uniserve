"""Build and render direct benchmark summaries."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

from .metrics.common import RequestRecord
from .metrics.image import summarize_image
from .metrics.stream import summarize_stream
from .spec import BenchmarkSpec, MetricDefinition
from .tasks.base import BenchmarkTask
from .validation import ValidationResult


def selected_rows_identity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    encoded = json.dumps(
        list(rows),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {"count": len(rows), "sha256": hashlib.sha256(encoded).hexdigest()}


def build_summary(
    spec: BenchmarkSpec,
    base_url: str,
    records: list[RequestRecord],
    dur_s: float,
    *,
    task: BenchmarkTask,
    selected_rows: dict[str, Any],
    tokenizer: Any | None = None,
    server_version: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if spec.is_stream_task:
        family = "stream"
        metrics = summarize_stream(records, dur_s, tokenizer=tokenizer)
    else:
        family = "image"
        metrics = summarize_image(records, dur_s)

    validation = task.validate(records).merged(_metric_validation(spec.metrics, metrics))
    warnings = _warnings(records, server_version, provenance or {})
    classifiers: dict[str, int] = {}
    for record in records:
        classifiers[record.classifier] = classifiers.get(record.classifier, 0) + 1
    return {
        "status": "completed",
        "benchmark": spec.name,
        "task": spec.task.value,
        "model": spec.model,
        "dataset": spec.dataset,
        "endpoint": spec.endpoint,
        "base_url": base_url,
        "workload": spec.workload_dict(),
        "selected_rows": selected_rows,
        "elapsed_s": dur_s,
        "request_count": len(records),
        "ok_count": sum(record.success for record in records),
        "failed_count": sum(not record.success for record in records),
        "classifiers": classifiers,
        "metric_family": family,
        "metrics": metrics,
        "metric_definitions": [definition.as_dict() for definition in spec.metrics],
        "validation": validation.as_dict(),
        "warnings": warnings,
        "server_version": server_version,
        "provenance": provenance or {},
    }


def metric_value(metrics: dict[str, Any], definition: MetricDefinition) -> float | None:
    value: Any = metrics
    for part in definition.path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['benchmark']}",
        "",
        f"- requests: {summary['ok_count']}/{summary['request_count']} successful",
        f"- elapsed: {summary['elapsed_s']:.3f} s",
        f"- validation: {'pass' if summary['validation']['valid'] else 'fail'}",
        "",
        "| Metric | Value | Direction |",
        "| --- | ---: | --- |",
    ]
    metrics = summary.get("metrics", {})
    for raw in summary.get("metric_definitions", []):
        definition = MetricDefinition(tuple(str(raw["path"]).split(".")), raw["direction"])
        value = metric_value(metrics, definition)
        rendered = "n/a" if value is None else f"{value:.6g}"
        lines.append(f"| `{definition.name}` | {rendered} | {definition.direction} |")
    failed = [name for name, passed in summary["validation"]["checks"].items() if not passed]
    if failed:
        lines.extend(["", "Failed checks: " + ", ".join(f"`{name}`" for name in failed) + "."])
    if summary.get("warnings"):
        lines.extend(["", "Warnings: " + ", ".join(summary["warnings"]) + "."])
    return "\n".join(lines) + "\n"


def _metric_validation(
    definitions: tuple[MetricDefinition, ...], metrics: dict[str, Any]
) -> ValidationResult:
    checks = {}
    for definition in definitions:
        value = metric_value(metrics, definition)
        checks[f"metric:{definition.name}"] = value is not None and value > 0
    return ValidationResult(checks=checks)


def _warnings(
    records: Sequence[RequestRecord],
    server_version: dict[str, Any] | None,
    provenance: dict[str, Any],
) -> list[str]:
    values = sorted({warning for record in records for warning in record.warnings})
    if server_version is None:
        values.append("server_version_unavailable")
    if provenance.get("dirty") is True:
        values.append("dirty_workspace")
    return values
