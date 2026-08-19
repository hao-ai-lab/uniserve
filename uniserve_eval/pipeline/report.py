"""Build and render a point summary."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ..metrics import summarize
from ..tasks.base import BenchmarkTask
from ..types import (
    BenchmarkPoint,
    MetricDefinition,
    RequestRecord,
    ValidationResult,
    selected_rows_identity,
)


def build_summary(
    point: BenchmarkPoint,
    base_url: str,
    records: list[RequestRecord],
    dur_s: float,
    *,
    task: BenchmarkTask,
    selected_rows: dict[str, Any],
    tokenizer: Any | None = None,
    server_version: dict[str, Any] | None = None,
    launch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = summarize(records, dur_s, tokenizer=tokenizer)
    validation = task.validate(records).merged(_metric_validation(point.metrics, metrics))
    warnings = _warnings(records, server_version, launch or {})
    classifiers: dict[str, int] = {}
    for record in records:
        classifiers[record.classifier] = classifiers.get(record.classifier, 0) + 1
    return {
        "status": "completed",
        "benchmark": point.name,
        "task": point.task.value,
        "model": point.model,
        "dataset": point.dataset,
        "endpoint": point.endpoint,
        "base_url": base_url,
        "workload": point.workload_dict(),
        "selected_rows": selected_rows,
        "elapsed_s": dur_s,
        "request_count": len(records),
        "ok_count": sum(record.success for record in records),
        "failed_count": sum(not record.success for record in records),
        "classifiers": classifiers,
        "metrics": metrics,
        "metric_definitions": [definition.as_dict() for definition in point.metrics],
        "validation": validation.as_dict(),
        "warnings": warnings,
        "server_version": server_version,
        "launch": launch or {},
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
    launch: dict[str, Any],
) -> list[str]:
    values = sorted({warning for record in records for warning in record.warnings})
    if server_version is None:
        values.append("server_version_unavailable")
    if launch.get("dirty") is True:
        values.append("dirty_workspace")
    return values


__all__ = ["build_summary", "metric_value", "render_markdown", "selected_rows_identity"]
