"""Deterministic image-generation and image-understanding workload composition."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import gcd
from typing import Any

from ..spec import BenchmarkSpec, TaskName
from .mjhq import load_mjhq
from .synthetic_images import load_image_dir


@dataclass(frozen=True)
class MixedImageTextRows:
    measured: list[dict[str, Any]]
    warmup: list[dict[str, Any]]


def load_mixed_image_text(spec: BenchmarkSpec) -> MixedImageTextRows:
    """Load disjoint measured and warmup rows for a declared task mixture."""
    measured_counts = spec.workload_mix
    warmup_counts = spec.warmup_mix
    t2i_total = measured_counts[TaskName.T2I.value] + warmup_counts[TaskName.T2I.value]
    i2t_total = measured_counts[TaskName.I2T.value] + warmup_counts[TaskName.I2T.value]

    t2i_rows = load_mjhq(
        None,
        t2i_total,
        seed=spec.seed,
        revision=spec.t2i_dataset_revision,
    )
    i2t_rows = load_image_dir(
        spec.dataset_path or "",
        i2t_total,
        seed=spec.seed,
        question=spec.i2t_question,
    )
    for row in i2t_rows:
        row["task"] = TaskName.I2T.value
        row["max_tokens"] = spec.max_tokens

    measured_by_task = {
        TaskName.T2I.value: t2i_rows[: measured_counts[TaskName.T2I.value]],
        TaskName.I2T.value: i2t_rows[: measured_counts[TaskName.I2T.value]],
    }
    warmup_by_task = {
        TaskName.T2I.value: t2i_rows[measured_counts[TaskName.T2I.value] :],
        TaskName.I2T.value: i2t_rows[measured_counts[TaskName.I2T.value] :],
    }
    return MixedImageTextRows(
        measured=_interleave(measured_by_task),
        warmup=_interleave(warmup_by_task),
    )


def _interleave(rows_by_task: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Interleave a fixed mixture in its smallest integral repeating block."""
    queues = {task: deque(rows) for task, rows in rows_by_task.items()}
    positive_counts = [len(rows) for rows in rows_by_task.values() if rows]
    if not positive_counts:
        return []
    divisor = positive_counts[0]
    for count in positive_counts[1:]:
        divisor = gcd(divisor, count)
    block = [
        task
        for task in (TaskName.T2I.value, TaskName.I2T.value)
        for _ in range(len(rows_by_task.get(task, [])) // divisor)
    ]
    ordered: list[dict[str, Any]] = []
    while any(queues.values()):
        for task in block:
            if queues[task]:
                ordered.append(queues[task].popleft())
    return ordered
