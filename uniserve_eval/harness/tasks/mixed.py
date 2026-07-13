"""Row-dispatched mixed workload adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..spec import BenchmarkSpec, TaskName
from .base import BenchmarkTask, TaskRequest
from .i2t import I2TTask
from .t2i import T2ITask


def mixed_subtask_specs(spec: BenchmarkSpec) -> dict[str, BenchmarkSpec]:
    """Resolve the task-local contracts carried by one mixed workload."""

    common = {
        "workload_mix": {},
        "warmup_mix": {},
        "dataset_path": None,
    }
    return {
        TaskName.T2I.value: replace(
            spec,
            task=TaskName.T2I,
            dataset="mjhq",
            wire="openai_chat_json",
            endpoint="/v1/chat/completions",
            output_constraint="gen_only",
            max_tokens=None,
            top_k=None,
            min_p=None,
            repetition_penalty=None,
            frequency_penalty=None,
            presence_penalty=None,
            **common,
        ),
        TaskName.I2T.value: replace(
            spec,
            task=TaskName.I2T,
            dataset="image-dir",
            wire="openai_chat",
            endpoint="/v1/chat/completions",
            output_constraint="und_only",
            **common,
        ),
    }


class MixedTask(BenchmarkTask):
    """Route each row through its semantic task adapter."""

    def __init__(self, spec: BenchmarkSpec) -> None:
        super().__init__(spec)
        subtask_specs = mixed_subtask_specs(spec)
        self.adapters = {
            TaskName.T2I.value: T2ITask(subtask_specs[TaskName.T2I.value]),
            TaskName.I2T.value: I2TTask(subtask_specs[TaskName.I2T.value]),
        }

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        task = str(item.get("task", ""))
        adapter = self.adapters.get(task)
        if adapter is None:
            raise ValueError(f"mixed workload row has unsupported task {task!r}")
        return replace(adapter.build_request(item), semantic_task=task)
