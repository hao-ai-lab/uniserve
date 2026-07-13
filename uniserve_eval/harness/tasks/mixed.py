"""Row-dispatched mixed workload adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..spec import BenchmarkSpec, TaskName
from .base import BenchmarkTask, TaskRequest
from .i2t import I2TTask
from .t2i import T2ITask


class MixedTask(BenchmarkTask):
    """Route each row through its semantic task adapter."""

    def __init__(self, spec: BenchmarkSpec) -> None:
        super().__init__(spec)
        common = {
            "workload_mix": {},
            "warmup_mix": {},
            "dataset_path": None,
        }
        self.adapters = {
            TaskName.T2I.value: T2ITask(
                replace(
                    spec,
                    task=TaskName.T2I,
                    dataset="mjhq",
                    wire="openai_chat_json",
                    endpoint="/v1/chat/completions",
                    **common,
                )
            ),
            TaskName.I2T.value: I2TTask(
                replace(
                    spec,
                    task=TaskName.I2T,
                    dataset="image-dir",
                    wire="openai_chat",
                    endpoint="/v1/chat/completions",
                    **common,
                )
            ),
        }

    def build_request(self, item: dict[str, Any]) -> TaskRequest:
        task = str(item.get("task", ""))
        adapter = self.adapters.get(task)
        if adapter is None:
            raise ValueError(f"mixed workload row has unsupported task {task!r}")
        return replace(adapter.build_request(item), semantic_task=task)
