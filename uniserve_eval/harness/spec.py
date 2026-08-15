"""One explicit benchmark operating point."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal


class TaskName(StrEnum):
    TEXT = "text"
    T2I = "t2i"
    I2I = "i2i"
    I2T = "i2t"
    INTERLEAVE = "interleave"


MetricDirection = Literal["higher", "lower"]


@dataclass(frozen=True)
class MetricDefinition:
    path: tuple[str, ...]
    direction: MetricDirection

    @property
    def name(self) -> str:
        return ".".join(self.path)

    def as_dict(self) -> dict[str, str]:
        return {"path": self.name, "direction": self.direction}


STREAM_TASKS = frozenset({TaskName.TEXT, TaskName.I2T, TaskName.INTERLEAVE})

TASK_WIRES = {
    TaskName.TEXT: ("openai_chat",),
    TaskName.T2I: ("images_generations", "openai_chat_json"),
    TaskName.I2I: ("openai_chat_json",),
    TaskName.I2T: ("openai_chat", "openai_chat_json"),
    TaskName.INTERLEAVE: ("openai_chat",),
}

WIRE_ENDPOINTS = {
    "openai_chat": "/v1/chat/completions",
    "openai_chat_json": "/v1/chat/completions",
    "images_generations": "/v1/images/generations",
}

DEFAULT_DATASETS = {
    TaskName.TEXT: "sharegpt",
    TaskName.T2I: "mjhq",
    TaskName.I2I: "pie-bench",
    TaskName.I2T: "synthetic-images",
    TaskName.INTERLEAVE: "ueval",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    task: TaskName
    model: str
    name: str
    metrics: tuple[MetricDefinition, ...]
    server: str
    endpoint: str = ""
    dataset: str = ""

    num_prompts: int = 1000
    request_rate: float = float("inf")
    max_concurrency: int | None = None
    warmup_requests: int = 1
    seed: int = 42

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    sampling_seed: int | None = None
    ignore_eos: bool = True
    max_tokens: int | None = None

    width: int | None = None
    height: int | None = None
    steps: int | None = None
    image_count: int | None = None
    minimum_average_images: float | None = None
    guidance_scale: float | None = None
    image_guidance_scale: float | None = None
    cfg_norm: str | None = None
    cfg_interval: tuple[float, float] | None = None
    timestep_shift: float | None = None

    wire: str = ""
    i2t_question: str = "Describe this image in detail."
    sample_gpu_memory: bool = True
    tokenizer: str | None = None
    sharegpt_context_len: int | None = None
    sharegpt_output_len: int | None = None
    dataset_path: str | None = None
    dataset_revision: str | None = None
    extra_request_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName(self.task))
        if self.num_prompts < 1:
            raise ValueError("num_prompts must be positive")
        if self.request_rate <= 0 or math.isnan(self.request_rate):
            raise ValueError("request_rate must be positive")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative")
        if not self.wire:
            object.__setattr__(self, "wire", TASK_WIRES[self.task][0])
        if self.wire not in TASK_WIRES[self.task]:
            supported = ", ".join(TASK_WIRES[self.task])
            raise ValueError(
                f"task {self.task.value} does not support wire {self.wire!r}; expected: {supported}"
            )
        if not self.endpoint:
            object.__setattr__(self, "endpoint", WIRE_ENDPOINTS[self.wire])
        if not self.dataset:
            object.__setattr__(self, "dataset", DEFAULT_DATASETS[self.task])
        if not self.metrics:
            raise ValueError("a benchmark point must protect at least one metric")
        if self.cfg_interval is not None and len(self.cfg_interval) != 2:
            raise ValueError("cfg_interval must contain exactly two values")
        if self.steps is not None and self.steps < 1:
            raise ValueError("steps must be positive")
        if self.image_count is not None and self.image_count < 1:
            raise ValueError("image_count must be positive")
        if self.minimum_average_images is not None and self.minimum_average_images < 0:
            raise ValueError("minimum_average_images must be non-negative")
        if self.task == TaskName.T2I and self.image_count is None:
            raise ValueError("t2i requires image_count")
        if self.task == TaskName.INTERLEAVE and self.image_count is not None:
            raise ValueError("interleave does not declare a per-request image count")
        if self.task != TaskName.INTERLEAVE and self.minimum_average_images is not None:
            raise ValueError("minimum_average_images is only valid for interleave")

    @property
    def is_stream_task(self) -> bool:
        return self.task in STREAM_TASKS

    def workload_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["task"] = self.task.value
        value["metrics"] = [metric.as_dict() for metric in self.metrics]
        value["request_rate"] = "inf" if math.isinf(self.request_rate) else self.request_rate
        return value
