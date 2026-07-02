"""Benchmark operating-point configuration.

One :class:`BenchmarkSpec` describes exactly one operating point (task, dataset,
arrival rate, concurrency, generation/image knobs). Sweeping across rates or
concurrency levels is the CLI's job (one run per point), mirroring how
``refs/sglang`` is invoked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TaskName(StrEnum):
    TEXT = "text"
    T2I = "t2i"
    I2I = "i2i"
    I2T = "i2t"
    INTERLEAVE = "interleave"


# Streaming token metrics (Family A) vs image-speed metrics (Family B).
STREAM_TASKS = frozenset({TaskName.TEXT, TaskName.INTERLEAVE, TaskName.I2T})
IMAGE_TASKS = frozenset({TaskName.T2I, TaskName.I2I})

DEFAULT_ENDPOINTS = {
    TaskName.TEXT: "/v1/chat/completions",
    TaskName.T2I: "/v1/images/generations",
    TaskName.I2I: "/generate",
    TaskName.I2T: "/generate",
    TaskName.INTERLEAVE: "/generate",
}

# Default real dataset backing each task. i2t defaults to deterministic
# synthetic images so the comparison runs from a clean checkout with no
# downloads; pass --dataset image-dir --dataset-path <dir> for real photos.
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
    endpoint: str = ""
    dataset: str = ""
    name: str = ""

    # Load / arrival.
    num_prompts: int = 1000
    request_rate: float = float("inf")
    max_concurrency: int | None = None
    warmup_requests: int = 1
    seed: int = 42

    # Text generation.
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = True
    max_tokens: int | None = None  # output cap; ShareGPT uses per-row output_len when None.

    # Image generation.
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    max_images: int = 1
    i2i_mode: str = "image"

    # Image understanding (i2t). ``i2t_wire`` selects the request shape:
    # "native"      -> UniServe /generate mode:"understand" SSE
    # "openai_chat" -> OpenAI chat completions with image_url content
    #                  (vLLM-Omni style, modalities:["text"])
    i2t_wire: str = "native"
    i2t_question: str = "Describe this image in detail."

    # Harness-side GPU memory sampling (nvidia-smi poll) during the timed
    # region; backend-agnostic so both servers are measured identically.
    sample_gpu_memory: bool = True

    # Tokenizer (ShareGPT length shaping + retokenized cross-check).
    tokenizer: str | None = None

    # ShareGPT shaping.
    sharegpt_context_len: int | None = None
    sharegpt_output_len: int | None = None

    # Dataset access: local file/dir override (else HF auto-download).
    dataset_path: str | None = None

    # Per-request body extras (rarely needed).
    extra_request_body: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName(self.task))
        if not self.endpoint:
            object.__setattr__(self, "endpoint", DEFAULT_ENDPOINTS[self.task])
        if not self.dataset:
            object.__setattr__(self, "dataset", DEFAULT_DATASETS[self.task])
        if not self.name:
            object.__setattr__(self, "name", f"{self.model}_{self.task.value}_{self.dataset}")

    @property
    def is_stream_task(self) -> bool:
        return self.task in STREAM_TASKS
