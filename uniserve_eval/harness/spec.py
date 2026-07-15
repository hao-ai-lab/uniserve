"""Benchmark operating-point configuration.

One :class:`BenchmarkSpec` describes exactly one operating point: task, dataset,
request-arrival behavior, concurrency bound, and generation controls. The matrix
runner creates one spec for each named load case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TaskName(StrEnum):
    TEXT = "text"
    T2I = "t2i"
    I2I = "i2i"
    I2T = "i2t"
    MIXED = "mixed"
    INTERLEAVE = "interleave"
    DEFAULT = "default"


# Streaming token metrics (Family A) vs image-speed metrics (Family B).
STREAM_TASKS = frozenset({TaskName.TEXT, TaskName.DEFAULT, TaskName.I2T, TaskName.INTERLEAVE})
IMAGE_TASKS = frozenset({TaskName.T2I, TaskName.I2I})

# Wire = request/response shape used to exercise one task over one endpoint.
#
# * "openai_chat"       -> OpenAI chat completions SSE, streamed; per-chunk
#                          timing (TTFT/ITL) and delta.images image counting
# * "openai_chat_json"  -> OpenAI chat completions, one non-streamed JSON
#                          response; E2E + counts only (diffusion-pipeline
#                          backends such as vLLM-Omni, and image-only chat)
# * "images_generations"-> OpenAI-style /v1/images/generations JSON
#
# Endpoint is derived from (task, wire) unless --endpoint overrides it.
TASK_WIRES = {
    TaskName.TEXT: ("openai_chat",),
    TaskName.T2I: ("images_generations", "openai_chat_json"),
    TaskName.I2I: ("openai_chat_json",),
    TaskName.I2T: ("openai_chat", "openai_chat_json"),
    TaskName.MIXED: ("mixed_chat",),
    TaskName.INTERLEAVE: ("openai_chat",),
    TaskName.DEFAULT: ("openai_chat",),
}

WIRE_ENDPOINTS = {
    "openai_chat": "/v1/chat/completions",
    "openai_chat_json": "/v1/chat/completions",
    "images_generations": "/v1/images/generations",
    "mixed_chat": "/v1/chat/completions",
}

# Default real dataset backing each task. i2t defaults to deterministic
# synthetic images so the comparison runs from a clean checkout with no
# downloads; pass --dataset image-dir --dataset-path <dir> for real photos.
DEFAULT_DATASETS = {
    TaskName.TEXT: "sharegpt",
    TaskName.T2I: "mjhq",
    TaskName.I2I: "pie-bench",
    TaskName.I2T: "synthetic-images",
    TaskName.MIXED: "mixed-image-text",
    TaskName.INTERLEAVE: "mjhq",
    TaskName.DEFAULT: "ueval",
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
    workload_mix: dict[str, int] = field(default_factory=dict)
    warmup_mix: dict[str, int] = field(default_factory=dict)

    # Text generation.
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    sampling_seed: int | None = None
    chat_template_kwargs: dict[str, object] = field(default_factory=dict)
    ignore_eos: bool = True
    max_tokens: int | None = None  # output cap; ShareGPT uses per-row output_len when None.

    # Image generation.
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    denoise_updates: int | None = None
    max_images: int | None = None
    guidance_scale: float | None = None
    image_guidance_scale: float | None = None
    cfg_norm: str | None = None
    cfg_interval: tuple[float, float] | None = None
    timestep_shift: float | None = None
    image_think: bool | None = None
    image_t_eps: float | None = None

    # Request/response shape for this task; see TASK_WIRES. Empty selects the
    # task's first (default) wire.
    wire: str = ""
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
    dataset_revision: str | None = None
    t2i_dataset_revision: str | None = None
    i2t_dataset_revision: str | None = None

    # Per-request body extras (rarely needed).
    extra_request_body: dict = field(default_factory=dict)

    # Formal measurement contract and artifact provenance.
    runtime_profile_id: str = "unspecified"
    measurement_interface: str = "public_protocol_adapter"
    cache_read_policy: str = "enabled"
    cache_write_policy: str = "enabled"
    adapter_selection: str = "base"
    structured_output_policy: str = "none"
    output_constraint: str = "default"
    preprocessing: str = "dataset_default"
    measured_runs: int = 1
    server_topology: str = "single_server"
    plan_evidence_policy: str = "declared_contract"
    acceptance_min_success: int = 1
    acceptance_max_failed: int = 0
    acceptance_min_images_per_success: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName(self.task))
        if self.num_prompts < 1:
            raise ValueError("num_prompts must be positive")
        self._validate_workload_mix()
        if not self.wire:
            object.__setattr__(self, "wire", TASK_WIRES[self.task][0])
        if self.wire not in TASK_WIRES[self.task]:
            supported = ", ".join(TASK_WIRES[self.task])
            raise ValueError(
                f"task {self.task.value} does not support wire {self.wire!r}; expected one of: {supported}"
            )
        if not self.endpoint:
            object.__setattr__(self, "endpoint", WIRE_ENDPOINTS[self.wire])
        if not self.dataset:
            object.__setattr__(self, "dataset", DEFAULT_DATASETS[self.task])
        if not self.name:
            object.__setattr__(self, "name", f"{self.model}_{self.task.value}_{self.dataset}")
        if self.cfg_interval is not None and len(self.cfg_interval) != 2:
            raise ValueError("cfg_interval must contain exactly two values")
        if self.denoise_updates is not None:
            if self.denoise_updates < 1:
                raise ValueError("denoise_updates must be positive")
            if self.steps is None:
                raise ValueError("denoise_updates requires a backend steps value")
        if self.measured_runs != 1:
            raise ValueError("one harness invocation is exactly one measured run")
        if self.plan_evidence_policy not in {
            "declared_contract",
            "runtime_inspection",
            "reference_protocol",
        }:
            raise ValueError("unsupported plan evidence policy")
        if (
            self.acceptance_min_success < 1
            or self.acceptance_max_failed < 0
            or self.acceptance_min_images_per_success < 0
        ):
            raise ValueError(
                "acceptance criteria must require success and a non-negative failure bound"
            )

    def _validate_workload_mix(self) -> None:
        if self.task != TaskName.MIXED:
            if self.workload_mix or self.warmup_mix:
                raise ValueError("workload_mix and warmup_mix require task 'mixed'")
            return
        supported = {TaskName.T2I.value, TaskName.I2T.value}
        for field_name, values, expected_total in (
            ("workload_mix", self.workload_mix, self.num_prompts),
            ("warmup_mix", self.warmup_mix, self.warmup_requests),
        ):
            if set(values) != supported:
                raise ValueError(f"{field_name} must declare exactly t2i and i2t")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values.values()
            ):
                raise ValueError(f"{field_name} counts must be non-negative integers")
            if sum(values.values()) != expected_total:
                raise ValueError(f"{field_name} counts must sum to {expected_total}")
        if not all(self.workload_mix.values()):
            raise ValueError("mixed workloads require at least one measured request per task")
        if self.dataset_path is None:
            raise ValueError("mixed-image-text requires an image dataset directory")

    @property
    def is_stream_task(self) -> bool:
        return self.task in STREAM_TASKS
