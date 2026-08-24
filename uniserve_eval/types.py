"""Contracts shared by configuration, tasks, transport, and reports."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeGuard

CHAT_COMPLETIONS = "/v1/chat/completions"
IMAGES_GENERATIONS = "/v1/images/generations"
DEFAULT_I2T_QUESTION = "Describe this image in detail."

MetricDirection = Literal["higher", "lower"]


class TaskName(StrEnum):
    TEXT = "text"
    T2I = "t2i"
    I2I = "i2i"
    I2T = "i2t"
    INTERLEAVE = "interleave"


@dataclass(frozen=True)
class MetricDefinition:
    path: tuple[str, ...]
    direction: MetricDirection

    @property
    def name(self) -> str:
        return ".".join(self.path)

    def as_dict(self) -> dict[str, str]:
        return {"path": self.name, "direction": self.direction}


@dataclass(frozen=True)
class LoadConfig:
    num_prompts: int = 1000
    request_rate: float = float("inf")
    max_concurrency: int | None = None
    warmup_requests: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_prompts < 1:
            raise ValueError("num_prompts must be positive")
        if self.request_rate <= 0 or math.isnan(self.request_rate):
            raise ValueError("request_rate must be positive")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative")


@dataclass(frozen=True)
class SamplingConfig:
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
    stream: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageConfig:
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    image_count: int | None = None
    guidance_scale: float | None = None
    image_guidance_scale: float | None = None
    cfg_norm: str | None = None
    cfg_interval: tuple[float, float] | None = None
    timestep_shift: float | None = None

    def __post_init__(self) -> None:
        if self.cfg_interval is not None and len(self.cfg_interval) != 2:
            raise ValueError("cfg_interval must contain exactly two values")
        if self.steps is not None and self.steps < 1:
            raise ValueError("steps must be positive")
        if self.image_count is not None and self.image_count < 1:
            raise ValueError("image_count must be positive")


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    messages: list[dict[str, Any]] | None = None
    prompt_len: int | None = None
    output_len: int | None = None
    max_tokens: int | None = None
    input_image_b64: str | None = None
    input_image_mime: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    seed: int | None = None
    aspect_ratio: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class TaskRequest:
    endpoint: str
    payload: dict[str, Any]
    stream: bool


@dataclass(frozen=True)
class DecodedImage:
    data: bytes
    sha256: str
    byte_size: int
    mime: str
    width: int
    height: int
    sample_filename: str

    def metadata_dict(self) -> dict[str, int | str]:
        return {
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "sample_filename": self.sample_filename,
        }


@dataclass
class RequestRecord:
    request_id: str
    task: str
    success: bool = False
    classifier: str = ""
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    endpoint: str = ""
    scheduled_time: float | None = None
    start_time: float = 0.0
    http_response_time: float | None = None
    first_text_time: float | None = None
    first_image_done_time: float | None = None
    final_event_time: float | None = None
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    token_timing_available: bool = False
    prompt_len: int = 0
    output_len: int = 0
    requested_output_len: int = 0
    prompt_len_source: str = "request_fallback"
    output_len_source: str = "requested_fallback"
    cached_prompt_tokens: int | None = None
    cached_prompt_tokens_source: str = "unavailable"
    generated_text: str = ""
    images: int = 0
    image_latencies: list[float] = field(default_factory=list)
    first_image_latency: float | None = None
    image_steps: list[int] = field(default_factory=list)
    decoded_images: list[DecodedImage] = field(default_factory=list, repr=False)
    status_code: int | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None

    def begin(
        self,
        *,
        endpoint: str,
        scheduled_time: float | None,
        requested_output_len: int,
    ) -> None:
        self.endpoint = endpoint
        self.scheduled_time = scheduled_time
        self.requested_output_len = requested_output_len
        self.start_time = time.perf_counter()

    def note_http(self, status_code: int) -> None:
        self.http_response_time = time.perf_counter()
        self.status_code = status_code

    def close_now(self) -> None:
        now = time.perf_counter()
        self.latency = now - self.start_time
        self.final_event_time = now

    def close_at(self, timestamp: float) -> None:
        self.final_event_time = timestamp
        self.latency = timestamp - self.start_time

    def mark_failure(self, classifier: str, error: str | None = None) -> None:
        self.success = False
        self.classifier = classifier
        if error is not None:
            self.error = error

    def mark_success(self) -> None:
        self.success = True
        self.classifier = "ok"

    def mark_transport_exception(self, error: BaseException) -> None:
        self.close_now()
        self.mark_failure("transport_failure", f"{type(error).__name__}: {error}")

    def apply_choice_metadata(self, choice: dict[str, Any]) -> None:
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            self.finish_reason = finish_reason
        stop_reason = choice.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason

    def apply_usage(self, usage: dict[str, Any]) -> None:
        if isinstance(usage.get("completion_tokens"), int):
            self.output_len = int(usage["completion_tokens"])
            self.output_len_source = "server_usage"
        if isinstance(usage.get("prompt_tokens"), int):
            self.prompt_len = int(usage["prompt_tokens"])
            self.prompt_len_source = "server_usage"
        steps = usage.get("image_steps_per_image")
        if isinstance(steps, list) and all(_is_token_count(step) for step in steps):
            self.image_steps = [int(step) for step in steps]

    def apply_cached_prompt_tokens(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return
        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            return
        cached = details.get("cached_tokens")
        if _is_token_count(cached):
            self.cached_prompt_tokens = int(cached)
            self.cached_prompt_tokens_source = "openai_usage_prompt_tokens_details"

    def apply_token_fallbacks(self, *, prompt_len: int, output_len_fallback: int) -> None:
        if self.output_len_source != "server_usage":
            self.output_len = output_len_fallback
        if self.prompt_len_source != "server_usage":
            self.prompt_len = prompt_len

    def add_text(
        self,
        content: str,
        timestamp: float | None,
        *,
        last_text_time: float | None,
        count_itl: bool,
    ) -> None:
        self.token_timing_available = True
        self.generated_text += content
        if timestamp is None:
            return
        if last_text_time is None:
            self.ttft = timestamp - self.start_time
            self.first_text_time = timestamp
        elif count_itl:
            self.itl.append(timestamp - last_text_time)

    def add_image_arrival(self, count: int, timestamp: float | None) -> None:
        if timestamp is None:
            return
        latency = timestamp - self.start_time
        if self.first_image_latency is None:
            self.first_image_latency = latency
            self.first_image_done_time = timestamp
        self.image_latencies.extend([latency] * count)

    def attach_images(self, decoded: list[DecodedImage], *, assign_json_latency: bool = False) -> None:
        self.decoded_images = decoded
        self.images = len(decoded)
        if assign_json_latency and decoded:
            self.image_latencies = [self.latency] * self.images

    def record_dict(self) -> dict[str, Any]:
        generated_text_bytes = self.generated_text.encode("utf-8")
        http_response = (
            self.http_response_time - self.start_time
            if self.http_response_time is not None
            else None
        )
        dispatch_wait = (
            self.start_time - self.scheduled_time if self.scheduled_time is not None else None
        )
        return {
            "request_id": self.request_id,
            "task": self.task,
            "success": self.success,
            "classifier": self.classifier,
            "error": self.error,
            "warnings": list(self.warnings),
            "endpoint": self.endpoint,
            "scheduled_time": self.scheduled_time,
            "client_send_time": self.start_time,
            "http_response_time": self.http_response_time,
            "first_text_time": self.first_text_time,
            "first_image_done_time": self.first_image_done_time,
            "final_event_time": self.final_event_time,
            "client_dispatch_wait_ms": (
                dispatch_wait * 1000.0 if dispatch_wait is not None else None
            ),
            "http_response_ms": (http_response * 1000.0 if http_response is not None else None),
            "e2e_ms": self.latency * 1000.0,
            "token_timing_available": self.token_timing_available,
            "ttft_ms": (
                self.ttft * 1000.0
                if self.token_timing_available and self.ttft
                else None
            ),
            "tpot_ms": (
                (self.latency - self.ttft) / (self.output_len - 1) * 1000.0
                if self.token_timing_available and self.output_len > 1
                else None
            ),
            "itl_count": len(self.itl),
            "prompt_len": self.prompt_len,
            "output_len": self.output_len,
            "requested_output_len": self.requested_output_len,
            "prompt_len_source": self.prompt_len_source,
            "output_len_source": self.output_len_source,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "cached_prompt_tokens_source": self.cached_prompt_tokens_source,
            "generated_text_bytes": len(generated_text_bytes),
            "generated_text_sha256": (
                hashlib.sha256(generated_text_bytes).hexdigest() if generated_text_bytes else None
            ),
            "images": self.images,
            "image_outputs": [image.metadata_dict() for image in self.decoded_images],
            "first_image_latency_ms": (
                self.first_image_latency * 1000.0 if self.first_image_latency is not None else None
            ),
            "image_latencies_ms": [value * 1000.0 for value in self.image_latencies],
            "image_steps": list(self.image_steps),
            "status_code": self.status_code,
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class BenchmarkPoint:
    name: str
    server: str
    task: TaskName
    model: str
    dataset: str
    metrics: tuple[MetricDefinition, ...]
    load: LoadConfig = field(default_factory=LoadConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    dataset_revision: str | None = None
    dataset_path: str | None = None
    tokenizer: str | None = None
    endpoint: str = CHAT_COMPLETIONS
    question: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName(self.task))
        if not self.metrics:
            raise ValueError("a benchmark point must protect at least one metric")

    def workload_dict(self) -> dict[str, Any]:
        load = asdict(self.load)
        load["request_rate"] = "inf" if math.isinf(self.load.request_rate) else self.load.request_rate
        return {
            "name": self.name,
            "server": self.server,
            "task": self.task.value,
            "model": self.model,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "dataset_path": self.dataset_path,
            "tokenizer": self.tokenizer,
            "endpoint": self.endpoint,
            "question": self.question,
            "load": load,
            "sampling": asdict(self.sampling),
            "image": asdict(self.image),
            "metrics": [metric.as_dict() for metric in self.metrics],
        }


@dataclass(frozen=True)
class ValidationResult:
    checks: dict[str, bool]
    statistics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": dict(self.checks),
            "statistics": dict(self.statistics),
            "warnings": list(self.warnings),
        }

    def merged(self, other: ValidationResult) -> ValidationResult:
        overlap = set(self.checks) & set(other.checks)
        if overlap:
            raise ValueError(f"duplicate validation checks: {', '.join(sorted(overlap))}")
        return ValidationResult(
            checks={**self.checks, **other.checks},
            statistics={**self.statistics, **other.statistics},
            warnings=(*self.warnings, *other.warnings),
        )


@dataclass
class RunResult:
    summary: dict[str, Any]
    output_dir: Path


def _is_token_count(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def selected_rows_identity(rows: list[Example]) -> dict[str, Any]:
    encoded = json.dumps(
        [row.as_dict() for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {"count": len(rows), "sha256": hashlib.sha256(encoded).hexdigest()}
