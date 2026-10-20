"""Defines evaluator configuration, request, media, and result value types."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

# The synchronous video route, which answers with the encoded MP4.
VIDEOS_SYNC = "/v1/videos/sync"

MetricDirection = Literal["higher", "lower"]


class TaskName(StrEnum):
    """Identifies the request and validation contract for a benchmark point."""

    VIDEO = "video"


@dataclass(frozen=True)
class MetricDefinition:
    """Selects a numeric summary metric and its preferred direction.

    ``path`` addresses a value in the nested run-summary metrics mapping, one
    key per element. The profile loader in ``uniserve_eval.config`` rejects
    empty path segments; this type does not validate them.
    """

    path: tuple[str, ...]
    direction: MetricDirection

    @property
    def name(self) -> str:
        """Return the dotted metric path."""
        return ".".join(self.path)

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-compatible metric declaration."""
        return {"path": self.name, "direction": self.direction}


@dataclass(frozen=True)
class LoadConfig:
    """Configures warmup, arrivals, concurrency, and deterministic sampling.

    Attributes:
        num_prompts: Number of measured rows; dataset loading fails unless
            exactly this many rows resolve.
        request_rate: Mean Poisson arrival rate in requests per second;
            ``inf`` submits every request without delay.
        max_concurrency: In-flight request limit shared by warmup and
            measured requests, or ``None`` for no limit.
        warmup_requests: Number of concurrent warmup requests, all built
            from the first row and completed before measurement; ``0`` skips
            warmup. A point with a ``warmup_manifest`` sends none of them.
        warmup_manifest: JSON Lines file whose rows are the point's warmup
            requests, in file order, in place of the repeated first row.
        seed: Seeds arrival intervals and the generation seed of rows that
            carry none.
    """

    num_prompts: int = 1000
    request_rate: float = float("inf")
    max_concurrency: int | None = None
    warmup_requests: int = 1
    seed: int = 42
    warmup_manifest: str | None = None
    priming_manifest: str | None = None
    request_timeout_s: float = 6 * 60 * 60

    def __post_init__(self) -> None:
        """Validate load parameters that govern request scheduling."""
        if self.num_prompts < 1:
            raise ValueError("num_prompts must be positive")
        if self.request_rate <= 0 or math.isnan(self.request_rate):
            raise ValueError("request_rate must be positive")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative")
        if (
            not math.isfinite(self.request_timeout_s)
            or self.request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be finite and positive")


# Video tasks the video benchmark builds requests for. A t2va request carries
# no conditions; fl2va and ref2va rows carry their condition media.
VIDEO_TASKS = frozenset({"t2va", "fl2va", "ref2va"})


@dataclass(frozen=True)
class VideoConfig:
    """Configures the generated video work and how UniServe is asked for it.

    ``task``, ``short_edge`` and ``aspect_ratio`` form each request's target;
    ``seconds`` is the target duration of rows without a per-row override.
    ``short_edge`` is 768, the canvas rule's, or 480, the FastH3 buckets'.
    ``aspect_ratio`` is ``auto`` or a named ``W:H`` ratio the task accepts,
    and ``auto`` needs the 768-pixel short edge; the video task resolves the
    target to the generated canvas when the profile loads. A request states
    no sampling schedule: the server runs its checkpoint's own.

    ``condition_root`` is the directory that the condition media paths of
    fl2va and ref2va rows are relative to; a t2va point has none.

    ``prompt_tokens`` is the tokenizer length of the prompts that the
    MiniMax H3 dataset synthesizes. ``poll_interval_s`` is the wait between
    status reads of a ``/v1/videos`` job. ``media_dir``, when set, holds the
    measured videos, each named by its request ID.
    """

    seconds: float = 5.0
    task: str = "t2va"
    short_edge: int = 768
    aspect_ratio: str = "16:9"
    prompt_tokens: int = 1000
    poll_interval_s: float = 0.1
    media_dir: str | None = None
    condition_root: str | None = None

    def __post_init__(self) -> None:
        """Validate the target, the condition root and the polling interval."""
        if not math.isfinite(self.seconds) or self.seconds <= 0.0:
            raise ValueError("video seconds must be finite and positive")
        if self.task not in VIDEO_TASKS:
            raise ValueError(f"video task must be one of {sorted(VIDEO_TASKS)}")
        # Conditioned tasks read their rows' media from the root; t2va rows
        # carry none, so a root there would be configuration without effect.
        if (self.task == "t2va") != (self.condition_root is None):
            raise ValueError(
                "video condition_root names the condition media of fl2va "
                "and ref2va rows and is required exactly for those tasks"
            )
        if self.short_edge not in (768, 480):
            raise ValueError("video short_edge must be 768 or 480")
        if not isinstance(self.aspect_ratio, str):
            raise ValueError("video aspect_ratio must be a string")
        if self.prompt_tokens < 1:
            raise ValueError("video prompt_tokens must be positive")
        if not math.isfinite(self.poll_interval_s) or self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be finite and positive")


@dataclass(frozen=True)
class Example:
    """A dataset row and its per-request overrides.

    ``seed`` and ``seconds`` take precedence over the point's load seed and
    video duration when the request is built. ``prompt_len`` is the prompt's
    dataset-side token count, recorded with the request.

    ``conditions`` lists a video row's condition media in request order, each
    an object with the request's ``type``, ``role`` and, where they apply,
    ``frame_index`` and ``start_time_seconds``, plus ``media``: the file's
    path relative to the point's ``video.condition_root``.
    """

    id: str
    prompt: str
    prompt_len: int | None = None
    seed: int | None = None
    seconds: float | None = None
    conditions: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return populated fields as a JSON-compatible mapping."""
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True)
class VideoShape:
    """The canvas and frame count a video request's target resolves to."""

    width: int
    height: int
    frames: int


@dataclass(frozen=True)
class TaskRequest:
    """Contains an endpoint payload and the video work it asks for.

    ``payload`` is the canonical MiniMax-H3 body (``model``, ``prompt``,
    ``task``, ``conditions``, ``target`` and ``seed``), which the native
    video transport sends as JSON. Its conditions name their media by
    ``file://`` URI. ``video_shape`` is the canvas and frame count the
    target resolves to, and ``poll_interval_s`` the point's
    ``VideoConfig.poll_interval_s``.
    """

    endpoint: str
    payload: dict[str, Any]
    video_shape: VideoShape
    poll_interval_s: float = 0.1


@dataclass(frozen=True)
class DecodedVideo:
    """Contains validated MP4 bytes and decoded stream metadata.

    Built by ``inspect_video_bytes``. The frame rate is the video stream's
    average rate kept as an exact rational, and ``audio_samples`` counts
    samples per channel.
    """

    data: bytes
    sha256: str
    byte_size: int
    mime: str
    width: int
    height: int
    frame_count: int
    fps_numerator: int
    fps_denominator: int
    video_codec: str
    audio_codec: str
    audio_channels: int
    audio_sample_rate: int
    audio_samples: int
    sample_filename: str
    video_variance: float = 0.0
    audio_rms: float = 0.0

    @property
    def fps(self) -> float:
        """Return the exact rational frame rate as a float."""
        return self.fps_numerator / self.fps_denominator

    @property
    def video_duration_s(self) -> float:
        """Return duration derived from decoded frames and frame rate."""
        return self.frame_count / self.fps

    @property
    def audio_duration_s(self) -> float:
        """Return duration derived from decoded samples and sample rate."""
        return self.audio_samples / self.audio_sample_rate

    def metadata_dict(self) -> dict[str, float | int | str]:
        """Return persistent media metadata without the encoded bytes."""
        return {
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "fps_numerator": self.fps_numerator,
            "fps_denominator": self.fps_denominator,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "audio_channels": self.audio_channels,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_samples": self.audio_samples,
            "video_variance": self.video_variance,
            "audio_rms": self.audio_rms,
            "video_duration_s": self.video_duration_s,
            "audio_duration_s": self.audio_duration_s,
            "sample_filename": self.sample_filename,
        }


@dataclass
class RequestRecord:
    """Accumulates transport, timing, and decoded-output observations.

    ``send_request`` in ``uniserve_eval.transport.client`` creates one record
    per request and calls ``begin`` before dispatch; the video transport then
    records the HTTP status, closes the record, and classifies it, and
    ``send_request`` itself closes and classifies a record whose transport
    raised an ``Exception``.

    Absolute times are ``time.perf_counter()`` readings in seconds and are
    comparable only with other readings of that clock in the same process.
    ``scheduled_time`` is the arrival time assigned by the load generator,
    or ``None`` for warmup; ``start_time`` is taken after any concurrency
    queueing, so their difference is the client dispatch wait. ``latency``
    is seconds from ``start_time``. All durations become milliseconds only
    in ``record_dict``.
    """

    request_id: str
    task: str
    success: bool = False
    classifier: str = ""
    error: str | None = None
    endpoint: str = ""
    # Origin the request was routed to; replicas make this per request.
    base_url: str = ""
    scheduled_time: float | None = None
    start_time: float = 0.0
    http_response_time: float | None = None
    final_event_time: float | None = None
    latency: float = 0.0
    prompt_len: int = 0
    decoded_video: DecodedVideo | None = field(default=None, repr=False)
    video_body: bytes | None = field(default=None, repr=False)
    video_mime: str = ""
    requested_seconds: float | None = None
    # The canvas and frame count the request's target resolves to, which
    # the output is validated against.
    video_shape: VideoShape | None = None
    # Timings the server reported for this request, as it defines them: its
    # inference time and its named stage durations. A response that reports
    # none leaves them unset.
    server_inference_s: float | None = None
    server_stage_s: dict[str, float] = field(default_factory=dict)
    media_checks: dict[str, bool] = field(default_factory=dict)
    original_output: dict[str, Any] | None = None
    example: dict[str, Any] | None = None
    status_code: int | None = None

    def begin(self, *, endpoint: str, scheduled_time: float | None) -> None:
        """Initialize endpoint metadata and the monotonic request clock."""
        self.endpoint = endpoint
        self.scheduled_time = scheduled_time
        self.start_time = time.perf_counter()

    def note_http(self, status_code: int) -> None:
        """Record when the HTTP response headers become available."""
        self.http_response_time = time.perf_counter()
        self.status_code = status_code

    def close_now(self) -> None:
        """Close the request at the current monotonic time."""
        now = time.perf_counter()
        self.latency = now - self.start_time
        self.final_event_time = now

    def close_at(self, timestamp: float) -> None:
        """Close the request at a supplied monotonic event timestamp."""
        self.final_event_time = timestamp
        self.latency = timestamp - self.start_time

    def mark_failure(self, classifier: str, error: str | None = None) -> None:
        """Mark the request unsuccessful with a stable classifier.

        An ``error`` of ``None`` keeps any previously recorded detail.
        """
        self.success = False
        self.classifier = classifier
        if error is not None:
            self.error = error

    def mark_success(self) -> None:
        """Mark the request successful."""
        self.success = True
        self.classifier = "ok"

    def mark_transport_exception(self, error: BaseException) -> None:
        """Close and classify an exception raised by the transport path."""
        self.close_now()
        self.mark_failure(
            "transport_failure", f"{type(error).__name__}: {error}"
        )

    def record_dict(self) -> dict[str, Any]:
        """Return the durable request record, including media metadata."""
        http_response = (
            self.http_response_time - self.start_time
            if self.http_response_time is not None
            else None
        )
        dispatch_wait = (
            self.start_time - self.scheduled_time
            if self.scheduled_time is not None
            else None
        )

        return {
            # Stable identity and terminal classification.
            "request_id": self.request_id,
            "task": self.task,
            "success": self.success,
            "classifier": self.classifier,
            "error": self.error,
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            # Absolute event timestamps are raw `time.perf_counter()` seconds.
            "scheduled_time": self.scheduled_time,
            "client_send_time": self.start_time,
            "http_response_time": self.http_response_time,
            "final_event_time": self.final_event_time,
            # Derived durations use milliseconds in durable artifacts.
            "client_dispatch_wait_ms": (
                dispatch_wait * 1000.0 if dispatch_wait is not None else None
            ),
            "http_response_ms": (
                http_response * 1000.0 if http_response is not None else None
            ),
            "e2e_ms": self.latency * 1000.0,
            "prompt_len": self.prompt_len,
            # Media entries retain verified metadata while sample bytes live in
            # their own content-addressed artifacts.
            "video_output": (
                self.decoded_video.metadata_dict()
                if self.decoded_video is not None
                else None
            ),
            "requested_seconds": self.requested_seconds,
            "video_shape": (
                asdict(self.video_shape)
                if self.video_shape is not None
                else None
            ),
            "server_inference_s": self.server_inference_s,
            "server_stage_s": dict(self.server_stage_s),
            "media_checks": self.media_checks,
            "original_output": self.original_output,
            "example": self.example,
            "real_time_factor": (
                self.latency / self.decoded_video.video_duration_s
                if self.decoded_video is not None
                else None
            ),
            # Transport terminal state remains available for validation reports.
            "status_code": self.status_code,
        }


@dataclass(frozen=True)
class BenchmarkPoint:
    """Defines one fully resolved workload, task, server, and metric contract."""  # noqa: E501

    name: str
    server: str
    task: TaskName
    model: str
    dataset: str
    metrics: tuple[MetricDefinition, ...]
    load: LoadConfig = field(default_factory=LoadConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    dataset_revision: str | None = None
    dataset_path: str | None = None
    tokenizer: str | None = None
    endpoint: str = VIDEOS_SYNC
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the task identifier and require protected metrics."""
        # The dataclass is frozen, so normalization bypasses its __setattr__.
        object.__setattr__(self, "task", TaskName(self.task))
        if not self.metrics:
            raise ValueError(
                "a benchmark point must protect at least one metric"
            )

    def workload_dict(self) -> dict[str, Any]:
        """Return the benchmark workload as a JSON-compatible mapping."""
        load = asdict(self.load)
        # JSON has no infinity; "inf" is the spelling that
        # `uniserve_eval.config` accepts for an unlimited request rate.
        load["request_rate"] = (
            "inf"
            if math.isinf(self.load.request_rate)
            else self.load.request_rate
        )
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
            "provenance": self.provenance,
            "load": load,
            "video": asdict(self.video),
            "metrics": [metric.as_dict() for metric in self.metrics],
        }


@dataclass(frozen=True)
class ValidationResult:
    """Collects named observable checks, statistics, and warnings."""

    checks: dict[str, bool]
    statistics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Report whether at least one check exists and every check passes."""
        return bool(self.checks) and all(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible validation report."""
        return {
            "valid": self.valid,
            "checks": dict(self.checks),
            "statistics": dict(self.statistics),
            "warnings": list(self.warnings),
        }

    def merged(self, other: ValidationResult) -> ValidationResult:
        """Combine validation reports whose check names do not overlap.

        Statistics are merged with ``other`` winning on duplicate keys, and
        warnings are concatenated.

        Raises:
            ValueError: The two reports share a check name.
        """
        overlap = set(self.checks) & set(other.checks)
        if overlap:
            raise ValueError(
                f"duplicate validation checks: {', '.join(sorted(overlap))}"
            )
        return ValidationResult(
            checks={**self.checks, **other.checks},
            statistics={**self.statistics, **other.statistics},
            warnings=(*self.warnings, *other.warnings),
        )


@dataclass
class RunResult:
    """Pairs the completed summary with its artifact directory."""

    summary: dict[str, Any]
    output_dir: Path


def selected_rows_identity(rows: list[Example]) -> dict[str, Any]:
    """Return a deterministic count and digest for selected dataset rows.

    The digest covers each row's populated fields in row order, serialized
    as canonical JSON, so the same rows in the same order yield the same
    identity across runs.
    """
    encoded = json.dumps(
        [row.as_dict() for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {"count": len(rows), "sha256": hashlib.sha256(encoded).hexdigest()}
