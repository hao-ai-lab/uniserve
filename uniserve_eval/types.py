"""Defines evaluator configuration, request, media, and result value types."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeGuard

# Endpoint paths. `send_request` in `uniserve_eval.transport.client` selects
# the video, image-generations, and decision-readout transports by path; any
# other endpoint uses a chat transport.
CHAT_COMPLETIONS = "/v1/chat/completions"
IMAGES_GENERATIONS = "/v1/images/generations"
VIDEOS_SYNC = "/v1/videos/sync"
# TypeSafe System One decision readout (OpenAPI 0.2.0): one state per request.
SYSTEMONE = "/v1/systemone"
# DJev's multi-state readout, the reference implementation of the same
# decision semantics under its NanoJev-compatible schema.
DJEV_EVALUATE = "/api/evaluate"

DEFAULT_I2T_QUESTION = "Describe this image in detail."

MetricDirection = Literal["higher", "lower"]


class TaskName(StrEnum):
    """Identifies the request and validation contract for a benchmark point."""

    TEXT = "text"
    T2I = "t2i"
    I2I = "i2i"
    I2T = "i2t"
    INTERLEAVE = "interleave"
    VIDEO = "video"
    SYSTEMONE = "systemone"


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
            warmup.
        seed: Seeds row shuffling in the datasets that shuffle, arrival
            intervals, and the image or video generation seed of rows that
            carry none. The chat sampling ``seed`` comes from
            ``SamplingConfig.sampling_seed`` instead.
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
        if self.warmup_manifest and self.warmup_requests:
            raise ValueError("warmup_manifest requires warmup_requests = 0")
        if (
            not math.isfinite(self.request_timeout_s)
            or self.request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be finite and positive")


@dataclass(frozen=True)
class SamplingConfig:
    """Configures model sampling and endpoint-specific request extensions.

    Chat requests receive the sampling parameters and ``extra_body`` through
    ``BenchmarkTask.apply_text_sampling``, which omits optional parameters
    left as ``None`` and merges ``extra_body`` last, so its keys override
    any request field the task sets, including ``max_completion_tokens``.
    Image-generations requests carry only ``extra_body``, and video requests
    carry no field of this config.

    ``max_tokens`` is the output limit of rows without their own; when both
    are unset, the text task sends no limit and the i2t and interleave tasks
    send 512. ``stream`` defaults to the task's ``default_stream`` when a
    profile omits it; only the i2t task reads it, while the text and
    interleave tasks always stream and the other tasks never do. With
    ``ignore_eos`` set, the text and i2t tasks also validate fixed-length
    output.

    ``temperature``, ``top_p``, and ``ignore_eos`` of ``None`` leave the
    field out of the request, like the optional controls. A block-diffusion
    server samples every canvas under its own configured schedule and
    refuses these controls, so its points send none of them; a profile
    declares that with ``server_sampling = true`` (see
    ``uniserve_eval.config``).
    """

    temperature: float | None = 0.0
    top_p: float | None = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    sampling_seed: int | None = None
    ignore_eos: bool | None = True
    max_tokens: int | None = None
    stream: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageConfig:
    """Configures image geometry, denoising, guidance, and output count."""

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
        """Validate image settings with structural or positivity constraints."""
        if self.cfg_interval is not None and len(self.cfg_interval) != 2:
            raise ValueError("cfg_interval must contain exactly two values")
        if self.steps is not None and self.steps < 1:
            raise ValueError("steps must be positive")
        if self.image_count is not None and self.image_count < 1:
            raise ValueError("image_count must be positive")


# Request fields that define the generated work, per backend accepting
# configured serving options; the video task sets them from each request, so
# configured options cannot.
VIDEO_WORK_FIELDS = {
    # Fields of vLLM-Omni's `extra_params` form field.
    "vllm-omni": frozenset({"task", "duration", "audio_flow_shift"}),
    # Top-level fields of SGLang's native JSON request.
    "sglang": frozenset(
        {
            "model",
            "prompt",
            "seed",
            "task",
            "conditions",
            "target",
            "num_inference_steps",
            "flow_shift",
            "audio_flow_shift",
        }
    ),
}


# Video tasks the video benchmark builds requests for. A t2va request carries
# no conditions; fl2va and ref2va rows carry their condition media.
VIDEO_TASKS = frozenset({"t2va", "fl2va", "ref2va"})

# The schedule fields of the canonical request body: sigma points including
# the clean endpoint, then the video and audio schedule shifts.
VIDEO_SCHEDULE_FIELDS = (
    "num_inference_steps",
    "flow_shift",
    "audio_flow_shift",
)

# Schedule fields a baseline backend receives. A baseline runs whatever
# schedule it is sent, so a point measuring one must state each of these.
# FastVideo takes the sigma point count alone; its shifts come from the
# checkpoint's inference contract and cannot be sent.
_BASELINE_SCHEDULE_FIELDS = {
    "sglang": frozenset(VIDEO_SCHEDULE_FIELDS),
    "vllm-omni": frozenset(VIDEO_SCHEDULE_FIELDS),
    "fastvideo": frozenset({"num_inference_steps"}),
}


@dataclass(frozen=True)
class VideoConfig:
    """Configures the generated video work and how a backend is asked for it.

    ``task`` and ``aspect_ratio`` form each request's target together with the
    768-pixel short edge; ``seconds`` is the target duration of rows without
    a per-row override. ``aspect_ratio`` is ``auto`` or a named ``W:H`` ratio
    the task accepts; the video task resolves it to the generated canvas with
    the model package's canvas rule when the profile loads.

    ``num_inference_steps``, ``flow_shift`` and ``audio_flow_shift`` state the
    sampling schedule the point measures: sigma points including the clean
    endpoint, and the video and audio schedule shifts. UniServe receives them
    when set and refuses a schedule other than its checkpoint's, so they are
    optional there. A baseline runs the schedule it is sent and requires
    them: SGLang and vLLM-Omni need all three, and FastVideo, whose shifts
    come from its checkpoint, needs the point count and refuses the shifts.

    ``condition_root`` is the directory that the condition media paths of
    fl2va and ref2va rows are relative to; a t2va point has none. Every
    backend receives the same files from it, each in its own request form.

    ``parallel_decoding`` marks a FastVideo point that measures a parallel
    decoding (PDD) student, such as FastH3 OmniRef. FastVideo counts such a
    checkpoint's schedule in fused-block forwards, one fewer than its sigma
    points, where it counts the DMD and uniform schedules in sigma points.

    ``prompt_tokens`` is the tokenizer length of the prompts that the
    MiniMax H3 dataset synthesizes. ``extra_params`` adds serving options to
    every request of a backend that accepts them: vLLM-Omni receives them in
    its ``extra_params`` form field (for example ``{"preencode_mp4": true}``),
    SGLang as additional JSON fields (for example ``{"x264_preset":
    "ultrafast"}``). They may not restate the fields that fix the generated
    work, and other backends refuse them rather than ignoring them.
    """

    seconds: float = 5.0
    task: str = "t2va"
    aspect_ratio: str = "16:9"
    num_inference_steps: int | None = None
    flow_shift: float | None = None
    audio_flow_shift: float | None = None
    prompt_tokens: int = 1000
    backend: str = "uniserve"
    poll_interval_s: float = 0.1
    media_dir: str | None = None
    condition_root: str | None = None
    parallel_decoding: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the work, the schedule a backend needs, and its options."""
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
        if not isinstance(self.aspect_ratio, str):
            raise ValueError("video aspect_ratio must be a string")
        if self.prompt_tokens < 1:
            raise ValueError("video prompt_tokens must be positive")
        if self.backend not in {"uniserve", "vllm-omni", "sglang", "fastvideo"}:
            raise ValueError("unknown video backend")
        if not math.isfinite(self.poll_interval_s) or self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be finite and positive")
        # Only FastVideo's request counts steps by the checkpoint's kind.
        if self.parallel_decoding and self.backend != "fastvideo":
            raise ValueError(
                "video parallel_decoding applies only to fastvideo"
            )
        self._check_schedule()
        if self.extra_params and self.backend not in VIDEO_WORK_FIELDS:
            raise ValueError(
                "video extra_params apply only to "
                f"{' and '.join(sorted(VIDEO_WORK_FIELDS))}"
            )
        fixed = VIDEO_WORK_FIELDS.get(self.backend, frozenset()) & set(
            self.extra_params
        )
        if fixed:
            raise ValueError(f"video extra_params may not set {sorted(fixed)}")

    def _check_schedule(self) -> None:
        """Validate the stated schedule and the fields the backend takes."""
        steps = self.num_inference_steps
        # One interval between the noise and the clean endpoint is the
        # shortest schedule.
        if steps is not None and (
            isinstance(steps, bool) or not isinstance(steps, int) or steps < 2
        ):
            raise ValueError(
                "video num_inference_steps counts sigma points including the "
                "clean endpoint and must be an integer of at least 2"
            )
        for name in ("flow_shift", "audio_flow_shift"):
            shift = getattr(self, name)
            if shift is not None and (
                isinstance(shift, bool)
                or not isinstance(shift, int | float)
                or not math.isfinite(shift)
                or shift <= 0
            ):
                raise ValueError(f"video {name} must be finite and positive")

        required = _BASELINE_SCHEDULE_FIELDS.get(self.backend)
        if required is None:
            return
        stated = {
            name
            for name in VIDEO_SCHEDULE_FIELDS
            if getattr(self, name) is not None
        }
        if missing := sorted(required - stated):
            raise ValueError(
                f"video backend {self.backend} runs the schedule it is sent "
                f"and requires {missing}"
            )
        if unsent := sorted(stated - required):
            raise ValueError(
                f"video backend {self.backend} cannot receive {unsent}"
            )


@dataclass(frozen=True)
class Example:
    """A dataset row with optional task-specific modality overrides.

    Populated override fields take precedence over the benchmark point's
    configuration when a task builds the request. ``prompt_len`` and
    ``output_len`` are dataset-side token counts that stand in for
    server-reported usage when a chat response carries none; the text task
    also uses ``output_len`` as its output limit.

    ``conditions`` lists a video row's condition media in request order, each
    an object with the request's ``type``, ``role`` and, where they apply,
    ``frame_index`` and ``start_time_seconds``, plus ``media``: the file's
    path relative to the point's ``video.condition_root``.
    """

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
    seconds: float | None = None
    conditions: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state: Any = None
    questions: dict[str, Any] | None = None
    images: list[str] | None = None
    # Rows in one session are successive decisions: the next arrives only
    # after the preceding response. This identity never enters the request.
    session_id: str | None = None

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
class ConditionMedia:
    """A condition's media file, read before the request is measured.

    Attributes:
        path: Absolute local path of the file.
        mime: The file's media type, from its suffix.
        data: The file's bytes, for a backend that receives media as
            request parts.
    """

    path: str
    mime: str
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class TaskRequest:
    """Contains an endpoint payload and its streaming mode.

    A video request's ``payload`` is the canonical MiniMax-H3 body (``task``,
    ``conditions``, ``target``, ``seed`` and any stated schedule), which the
    native video transport sends to UniServe and SGLang as is and translates
    for the other backends; those take explicit dimensions or a frame count
    from ``video_shape``. Its conditions name their media by ``file://`` URI;
    ``condition_media`` holds the same files in the same order for backends
    that take uploads or plain paths instead. ``video_parallel_decoding`` is
    the point's ``VideoConfig.parallel_decoding``.
    """

    endpoint: str
    payload: dict[str, Any]
    stream: bool
    video_backend: str | None = None
    poll_interval_s: float = 0.1
    video_extra_params: dict[str, Any] = field(default_factory=dict)
    video_shape: VideoShape | None = None
    condition_media: tuple[ConditionMedia, ...] = ()
    video_parallel_decoding: bool = False


@dataclass(frozen=True)
class DecodedImage:
    """Contains validated image bytes and content-derived metadata.

    Built by ``inspect_image_bytes`` in ``uniserve_eval.transport.images``:
    ``mime`` is the detected format rather than the declared one, and
    ``sample_filename`` is the SHA-256 digest plus the format's extension.
    """

    data: bytes
    sha256: str
    byte_size: int
    mime: str
    width: int
    height: int
    sample_filename: str

    def metadata_dict(self) -> dict[str, int | str]:
        """Return persistent image metadata without the encoded bytes."""
        return {
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "sample_filename": self.sample_filename,
        }


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
    """Accumulates transport, timing, usage, and decoded-output observations.

    ``send_request`` in ``uniserve_eval.transport.client`` creates one record
    per request and calls ``begin`` before dispatch; the endpoint transports
    then record the HTTP status, close the record, and classify it, and
    ``send_request`` itself closes and classifies a record whose transport
    raised an ``Exception``.

    Absolute times are ``time.perf_counter()`` readings in seconds and are
    comparable only with other readings of that clock in the same process.
    ``scheduled_time`` is the arrival time assigned by the load generator,
    or ``None`` for warmup; ``start_time`` is taken after any concurrency
    queueing, so their difference is the client dispatch wait. ``latency``,
    ``ttft``, and the image latencies are seconds from ``start_time``. All
    durations become milliseconds only in ``record_dict``.

    ``itl`` holds gaps in seconds between consecutive text-bearing stream
    events, not per-token gaps, since one event may carry several tokens;
    for a block-diffusion server that streams one event per committed
    canvas, they are the intervals between consecutive blocks.
    The streaming chat transport excludes gaps that span an image event.
    ``text_times`` holds the absolute arrival time of every stamped
    text-bearing event, including one that follows an image, so it keeps
    the timeline that ``itl`` omits across images.
    """

    request_id: str
    task: str
    success: bool = False
    classifier: str = ""
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    endpoint: str = ""
    # Origin the request was routed to; replicas make this per request.
    base_url: str = ""
    scheduled_time: float | None = None
    start_time: float = 0.0
    http_response_time: float | None = None
    first_text_time: float | None = None
    first_image_done_time: float | None = None
    final_event_time: float | None = None
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    text_times: list[float] = field(default_factory=list)
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
    decoded_video: DecodedVideo | None = field(default=None, repr=False)
    video_body: bytes | None = field(default=None, repr=False)
    video_mime: str = ""
    requested_seconds: float | None = None
    # The canvas and frame count the request's target resolves to, which
    # the output is validated against.
    video_shape: VideoShape | None = None
    # Timings a video server reported for this request, as it defines them:
    # its inference time, its named stage durations, and its peak device
    # memory in MiB. Servers that report none leave them unset.
    server_inference_s: float | None = None
    server_stage_s: dict[str, float] = field(default_factory=dict)
    server_peak_memory_mib: float | None = None
    media_checks: dict[str, bool] = field(default_factory=dict)
    original_output: dict[str, Any] | None = None
    example: dict[str, Any] | None = None
    status_code: int | None = None
    finish_reason: str | None = None
    stop_reason: str | None = None
    # A decision readout answers every question of each state it carries;
    # `answers` keeps the server's answer objects keyed by question id.
    decision_states: int = 0
    decision_questions: int = 0
    answers: dict[str, Any] | None = None

    def begin(
        self,
        *,
        endpoint: str,
        scheduled_time: float | None,
        requested_output_len: int,
    ) -> None:
        """Initialize endpoint metadata and the monotonic request clock."""
        self.endpoint = endpoint
        self.scheduled_time = scheduled_time
        self.requested_output_len = requested_output_len
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

    def apply_choice_metadata(self, choice: dict[str, Any]) -> None:
        """Record finish and stop reasons from a completion choice."""
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            self.finish_reason = finish_reason
        stop_reason = choice.get("stop_reason")
        if isinstance(stop_reason, str):
            self.stop_reason = stop_reason

    def apply_usage(self, usage: dict[str, Any]) -> None:
        """Apply authoritative token and image-step usage fields.

        Reported token counts replace any earlier value and mark their source
        as ``server_usage``, which ``apply_token_fallbacks`` does not
        overwrite.
        ``image_steps_per_image`` is recorded only when every entry is a
        non-negative integer.
        """
        if isinstance(usage.get("completion_tokens"), int):
            self.output_len = int(usage["completion_tokens"])
            self.output_len_source = "server_usage"
        if isinstance(usage.get("prompt_tokens"), int):
            self.prompt_len = int(usage["prompt_tokens"])
            self.prompt_len_source = "server_usage"
        steps = usage.get("image_steps_per_image")
        if isinstance(steps, list) and all(
            _is_token_count(step) for step in steps
        ):
            self.image_steps = [int(step) for step in steps]

    def apply_cached_prompt_tokens(self, payload: dict[str, Any]) -> None:
        """Extract cached-token usage from an OpenAI-compatible payload."""
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return
        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            return
        cached = details.get("cached_tokens")
        if _is_token_count(cached):
            self.cached_prompt_tokens = int(cached)
            self.cached_prompt_tokens_source = (
                "openai_usage_prompt_tokens_details"
            )

    def apply_token_fallbacks(
        self, *, prompt_len: int, output_len_fallback: int
    ) -> None:
        """Fill token counts that the server did not report."""
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
        """Append streamed text and update first-token or inter-token timing.

        Args:
            content: Text carried by one stream event.
            timestamp: The event's client arrival time, or ``None`` when the
                event is unstamped; unstamped text updates no timing.
            last_text_time: Arrival time of the previous stamped text event,
                owned by the caller; ``None`` makes this event the first
                token for TTFT.
            count_itl: Whether the gap since ``last_text_time`` is an
                inter-token interval. The streaming chat transport passes
                false after an image event so image generation time does not
                enter ``itl``; the event still enters ``text_times``.
        """
        self.token_timing_available = True
        self.generated_text += content
        if timestamp is None:
            return
        self.text_times.append(timestamp)
        if last_text_time is None:
            self.ttft = timestamp - self.start_time
            self.first_text_time = timestamp
        elif count_itl:
            self.itl.append(timestamp - last_text_time)

    def add_image_arrival(self, count: int, timestamp: float | None) -> None:
        """Record completion latency for newly observed image parts.

        All ``count`` parts share the event's latency from ``start_time``.
        """
        if timestamp is None:
            return
        latency = timestamp - self.start_time
        if self.first_image_latency is None:
            self.first_image_latency = latency
            self.first_image_done_time = timestamp
        self.image_latencies.extend([latency] * count)

    def attach_images(
        self, decoded: list[DecodedImage], *, assign_json_latency: bool = False
    ) -> None:
        """Attach validated images and optionally assign response latency to each.

        With ``assign_json_latency``, ``image_latencies`` is replaced by the
        whole-request ``latency`` for every image, so the record must already
        be closed.
        """  # noqa: E501
        self.decoded_images = decoded
        self.images = len(decoded)
        if assign_json_latency and decoded:
            self.image_latencies = [self.latency] * self.images

    def record_dict(self) -> dict[str, Any]:
        """Return the durable request record, including generated text and media metadata."""  # noqa: E501
        generated_text_bytes = self.generated_text.encode("utf-8")
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
            "warnings": list(self.warnings),
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            # Absolute event timestamps are raw `time.perf_counter()` seconds.
            "scheduled_time": self.scheduled_time,
            "client_send_time": self.start_time,
            "http_response_time": self.http_response_time,
            "first_text_time": self.first_text_time,
            "first_image_done_time": self.first_image_done_time,
            "final_event_time": self.final_event_time,
            # Derived durations use milliseconds in durable artifacts.
            "client_dispatch_wait_ms": (
                dispatch_wait * 1000.0 if dispatch_wait is not None else None
            ),
            "http_response_ms": (
                http_response * 1000.0 if http_response is not None else None
            ),
            "e2e_ms": self.latency * 1000.0,
            "token_timing_available": self.token_timing_available,
            # `ttft` stays 0.0 unless a stamped text event set it. TPOT divides
            # the time from the first text event to the final event by the
            # output tokens after the first.
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
            # Token provenance distinguishes authoritative server counts from
            # client-side fallbacks. The full text is persisted along with its
            # UTF-8 size and SHA-256 digest.
            "prompt_len": self.prompt_len,
            "output_len": self.output_len,
            "requested_output_len": self.requested_output_len,
            "prompt_len_source": self.prompt_len_source,
            "output_len_source": self.output_len_source,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "cached_prompt_tokens_source": self.cached_prompt_tokens_source,
            "generated_text": self.generated_text,
            "generated_text_bytes": len(generated_text_bytes),
            "generated_text_sha256": (
                hashlib.sha256(generated_text_bytes).hexdigest()
                if generated_text_bytes
                else None
            ),
            # Media entries retain verified metadata while sample bytes live in
            # their own content-addressed artifacts.
            "images": self.images,
            "image_outputs": [
                image.metadata_dict() for image in self.decoded_images
            ],
            "first_image_latency_ms": (
                self.first_image_latency * 1000.0
                if self.first_image_latency is not None
                else None
            ),
            "image_latencies_ms": [
                value * 1000.0 for value in self.image_latencies
            ],
            "image_steps": list(self.image_steps),
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
            "server_peak_memory_mib": self.server_peak_memory_mib,
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
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
            # Decision readouts: answered states and questions, and the
            # answers as the server returned them.
            "decision_states": self.decision_states,
            "decision_questions": self.decision_questions,
            "answers": self.answers,
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
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    dataset_revision: str | None = None
    dataset_path: str | None = None
    tokenizer: str | None = None
    endpoint: str = CHAT_COMPLETIONS
    question: str | None = None
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
            "question": self.question,
            "provenance": self.provenance,
            "load": load,
            "sampling": asdict(self.sampling),
            "image": asdict(self.image),
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


def _is_token_count(value: Any) -> TypeGuard[int]:
    """Recognize non-negative integer token counts while excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


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
