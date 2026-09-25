"""Worker completion records and response serialization.

A rank reports each submitted batch as one `BatchOutput`: a `RequestOutput`
per call plus the tensor products successful calls published. The execution
package builds these records once every call's output has materialized on the
host; `messages.finalize_response` serializes them with `to_mapping` for the
PyO3 transport (`crates/worker-ipc-py`), which decodes the mapping into the
Rust `BatchOutput`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol import identity, transfer
from uniserve_worker.protocol.batch import TensorPublication
from uniserve_worker.protocol.call import (
    CallKind,
    CallStatus,
    ErrorCode,
    TransferMode,
    computation,
)
from uniserve_worker.protocol.validation import (
    _bool,
    _enum,
    _map,
    _optional_uint,
    _seq,
    _str,
    _uint,
    _uints,
)


def _logprob_entries(value: object) -> tuple[tuple[int, float, int], ...]:
    """Decode ranked candidates as ``(token_id, logprob, rank)`` tuples."""
    entries = []
    for item in _seq(value, "logprob entries"):
        data = _map(item, "logprob entry")
        entries.append(
            (
                _uint(data.get("token_id"), "logprob.token_id"),
                _logprob_value(data.get("logprob")),
                _uint(data.get("rank"), "logprob.rank"),
            )
        )

    return tuple(entries)


def _logprob_value(value: object) -> float:
    """Read a score, including negative infinity for zero-probability tokens."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise invalid_descriptor("logprob value must be numeric")
    return float(value)


@dataclass(frozen=True, slots=True)
class FinishFlags:
    """Device-observed finish candidates for a token call.

    `eos` means an end-of-sequence token was selected, `length` that the
    length limit was reached, and `stop` that a stop condition matched.
    """

    eos: bool = False
    length: bool = False
    stop: bool = False

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "finish_flags"
    ) -> FinishFlags:
        """Parse EOS, length-limit, and stop-sequence termination flags."""
        data = _map(value, where)
        return cls(
            eos=_bool(data.get("eos", False), f"{where}.eos"),
            length=_bool(data.get("length", False), f"{where}.length"),
            stop=_bool(data.get("stop", False), f"{where}.stop"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize generation termination flags for IPC."""
        return {"eos": self.eos, "length": self.length, "stop": self.stop}


@dataclass(frozen=True, slots=True)
class TimingCounters:
    """Per-call queue, device, copy, and host time in microseconds.

    Accounting only; timings are not part of any call identity.
    """

    queued_us: int = 0
    device_us: int = 0
    copy_us: int = 0
    host_us: int = 0

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "timing_counters"
    ) -> TimingCounters:
        """Parse non-negative microsecond timings; absent fields are zero."""
        data = _map(value, where)
        return cls(
            queued_us=_uint(data.get("queued_us", 0), f"{where}.queued_us"),
            device_us=_uint(data.get("device_us", 0), f"{where}.device_us"),
            copy_us=_uint(data.get("copy_us", 0), f"{where}.copy_us"),
            host_us=_uint(data.get("host_us", 0), f"{where}.host_us"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize execution-stage timings in microseconds for IPC."""
        return {
            "queued_us": self.queued_us,
            "device_us": self.device_us,
            "copy_us": self.copy_us,
            "host_us": self.host_us,
        }


@dataclass(frozen=True, slots=True)
class PosixShmArtifact:
    """Identifies a completed media artifact stored in POSIX shared storage.

    The worker writes the bytes and closes its mapping before publishing the
    name (`uniserve_worker.media.storage.publish_media_bytes`); the engine
    claims the object by name, which unlinks it, when it receives the batch
    result.
    """

    name: str

    def __post_init__(self) -> None:
        """Require a non-empty object name without a path separator."""
        if not self.name or "/" in self.name:
            raise invalid_descriptor(
                "POSIX shared-storage artifact name is invalid"
            )

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "artifact handle"
    ) -> PosixShmArtifact:
        """Parse a ``{"transport": "posix_shm", "value": {...}}`` handle."""
        data = _map(value, where)
        if data.get("transport") != "posix_shm":
            raise invalid_descriptor(f"{where}.transport is invalid")
        payload = _map(data.get("value"), f"{where}.value")
        return cls(name=_str(payload.get("name"), f"{where}.value.name"))

    def to_mapping(self) -> dict[str, object]:
        """Serialize the shared-storage handle as a tagged transport value."""
        return {"transport": "posix_shm", "value": {"name": self.name}}


@dataclass(frozen=True, slots=True)
class MediaOutput:
    """A completed media artifact: its storage handle and length in bytes."""

    handle: PosixShmArtifact
    bytes: int

    def __post_init__(self) -> None:
        """Require a positive byte length."""
        if self.bytes < 1:
            raise invalid_descriptor("media output locator is invalid")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "media_output"
    ) -> MediaOutput:
        """Parse a validated media artifact handle and byte extent."""
        data = _map(value, where)
        return cls(
            handle=PosixShmArtifact.from_mapping(
                data.get("handle"), f"{where}.handle"
            ),
            bytes=_uint(data.get("bytes"), f"{where}.bytes"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize a completed media artifact for IPC."""
        return {"handle": self.handle.to_mapping(), "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """One call's completion: status, accepted progress, tokens, and products.

    A successful call reports the request's coordinates after it ran; a
    predicated call did not run and reports the request's accepted
    coordinates.
    """

    request_key: identity.RequestKey
    call_id: identity.CallId
    status: CallStatus
    # Allocation generations of the products the call emitted.
    product_generations: tuple[int, ...]
    # Set exactly when status is ERROR.
    error_code: ErrorCode | None
    timing_counters: TimingCounters
    # Kind of the call that produced this result; the engine requires it to
    # match the submitted call.
    kind: CallKind
    # Logical position of the request's next input token.
    position: int
    # Accepted KV prefix a successor may attend to, in tokens.
    kv_visible_len: int
    # KV extent execution initialized, in tokens, including rejected
    # speculative positions.
    kv_computed_len: int
    # Denoising steps completed for the request.
    num_completed_steps: int
    # Token ids the sampler accepted.
    committed_tokens: tuple[int, ...]
    finish_flags: FinishFlags
    media_output: MediaOutput | None = None
    # KV publication; only a successful KV_PUBLISH call carries one.
    kv_output: transfer.KvTransfer | None = None
    # Natural-log probability of the final accepted token, when requested.
    sampled_logprob: float | None = None
    # Ranked (token_id, logprob, rank) candidates for the final accepted token.
    top_logprobs: tuple[tuple[int, float, int], ...] = ()
    # Ranked candidates for each scored prompt position, in input order.
    prompt_logprobs: tuple[tuple[tuple[int, float, int], ...], ...] = ()

    def validate(self) -> None:
        """Check that identity, status, coordinates, and outputs agree.

        `PendingOutput.materialize` in `uniserve_worker.execution.output`
        calls this on every completion it builds, and `from_mapping` on every
        parsed one.

        Raises:
            WorkerError: The call id is not positive; a KV publication is
                attached to anything but a successful KV_PUBLISH of this
                call; the visible KV extent exceeds the computed one; a
                coordinate is negative; the error code does not match the
                status; or a predicated completion carries tokens,
                logprobs, products, or finish flags.
        """
        if self.call_id.batch_id < 1:
            raise invalid_descriptor("completion call id must be positive")

        if self.kv_output is not None and (
            self.status is not CallStatus.OK
            or self.kind is not TransferMode.KV_PUBLISH
            or self.kv_output.source.owner != self.request_key
            or self.kv_output.source.producer_call_id != self.call_id
        ):
            raise invalid_descriptor(
                "KV publication does not belong to its successful completion"
            )

        if self.kv_visible_len > self.kv_computed_len:
            raise invalid_descriptor(
                "completion selected KV length exceeds computed length"
            )
        if (
            min(
                self.position,
                self.kv_visible_len,
                self.kv_computed_len,
                self.num_completed_steps,
            )
            < 0
        ):
            raise invalid_descriptor(
                "completion execution coordinates must be non-negative"
            )
        if self.status is CallStatus.ERROR:
            if self.error_code is None:
                raise invalid_descriptor(
                    "an error completion must carry an error code"
                )
        elif self.error_code is not None:
            raise invalid_descriptor(
                "a non-error completion must not carry an error code"
            )
        if self.status is CallStatus.PREDICATED and (
            self.committed_tokens
            or self.sampled_logprob is not None
            or self.top_logprobs
            or self.prompt_logprobs
            or self.product_generations
            or self.finish_flags.eos
            or self.finish_flags.length
            or self.finish_flags.stop
        ):
            raise invalid_descriptor(
                "a predicated completion must select its parent without "
                "semantic output"
            )

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "completion"
    ) -> RequestOutput:
        """Parse a completion mapping and run `validate` on the result."""
        data = _map(value, where)
        record = cls(
            request_key=identity.RequestKey.from_mapping(
                data.get("request_key"), f"{where}.request_key"
            ),
            call_id=identity.CallId.from_mapping(
                data.get("call_id"), f"{where}.call_id"
            ),
            status=_enum(CallStatus, data.get("status"), f"{where}.status"),
            product_generations=_uints(
                data.get("product_generations", ()),
                f"{where}.product_generations",
            ),
            error_code=(
                None
                if data.get("error_code") is None
                else _enum(ErrorCode, data["error_code"], f"{where}.error_code")
            ),
            timing_counters=TimingCounters.from_mapping(
                data.get("timing_counters"), f"{where}.timing_counters"
            ),
            kind=computation(data.get("code"), f"{where}.code"),
            position=_uint(data.get("position"), f"{where}.position"),
            kv_visible_len=_uint(
                data.get("kv_visible_len"), f"{where}.kv_visible_len"
            ),
            kv_computed_len=_uint(
                data.get("kv_computed_len"), f"{where}.kv_computed_len"
            ),
            num_completed_steps=_uint(
                data.get("num_completed_steps"), f"{where}.num_completed_steps"
            ),
            sampled_logprob=(
                None
                if data.get("sampled_logprob") is None
                else _logprob_value(data["sampled_logprob"])
            ),
            top_logprobs=_logprob_entries(data.get("top_logprobs", ())),
            prompt_logprobs=tuple(
                _logprob_entries(entries)
                for entries in _seq(
                    data.get("prompt_logprobs", ()), "prompt_logprobs"
                )
            ),
            committed_tokens=_uints(
                data.get("committed_tokens", ()), f"{where}.committed_tokens"
            ),
            finish_flags=FinishFlags.from_mapping(
                data.get("finish_flags"), f"{where}.finish_flags"
            ),
            kv_output=None
            if data.get("kv_output") is None
            else transfer.KvTransfer.from_mapping(
                data["kv_output"], f"{where}.kv_output"
            ),
            media_output=None
            if data.get("media_output") is None
            else MediaOutput.from_mapping(
                data["media_output"], f"{where}.media_output"
            ),
        )

        record.validate()
        return record

    def to_mapping(self) -> dict[str, object]:
        """Encode the completion into the wire mapping the transport decodes.

        The request-key, finish-flag, and timing mappings are written
        inline; they must stay identical to those records' `to_mapping`
        output.
        """
        flags = self.finish_flags
        key = self.request_key
        timing = self.timing_counters
        error_code = self.error_code
        return {
            "request_key": {
                "engine_id": key.engine_id,
                "request_id": key.request_id,
                "request_epoch": key.request_epoch,
            },
            "call_id": self.call_id.to_mapping(),
            "status": self.status.value,
            "code": self.kind.value,
            "position": self.position,
            "kv_visible_len": self.kv_visible_len,
            "kv_computed_len": self.kv_computed_len,
            "num_completed_steps": self.num_completed_steps,
            "committed_tokens": list(self.committed_tokens),
            "sampled_logprob": self.sampled_logprob,
            "top_logprobs": [
                {"token_id": token, "logprob": value, "rank": rank}
                for token, value, rank in self.top_logprobs
            ],
            "prompt_logprobs": [
                [
                    {"token_id": token, "logprob": value, "rank": rank}
                    for token, value, rank in entries
                ]
                for entries in self.prompt_logprobs
            ],
            "finish_flags": {
                "eos": flags.eos,
                "length": flags.length,
                "stop": flags.stop,
            },
            "media_output": None
            if self.media_output is None
            else self.media_output.to_mapping(),
            "kv_output": None
            if self.kv_output is None
            else self.kv_output.to_mapping(),
            "product_generations": list(self.product_generations),
            "error_code": None if error_code is None else error_code.value,
            "timing_counters": {
                "queued_us": timing.queued_us,
                "device_us": timing.device_us,
                "copy_us": timing.copy_us,
                "host_us": timing.host_us,
            },
        }


@dataclass(frozen=True, slots=True)
class ForwardStats:
    """Aggregate model-forward counters reported with a batch result.

    Covers forward modes, component time, attention, CUDA graphs, decode
    relays, FlashInfer decode planning, and speculative verification. Every
    field is either an ``int`` or a ``str -> int`` mapping; `combine` and
    `to_mapping` iterate the dataclass fields generically and depend on that.
    `to_mapping` emits every field by name, and the PyO3 decoder
    (`forward_stats_from_py` in `crates/worker-ipc-py`) requires each counter
    of the Rust `ForwardStats`, so a field renamed or removed here fails the
    whole batch result.

    ``mode_counts`` and ``mode_us`` count calls and their time by forward
    mode. ``mode_tokens`` counts the query tokens those calls computed, so a
    mode without a token notion (encoder and decoder calls, standalone module
    invocations) has no entry there.
    """

    mode_counts: Mapping[str, int] = field(default_factory=dict)
    mode_tokens: Mapping[str, int] = field(default_factory=dict)
    mode_us: Mapping[str, int] = field(default_factory=dict)
    component_us: Mapping[str, int] = field(default_factory=dict)
    attention_launches: int = 0
    attention_us: int = 0
    attention_backend_counts: Mapping[str, int] = field(default_factory=dict)
    cuda_graph_captures: int = 0
    cuda_graph_replays: int = 0
    cuda_graph_misses: int = 0
    cuda_graph_fallbacks: int = 0
    cuda_graph_unpadded_tokens: int = 0
    cuda_graph_padded_tokens: int = 0
    cuda_graph_runtime_mode_counts: Mapping[str, int] = field(
        default_factory=dict
    )
    text_decode_token_relay_hits: int = 0
    text_decode_token_relay_misses: int = 0
    text_decode_position_relay_hits: int = 0
    text_decode_position_relay_misses: int = 0
    flashinfer_decode_plan_calls: int = 0
    flashinfer_decode_plan_reuses: int = 0
    flashinfer_decode_plan_rows: int = 0
    flashinfer_decode_plan_indices: int = 0
    flashinfer_decode_graph_plan_calls: int = 0
    flashinfer_decode_graph_plan_reuses: int = 0
    spec_verify_rows: int = 0
    spec_verify_draft_tokens: int = 0
    spec_verify_accepted_tokens: int = 0
    spec_verify_rejected_tokens: int = 0
    spec_verify_committed_tokens: int = 0
    spec_verify_path_counts: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def combine(cls, values: Sequence[ForwardStats]) -> ForwardStats:
        """Sum scalar counters and per-key mapping counters across values.

        An empty sequence yields zeroed stats, and a single value is returned
        as-is. The first value's field type decides how each field is merged.
        """
        if not values:
            return cls()
        if len(values) == 1:
            return values[0]

        merged: dict[str, object] = {}
        for name in cls.__dataclass_fields__:
            fields = tuple(getattr(value, name) for value in values)
            if isinstance(fields[0], Mapping):
                totals: dict[str, int] = {}
                for field_value in fields:
                    for key, count in cast(
                        Mapping[str, int], field_value
                    ).items():
                        totals[key] = totals.get(key, 0) + count
                merged[name] = totals
            else:
                merged[name] = sum(cast(tuple[int, ...], fields))

        return cls(**cast(Any, merged))

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "worker forward stats"
    ) -> ForwardStats:
        """Parse forward counters; absent fields default to zero or empty.

        Raises:
            WorkerError: `value` or a keyed field is not a mapping, a
                mapping key is not a string, or a counter is not a
                non-negative integer.
        """
        data = _map(value, where)

        def counter_map(name: str) -> dict[str, int]:
            """Parse one string-keyed map of nonnegative execution counters."""
            values = _map(data.get(name, {}), f"{where}.{name}")
            return {
                _str(key, f"{where}.{name}.key"): _uint(
                    raw, f"{where}.{name}.{key}"
                )
                for key, raw in values.items()
            }

        map_fields = {
            name: counter_map(name)
            for name in (
                "mode_counts",
                "mode_tokens",
                "mode_us",
                "component_us",
                "attention_backend_counts",
                "cuda_graph_runtime_mode_counts",
                "spec_verify_path_counts",
            )
        }
        scalar_fields = {
            name: _uint(data.get(name, 0), f"{where}.{name}")
            for name in (
                "attention_launches",
                "attention_us",
                "cuda_graph_captures",
                "cuda_graph_replays",
                "cuda_graph_misses",
                "cuda_graph_fallbacks",
                "cuda_graph_unpadded_tokens",
                "cuda_graph_padded_tokens",
                "text_decode_token_relay_hits",
                "text_decode_token_relay_misses",
                "text_decode_position_relay_hits",
                "text_decode_position_relay_misses",
                "flashinfer_decode_plan_calls",
                "flashinfer_decode_plan_reuses",
                "flashinfer_decode_plan_rows",
                "flashinfer_decode_plan_indices",
                "flashinfer_decode_graph_plan_calls",
                "flashinfer_decode_graph_plan_reuses",
                "spec_verify_rows",
                "spec_verify_draft_tokens",
                "spec_verify_accepted_tokens",
                "spec_verify_rejected_tokens",
                "spec_verify_committed_tokens",
            )
        }

        return cls(
            mode_counts=map_fields["mode_counts"],
            mode_tokens=map_fields["mode_tokens"],
            mode_us=map_fields["mode_us"],
            component_us=map_fields["component_us"],
            attention_launches=scalar_fields["attention_launches"],
            attention_us=scalar_fields["attention_us"],
            attention_backend_counts=map_fields["attention_backend_counts"],
            cuda_graph_captures=scalar_fields["cuda_graph_captures"],
            cuda_graph_replays=scalar_fields["cuda_graph_replays"],
            cuda_graph_misses=scalar_fields["cuda_graph_misses"],
            cuda_graph_fallbacks=scalar_fields["cuda_graph_fallbacks"],
            cuda_graph_unpadded_tokens=scalar_fields[
                "cuda_graph_unpadded_tokens"
            ],
            cuda_graph_padded_tokens=scalar_fields["cuda_graph_padded_tokens"],
            cuda_graph_runtime_mode_counts=map_fields[
                "cuda_graph_runtime_mode_counts"
            ],
            text_decode_token_relay_hits=scalar_fields[
                "text_decode_token_relay_hits"
            ],
            text_decode_token_relay_misses=scalar_fields[
                "text_decode_token_relay_misses"
            ],
            text_decode_position_relay_hits=scalar_fields[
                "text_decode_position_relay_hits"
            ],
            text_decode_position_relay_misses=scalar_fields[
                "text_decode_position_relay_misses"
            ],
            flashinfer_decode_plan_calls=scalar_fields[
                "flashinfer_decode_plan_calls"
            ],
            flashinfer_decode_plan_reuses=scalar_fields[
                "flashinfer_decode_plan_reuses"
            ],
            flashinfer_decode_plan_rows=scalar_fields[
                "flashinfer_decode_plan_rows"
            ],
            flashinfer_decode_plan_indices=scalar_fields[
                "flashinfer_decode_plan_indices"
            ],
            flashinfer_decode_graph_plan_calls=scalar_fields[
                "flashinfer_decode_graph_plan_calls"
            ],
            flashinfer_decode_graph_plan_reuses=scalar_fields[
                "flashinfer_decode_graph_plan_reuses"
            ],
            spec_verify_rows=scalar_fields["spec_verify_rows"],
            spec_verify_draft_tokens=scalar_fields["spec_verify_draft_tokens"],
            spec_verify_accepted_tokens=scalar_fields[
                "spec_verify_accepted_tokens"
            ],
            spec_verify_rejected_tokens=scalar_fields[
                "spec_verify_rejected_tokens"
            ],
            spec_verify_committed_tokens=scalar_fields[
                "spec_verify_committed_tokens"
            ],
            spec_verify_path_counts=map_fields["spec_verify_path_counts"],
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize every counter field, copying mappings into plain dicts."""
        return {
            name: dict(value) if isinstance(value, Mapping) else value
            for name, value in (
                (name, getattr(self, name))
                for name in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class BatchOutput:
    """One batch's complete result, carrying only materialized host values.

    `BatchState.take_output` in `uniserve_worker.execution.batch` builds it
    with only the product publications of calls that completed with status
    OK; the record itself does not check this.
    """

    batch_id: int
    completions: tuple[RequestOutput, ...] = ()
    products: tuple[TensorPublication, ...] = ()
    worker_exec_us: int | None = None
    forward_stats: ForwardStats | None = None

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "completion report"
    ) -> BatchOutput:
        """Parse a batch result mapping, validating each completion."""
        data = _map(value, where)
        return cls(
            batch_id=_uint(data.get("batch_id"), f"{where}.batch_id"),
            completions=tuple(
                RequestOutput.from_mapping(
                    item, f"{where}.completions[{index}]"
                )
                for index, item in enumerate(
                    _seq(data.get("completions", ()), f"{where}.completions")
                )
            ),
            products=tuple(
                TensorPublication.from_mapping(
                    item, f"{where}.products[{index}]"
                )
                for index, item in enumerate(
                    _seq(data.get("products", ()), f"{where}.products")
                )
            ),
            worker_exec_us=_optional_uint(
                data.get("worker_exec_us"), f"{where}.worker_exec_us"
            ),
            forward_stats=(
                None
                if data.get("forward_stats") is None
                else ForwardStats.from_mapping(
                    data["forward_stats"], f"{where}.forward_stats"
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode the result into the mapping the transport decodes."""
        return {
            "batch_id": self.batch_id,
            "completions": [value.to_mapping() for value in self.completions],
            "products": [value.to_mapping() for value in self.products],
            "worker_exec_us": self.worker_exec_us,
            "forward_stats": None
            if self.forward_stats is None
            else self.forward_stats.to_mapping(),
        }
