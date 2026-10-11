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

from uniserve_worker.errors import ErrorCode
from uniserve_worker.protocol import identity, transfer
from uniserve_worker.protocol.batch import TensorPublication
from uniserve_worker.protocol.call import CallStatus


@dataclass(frozen=True, slots=True)
class FinishFlags:
    """Device-observed finish candidates for a token call.

    `eos` means an end-of-sequence token was selected, `length` that the
    length limit was reached, and `stop` that a stop condition matched.
    """

    eos: bool = False
    length: bool = False
    stop: bool = False


@dataclass(frozen=True, slots=True)
class TimingCounters:
    """Per-call queue, device, copy, and host time in microseconds.

    Accounting only; timings are not part of any call identity.
    """

    queued_us: int = 0
    device_us: int = 0
    copy_us: int = 0
    host_us: int = 0


@dataclass(frozen=True, slots=True)
class PosixShmArtifact:
    """Identifies a completed media artifact stored in POSIX shared storage.

    The worker writes the bytes and closes its mapping before publishing the
    name (`uniserve_worker.media.storage.publish_media_bytes`); the engine
    claims the object by name, which unlinks it, when it receives the batch
    result.
    """

    name: str

    def to_mapping(self) -> dict[str, object]:
        """Serialize the shared-storage handle as a tagged transport value."""
        return {"transport": "posix_shm", "value": {"name": self.name}}


@dataclass(frozen=True, slots=True)
class MediaOutput:
    """A completed media artifact: its storage handle and length in bytes."""

    handle: PosixShmArtifact
    bytes: int

    def to_mapping(self) -> dict[str, object]:
        """Serialize a completed media artifact for IPC."""
        return {"handle": self.handle.to_mapping(), "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """One call's completion: status, accepted progress, tokens, and products.

    A successful call reports the request's coordinates after it ran; a
    predicated call did not run and reports the request's accepted
    coordinates. The record is built to satisfy the wire's completion rules,
    which the engine-side decoder checks once, with the call that answered.
    """

    request_key: identity.RequestKey
    call_id: identity.CallId
    status: CallStatus
    # Set exactly when status is ERROR.
    error_code: ErrorCode | None
    timing_counters: TimingCounters
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
    # Natural-log probability of every candidate of a token-denoising
    # readout, in the call's `Readout.candidate_ids` order.
    candidate_logprobs: tuple[float, ...] = ()
    # Why the worker refused the request; set exactly when error_code is
    # INVALID_REQUEST.
    error_message: str | None = None

    def to_mapping(self) -> dict[str, object]:
        """Encode the completion into the wire mapping the transport decodes.

        The request-key, finish-flag, and timing mappings are written
        inline.
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
            "candidate_logprobs": list(self.candidate_logprobs),
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
            "error_code": error_code,
            "error_message": self.error_message,
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
