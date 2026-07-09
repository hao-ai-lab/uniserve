"""Typed per-forward statistics accumulator.

Collects per-forward counters in semantically grouped nested accumulators
(:class:`ModeStats`, :class:`AttentionStats`, :class:`CudaGraphStats`,
:class:`TextRelayStats`, :class:`FlashinferPlanStats`, :class:`SpecVerifyStats`).
Typing makes the live counter surface visible, turns typos into attribute errors
at author time, and gives the wire projection a single owner
(:meth:`ForwardStats.to_wire`).

The flat field and method surface is preserved on :class:`ForwardStats` via
forwarding properties so existing read/write/``+=`` call sites are unaffected.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

__all__ = [
    'ForwardStats',
    'ModeStats',
    'AttentionStats',
    'CudaGraphStats',
    'TextRelayStats',
    'FlashinferPlanStats',
    'OperatorStats',
    'SpecVerifyStats',
]


@dataclass
class ModeStats:
    """Per-mode shape/throughput/wall-time and per-component wall-time."""

    mode_counts: dict[str, int] = field(default_factory=dict)
    mode_tokens: dict[str, int] = field(default_factory=dict)
    mode_ns: dict[str, int] = field(default_factory=dict)
    component_ns: dict[str, int] = field(default_factory=dict)


@dataclass
class AttentionStats:
    """Attention launches, timing, backend usage, padding and metadata reuse."""

    launches: int = 0
    ns: int = 0
    backend_counts: dict[str, int] = field(default_factory=dict)
    padded_tokens: int = 0
    padded_calls: int = 0
    metadata_hits: int = 0
    metadata_misses: int = 0


@dataclass
class CudaGraphStats:
    """CUDA-graph capture/replay accounting."""

    captures: int = 0
    replays: int = 0
    misses: int = 0
    fallbacks: int = 0
    capture_failures: int = 0
    replay_failures: int = 0
    unpadded_tokens: int = 0
    padded_tokens: int = 0
    eager_fallbacks: int = 0
    eager_tokens: int = 0
    eager_rows: int = 0
    runtime_mode_counts: dict[str, int] = field(default_factory=dict)
    shape_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class TextRelayStats:
    """Text decode relay hit/miss accounting and text padding."""

    token_relay_hits: int = 0
    token_relay_misses: int = 0
    position_relay_hits: int = 0
    position_relay_misses: int = 0
    padded_tokens: int = 0
    padded_batches: int = 0


@dataclass
class FlashinferPlanStats:
    """flashinfer plan-build accounting."""

    decode_plan_calls: int = 0
    decode_plan_reuses: int = 0
    decode_plan_rows: int = 0
    decode_plan_indices: int = 0
    decode_graph_plan_calls: int = 0
    decode_graph_plan_reuses: int = 0
    prefill_plan_calls: int = 0
    prefill_plan_reuses: int = 0
    prefill_plan_rows: int = 0
    prefill_plan_indices: int = 0


@dataclass
class OperatorStats:
    """Generic op-dispatch launch accounting for non-attention providers."""

    launches: int = 0
    ns: int = 0
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class SpecVerifyStats:
    """Speculative-verify accounting."""

    rows: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    committed_tokens: int = 0
    path_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ForwardStats:
    """Root accumulator for one forward pass; groups counters and projects to wire."""

    mode: ModeStats = field(default_factory=ModeStats)
    attention: AttentionStats = field(default_factory=AttentionStats)
    cuda_graph: CudaGraphStats = field(default_factory=CudaGraphStats)
    text_relay: TextRelayStats = field(default_factory=TextRelayStats)
    flashinfer: FlashinferPlanStats = field(default_factory=FlashinferPlanStats)
    operators: OperatorStats = field(default_factory=OperatorStats)
    spec_verify: SpecVerifyStats = field(default_factory=SpecVerifyStats)

    # --- flat-field forwarders (preserved external surface) ---

    @property
    def mode_counts(self) -> dict[str, int]:
        return self.mode.mode_counts

    @mode_counts.setter
    def mode_counts(self, value: dict[str, int]) -> None:
        self.mode.mode_counts = value

    @property
    def mode_tokens(self) -> dict[str, int]:
        return self.mode.mode_tokens

    @mode_tokens.setter
    def mode_tokens(self, value: dict[str, int]) -> None:
        self.mode.mode_tokens = value

    @property
    def mode_ns(self) -> dict[str, int]:
        return self.mode.mode_ns

    @mode_ns.setter
    def mode_ns(self, value: dict[str, int]) -> None:
        self.mode.mode_ns = value

    @property
    def component_ns(self) -> dict[str, int]:
        return self.mode.component_ns

    @component_ns.setter
    def component_ns(self, value: dict[str, int]) -> None:
        self.mode.component_ns = value

    @property
    def attention_launches(self) -> int:
        return self.attention.launches

    @attention_launches.setter
    def attention_launches(self, value: int) -> None:
        self.attention.launches = value

    @property
    def attention_ns(self) -> int:
        return self.attention.ns

    @attention_ns.setter
    def attention_ns(self, value: int) -> None:
        self.attention.ns = value

    @property
    def attention_backend_counts(self) -> dict[str, int]:
        return self.attention.backend_counts

    @attention_backend_counts.setter
    def attention_backend_counts(self, value: dict[str, int]) -> None:
        self.attention.backend_counts = value

    @property
    def attention_padded_tokens(self) -> int:
        return self.attention.padded_tokens

    @attention_padded_tokens.setter
    def attention_padded_tokens(self, value: int) -> None:
        self.attention.padded_tokens = value

    @property
    def attention_padded_calls(self) -> int:
        return self.attention.padded_calls

    @attention_padded_calls.setter
    def attention_padded_calls(self, value: int) -> None:
        self.attention.padded_calls = value

    @property
    def attention_metadata_hits(self) -> int:
        return self.attention.metadata_hits

    @attention_metadata_hits.setter
    def attention_metadata_hits(self, value: int) -> None:
        self.attention.metadata_hits = value

    @property
    def attention_metadata_misses(self) -> int:
        return self.attention.metadata_misses

    @attention_metadata_misses.setter
    def attention_metadata_misses(self, value: int) -> None:
        self.attention.metadata_misses = value

    @property
    def cuda_graph_captures(self) -> int:
        return self.cuda_graph.captures

    @cuda_graph_captures.setter
    def cuda_graph_captures(self, value: int) -> None:
        self.cuda_graph.captures = value

    @property
    def cuda_graph_replays(self) -> int:
        return self.cuda_graph.replays

    @cuda_graph_replays.setter
    def cuda_graph_replays(self, value: int) -> None:
        self.cuda_graph.replays = value

    @property
    def cuda_graph_misses(self) -> int:
        return self.cuda_graph.misses

    @cuda_graph_misses.setter
    def cuda_graph_misses(self, value: int) -> None:
        self.cuda_graph.misses = value

    @property
    def cuda_graph_fallbacks(self) -> int:
        return self.cuda_graph.fallbacks

    @cuda_graph_fallbacks.setter
    def cuda_graph_fallbacks(self, value: int) -> None:
        self.cuda_graph.fallbacks = value

    @property
    def cuda_graph_unpadded_tokens(self) -> int:
        return self.cuda_graph.unpadded_tokens

    @cuda_graph_unpadded_tokens.setter
    def cuda_graph_unpadded_tokens(self, value: int) -> None:
        self.cuda_graph.unpadded_tokens = value

    @property
    def cuda_graph_padded_tokens(self) -> int:
        return self.cuda_graph.padded_tokens

    @cuda_graph_padded_tokens.setter
    def cuda_graph_padded_tokens(self, value: int) -> None:
        self.cuda_graph.padded_tokens = value

    @property
    def cuda_graph_runtime_mode_counts(self) -> dict[str, int]:
        return self.cuda_graph.runtime_mode_counts

    @cuda_graph_runtime_mode_counts.setter
    def cuda_graph_runtime_mode_counts(self, value: dict[str, int]) -> None:
        self.cuda_graph.runtime_mode_counts = value

    @property
    def forward_graph_capture_failures(self) -> int:
        return self.cuda_graph.capture_failures

    @forward_graph_capture_failures.setter
    def forward_graph_capture_failures(self, value: int) -> None:
        self.cuda_graph.capture_failures = value

    @property
    def forward_graph_replay_failures(self) -> int:
        return self.cuda_graph.replay_failures

    @forward_graph_replay_failures.setter
    def forward_graph_replay_failures(self, value: int) -> None:
        self.cuda_graph.replay_failures = value

    @property
    def forward_eager_fallbacks(self) -> int:
        return self.cuda_graph.eager_fallbacks

    @forward_eager_fallbacks.setter
    def forward_eager_fallbacks(self, value: int) -> None:
        self.cuda_graph.eager_fallbacks = value

    @property
    def forward_eager_tokens(self) -> int:
        return self.cuda_graph.eager_tokens

    @forward_eager_tokens.setter
    def forward_eager_tokens(self, value: int) -> None:
        self.cuda_graph.eager_tokens = value

    @property
    def forward_eager_rows(self) -> int:
        return self.cuda_graph.eager_rows

    @forward_eager_rows.setter
    def forward_eager_rows(self, value: int) -> None:
        self.cuda_graph.eager_rows = value

    @property
    def forward_graph_shape_counts(self) -> dict[str, int]:
        return self.cuda_graph.shape_counts

    @forward_graph_shape_counts.setter
    def forward_graph_shape_counts(self, value: dict[str, int]) -> None:
        self.cuda_graph.shape_counts = value

    @property
    def text_decode_token_relay_hits(self) -> int:
        return self.text_relay.token_relay_hits

    @text_decode_token_relay_hits.setter
    def text_decode_token_relay_hits(self, value: int) -> None:
        self.text_relay.token_relay_hits = value

    @property
    def text_decode_token_relay_misses(self) -> int:
        return self.text_relay.token_relay_misses

    @text_decode_token_relay_misses.setter
    def text_decode_token_relay_misses(self, value: int) -> None:
        self.text_relay.token_relay_misses = value

    @property
    def text_decode_position_relay_hits(self) -> int:
        return self.text_relay.position_relay_hits

    @text_decode_position_relay_hits.setter
    def text_decode_position_relay_hits(self, value: int) -> None:
        self.text_relay.position_relay_hits = value

    @property
    def text_decode_position_relay_misses(self) -> int:
        return self.text_relay.position_relay_misses

    @text_decode_position_relay_misses.setter
    def text_decode_position_relay_misses(self, value: int) -> None:
        self.text_relay.position_relay_misses = value

    @property
    def text_padded_tokens(self) -> int:
        return self.text_relay.padded_tokens

    @text_padded_tokens.setter
    def text_padded_tokens(self, value: int) -> None:
        self.text_relay.padded_tokens = value

    @property
    def text_padded_batches(self) -> int:
        return self.text_relay.padded_batches

    @text_padded_batches.setter
    def text_padded_batches(self, value: int) -> None:
        self.text_relay.padded_batches = value

    @property
    def flashinfer_decode_plan_calls(self) -> int:
        return self.flashinfer.decode_plan_calls

    @flashinfer_decode_plan_calls.setter
    def flashinfer_decode_plan_calls(self, value: int) -> None:
        self.flashinfer.decode_plan_calls = value

    @property
    def flashinfer_decode_plan_reuses(self) -> int:
        return self.flashinfer.decode_plan_reuses

    @flashinfer_decode_plan_reuses.setter
    def flashinfer_decode_plan_reuses(self, value: int) -> None:
        self.flashinfer.decode_plan_reuses = value

    @property
    def flashinfer_decode_plan_rows(self) -> int:
        return self.flashinfer.decode_plan_rows

    @flashinfer_decode_plan_rows.setter
    def flashinfer_decode_plan_rows(self, value: int) -> None:
        self.flashinfer.decode_plan_rows = value

    @property
    def flashinfer_decode_plan_indices(self) -> int:
        return self.flashinfer.decode_plan_indices

    @flashinfer_decode_plan_indices.setter
    def flashinfer_decode_plan_indices(self, value: int) -> None:
        self.flashinfer.decode_plan_indices = value

    @property
    def flashinfer_decode_graph_plan_calls(self) -> int:
        return self.flashinfer.decode_graph_plan_calls

    @flashinfer_decode_graph_plan_calls.setter
    def flashinfer_decode_graph_plan_calls(self, value: int) -> None:
        self.flashinfer.decode_graph_plan_calls = value

    @property
    def flashinfer_decode_graph_plan_reuses(self) -> int:
        return self.flashinfer.decode_graph_plan_reuses

    @flashinfer_decode_graph_plan_reuses.setter
    def flashinfer_decode_graph_plan_reuses(self, value: int) -> None:
        self.flashinfer.decode_graph_plan_reuses = value

    @property
    def flashinfer_prefill_plan_calls(self) -> int:
        return self.flashinfer.prefill_plan_calls

    @flashinfer_prefill_plan_calls.setter
    def flashinfer_prefill_plan_calls(self, value: int) -> None:
        self.flashinfer.prefill_plan_calls = value

    @property
    def flashinfer_prefill_plan_reuses(self) -> int:
        return self.flashinfer.prefill_plan_reuses

    @flashinfer_prefill_plan_reuses.setter
    def flashinfer_prefill_plan_reuses(self, value: int) -> None:
        self.flashinfer.prefill_plan_reuses = value

    @property
    def flashinfer_prefill_plan_rows(self) -> int:
        return self.flashinfer.prefill_plan_rows

    @flashinfer_prefill_plan_rows.setter
    def flashinfer_prefill_plan_rows(self, value: int) -> None:
        self.flashinfer.prefill_plan_rows = value

    @property
    def flashinfer_prefill_plan_indices(self) -> int:
        return self.flashinfer.prefill_plan_indices

    @flashinfer_prefill_plan_indices.setter
    def flashinfer_prefill_plan_indices(self, value: int) -> None:
        self.flashinfer.prefill_plan_indices = value

    @property
    def spec_verify_rows(self) -> int:
        return self.spec_verify.rows

    @spec_verify_rows.setter
    def spec_verify_rows(self, value: int) -> None:
        self.spec_verify.rows = value

    @property
    def spec_verify_draft_tokens(self) -> int:
        return self.spec_verify.draft_tokens

    @spec_verify_draft_tokens.setter
    def spec_verify_draft_tokens(self, value: int) -> None:
        self.spec_verify.draft_tokens = value

    @property
    def spec_verify_accepted_tokens(self) -> int:
        return self.spec_verify.accepted_tokens

    @spec_verify_accepted_tokens.setter
    def spec_verify_accepted_tokens(self, value: int) -> None:
        self.spec_verify.accepted_tokens = value

    @property
    def spec_verify_rejected_tokens(self) -> int:
        return self.spec_verify.rejected_tokens

    @spec_verify_rejected_tokens.setter
    def spec_verify_rejected_tokens(self, value: int) -> None:
        self.spec_verify.rejected_tokens = value

    @property
    def spec_verify_committed_tokens(self) -> int:
        return self.spec_verify.committed_tokens

    @spec_verify_committed_tokens.setter
    def spec_verify_committed_tokens(self, value: int) -> None:
        self.spec_verify.committed_tokens = value

    @property
    def spec_verify_path_counts(self) -> dict[str, int]:
        return self.spec_verify.path_counts

    @spec_verify_path_counts.setter
    def spec_verify_path_counts(self, value: dict[str, int]) -> None:
        self.spec_verify.path_counts = value

    # --- record helpers (the small set of non-trivial accumulations) ---

    def add_component_elapsed(self, component: str, start_ns: int) -> None:
        key = str(component)
        self.mode.component_ns[key] = int(self.mode.component_ns.get(key, 0)) + (
            time.perf_counter_ns() - int(start_ns)
        )

    def record_mode_shape(self, mode: str, *, ops: int, tokens: int) -> None:
        self.mode.mode_counts[mode] = int(self.mode.mode_counts.get(mode, 0)) + int(ops)
        self.mode.mode_tokens[mode] = int(self.mode.mode_tokens.get(mode, 0)) + int(tokens)

    def record_mode_wall_time(self, mode: str, elapsed_ns: int) -> None:
        self.mode.mode_ns[mode] = int(self.mode.mode_ns.get(mode, 0)) + int(elapsed_ns)

    def record_attention_launch(self, backend: str, elapsed_ns: int) -> None:
        self.attention.launches += 1
        self.attention.ns += int(elapsed_ns)
        name = str(backend)
        self.attention.backend_counts[name] = int(self.attention.backend_counts.get(name, 0)) + 1

    def record_operator_launch(self, operator: str, provider: str, elapsed_ns: int) -> None:
        self.operators.launches += 1
        self.operators.ns += int(elapsed_ns)
        key = f"{operator}:{provider}"
        self.operators.counts[key] = int(self.operators.counts.get(key, 0)) + 1

    def record_runtime_graph_mode(self, mode: str) -> None:
        name = str(mode)
        self.cuda_graph.runtime_mode_counts[name] = int(
            self.cuda_graph.runtime_mode_counts.get(name, 0)
        ) + 1

    def record_spec_path(self, path: str) -> None:
        name = str(path)
        self.spec_verify.path_counts[name] = int(self.spec_verify.path_counts.get(name, 0)) + 1

    def to_wire(self) -> dict[str, object]:
        """Project the accumulated counters to the host wire schema.

        Nanosecond timers are emitted as microseconds; everything else is copied
        out as plain ints / dicts. This is the single source of truth for the
        forward-stats wire shape.
        """
        return {
            "mode_counts": dict(self.mode.mode_counts),
            "mode_tokens": dict(self.mode.mode_tokens),
            "mode_us": {str(k): int(v) // 1000 for k, v in self.mode.mode_ns.items()},
            "component_us": {str(k): int(v) // 1000 for k, v in self.mode.component_ns.items()},
            "attention_launches": self.attention.launches,
            "attention_us": self.attention.ns // 1000,
            "attention_backend_counts": dict(self.attention.backend_counts),
            "operator_launches": self.operators.launches,
            "operator_us": self.operators.ns // 1000,
            "operator_counts": dict(self.operators.counts),
            "attention_padded_tokens": self.attention.padded_tokens,
            "attention_padded_calls": self.attention.padded_calls,
            "attention_metadata_hits": self.attention.metadata_hits,
            "attention_metadata_misses": self.attention.metadata_misses,
            "cuda_graph_captures": self.cuda_graph.captures,
            "cuda_graph_replays": self.cuda_graph.replays,
            "cuda_graph_misses": self.cuda_graph.misses,
            "cuda_graph_fallbacks": self.cuda_graph.fallbacks,
            "cuda_graph_unpadded_tokens": self.cuda_graph.unpadded_tokens,
            "cuda_graph_padded_tokens": self.cuda_graph.padded_tokens,
            "cuda_graph_runtime_mode_counts": dict(self.cuda_graph.runtime_mode_counts),
            "forward_graph_captures": self.cuda_graph.captures,
            "forward_graph_replays": self.cuda_graph.replays,
            "forward_graph_misses": self.cuda_graph.misses,
            "forward_graph_fallbacks": self.cuda_graph.fallbacks,
            "forward_graph_capture_failures": self.cuda_graph.capture_failures,
            "forward_graph_replay_failures": self.cuda_graph.replay_failures,
            "forward_eager_fallbacks": self.cuda_graph.eager_fallbacks,
            "forward_eager_tokens": self.cuda_graph.eager_tokens,
            "forward_eager_rows": self.cuda_graph.eager_rows,
            "forward_graph_runtime_mode_counts": dict(self.cuda_graph.runtime_mode_counts),
            "forward_graph_shape_counts": dict(self.cuda_graph.shape_counts),
            "forward_graph_unpadded_tokens": self.cuda_graph.unpadded_tokens,
            "forward_graph_padded_tokens": self.cuda_graph.padded_tokens,
            "text_decode_token_relay_hits": self.text_relay.token_relay_hits,
            "text_decode_token_relay_misses": self.text_relay.token_relay_misses,
            "text_decode_position_relay_hits": self.text_relay.position_relay_hits,
            "text_decode_position_relay_misses": self.text_relay.position_relay_misses,
            "text_padded_tokens": self.text_relay.padded_tokens,
            "text_padded_batches": self.text_relay.padded_batches,
            "flashinfer_decode_plan_calls": self.flashinfer.decode_plan_calls,
            "flashinfer_decode_plan_reuses": self.flashinfer.decode_plan_reuses,
            "flashinfer_decode_plan_rows": self.flashinfer.decode_plan_rows,
            "flashinfer_decode_plan_indices": self.flashinfer.decode_plan_indices,
            "flashinfer_decode_graph_plan_calls": self.flashinfer.decode_graph_plan_calls,
            "flashinfer_decode_graph_plan_reuses": self.flashinfer.decode_graph_plan_reuses,
            "flashinfer_prefill_plan_calls": self.flashinfer.prefill_plan_calls,
            "flashinfer_prefill_plan_reuses": self.flashinfer.prefill_plan_reuses,
            "flashinfer_prefill_plan_rows": self.flashinfer.prefill_plan_rows,
            "flashinfer_prefill_plan_indices": self.flashinfer.prefill_plan_indices,
            "spec_verify_rows": self.spec_verify.rows,
            "spec_verify_draft_tokens": self.spec_verify.draft_tokens,
            "spec_verify_accepted_tokens": self.spec_verify.accepted_tokens,
            "spec_verify_rejected_tokens": self.spec_verify.rejected_tokens,
            "spec_verify_committed_tokens": self.spec_verify.committed_tokens,
            "spec_verify_path_counts": dict(self.spec_verify.path_counts),
        }
