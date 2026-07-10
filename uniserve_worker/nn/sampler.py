"""Model-independent token sampling from logits.

Processing order: allowed mask, suppress, bias, penalties, temperature, min-p,
top-k, top-p, sample, logprobs. Seeded requests use a per-request
``torch.Generator`` for reproducible worker-side draws.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypeGuard, cast, overload

import torch

from ..foundation.env import env_flag
from ..foundation.errors import capability_mismatch, invalid_descriptor
from .mesh import get_current_mesh

__all__ = [
    'NEG_INF',
    'Truncation',
    'Penalties',
    'Strategy',
    'resolve_sampling_strategy',
    'TokenSample',
    'BatchedSamplingResult',
    'DeferredBatchedSamplingResult',
    'is_deferred_sampling_result',
    'finalize_sampling_result',
    'sample_one_from_logits',
    'apply_sampling_batched',
    'apply_sampling_batched_with_device_tokens',
    'apply_allowed_mask_',
    'apply_suppress_',
    'apply_logit_bias_',
    'shape_logits_for_sampling',
    'sync_tp_sampled_tokens',
    'Sampler',
]

NEG_INF = float("-inf")


@dataclass(frozen=True)
class Truncation:
    """Active top-k / top-p truncation knobs for one request.

    The four top-k/top-p combinations are factory variants; inactive knobs use
    identity values (``top_k == 0``, ``top_p == 1.0``).
    """

    top_k: int
    top_p: float

    @classmethod
    def none(cls) -> "Truncation":
        return cls(top_k=0, top_p=1.0)

    @classmethod
    def top_k_only(cls, k: int) -> "Truncation":
        return cls(top_k=int(k), top_p=1.0)

    @classmethod
    def top_p_only(cls, p: float) -> "Truncation":
        return cls(top_k=0, top_p=float(p))

    @classmethod
    def top_k_top_p(cls, k: int, p: float) -> "Truncation":
        return cls(top_k=int(k), top_p=float(p))


@dataclass(frozen=True)
class Penalties:
    """Repetition / frequency / presence penalty knobs for one request.

    ``active`` is true when any penalty differs from its neutral value.
    """

    repetition: float
    frequency: float
    presence: float

    @property
    def active(self) -> bool:
        return self.repetition != 1.0 or self.frequency != 0.0 or self.presence != 0.0


@dataclass(frozen=True)
class Strategy:
    """Resolved sampling plan for one request, parsed from the params dict once.

    Fields are parsed from the params dict with ``float(... or default) > 0`` coercion
    for temperature and min_p.
    """

    temperature: float | None
    min_p: float
    truncation: Truncation
    penalties: Penalties
    return_logprobs: bool
    n_logprobs: int
    logprob_token_ids: tuple[int, ...]

    @property
    def is_greedy(self) -> bool:
        return self.temperature is None

    @property
    def has_min_p(self) -> bool:
        return self.min_p > 0.0

    @property
    def logprobs_requested(self) -> bool:
        return self.return_logprobs or self.n_logprobs > 0 or bool(self.logprob_token_ids)


def resolve_sampling_strategy(sp: dict[str, Any]) -> Strategy:
    """Parse sampling params into a :class:`Strategy` for one request."""
    temperature = float(sp.get("temperature", 0.0) or 0.0)
    min_p = float(sp.get("min_p", 0.0) or 0.0)
    top_k = int(sp.get("top_k", 0) or 0)
    top_p = float(sp.get("top_p", 1.0) or 1.0)
    if top_k and (top_p != 1.0):
        truncation = Truncation.top_k_top_p(top_k, top_p)
    elif top_k:
        truncation = Truncation.top_k_only(top_k)
    elif top_p != 1.0:
        truncation = Truncation.top_p_only(top_p)
    else:
        truncation = Truncation.none()
    penalties = Penalties(
        repetition=float(sp.get("repetition_penalty", 1.0) or 1.0),
        frequency=float(sp.get("frequency_penalty", 0.0) or 0.0),
        presence=float(sp.get("presence_penalty", 0.0) or 0.0),
    )
    return Strategy(
        temperature=temperature if temperature > 0.0 else None,
        min_p=min_p,
        truncation=truncation,
        penalties=penalties,
        return_logprobs=bool(sp.get("return_logprobs", False)),
        n_logprobs=max(0, int(sp.get("n_logprobs", 0) or 0)),
        logprob_token_ids=tuple(
            dict.fromkeys(int(token_id) for token_id in (sp.get("logprob_token_ids") or ()))
        ),
    )


class TokenSample(NamedTuple):
    """One sampled token plus optional logprob detail.

    ``top_logprobs`` is the ``[token_id, logprob, rank]`` list-of-lists carried in the
    output protocol; positional unpacking ``token, logprob, top = sample`` and
    index access stay valid because this is a tuple subtype.
    """

    token_id: int
    logprob: float | None
    top_logprobs: list[list[float | int]] | None
_COPY_STREAMS: dict[int, torch.cuda.Stream] = {}
_COPY_STREAMS_LOCK = threading.Lock()
_ENABLE_ASYNC_ASSERT = env_flag("UNISERVE_ENABLE_ASYNC_ASSERT")


@dataclass(frozen=True)
class BatchedSamplingResult:
    """GPU/CPU token samples plus the device-resident token tensor."""

    samples: list[TokenSample]
    device_tokens: torch.Tensor


def _competition_rank_from_top_values(values: Sequence[float], value: float) -> int:
    """Rank by one plus the number of candidates with a strictly greater score."""

    return sum(candidate > value for candidate in values) + 1


def score_prompt_token_logprobs(
    logits: torch.Tensor,
    target_token_ids: Sequence[int],
    *,
    n_logprobs: int,
    logprob_token_ids: Sequence[int] = (),
) -> list[list[tuple[int, float, int]]]:
    """Score prompt targets against their left-context logits.

    Each position starts with its actual target token, then adds the highest-ranked
    requested candidates and explicit token IDs without duplication.
    """

    if logits.ndim != 2:
        raise invalid_descriptor("prompt-scoring logits must be shaped [positions, vocab]")
    position_count, vocab = int(logits.shape[0]), int(logits.shape[1])
    targets = [int(token_id) for token_id in target_token_ids]
    if len(targets) != position_count:
        raise invalid_descriptor("prompt-scoring target count must match logits positions")
    if any(token_id < 0 or token_id >= vocab for token_id in targets):
        raise invalid_descriptor("prompt-scoring target token is outside the model vocabulary")
    if position_count == 0:
        return []

    requested_ids = list(
        dict.fromkeys(
            int(token_id)
            for token_id in logprob_token_ids
            if 0 <= int(token_id) < vocab
        )
    )
    top_count = min(max(0, int(n_logprobs)), vocab)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    target_indices = torch.tensor(targets, dtype=torch.long, device=logits.device)
    selected = logprobs.gather(1, target_indices[:, None]).squeeze(1)
    selected_ranks = (logprobs > selected[:, None]).sum(dim=-1) + 1
    top_values = top_indices = None
    if top_count > 0:
        top_values, top_indices = torch.topk(logprobs, top_count, dim=-1)
    requested_values = requested_ranks = None
    if requested_ids:
        requested_indices = torch.tensor(
            requested_ids, dtype=torch.long, device=logits.device
        ).expand(position_count, -1)
        requested_values = logprobs.gather(1, requested_indices)
        requested_ranks = torch.empty_like(requested_indices)
        for offset in range(len(requested_ids)):
            requested_ranks[:, offset] = (
                logprobs > requested_values[:, offset, None]
            ).sum(dim=-1) + 1

    selected_cpu = selected.detach().to("cpu")
    selected_ranks_cpu = selected_ranks.detach().to("cpu")
    top_values_cpu = top_values.detach().to("cpu") if top_values is not None else None
    top_indices_cpu = top_indices.detach().to("cpu") if top_indices is not None else None
    requested_values_cpu = (
        requested_values.detach().to("cpu") if requested_values is not None else None
    )
    requested_ranks_cpu = (
        requested_ranks.detach().to("cpu") if requested_ranks is not None else None
    )

    positions: list[list[tuple[int, float, int]]] = []
    for row, target in enumerate(targets):
        entries = [
            (
                target,
                float(selected_cpu[row].item()),
                int(selected_ranks_cpu[row].item()),
            )
        ]
        seen = {target}
        if top_values_cpu is not None and top_indices_cpu is not None:
            row_top_values = [float(value) for value in top_values_cpu[row].tolist()]
            for token_id, logprob in zip(
                top_indices_cpu[row].tolist(), row_top_values
            ):
                token_id = int(token_id)
                if token_id not in seen:
                    entries.append(
                        (
                            token_id,
                            logprob,
                            _competition_rank_from_top_values(
                                row_top_values, logprob
                            ),
                        )
                    )
                    seen.add(token_id)
        if requested_values_cpu is not None and requested_ranks_cpu is not None:
            for offset, token_id in enumerate(requested_ids):
                if token_id not in seen:
                    entries.append(
                        (
                            token_id,
                            float(requested_values_cpu[row, offset].item()),
                            int(requested_ranks_cpu[row, offset].item()),
                        )
                    )
                    seen.add(token_id)
        positions.append(entries)
    return positions


@dataclass(frozen=True)
class _SamplingBatch:
    """Validated, immutable inputs for one batched-sampling call.

    Carries the original ``[batch, vocab]`` logits plus the per-row descriptor
    lists and the explicit ``batch``/``vocab`` extents. The pipeline stages read
    this and mutate a separate float ``work`` tensor; the batch object itself
    stays constant so each stage is a small, typed, CPU-testable function over
    explicit inputs.
    """

    logits: torch.Tensor
    sampling_params: list[dict[str, Any]]
    recent: list[list[int] | tuple[int, ...]]
    allowed: list[list[int] | tuple[int, ...] | None]
    suppress: list[list[int] | tuple[int, ...] | None]
    batch: int
    vocab: int


class DeferredBatchedSamplingResult:
    """Batched samples whose CPU token/logprob copies finalize later.

    Generalizes the greedy fast path to the full sampler: the chosen tokens and
    (optionally) the selected/top logprob tensors are copied to pinned CPU on the
    copy stream behind a single CUDA event, and ``finalize()`` synchronizes that
    event once before reading them. The synchronize/``.item()``/``.tolist()``
    cost is deferred out of the forward and hidden behind the next batch's GPU
    launch. ``device_tokens`` stays available for the decode relay regardless of
    when the CPU copy is read.
    """

    def __init__(
        self,
        *,
        tokens_cpu: torch.Tensor,
        device_tokens: torch.Tensor,
        copy_event: torch.cuda.Event | None,
        ready_start_event: torch.cuda.Event | None = None,
        return_logprobs: list[bool] | None = None,
        n_logprobs: list[int] | None = None,
        logprob_token_ids: list[list[int]] | None = None,
        selected_cpu: torch.Tensor | None = None,
        top_values_cpu: torch.Tensor | None = None,
        top_indices_cpu: torch.Tensor | None = None,
        selected_ranks_cpu: torch.Tensor | None = None,
        requested_values_cpu: torch.Tensor | None = None,
        requested_ranks_cpu: torch.Tensor | None = None,
    ) -> None:
        self._tokens_cpu = tokens_cpu
        self.device_tokens = device_tokens
        self._copy_event = copy_event
        self._ready_start_event = ready_start_event
        self._return_logprobs = return_logprobs
        self._n_logprobs = n_logprobs
        self._logprob_token_ids = logprob_token_ids
        self._selected_cpu = selected_cpu
        self._top_values_cpu = top_values_cpu
        self._top_indices_cpu = top_indices_cpu
        self._selected_ranks_cpu = selected_ranks_cpu
        self._requested_values_cpu = requested_values_cpu
        self._requested_ranks_cpu = requested_ranks_cpu
        self._finalized: BatchedSamplingResult | None = None
        self._token_ids: list[int] | None = None

    def finalize(self) -> BatchedSamplingResult:
        if self._finalized is None:
            if self._copy_event is not None:
                self._copy_event.synchronize()
            self._finalized = BatchedSamplingResult(
                samples=[
                    self._sample_for_row(row)
                    for row in range(int(self._tokens_cpu.numel()))
                ],
                device_tokens=self.device_tokens,
            )
        return self._finalized

    def _ensure_token_ids(self) -> list[int]:
        if self._token_ids is None:
            if self._copy_event is not None:
                self._copy_event.synchronize()
            tokens = self._tokens_cpu.reshape(-1)
            if tokens.device.type != "cpu":
                tokens = tokens.detach().to("cpu")
            self._token_ids = [int(token) for token in tokens.tolist()]
        return self._token_ids

    def token_ids(self) -> list[int]:
        """Return sampled token ids without materializing logprob payloads."""

        return list(self._ensure_token_ids())

    def ready(self) -> bool:
        if self._finalized is not None or self._copy_event is None:
            return True
        return bool(self._copy_event.query())

    def set_ready_start_event(self, event: torch.cuda.Event | None) -> None:
        self._ready_start_event = event

    def cuda_ready_elapsed_us(self) -> int | None:
        if self._ready_start_event is None or self._copy_event is None:
            return None
        if not self.ready():
            return None
        try:
            return int(round(float(self._ready_start_event.elapsed_time(self._copy_event)) * 1000.0))
        except (RuntimeError, ValueError):
            return None

    def _sample_for_row(self, row: int) -> TokenSample:
        token_id = self._ensure_token_ids()[int(row)]
        n = int(self._n_logprobs[row]) if self._n_logprobs is not None else 0
        requested_ids = (
            self._logprob_token_ids[row]
            if self._logprob_token_ids is not None
            else []
        )
        return_logprobs = (
            bool(self._return_logprobs[row])
            if self._return_logprobs is not None
            else False
        )
        if not return_logprobs and n <= 0 and not requested_ids:
            return TokenSample(token_id, None, None)
        logprob = (
            float(self._selected_cpu[row].item())
            if self._selected_cpu is not None
            else None
        )
        top = []
        existing: set[int] = set()
        if logprob is not None and self._selected_ranks_cpu is not None:
            top.append(
                [
                    token_id,
                    logprob,
                    int(self._selected_ranks_cpu[row].item()),
                ]
            )
            existing.add(token_id)
        if self._top_values_cpu is not None and self._top_indices_cpu is not None:
            row_top_values = [
                float(value) for value in self._top_values_cpu[row, :n].tolist()
            ]
            for token, value in zip(
                self._top_indices_cpu[row, :n].tolist(), row_top_values
            ):
                token = int(token)
                if token not in existing:
                    top.append(
                        [
                            token,
                            value,
                            _competition_rank_from_top_values(
                                row_top_values, value
                            ),
                        ]
                    )
                    existing.add(token)
        if (
            requested_ids
            and self._requested_values_cpu is not None
            and self._requested_ranks_cpu is not None
        ):
            for offset, requested_id in enumerate(requested_ids):
                if requested_id not in existing:
                    top.append(
                        [
                            int(requested_id),
                            float(self._requested_values_cpu[row, offset].item()),
                            int(self._requested_ranks_cpu[row, offset].item()),
                        ]
                    )
                    existing.add(requested_id)
        return TokenSample(token_id, logprob, top or None)


def is_deferred_sampling_result(result: object) -> TypeGuard[DeferredBatchedSamplingResult]:
    """Identify deferred results without coupling consumers to class identity."""
    return callable(getattr(result, "finalize", None)) and hasattr(result, "device_tokens")


def finalize_sampling_result(result: object) -> BatchedSamplingResult:
    """Return an immediate sampling result across module reload boundaries."""
    finalize = getattr(result, "finalize", None)
    resolved = finalize() if is_deferred_sampling_result(result) and callable(finalize) else result
    if not hasattr(resolved, "samples") or not hasattr(resolved, "device_tokens"):
        raise invalid_descriptor("sampler returned an invalid batched result")
    return cast(BatchedSamplingResult, resolved)


def _seed_generator(
    sp: dict[str, Any],
    device: torch.device,
) -> Any | None:
    """Build a per-request ``torch.Generator`` from ``sp['seed']``.

    Returns ``None`` when the request carries no seed (uses the global default RNG).
    """
    seed = sp.get("seed")
    if seed is None:
        return None
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except (TypeError, ValueError, RuntimeError):
        return None


def sample_one_from_logits(
    logits: torch.Tensor,
    sp: dict[str, Any],
    *,
    recent: list[int] | tuple[int, ...],
    allowed: list[int] | tuple[int, ...] | None,
    suppress: list[int] | tuple[int, ...] | None,
    n_logprobs: int,
) -> TokenSample:
    """Sample one token from 1-D vocabulary logits via the batched ``[1, V]`` path.

    The ``n_logprobs`` argument overrides ``sp['n_logprobs']`` so callers can
    request the logprob detail without mutating their params dict.
    """
    params = dict(sp)
    params["n_logprobs"] = int(n_logprobs or 0)
    result = apply_sampling_batched_with_device_tokens(
        logits.reshape(1, -1),
        [params],
        [recent],
        [allowed],
        [suppress],
    )
    return result.samples[0]


def apply_sampling_batched(
    logits: torch.Tensor,
    sampling_params: list[dict[str, Any]],
    recent: list[list[int] | tuple[int, ...]],
    allowed: list[list[int] | tuple[int, ...] | None],
    suppress: list[list[int] | tuple[int, ...] | None],
) -> list[TokenSample]:
    return apply_sampling_batched_with_device_tokens(
        logits,
        sampling_params,
        recent,
        allowed,
        suppress,
    ).samples


@overload
def apply_sampling_batched_with_device_tokens(
    logits: torch.Tensor,
    sampling_params: list[dict[str, Any]],
    recent: list[list[int] | tuple[int, ...]],
    allowed: list[list[int] | tuple[int, ...] | None],
    suppress: list[list[int] | tuple[int, ...] | None],
    *,
    defer_cpu: Literal[False] = False,
    enable_cuda_timing: bool = False,
) -> BatchedSamplingResult: ...


@overload
def apply_sampling_batched_with_device_tokens(
    logits: torch.Tensor,
    sampling_params: list[dict[str, Any]],
    recent: list[list[int] | tuple[int, ...]],
    allowed: list[list[int] | tuple[int, ...] | None],
    suppress: list[list[int] | tuple[int, ...] | None],
    *,
    defer_cpu: Literal[True],
    enable_cuda_timing: bool = False,
) -> BatchedSamplingResult | DeferredBatchedSamplingResult: ...


@overload
def apply_sampling_batched_with_device_tokens(
    logits: torch.Tensor,
    sampling_params: list[dict[str, Any]],
    recent: list[list[int] | tuple[int, ...]],
    allowed: list[list[int] | tuple[int, ...] | None],
    suppress: list[list[int] | tuple[int, ...] | None],
    *,
    defer_cpu: bool,
    enable_cuda_timing: bool = False,
) -> BatchedSamplingResult | DeferredBatchedSamplingResult: ...


def apply_sampling_batched_with_device_tokens(
    logits: torch.Tensor,
    sampling_params: list[dict[str, Any]],
    recent: list[list[int] | tuple[int, ...]],
    allowed: list[list[int] | tuple[int, ...] | None],
    suppress: list[list[int] | tuple[int, ...] | None],
    *,
    defer_cpu: bool = False,
    enable_cuda_timing: bool = False,
) -> BatchedSamplingResult | DeferredBatchedSamplingResult:
    """Sample one token per row from a ``[B, V]`` logits tensor.

    This preserves the scalar sampler's operation order while batching the
    heavy tensor operations and collapsing token/logprob D2H reads into one
    transfer per output field.  The returned ``device_tokens`` tensor is the
    same chosen-token tensor before the protocol CPU copy, so decode can feed
    the next step directly from device when the scheduler sends that token back.
    """

    if logits.ndim != 2:
        raise invalid_descriptor("batched sampler expects logits shaped [batch, vocab]")
    _maybe_async_assert_valid_logits(logits, "batched sampler input")
    batch, vocab = int(logits.shape[0]), int(logits.shape[1])
    if not (
        len(sampling_params) == len(recent) == len(allowed) == len(suppress) == batch
    ):
        raise invalid_descriptor("batched sampler metadata length mismatch")
    state = _SamplingBatch(
        logits=logits,
        sampling_params=sampling_params,
        recent=recent,
        allowed=allowed,
        suppress=suppress,
        batch=batch,
        vocab=vocab,
    )

    greedy_logits = _greedy_device_fast_path_logits(
        logits,
        sampling_params,
        recent,
        allowed,
        suppress,
        vocab,
    )
    if greedy_logits is not None:
        return _draw_greedy_fast_path(
            state,
            greedy_logits,
            defer_cpu=defer_cpu,
            enable_cuda_timing=enable_cuda_timing,
        )

    work = _writable_float_work(logits)
    _apply_mask_bias_penalty_stage(state, work)
    sampled_rows = _apply_temperature_stage(state, work)
    _apply_truncation_stage(state, work)
    tokens = _draw_stage(state, work, sampled_rows)
    logprobs = _logprobs_stage(state, work, tokens)
    return _device_to_host_stage(
        state,
        work,
        tokens,
        logprobs,
        defer_cpu=defer_cpu,
        enable_cuda_timing=enable_cuda_timing,
    )


@dataclass(frozen=True)
class _BatchLogprobs:
    """Optional per-row logprob detail produced by the logprobs stage.

    All three tensors are ``None`` when no row requested logprobs. ``selected``
    is ``[batch]`` log-probabilities of the drawn tokens; ``top_values`` /
    ``top_indices`` are the ``[batch, k]`` top-``max_top`` log-probabilities and
    their token ids.
    """

    selected: torch.Tensor | None
    top_values: torch.Tensor | None
    top_indices: torch.Tensor | None
    selected_ranks: torch.Tensor | None
    requested_values: torch.Tensor | None
    requested_ranks: torch.Tensor | None
    requested_token_ids: list[list[int]]


def _draw_greedy_fast_path(
    state: _SamplingBatch,
    greedy_logits: torch.Tensor,
    *,
    defer_cpu: bool,
    enable_cuda_timing: bool = False,
) -> BatchedSamplingResult | DeferredBatchedSamplingResult:
    """Greedy-only draw + device-to-host copy (the fast-path stage subset).

    Argmax over the already-shaped ``greedy_logits``, TP-sync the tokens, then
    copy them to CPU via the shared async pinned path. Returns a deferred result
    when ``defer_cpu`` is set and the copy is event-backed.
    """
    tokens = torch.argmax(greedy_logits, dim=-1)
    tokens = sync_tp_sampled_tokens(tokens)
    tokens_cpu, copy_event = _copy_tensor_to_cpu_async(
        tokens,
        state.logits.device,
        enable_cuda_timing=enable_cuda_timing,
    )
    if defer_cpu and copy_event is not None:
        return DeferredBatchedSamplingResult(
            tokens_cpu=tokens_cpu,
            device_tokens=tokens.detach(),
            copy_event=copy_event,
        )
    if copy_event is not None:
        copy_event.synchronize()
    return BatchedSamplingResult(
        samples=[
            TokenSample(int(tokens_cpu[row].item()), None, None)
            for row in range(state.batch)
        ],
        device_tokens=tokens.detach(),
    )


def _apply_mask_bias_penalty_stage(state: _SamplingBatch, work: torch.Tensor) -> None:
    """Per-row allowed mask, suppress, logit bias and penalties, in place.

    Runs the canonical pre-temperature processors on each ``work`` row through
    the shared :func:`apply_allowed_mask_` / :func:`apply_suppress_` /
    :func:`apply_logit_bias_` helpers and :func:`_apply_penalties_in_place`.
    """
    for row in range(state.batch):
        row_logits = work[row]
        sp = state.sampling_params[row]
        apply_allowed_mask_(row_logits, state.allowed[row], state.vocab)
        apply_suppress_(row_logits, state.suppress[row], state.vocab)
        apply_logit_bias_(row_logits, sp.get("logit_bias"), state.vocab)
        _apply_penalties_in_place(row_logits, sp, state.recent[row], state.vocab)


def _apply_temperature_stage(state: _SamplingBatch, work: torch.Tensor) -> list[int]:
    """Scale sampled rows by their temperature in place; return the sampled rows.

    Greedy rows (``temperature is None``) are left untouched and excluded from
    the returned list, which the draw stage uses to partition argmax vs.
    multinomial rows.
    """
    sampled_rows: list[int] = []
    for row, sp in enumerate(state.sampling_params):
        strategy = resolve_sampling_strategy(sp)
        if strategy.temperature is not None:
            work[row] /= strategy.temperature
            sampled_rows.append(row)
    return sampled_rows


def _apply_truncation_stage(state: _SamplingBatch, work: torch.Tensor) -> None:
    """Min-p / top-k / top-p truncation in place over the batch.

    Uses the batched common-params fast path when every row shares one
    truncation config, else falls back to the per-row 1-D kernel.
    """
    if not _apply_common_min_p_top_k_top_p_in_place(work, state.sampling_params, state.vocab):
        for row, sp in enumerate(state.sampling_params):
            _apply_min_p_top_k_top_p_in_place(work[row], sp, state.vocab)


def _draw_stage(
    state: _SamplingBatch,
    work: torch.Tensor,
    sampled_rows: list[int],
) -> torch.Tensor:
    """Draw one token per row, then TP-sync the result.

    Greedy rows take argmax; sampled rows draw via ``torch.multinomial`` with
    per-request seeded generators (one shared generator when all seeded rows
    carry the same seed, else one generator per row).
    """
    sampling_params = state.sampling_params
    sampled_row_set = set(sampled_rows)
    greedy_rows = [row for row in range(state.batch) if row not in sampled_row_set]
    tokens = torch.empty(state.batch, dtype=torch.long, device=work.device)
    if greedy_rows:
        idx = torch.tensor(greedy_rows, dtype=torch.long, device=work.device)
        tokens[idx] = torch.argmax(work[idx], dim=-1)
    if sampled_rows:
        idx = torch.tensor(sampled_rows, dtype=torch.long, device=work.device)
        probs = torch.softmax(work[idx], dim=-1)
        seeds = [sampling_params[row].get("seed") for row in sampled_rows]
        if all(s is None for s in seeds):
            tokens[idx] = torch.multinomial(probs, 1).squeeze(-1)
        elif all(s == seeds[0] for s in seeds):
            # All seeded rows share one seed: a single generator is reproducible.
            generator = _seed_generator(sampling_params[sampled_rows[0]], probs.device)
            tokens[idx] = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        else:
            # Per-row seeds differ: draw each row with its own generator so the
            # per-request seed is honoured independently.
            for offset, row in enumerate(sampled_rows):
                generator = _seed_generator(sampling_params[row], probs.device)
                tokens[row] = torch.multinomial(
                    probs[offset], 1, generator=generator
                ).squeeze(-1)
    return sync_tp_sampled_tokens(tokens)


def _logprobs_stage(
    state: _SamplingBatch,
    work: torch.Tensor,
    tokens: torch.Tensor,
) -> _BatchLogprobs:
    """Compute selected and top-``max_top`` logprobs over the shaped logits.

    Returns all-``None`` detail when no row requested logprobs.
    """
    strategies = [resolve_sampling_strategy(sp) for sp in state.sampling_params]
    requested_token_ids = [
        [token_id for token_id in strategy.logprob_token_ids if 0 <= token_id < state.vocab]
        for strategy in strategies
    ]
    max_top = min(
        max((strategy.n_logprobs for strategy in strategies), default=0),
        state.vocab,
    )
    if not any(strategy.logprobs_requested for strategy in strategies):
        return _BatchLogprobs(None, None, None, None, None, None, requested_token_ids)
    logprobs = torch.log_softmax(work, dim=-1)
    selected = logprobs.gather(1, tokens[:, None]).squeeze(1)
    selected_ranks = (logprobs > selected[:, None]).sum(dim=-1) + 1
    top_values = top_indices = None
    if max_top > 0:
        top_values, top_indices = torch.topk(logprobs, max_top, dim=-1)
    max_requested = max((len(token_ids) for token_ids in requested_token_ids), default=0)
    requested_values = requested_ranks = None
    if max_requested > 0:
        requested_indices = torch.zeros(
            (state.batch, max_requested), dtype=torch.long, device=work.device
        )
        for row, token_ids in enumerate(requested_token_ids):
            if token_ids:
                requested_indices[row, : len(token_ids)] = torch.tensor(
                    token_ids, dtype=torch.long, device=work.device
                )
        requested_values = logprobs.gather(1, requested_indices)
        requested_ranks = torch.empty_like(requested_indices)
        for offset in range(max_requested):
            requested_ranks[:, offset] = (
                logprobs > requested_values[:, offset, None]
            ).sum(dim=-1) + 1
    return _BatchLogprobs(
        selected,
        top_values,
        top_indices,
        selected_ranks,
        requested_values,
        requested_ranks,
        requested_token_ids,
    )


def _device_to_host_stage(
    state: _SamplingBatch,
    work: torch.Tensor,
    tokens: torch.Tensor,
    logprobs: _BatchLogprobs,
    *,
    defer_cpu: bool = False,
    enable_cuda_timing: bool = False,
) -> BatchedSamplingResult | DeferredBatchedSamplingResult:
    """Copy tokens/logprobs to CPU on the copy stream and assemble samples.

    The tokens and (when requested) the selected/top logprob tensors are staged
    to pinned CPU on the dedicated copy stream behind a single CUDA event. When
    ``defer_cpu`` is set and the copy is event-backed, the synchronize and the
    ``.item()``/``.tolist()`` reads are deferred to ``finalize()`` so they hide
    behind the next batch's GPU launch; otherwise the event is synchronized inline
    and the samples are assembled here.
    """
    strategies = [resolve_sampling_strategy(sp) for sp in state.sampling_params]
    return_logprobs = [strategy.return_logprobs for strategy in strategies]
    n_logprobs = [min(strategy.n_logprobs, state.vocab) for strategy in strategies]
    copy_stream = _cuda_copy_stream(work.device)
    selected_cpu = top_values_cpu = top_indices_cpu = None
    selected_ranks_cpu = requested_values_cpu = requested_ranks_cpu = None
    copy_event: torch.cuda.Event | None = None
    if copy_stream is not None:
        copy_stream.wait_stream(torch.cuda.current_stream(work.device))
        with torch.cuda.stream(copy_stream):
            tokens_cpu = _copy_to_pinned_cpu(tokens)
            if logprobs.selected is not None:
                selected_cpu = _copy_to_pinned_cpu(logprobs.selected)
            if logprobs.top_values is not None and logprobs.top_indices is not None:
                top_values_cpu = _copy_to_pinned_cpu(logprobs.top_values)
                top_indices_cpu = _copy_to_pinned_cpu(logprobs.top_indices)
            if logprobs.selected_ranks is not None:
                selected_ranks_cpu = _copy_to_pinned_cpu(logprobs.selected_ranks)
            if logprobs.requested_values is not None:
                requested_values_cpu = _copy_to_pinned_cpu(logprobs.requested_values)
            if logprobs.requested_ranks is not None:
                requested_ranks_cpu = _copy_to_pinned_cpu(logprobs.requested_ranks)
            copy_event = torch.cuda.Event(enable_timing=enable_cuda_timing)
            copy_event.record(copy_stream)
    else:
        tokens_cpu = tokens.detach().to("cpu")
        if logprobs.selected is not None:
            selected_cpu = logprobs.selected.detach().to("cpu")
        if logprobs.top_values is not None and logprobs.top_indices is not None:
            top_values_cpu = logprobs.top_values.detach().to("cpu")
            top_indices_cpu = logprobs.top_indices.detach().to("cpu")
        if logprobs.selected_ranks is not None:
            selected_ranks_cpu = logprobs.selected_ranks.detach().to("cpu")
        if logprobs.requested_values is not None:
            requested_values_cpu = logprobs.requested_values.detach().to("cpu")
        if logprobs.requested_ranks is not None:
            requested_ranks_cpu = logprobs.requested_ranks.detach().to("cpu")

    deferred = DeferredBatchedSamplingResult(
        tokens_cpu=tokens_cpu,
        device_tokens=tokens.detach(),
        copy_event=copy_event,
        return_logprobs=return_logprobs,
        n_logprobs=n_logprobs,
        logprob_token_ids=logprobs.requested_token_ids,
        selected_cpu=selected_cpu,
        top_values_cpu=top_values_cpu,
        top_indices_cpu=top_indices_cpu,
        selected_ranks_cpu=selected_ranks_cpu,
        requested_values_cpu=requested_values_cpu,
        requested_ranks_cpu=requested_ranks_cpu,
    )
    if defer_cpu and copy_event is not None:
        return deferred
    # Inline materialization (event synchronize happens inside finalize()).
    return deferred.finalize()


# Device index tensors for suppress/allowed lists, keyed by content. These
# lists are per-request constants consumed every decode step; rebuilding them
# per step issues a pageable host->device copy whose implicit
# cudaStreamSynchronize blocks the CPU behind the in-flight decode graph
# replay and serializes the whole decode pipeline. The cache bounds itself by
# clearing at capacity (lists are tiny and few per serving session).
_INDEX_TENSOR_CACHE: dict[tuple[str, tuple[int, ...], int], torch.Tensor] = {}
_INDEX_TENSOR_CACHE_MAX = 512


def _valid_index_tensor(
    values: Sequence[int],
    vocab: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    key = (str(device), tuple(int(token_id) for token_id in values), int(vocab))
    cached = _INDEX_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    tensor = torch.tensor(
        [token_id for token_id in key[1] if 0 <= token_id < vocab],
        dtype=torch.long,
        device=device,
    )
    if len(_INDEX_TENSOR_CACHE) >= _INDEX_TENSOR_CACHE_MAX:
        _INDEX_TENSOR_CACHE.clear()
    _INDEX_TENSOR_CACHE[key] = tensor
    return tensor


def apply_allowed_mask_(
    logits: torch.Tensor,
    allowed: Sequence[int] | None,
    vocab: int,
) -> None:
    """Restrict a 1-D logits row to ``allowed`` token ids in place.

    Every position not in ``allowed`` is set to ``-inf``; an empty/``None``
    ``allowed`` is a no-op. Out-of-range ids are dropped. Shared by the batched
    sampler, the greedy device fast path and the single-row shaping pipeline.
    """
    if not allowed:
        return
    mask = torch.full_like(logits, NEG_INF)
    idx = _valid_index_tensor(allowed, vocab, device=logits.device)
    if idx.numel() > 0:
        mask[idx] = logits[idx]
    logits.copy_(mask)


def apply_suppress_(
    logits: torch.Tensor,
    suppress: Sequence[int] | None,
    vocab: int,
) -> None:
    """Force ``suppress`` token ids in a 1-D logits row to ``-inf`` in place.

    An empty/``None`` ``suppress`` is a no-op; out-of-range ids are dropped.
    """
    if not suppress:
        return
    idx = _valid_index_tensor(suppress, vocab, device=logits.device)
    if idx.numel() > 0:
        # ``logits[idx] = NEG_INF`` wraps the Python float into a CPU scalar
        # tensor and copies it host->device with an implicit stream synchronize
        # every call; ``index_fill_`` takes the scalar by value with no copy.
        logits.index_fill_(0, idx, NEG_INF)


def apply_logit_bias_(
    logits: torch.Tensor,
    logit_bias: list[Any] | tuple[Any, ...] | None,
    vocab: int,
) -> None:
    """Add per-token bias to a 1-D logits row in place.

    ``logit_bias`` is the ``[token_id, bias]`` pair list from the params dict.
    Out-of-range ids are skipped, and positions already masked to ``-inf`` stay
    masked (bias is not allowed to revive a suppressed token).
    """
    for pair in logit_bias or []:
        token_id, bias = int(pair[0]), float(pair[1])
        if 0 <= token_id < vocab:
            logits[token_id] = torch.where(
                torch.isneginf(logits[token_id]),
                logits[token_id],
                logits[token_id] + bias,
            )


def _writable_float_work(logits: torch.Tensor) -> torch.Tensor:
    """Fresh fp32 working copy of ``logits`` with exactly one copy.

    ``.float()`` on a non-fp32 tensor already materializes a new fp32 tensor;
    cloning again would copy the full vocab row a second time per step. Only
    an already-fp32 input needs the explicit clone to stay writable.
    """
    work = logits.float()
    return logits.clone() if work is logits else work


def _greedy_device_fast_path_logits(
    logits: torch.Tensor,
    sampling_params: list[dict[str, Any]],
    recent: list[list[int] | tuple[int, ...]],
    allowed: list[list[int] | tuple[int, ...] | None],
    suppress: list[list[int] | tuple[int, ...] | None],
    vocab: int,
) -> torch.Tensor | None:
    if not _can_use_greedy_device_fast_path(sampling_params):
        return None
    if not _greedy_path_needs_argmax_processors(sampling_params, recent, allowed, suppress):
        return logits

    work = _writable_float_work(logits)
    for row in range(int(work.shape[0])):
        row_logits = work[row]
        sp = sampling_params[row]
        apply_allowed_mask_(row_logits, allowed[row], vocab)
        apply_suppress_(row_logits, suppress[row], vocab)
        apply_logit_bias_(row_logits, sp.get("logit_bias"), vocab)
        _apply_penalties_in_place(row_logits, sp, recent[row], vocab)
    return work


def _can_use_greedy_device_fast_path(sampling_params: list[dict[str, Any]]) -> bool:
    for sp in sampling_params:
        strategy = resolve_sampling_strategy(sp)
        if strategy.logprobs_requested:
            return False
        if not strategy.is_greedy:
            return False
    return True


def _greedy_path_needs_argmax_processors(
    sampling_params: list[dict[str, Any]],
    recent: list[list[int] | tuple[int, ...]],
    allowed: list[list[int] | tuple[int, ...] | None],
    suppress: list[list[int] | tuple[int, ...] | None],
) -> bool:
    for sp, recent_row, allowed_row, suppress_row in zip(
        sampling_params,
        recent,
        allowed,
        suppress,
    ):
        if allowed_row or suppress_row or sp.get("logit_bias"):
            return True
        if recent_row and resolve_sampling_strategy(sp).penalties.active:
            return True
    return False


def _maybe_async_assert_valid_logits(logits: torch.Tensor, context: str) -> None:
    if not _ENABLE_ASYNC_ASSERT or not logits.is_floating_point():
        return
    assert_async = getattr(torch, "_assert_async", None)
    if not callable(assert_async):
        return
    assert_async(~torch.any(torch.isnan(logits)), f"NaN detected in logits: {context}")
    assert_async(
        ~torch.any(torch.isinf(logits) & (logits > 0)),
        f"+Inf detected in logits: {context}",
    )


def _apply_penalties_in_place(
    logits: torch.Tensor,
    sp: dict[str, Any],
    recent: list[int] | tuple[int, ...],
    vocab: int,
) -> None:
    penalties = resolve_sampling_strategy(sp).penalties
    repetition, frequency, presence = (
        penalties.repetition,
        penalties.frequency,
        penalties.presence,
    )
    if not penalties.active or not recent:
        return
    counts: dict[int, int] = {}
    for token_id in recent:
        tid = int(token_id)
        if 0 <= tid < vocab:
            counts[tid] = counts.get(tid, 0) + 1
    if not counts:
        return
    idx = torch.tensor(list(counts), dtype=torch.long, device=logits.device)
    vals = logits[idx]
    finite = ~torch.isneginf(vals)
    if repetition != 1.0:
        vals = torch.where(vals > 0, vals / repetition, vals * repetition)
    counts_t = torch.tensor([counts[int(t)] for t in counts], dtype=logits.dtype, device=logits.device)
    vals = vals - frequency * counts_t - presence
    logits[idx] = torch.where(finite, vals, logits[idx])


def _apply_min_p_top_k_top_p_in_place(logits: torch.Tensor, sp: dict[str, Any], vocab: int) -> None:
    strategy = resolve_sampling_strategy(sp)
    if strategy.has_min_p:
        probs = torch.softmax(logits, dim=-1)
        threshold = strategy.min_p * probs.max()
        logits.masked_fill_(probs < threshold, NEG_INF)

    _apply_top_k_top_p_in_place(
        logits, strategy.truncation.top_k, strategy.truncation.top_p, vocab
    )


def shape_logits_for_sampling(
    logits: torch.Tensor,
    sp: dict[str, Any],
    *,
    recent: Sequence[int],
    allowed: Sequence[int] | None,
    suppress: Sequence[int] | None,
    vocab: int,
) -> None:
    """Apply the full single-row logits-shaping pipeline in place.

    Runs allowed mask, suppress, logit bias, penalties, temperature scaling and
    min-p/top-k/top-p truncation on a 1-D ``logits`` tensor, matching the scalar
    sampler order. Shared by the batched sampler's fallback row path and the
    speculative verifier so both go through one pipeline.
    """
    apply_allowed_mask_(logits, allowed, vocab)
    apply_suppress_(logits, suppress, vocab)
    apply_logit_bias_(logits, sp.get("logit_bias"), vocab)
    _apply_penalties_in_place(logits, sp, list(recent), vocab)
    strategy = resolve_sampling_strategy(sp)
    if strategy.temperature is not None:
        logits.div_(strategy.temperature)
    _apply_min_p_top_k_top_p_in_place(logits, sp, vocab)


def _apply_common_min_p_top_k_top_p_in_place(
    logits: torch.Tensor,
    sampling_params: list[dict[str, Any]],
    vocab: int,
) -> bool:
    if logits.ndim != 2 or not sampling_params:
        return False
    first = _truncation_params(sampling_params[0])
    if any(_truncation_params(sp) != first for sp in sampling_params[1:]):
        return False
    min_p, top_k, top_p = first
    if min_p <= 0.0 and not (0 < top_k < vocab) and not (0.0 < top_p < 1.0):
        return True

    if min_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        threshold = min_p * probs.max(dim=-1, keepdim=True).values
        logits.masked_fill_(probs < threshold, NEG_INF)

    _apply_top_k_top_p_2d(logits, top_k, top_p, vocab)
    return True


def _truncation_params(sp: dict[str, Any]) -> tuple[float, int, float]:
    strategy = resolve_sampling_strategy(sp)
    return (strategy.min_p, strategy.truncation.top_k, strategy.truncation.top_p)


def _apply_top_k_top_p_2d(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    vocab: int,
) -> None:
    """Top-k then nucleus (top-p) truncation, in place, on ``[rows, vocab]``."""
    top_values = top_indices = None
    if 0 < top_k < vocab:
        top_values, top_indices = torch.topk(logits, top_k, dim=-1, sorted=True)
        masked = torch.full_like(logits, NEG_INF)
        masked.scatter_(1, top_indices, top_values)
        logits.copy_(masked)

    if not (0.0 < top_p < 1.0):
        return

    if top_values is not None and top_indices is not None:
        sorted_logits = top_values
        sorted_idx = top_indices
    else:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff[:, 1:] = cutoff[:, :-1].clone()
    cutoff[:, 0] = False
    drop = torch.zeros_like(logits, dtype=torch.bool)
    drop.scatter_(1, sorted_idx, cutoff)
    logits.masked_fill_(drop, NEG_INF)


def _apply_top_k_top_p_in_place(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    vocab: int,
) -> None:
    """Scalar (1-D) top-k/top-p truncation routed through the 2-D core."""
    _apply_top_k_top_p_2d(logits.unsqueeze(0), top_k, top_p, vocab)


def _cuda_copy_stream(device: torch.device | str) -> torch.cuda.Stream | None:
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    index = dev.index
    if index is None:
        index = torch.cuda.current_device()
    with _COPY_STREAMS_LOCK:
        stream = _COPY_STREAMS.get(index)
        if stream is None:
            with torch.cuda.device(index):
                stream = torch.cuda.Stream()
            _COPY_STREAMS[index] = stream
    return stream


def _copy_tensor_to_cpu(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    out, event = _copy_tensor_to_cpu_async(tensor, device)
    if event is not None:
        event.synchronize()
    return out


def _copy_tensor_to_cpu_async(
    tensor: torch.Tensor,
    device: torch.device | str,
    *,
    enable_cuda_timing: bool = False,
) -> tuple[torch.Tensor, torch.cuda.Event | None]:
    copy_stream = _cuda_copy_stream(device)
    if copy_stream is None:
        return tensor.detach().to("cpu"), None
    copy_stream.wait_stream(torch.cuda.current_stream(torch.device(device)))
    with torch.cuda.stream(copy_stream):
        out = _copy_to_pinned_cpu(tensor)
        event = torch.cuda.Event(enable_timing=enable_cuda_timing)
        event.record(copy_stream)
    return out, event


def _copy_to_pinned_cpu(tensor: torch.Tensor) -> torch.Tensor:
    try:
        out = torch.empty(
            tuple(tensor.shape),
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )
        out.copy_(tensor.detach(), non_blocking=True)
        return out
    except RuntimeError:
        return tensor.detach().to("cpu")


def sync_tp_sampled_tokens(tokens: torch.Tensor) -> torch.Tensor:
    mesh = get_current_mesh()
    if mesh.tp_size <= 1:
        return tokens
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise capability_mismatch("sampling with tp_size > 1 requires an initialized torch.distributed collective")
    group = getattr(mesh.transport("tp"), "group", None)
    out = tokens.contiguous()
    torch.distributed.broadcast(out, src=_tp_group_zero_global_rank(group), group=group)
    return out


def _tp_group_zero_global_rank(group) -> int:
    if group is None:
        return 0
    get_global_rank = getattr(torch.distributed, "get_global_rank", None)
    if callable(get_global_rank):
        try:
            return int(get_global_rank(group, 0))
        except (RuntimeError, TypeError, ValueError):
            pass
    return 0


class Sampler:
    """Thin wrapper around :func:`sample_one_from_logits` for single-row sampling."""

    def sample(
        self,
        logits: torch.Tensor,
        sampling_params: dict[str, Any],
        *,
        recent: list[int] | None = None,
        allowed: list[int] | None = None,
        suppress: list[int] | None = None,
        n_logprobs: int = 0,
    ) -> TokenSample:
        return sample_one_from_logits(
            logits,
            sampling_params,
            recent=recent or [],
            allowed=allowed,
            suppress=suppress,
            n_logprobs=n_logprobs,
        )
