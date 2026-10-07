"""Block-diffusion denoising steps over resident token canvases.

A token denoiser refines canvases of ``C`` token ids; between its passes a
sampler scores the pass's logits, accepts the lowest-entropy candidates
within an entropy budget, re-noises the rest, decides whether the canvas
has converged and prepares the next pass's self-conditioning embedding.
This module defines one such step over ``R`` canvas rows as numerical
functions of borrowed device tensors. The caller keeps the per-row state
resident (:class:`CanvasState`) and supplies each row's Philox coordinates
(seed, block, step) as device columns; nothing here reads a device value on
the host, allocates, or knows request identities, so a CUDA graph can
capture ``start_canvas`` and ``denoise_canvas`` for a fixed row count. On
CUDA, run one eager step of each row count before capturing it: that step
configures the kernels and chooses the self-conditioning product's
algorithm, from the kernels' shipped table or by timing the candidates,
which synchronizes the device (see
``uniserve_kernels.diffusion.canvas.product``).

A block runs ``start_canvas`` before its first pass (rows at step 0 draw
their initial canvas and clear history and self-conditioning), then per
step: denoiser pass -> logits ``[R, C, V]`` -> ``denoise_canvas``. After a
step, ``CanvasDecision.finished[r, 0]`` says whether row ``r``'s block is
done (converged or at its last step); ``CanvasDecision.tokens[r]`` is then
the committed canvas, padded after its first end-of-sequence token, and
``finished[r, 1]`` says whether it holds one. Choosing which rows step,
advancing their step and block counters and committing canvases belong to
execution.

Semantics (per row, at step ``k`` of a block, ``remaining = steps - k``)
follow the pinned Transformers ``DiffusionGemmaGenerationMixin``:

- processed logits ``q = x / t`` with the FP32 linear temperature
  ``t = t_min + (t_max - t_min) * (remaining / steps)``;
- ``entropy``: entropy of ``softmax(q)`` per position; ``argmax``: first
  maximum of ``q``; ``sample``: Gumbel-max draw from ``softmax(q)`` with the
  SAMPLE stream's per-token uniforms (``uniserve.diffusion.tokens``
  randomness contract, token ``v`` using lane ``v % 4`` of word ``v // 4``);
- the new canvas takes ``sample`` at the positions ``accept_by_entropy``
  accepts and a RENOISE-stream token elsewhere (lane 0 of word 0);
- the row stops when its argmax canvas equals the previous ``stability``
  ones (always true for ``stability = 0``) and its mean entropy is below
  ``confidence`` (never true for ``confidence = 0``: entropies are
  non-negative); its block is done when it stops or ``remaining == 1``;
  history shifts in the new argmax canvas;
- ``tokens`` is the argmax canvas truncated after the first end-of-sequence
  id; the self-conditioning embedding for the next pass is
  ``softmax(bf16(q)) @ E * bf16(scale)``;
- a block's initial canvas uses the INITIAL stream at step 0 (lane 0 of
  word 0).

The CPU path composes the portable primitives of
``uniserve.diffusion.tokens`` and matches the reference's arithmetic. CUDA
calls run the fused kernels of ``uniserve_kernels.diffusion.canvas`` and
raise when no kernel accepts the operands. The kernels reproduce the
Philox draws, the renoise and initial tokens, the argmax and the Gumbel
comparison exactly (up to near-ties of perturbed scores within FP32
rounding of the logarithms); entropy, the self-conditioning weights and the
embedding carry FP32 rounding differences bounded by the registered
tolerances (sums reassociated, ``exp2`` approximated to 2 ulp, the scaled
logits ``x * RN(1 / t)`` within 1.5 ulp of ``x / t`` outside the exact
candidate and argmax evaluations, and the embedding normalized after an
FP32 product rather than before a BF16 one).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from uniserve_kernels.triton import require_kernel

from . import tokens

__all__ = [
    "CanvasDecision",
    "CanvasSampling",
    "CanvasScores",
    "CanvasState",
    "CanvasWorkspace",
    "advance_canvas",
    "condition_canvas",
    "denoise_canvas",
    "score_canvas",
    "start_canvas",
]


def _fp32(value: float) -> float:
    """The FP32 value nearest ``value``, as the reference's tensors hold it."""
    return float(torch.tensor(value, dtype=torch.float32))


@dataclass(frozen=True, slots=True)
class CanvasSampling:
    """Service-level constants of block-diffusion canvas sampling.

    ``steps`` bounds the denoising steps of a block, ``entropy_bound`` the
    accepted entropy budget, ``t_min``/``t_max`` the temperature schedule,
    ``confidence`` the mean-entropy stop threshold (zero never stops a row
    before its last step) and ``stability`` the number of previous argmax
    canvases a stop must match (zero: none). ``eos_ids`` end a sequence and
    ``pad_id`` replaces tokens after the first of them.
    """

    steps: int
    entropy_bound: float
    t_min: float
    t_max: float
    confidence: float
    stability: int
    eos_ids: tuple[int, ...]
    pad_id: int

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("canvas sampling needs a positive step count")
        if not 0 <= self.t_min < self.t_max:
            raise ValueError(
                "the temperature schedule needs 0 <= t_min < t_max"
            )
        if not self.entropy_bound > 0:
            raise ValueError("the entropy bound must be positive")
        if not self.confidence >= 0:
            raise ValueError("the confidence threshold must be non-negative")
        if type(self.stability) is not int or self.stability < 0:
            raise ValueError("stability counts previous canvases (>= 0)")
        if not self.eos_ids or any(
            type(token) is not int or token < 0
            for token in (*self.eos_ids, self.pad_id)
        ):
            raise ValueError("end-of-sequence and padding ids are token ids")

    @property
    def t_delta(self) -> float:
        """``t_max - t_min`` rounded to FP32, as the reference evaluates it."""
        return _fp32(self.t_max - self.t_min)


@dataclass(frozen=True, slots=True)
class CanvasState:
    """Per-row sampler state of ``R`` canvas rows on one device.

    The caller allocates and keeps these tensors resident; the step
    functions read and update them in place.

    - ``seed``, ``block``, ``step``: int64 ``[R]`` Philox coordinates of
      each row: the request seed, the block (committed canvases so far) and
      the denoising step within the block, from 0. The caller sets them
      before each call; the functions never advance them.
    - ``canvas``: int64 ``[R, C]``, the canvas the next denoiser pass reads.
    - ``history``: int64 ``[R, stability, C]``, the previous argmax
      canvases, oldest first; -1 at the start of a block.
    - ``self_conditioning``: ``[R * C, H]`` in the embedding dtype, the soft
      embedding the next pass reads (``CanvasInput.self_conditioning``); zero
      on a block's first pass.
    """

    seed: torch.Tensor
    block: torch.Tensor
    step: torch.Tensor
    canvas: torch.Tensor
    history: torch.Tensor
    self_conditioning: torch.Tensor

    def __post_init__(self) -> None:
        if self.canvas.ndim != 2:
            raise ValueError("the canvas state holds [rows, canvas] token ids")
        rows, length = self.canvas.shape
        for column in (self.seed, self.block, self.step):
            if column.shape != (rows,) or column.dtype != torch.int64:
                raise ValueError(
                    "seed, block and step are int64 [rows] columns"
                )
        if (
            self.canvas.dtype != torch.int64
            or self.history.dtype != torch.int64
        ):
            raise ValueError("canvas and history hold int64 token ids")
        if self.history.ndim != 3 or (
            self.history.shape[0],
            self.history.shape[2],
        ) != (rows, length):
            raise ValueError("history is [rows, stability, canvas]")
        if (
            self.self_conditioning.ndim != 2
            or self.self_conditioning.shape[0] != rows * length
        ):
            raise ValueError("self-conditioning is [rows * canvas, hidden]")

    @classmethod
    def empty(
        cls,
        rows: int,
        canvas_length: int,
        hidden_size: int,
        *,
        stability: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> CanvasState:
        """Allocate uninitialized state for ``rows`` canvas rows."""

        def long(*shape: int) -> torch.Tensor:
            return torch.empty(*shape, dtype=torch.int64, device=device)

        return cls(
            seed=long(rows),
            block=long(rows),
            step=long(rows),
            canvas=long(rows, canvas_length),
            history=long(rows, stability, canvas_length),
            self_conditioning=torch.empty(
                rows * canvas_length, hidden_size, dtype=dtype, device=device
            ),
        )


@dataclass(frozen=True, slots=True)
class CanvasScores:
    """Per-position results of scoring one step's logits.

    ``entropy`` FP32, ``argmax`` and ``sample`` int64, each ``[R, C]``.
    """

    entropy: torch.Tensor
    argmax: torch.Tensor
    sample: torch.Tensor

    @classmethod
    def empty(
        cls,
        rows: int,
        canvas_length: int,
        *,
        device: torch.device | str | None = None,
    ) -> CanvasScores:
        shape = (rows, canvas_length)
        return cls(
            entropy=torch.empty(shape, dtype=torch.float32, device=device),
            argmax=torch.empty(shape, dtype=torch.int64, device=device),
            sample=torch.empty(shape, dtype=torch.int64, device=device),
        )


@dataclass(frozen=True, slots=True)
class CanvasDecision:
    """Per-row outcome of one step.

    ``tokens``: int64 ``[R, C]``, the argmax canvas padded after its first
    end-of-sequence token (the committed canvas once the block is done).
    ``finished``: bool ``[R, 2]``, (block done, end of sequence seen): the
    one small tensor a caller copies to the host after each step.
    """

    tokens: torch.Tensor
    finished: torch.Tensor

    @classmethod
    def empty(
        cls,
        rows: int,
        canvas_length: int,
        *,
        device: torch.device | str | None = None,
    ) -> CanvasDecision:
        return cls(
            tokens=torch.empty(
                rows, canvas_length, dtype=torch.int64, device=device
            ),
            finished=torch.empty(rows, 2, dtype=torch.bool, device=device),
        )


def _scratch_bytes(positions: int, hidden_size: int, device_type: str) -> int:
    # The CUDA self-conditioning product's cuBLASLt workspace; the kernels
    # size it and key their shipped algorithm table by that size.
    if device_type != "cuda":
        return 0
    return _kernels().product_scratch_bytes(positions, hidden_size)


@dataclass(frozen=True, slots=True)
class CanvasWorkspace:
    """Scratch of one step: the self-conditioning distribution in transit.

    ``weights`` ``[R * C, V]`` in the embedding dtype, ``normalizer`` FP32
    ``[R * C]``, ``product`` FP32 ``[R * C, H]`` and ``scratch`` uint8 bytes
    the CUDA product uses as its cuBLASLt workspace (empty on the CPU; see
    ``uniserve_kernels.diffusion.canvas.product_scratch_bytes``).
    ``score_canvas`` fills the first two in an implementation-defined form
    that ``condition_canvas`` consumes; the contents are otherwise
    unspecified. ``weights`` must not share storage with the logits. Steps
    that may run concurrently need separate workspaces.
    """

    weights: torch.Tensor
    normalizer: torch.Tensor
    product: torch.Tensor
    scratch: torch.Tensor

    @classmethod
    def empty(
        cls,
        rows: int,
        canvas_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> CanvasWorkspace:
        positions = rows * canvas_length
        device_type = torch.empty(0, device=device).device.type
        scratch = _scratch_bytes(positions, hidden_size, device_type)
        return cls(
            weights=torch.empty(
                positions, vocab_size, dtype=dtype, device=device
            ),
            normalizer=torch.empty(
                positions, dtype=torch.float32, device=device
            ),
            product=torch.empty(
                positions, hidden_size, dtype=torch.float32, device=device
            ),
            scratch=torch.empty(scratch, dtype=torch.uint8, device=device),
        )

    @staticmethod
    def nbytes(
        rows: int,
        canvas_length: int,
        vocab_size: int,
        hidden_size: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device_type: str = "cuda",
    ) -> int:
        """Bytes of the tensors ``empty`` allocates for these arguments.

        Sums ``weights``, ``normalizer``, ``product`` and ``scratch`` of a
        workspace on a device of type ``device_type`` (``"cuda"`` or
        ``"cpu"``) without allocating; a caching allocator may round each
        tensor's allocation up.
        """
        positions = rows * canvas_length
        return (
            positions * vocab_size * dtype.itemsize
            + positions * 4
            + positions * hidden_size * 4
            + _scratch_bytes(positions, hidden_size, device_type)
        )


def _kernels():
    from uniserve_kernels.diffusion import canvas

    return canvas


def _require(function: str, logits: torch.Tensor, state, sampling) -> None:
    kernels = _kernels()
    require_kernel(
        function,
        kernels.unsupported(
            logits,
            canvas_length=state.canvas.shape[1],
            hidden_size=state.self_conditioning.shape[1],
            eos_ids=sampling.eos_ids,
        ),
        logits=logits,
    )


def start_canvas(state: CanvasState, *, vocab_size: int) -> None:
    """Begin a block on every row whose step is 0; leave the other rows.

    Such a row draws its initial canvas from the INITIAL stream, fills its
    history with -1 and zeroes its self-conditioning rows.
    """
    rows, length = state.canvas.shape
    if state.canvas.is_cuda:
        kernels = _kernels()
        require_kernel(
            "start_canvas",
            None
            if kernels.supported(state.canvas.device)
            else "the canvas kernels require an SM100 device",
            canvas=state.canvas,
        )
        kernels.start(
            state.seed,
            state.block,
            state.step,
            state.canvas,
            state.history,
            state.self_conditioning,
            vocab_size=vocab_size,
        )
        return

    starting = state.step == 0
    bits = tokens.canvas_bits(
        state.seed,
        state.block,
        torch.zeros_like(state.step),
        tokens.NoiseStream.INITIAL,
        positions=length,
        words=1,
    )
    initial = tokens.random_tokens(bits[..., 0], vocab_size)
    state.canvas.copy_(torch.where(starting[:, None], initial, state.canvas))
    state.history.masked_fill_(starting[:, None, None], -1)
    state.self_conditioning.view(rows, length, -1).masked_fill_(
        starting[:, None, None], 0
    )


def _processed(
    logits: torch.Tensor, state: CanvasState, sampling: CanvasSampling
):
    remaining = sampling.steps - state.step
    temperature = tokens.step_temperature(
        remaining,
        t_min=sampling.t_min,
        t_max=sampling.t_max,
        steps=sampling.steps,
    )
    return logits / temperature[:, None, None]


def score_canvas(
    logits: torch.Tensor,
    state: CanvasState,
    sampling: CanvasSampling,
    *,
    scores: CanvasScores,
    workspace: CanvasWorkspace,
) -> None:
    """Score the FP32 logits ``[R, C, V]`` of one denoiser pass.

    Writes ``scores`` and the self-conditioning distribution of every
    position into ``workspace``. Reads ``state.seed``, ``block`` and
    ``step``. ``logits`` are only read.
    """
    rows, length, vocab = logits.shape
    if logits.is_cuda:
        _require("score_canvas", logits, state, sampling)
        _kernels().score(
            logits,
            workspace.weights,
            workspace.normalizer,
            scores.entropy,
            scores.argmax,
            scores.sample,
            state.seed,
            state.block,
            state.step,
            steps=sampling.steps,
            t_min=_fp32(sampling.t_min),
            t_delta=sampling.t_delta,
        )
        return

    processed = _processed(logits.float(), state, sampling)
    scores.entropy.copy_(tokens.token_entropy(processed))
    scores.argmax.copy_(processed.argmax(dim=-1))
    bits = tokens.canvas_bits(
        state.seed,
        state.block,
        state.step,
        tokens.NoiseStream.SAMPLE,
        positions=length,
        words=vocab // 4,
    )
    scores.sample.copy_(tokens.gumbel_sample(processed, tokens.uniform(bits)))
    # The reference carries bf16(q) to the next pass and normalizes it with
    # an FP32 softmax rounded to the embedding dtype.
    dtype = workspace.weights.dtype
    workspace.weights.copy_(
        torch.softmax(processed.to(dtype), dim=-1, dtype=torch.float32)
        .to(dtype)
        .reshape(rows * length, vocab)
    )
    workspace.normalizer.fill_(1.0)


def advance_canvas(
    scores: CanvasScores,
    state: CanvasState,
    sampling: CanvasSampling,
    *,
    decision: CanvasDecision,
    vocab_size: int,
) -> None:
    """Accept, re-noise and decide every row from its scores.

    Replaces ``state.canvas`` with the accepted samples and RENOISE-stream
    tokens, shifts the argmax canvas into ``state.history`` and writes
    ``decision``.
    """
    rows, length = state.canvas.shape
    if state.canvas.is_cuda:
        kernels = _kernels()
        require_kernel(
            "advance_canvas",
            None
            if kernels.supported(state.canvas.device)
            and length % 32 == 0
            and 0 < length <= 1024
            and len(sampling.eos_ids) <= 8
            else "the canvas kernels need an SM100 device, a canvas length "
            "divisible by 32 up to 1024 and at most 8 end-of-sequence ids",
            canvas=state.canvas,
        )
        kernels.advance(
            scores.entropy,
            scores.argmax,
            scores.sample,
            state.seed,
            state.block,
            state.step,
            state.history,
            state.canvas,
            decision.tokens,
            decision.finished,
            vocab_size=vocab_size,
            steps=sampling.steps,
            entropy_bound=_fp32(sampling.entropy_bound),
            confidence=_fp32(sampling.confidence),
            eos_ids=sampling.eos_ids,
            pad_id=sampling.pad_id,
        )
        return

    accepted = tokens.accept_by_entropy(scores.entropy, sampling.entropy_bound)
    bits = tokens.canvas_bits(
        state.seed,
        state.block,
        state.step,
        tokens.NoiseStream.RENOISE,
        positions=length,
        words=1,
    )
    noise = tokens.random_tokens(bits[..., 0], vocab_size)
    state.canvas.copy_(tokens.renoise(scores.sample, accepted, noise))
    stop, history = tokens.stable_and_confident(
        state.history.transpose(0, 1),
        scores.argmax,
        scores.entropy,
        confidence=sampling.confidence,
    )
    state.history.copy_(history.transpose(0, 1))
    done = stop | (state.step == sampling.steps - 1)
    eos = torch.tensor(
        sampling.eos_ids, dtype=torch.int64, device=state.canvas.device
    )
    truncated, ended = tokens.truncate_after_eos(
        scores.argmax, eos, sampling.pad_id
    )
    decision.tokens.copy_(truncated)
    decision.finished.copy_(torch.stack((done, ended), dim=-1))


def condition_canvas(
    workspace: CanvasWorkspace,
    embedding: torch.Tensor,
    embedding_scale: float,
    state: CanvasState,
) -> None:
    """Write the next pass's self-conditioning embedding into the state.

    ``embedding`` is the token embedding table ``[V', H]`` (``V' >= V``;
    its first ``V`` rows are used) and ``embedding_scale`` its scale, which
    rounds to the embedding dtype first as in the reference.
    """
    vocab = workspace.weights.shape[1]
    table = embedding[:vocab]
    scale = float(torch.tensor(embedding_scale, dtype=embedding.dtype))
    if workspace.weights.is_cuda:
        _kernels().product(
            workspace.weights, table, workspace.product, workspace.scratch
        )
        _kernels().condition(
            workspace.product,
            workspace.normalizer,
            state.self_conditioning,
            scale=scale,
        )
        return

    # The product accumulates in FP32 and rounds once, as the reference's
    # CUDA GEMM does; CPU BF16 matmul kernels need not accumulate in FP32.
    dtype = embedding.dtype
    product = torch.matmul(workspace.weights.float(), table.float())
    product = (product / workspace.normalizer[:, None]).to(dtype)
    state.self_conditioning.copy_(product * torch.tensor(scale, dtype=dtype))


def denoise_canvas(
    logits: torch.Tensor,
    embedding: torch.Tensor,
    embedding_scale: float,
    state: CanvasState,
    sampling: CanvasSampling,
    *,
    scores: CanvasScores,
    decision: CanvasDecision,
    workspace: CanvasWorkspace,
) -> None:
    """Run one sampler step after a denoiser pass: score, decide, condition.

    ``decision`` is complete before the self-conditioning product starts, so
    a caller may copy ``decision.finished`` to the host on another stream
    while the embedding computes.
    """
    score_canvas(logits, state, sampling, scores=scores, workspace=workspace)
    advance_canvas(
        scores, state, sampling, decision=decision, vocab_size=logits.shape[-1]
    )
    condition_canvas(workspace, embedding, embedding_scale, state)
