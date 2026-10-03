"""Numerical primitives of discrete token-canvas denoising.

Block-diffusion text generation repeatedly denoises a fixed-length canvas of
token ids: each step scores every canvas position, samples a candidate token
per position, accepts the lowest-entropy candidates within an entropy
budget, and re-noises the rest. These functions define that arithmetic on
tensors with a leading batch of canvas rows. They read no host values, keep
static shapes and draw randomness from counter-based Philox bits, so callers
may run them inside CUDA graphs and reproduce a request's draws from its
seed regardless of how requests are batched. Choosing which step runs and
committing its result belong to execution.

Randomness contract. Every draw is Philox4x32-10 keyed by the request seed
(``key = (seed mod 2^32, seed >> 32)``) at the counter
``(word, position, step, block << 2 | stream)``: ``stream`` separates the
initial canvas, the per-step candidate sample and the re-noise draw,
``block`` counts committed canvases of the request, ``step`` counts
denoising steps within the block from zero, ``position`` indexes the canvas
and ``word`` selects a group of four 32-bit outputs. A fused kernel that
follows this contract draws the same bits.
"""

from __future__ import annotations

from enum import IntEnum

import torch

__all__ = [
    "NoiseStream",
    "accept_by_entropy",
    "candidate_logprobs",
    "canvas_bits",
    "gumbel_sample",
    "philox",
    "random_tokens",
    "renoise",
    "self_conditioning_embedding",
    "stable_and_confident",
    "step_temperature",
    "token_entropy",
    "truncate_after_eos",
    "uniform",
]

_MASK = 0xFFFFFFFF
# Philox4x32 multipliers and Weyl key increments (Salmon et al., SC'11).
_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_ROUNDS = 10


class NoiseStream(IntEnum):
    """Independent random streams of one denoising block."""

    INITIAL = 0
    SAMPLE = 1
    RENOISE = 2


def _mulhilo(value: torch.Tensor, multiplier: int):
    """Return the high and low 32-bit words of ``value * multiplier``.

    ``value`` holds unsigned 32-bit integers in int64. The 64-bit product
    would overflow int64, so the value splits into 16-bit halves whose
    partial products stay below 2^48.
    """
    low_part = (value & 0xFFFF) * multiplier
    high_part = (value >> 16) * multiplier
    shifted = low_part + ((high_part & 0xFFFF) << 16)
    return (high_part >> 16) + (shifted >> 32), shifted & _MASK


def philox(key: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    """Philox4x32-10 block function on unsigned 32-bit words held in int64.

    ``key`` is ``[..., 2]`` and ``counter`` ``[..., 4]``; leading axes
    broadcast. Returns ``[..., 4]`` output words in ``[0, 2^32)``.
    """
    if key.shape[-1] != 2 or counter.shape[-1] != 4:
        raise ValueError("Philox takes two key words and four counter words")
    if key.dtype != torch.int64 or counter.dtype != torch.int64:
        raise ValueError("Philox words are held in int64 tensors")
    k0, k1 = key.unbind(-1)
    c0, c1, c2, c3 = counter.unbind(-1)
    for index in range(_ROUNDS):
        if index:
            k0 = (k0 + _W0) & _MASK
            k1 = (k1 + _W1) & _MASK
        hi0, lo0 = _mulhilo(c0, _M0)
        hi1, lo1 = _mulhilo(c2, _M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return torch.stack((c0, c1, c2, c3), dim=-1)


def canvas_bits(
    seed: torch.Tensor,
    block: torch.Tensor,
    step: torch.Tensor,
    stream: NoiseStream,
    *,
    positions: int,
    words: int,
) -> torch.Tensor:
    """Draw ``4 * words`` random 32-bit values per canvas position.

    ``seed``, ``block`` and ``step`` are int64 ``[R]`` per canvas row.
    Returns int64 ``[R, positions, 4 * words]``; value ``j`` of a position
    is output lane ``j % 4`` of counter word ``j // 4``.
    """
    rows = seed.shape[0]
    device = seed.device
    key = torch.stack((seed & _MASK, (seed >> 32) & _MASK), dim=-1)
    word = torch.arange(words, device=device)
    position = torch.arange(positions, device=device)
    counter = torch.stack(
        torch.broadcast_tensors(
            word[None, None, :],
            position[None, :, None],
            (step & _MASK)[:, None, None],
            ((block << 2 | int(stream)) & _MASK)[:, None, None],
        ),
        dim=-1,
    )
    bits = philox(key[:, None, None, :], counter)
    return bits.reshape(rows, positions, 4 * words)


def uniform(bits: torch.Tensor) -> torch.Tensor:
    """Map 32-bit values to FP32 uniforms strictly inside (0, 1).

    The top 23 bits select one of 2^23 cell midpoints ``(2k + 1) / 2^24``.
    Their odd numerators stay below 2^24, so every midpoint is exact in FP32
    and neither 0 nor 1 can occur.
    """
    return ((bits >> 9) * 2 + 1).to(torch.float32) * 2.0**-24


def random_tokens(bits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map 32-bit values to token ids in ``[0, vocab_size)`` (int64).

    Uses the multiply-shift reduction ``bits * vocab_size >> 32``.
    """
    if not 0 < vocab_size < 2**31:
        raise ValueError("vocabulary size must be positive and below 2^31")
    return (bits * vocab_size) >> 32


def step_temperature(
    remaining: torch.Tensor, *, t_min: float, t_max: float, steps: int
) -> torch.Tensor:
    """Linear temperature at a denoising step, in FP32.

    ``remaining`` counts steps still to run, from ``steps`` at the first
    step down to one, so the temperature falls from ``t_max`` towards
    ``t_min``: ``t_min + (t_max - t_min) * remaining / steps``. The
    operations and their FP32 rounding match Transformers'
    ``LinearTemperatureScheduleLogitsProcessor``.
    """
    return t_min + ((t_max - t_min) * (remaining / steps))


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of ``softmax(logits)`` over the last axis.

    Follows ``torch.distributions.Categorical(logits=...).entropy()`` step
    for step without its host-side argument validation: normalized
    log-probabilities are clamped at the dtype minimum so that zero
    probabilities contribute zero rather than NaN.
    """
    normalized = logits - logits.logsumexp(dim=-1, keepdim=True)
    probabilities = torch.softmax(normalized, dim=-1)
    clamped = normalized.clamp(min=torch.finfo(normalized.dtype).min)
    return -(clamped * probabilities).sum(dim=-1)


def accept_by_entropy(entropy: torch.Tensor, bound: float) -> torch.Tensor:
    """Select each row's lowest-entropy positions within an entropy budget.

    Positions sorted by ascending entropy ``h_1 <= ... <= h_k`` are accepted
    while ``sum_i h_i - h_k <= bound``: the sum minus the largest term upper
    bounds the mutual information among the accepted tokens. The first
    position is always accepted. Equal entropies keep their position order.
    Returns a boolean mask shaped like ``entropy``.
    """
    ordered, order = torch.sort(entropy, dim=-1, stable=True)
    keep = torch.cumsum(ordered, dim=-1) - ordered <= bound
    return torch.zeros_like(keep).scatter(-1, order, keep)


def gumbel_sample(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """Draw one token per row from ``softmax(logits)`` with given uniforms.

    Returns ``argmax(logits - log(-log(u)))`` over the last axis (int64),
    the Gumbel-max form of the exponential race; ties take the lowest id.
    ``uniforms`` lie strictly inside (0, 1), as ``uniform`` produces.
    """
    return torch.argmax(logits - torch.log(-torch.log(uniforms)), dim=-1)


def renoise(
    canvas: torch.Tensor, accepted: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """Keep accepted positions and replace every other with ``noise`` ids."""
    return torch.where(accepted, canvas, noise)


def stable_and_confident(
    history: torch.Tensor,
    argmax: torch.Tensor,
    entropy: torch.Tensor,
    *,
    confidence: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decide whether each canvas row has converged, and advance history.

    ``history`` is ``[S, R, C]``: the argmax canvases of the previous ``S``
    steps, oldest first, filled with -1 at the start of a block so that no
    row is stable before ``S`` steps. A row stops when its new ``argmax``
    ``[R, C]`` equals all ``S`` previous ones (always true for ``S = 0``)
    and the mean of ``entropy`` ``[R, C]`` over the canvas is below
    ``confidence``. Returns the stop mask ``[R]`` and the history with the
    oldest entry replaced by ``argmax``.
    """
    stable = (history == argmax[None]).all(dim=-1).all(dim=0)
    confident = entropy.mean(dim=-1) < confidence
    if history.shape[0]:
        history = torch.cat((history[1:], argmax[None]))
    return stable & confident, history


def truncate_after_eos(
    tokens: torch.Tensor, eos_ids: torch.Tensor, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad every token after a row's first end-of-sequence token.

    ``tokens`` is ``[R, C]`` and ``eos_ids`` a 1-D tensor of stop ids.
    Returns the truncated tokens and whether each row contains a stop id.
    """
    is_eos = torch.isin(tokens, eos_ids)
    seen = is_eos.cumsum(dim=-1)
    after = (seen > 0) & ~((seen == 1) & is_eos)
    return tokens.masked_fill(after, pad_id), is_eos.any(dim=-1)


def candidate_logprobs(
    logits: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """Log-probabilities of candidate ids under the full-vocabulary softmax.

    ``logits`` is ``[R, V]`` and ``candidates`` int64 ``[R, K]``; the result
    is FP32 ``[R, K]``.
    """
    normalized = torch.log_softmax(logits.float(), dim=-1)
    return normalized.gather(-1, candidates)


def self_conditioning_embedding(
    logits: torch.Tensor, embedding: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Expected input embedding under the previous step's distribution.

    ``logits`` are the temperature-processed FP32 logits ``[R, V]`` of the
    previous step. As in the reference decoder they round to the embedding
    dtype, take an FP32 softmax that rounds back to that dtype, multiply the
    embedding table ``[V, H]``, and scale by the embedding scale rounded to
    the embedding dtype. Returns ``[R, H]`` in the embedding dtype.

    The product accumulates in FP32 and rounds once, as the reference's
    CUDA GEMM does; CPU BF16 matmul kernels need not accumulate in FP32.
    """
    dtype = embedding.dtype
    probabilities = torch.softmax(
        logits.to(dtype), dim=-1, dtype=torch.float32
    ).to(dtype)
    product = torch.matmul(probabilities.float(), embedding.float()).to(dtype)
    return product * scale.to(dtype)
