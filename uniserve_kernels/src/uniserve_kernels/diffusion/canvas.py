"""SM100 kernels of one block-diffusion denoising step over token canvases.

The sources in ``csrc/`` build as a PyTorch JIT extension on the first
:func:`load`; :func:`unsupported` never compiles. The launches run on
the current CUDA stream with device-resident per-row values and static
shapes. Apart from the first :func:`product` of each shape, which chooses
its algorithm, they never synchronize with the host, so CUDA graphs capture
them:

- :func:`start` begins a block on the rows whose step is zero.
- :func:`score` sweeps the FP32 logits of every canvas position. A
  persistent grid of CTA clusters splits each position's vocabulary; a
  loading warp streams the slice twice through a shared-memory ring (HBM,
  then L2) while eight arithmetic warps compute the entropy, the first
  argmax, the Gumbel sample and the self-conditioning weights. Only tokens
  that can still win the Gumbel race draw Philox bits.
- :func:`advance` decides every row: entropy-bound acceptance, re-noise,
  the stable-and-confident stop, and end-of-sequence truncation.
- :func:`product` multiplies the self-conditioning weights by the
  embedding table through cuBLASLt, with the algorithm a shipped table
  names for the device model and shape, or else the fastest by timing.
- :func:`condition` turns the self-conditioning product into the next
  step's embedding.

``uniserve.diffusion.canvas`` calls these kernels and defines the portable
formulas they follow; its module documentation states the numerical
contract, including where the kernels round differently.
"""

from __future__ import annotations

import json
from functools import cache, lru_cache
from pathlib import Path

import torch

__all__ = [
    "SLICE_TOKENS",
    "advance",
    "cluster_size",
    "condition",
    "load",
    "PRODUCT_CONFIG_FIELDS",
    "PRODUCT_TABLE",
    "product",
    "product_algorithm",
    "product_scratch_bytes",
    "product_table_key",
    "score",
    "start",
    "supported",
    "unsupported",
]

# Tokens of one position each sweep CTA owns at most; a position splits over
# a cluster of vocab_size / SLICE_TOKENS CTAs (rounded up to a power of two).
SLICE_TOKENS = 65536
# Tokens per bulk-copy chunk (``kChunkFloats`` in the kernel source).
_CHUNK_TOKENS = 4096
_CLUSTERS = (1, 2, 4, 8, 16)
_MAX_EOS = 8


def supported(device: torch.device | None = None) -> bool:
    """Report whether ``device`` runs the extension, without compiling it.

    The extension targets sm_100a, whose architecture-conditional code has
    no cross-generation compatibility.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability(
        device
    ) == (10, 0)


def cluster_size(vocab_size: int) -> int:
    """CTAs per position: the smallest cluster whose slices fit SLICE_TOKENS.

    Each CTA owns ``vocab_size / cluster`` tokens in whole chunks of
    _CHUNK_TOKENS. Returns 0 when no supported cluster divides the
    vocabulary that way.
    """
    for cluster in _CLUSTERS:
        if (
            vocab_size % (_CHUNK_TOKENS * cluster) == 0
            and vocab_size // cluster <= SLICE_TOKENS
        ):
            return cluster
    return 0


def unsupported(
    logits: torch.Tensor,
    *,
    canvas_length: int,
    hidden_size: int,
    eos_ids: tuple[int, ...],
) -> str | None:
    """Return why the kernels cannot run a step on ``logits``, or ``None``.

    ``logits`` are the step's FP32 ``[rows, canvas, vocab]`` logits.
    """
    if not logits.is_cuda:
        return f"logits reside on {logits.device}, not a CUDA device"
    if not supported(logits.device):
        return "the canvas kernels require an SM100 (compute capability 10.0)"
    if logits.dtype != torch.float32 or logits.ndim != 3:
        return "logits must be FP32 [rows, canvas, vocab]"
    if not logits.is_contiguous() or logits.data_ptr() % 16:
        return "logits must be contiguous and 16-byte aligned"
    if cluster_size(logits.shape[-1]) == 0:
        return (
            f"the vocabulary must split into whole {_CHUNK_TOKENS}-token "
            f"chunks, at most {SLICE_TOKENS} tokens on each of at most 16 CTAs"
        )
    if canvas_length % 32 or not 0 < canvas_length <= 1024:
        return "the canvas length must be a multiple of 32 up to 1024"
    if hidden_size % 8:
        return "the hidden size must be a multiple of 8"
    if len(eos_ids) > _MAX_EOS:
        return f"at most {_MAX_EOS} end-of-sequence ids"
    return None


def load() -> None:
    """Compile or load the cached extension before serving or CUDA capture.

    The first :func:`score` launch of each cluster size also sets function
    attributes and queries occupancy on the host; run one step before
    capturing a graph.
    """
    _extension()


# Build flags. The sources compile without fast math: the Gumbel scores use
# the accurate logf, and the divisions and temperature use explicit
# round-to-nearest intrinsics.
_HOST_FLAGS = ("-O3", "-std=c++20")
_DEVICE_FLAGS = (
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "-gencode=arch=compute_100a,code=sm_100a",
)
_LINK_FLAGS = ("-lcublasLt",)


@lru_cache(maxsize=1)
def _extension():
    # One build or cache lookup per process, content-addressed by
    # uniserve_kernels.jit.load.
    from uniserve_kernels import jit

    directory = Path(__file__).parent / "csrc"
    return jit.load(
        "uniserve_canvas_sm100",
        [directory / "canvas.cu", directory / "product.cpp"],
        cxx_flags=_HOST_FLAGS,
        cuda_flags=_DEVICE_FLAGS,
        ldflags=_LINK_FLAGS,
    )


def score(
    logits: torch.Tensor,
    weights: torch.Tensor,
    normalizer: torch.Tensor,
    entropy: torch.Tensor,
    argmax: torch.Tensor,
    sample: torch.Tensor,
    seed: torch.Tensor,
    block: torch.Tensor,
    step: torch.Tensor,
    *,
    steps: int,
    t_min: float,
    t_delta: float,
) -> None:
    """Score every canvas position of ``logits`` in one persistent sweep.

    ``logits`` is contiguous FP32 ``[rows, canvas, vocab]``; ``seed``,
    ``block`` and ``step`` are int64 ``[rows]``. The row temperature is
    ``t_min + t_delta * ((steps - step) / steps)`` in FP32, where ``t_min``
    and ``t_delta`` are FP32 values. Writes per position ``entropy`` (FP32),
    the first ``argmax`` of the exact processed logits and the Gumbel
    ``sample`` (int64), each contiguous ``[rows, canvas]``, the BF16
    self-conditioning ``weights`` ``[positions, vocab]`` (unit column
    stride, row stride divisible by four; must not overlap ``logits``) and
    their FP32 ``normalizer`` ``[positions]``.
    """
    _extension().score(
        logits,
        weights,
        normalizer,
        entropy,
        argmax,
        sample,
        seed,
        block,
        step,
        int(steps),
        float(t_min),
        float(t_delta),
        cluster_size(logits.shape[-1]),
    )


def advance(
    entropy: torch.Tensor,
    argmax: torch.Tensor,
    sample: torch.Tensor,
    seed: torch.Tensor,
    block: torch.Tensor,
    step: torch.Tensor,
    history: torch.Tensor,
    canvas: torch.Tensor,
    tokens: torch.Tensor,
    finished: torch.Tensor,
    *,
    vocab_size: int,
    steps: int,
    entropy_bound: float,
    confidence: float,
    eos_ids: tuple[int, ...],
    pad_id: int,
) -> None:
    """Accept, re-noise and decide every canvas row from its scores.

    Reads the per-position ``entropy``, ``argmax`` and ``sample`` of
    :func:`score`; replaces ``canvas`` (int64 ``[rows, canvas]``) and
    shifts ``history`` (int64 ``[rows, stability, canvas]``) in place;
    writes the end-of-sequence truncated argmax canvas into ``tokens`` and
    ``(block done, end of sequence seen)`` into bool ``finished``
    ``[rows, 2]``. ``entropy_bound`` and ``confidence`` are FP32 values.
    """
    _extension().advance(
        entropy,
        argmax,
        sample,
        seed,
        block,
        step,
        history,
        canvas,
        tokens,
        finished,
        int(vocab_size),
        int(steps),
        float(entropy_bound),
        float(confidence),
        [int(token) for token in eos_ids],
        int(pad_id),
    )


# The product's cuBLASLt workspace holds split-K partial sums: room for
# eight FP32 products of the step's shape, within 32 MiB to 256 MiB.
_SCRATCH_PARTIALS = 8
_SCRATCH_BYTES = (32 << 20, 256 << 20)

# Algorithms measured per device model, cuBLASLt version and shape; the
# product_table module of this package writes the file.
PRODUCT_TABLE = Path(__file__).with_name("product_algorithms.json")
PRODUCT_CONFIG_FIELDS = (
    "algorithm",
    "tile",
    "stages",
    "split_k",
    "reduction",
    "swizzle",
    "custom",
    "inner_shape",
    "cluster_shape",
)


def product_scratch_bytes(positions: int, hidden_size: int) -> int:
    """Bytes of cuBLASLt workspace :func:`product` expects for its shape.

    ``positions`` rows of ``hidden_size`` FP32 outputs: eight such products
    of split-K partial sums, clamped to 32 MiB .. 256 MiB. The shipped
    algorithm table is keyed by this size.
    """
    partials = _SCRATCH_PARTIALS * positions * hidden_size * 4
    return min(max(partials, _SCRATCH_BYTES[0]), _SCRATCH_BYTES[1])


def product_table_key(
    device: torch.device, positions: int, hidden: int, vocab: int, scratch: int
) -> tuple[str, int, int, int, int, int]:
    """The shipped-table key of a product on ``device``.

    (device name, cuBLASLt version, positions, hidden, vocab, scratch
    bytes); loads the extension to read the cuBLASLt version.
    """
    return (
        torch.cuda.get_device_name(device),
        _extension().cublaslt_version(),
        positions,
        hidden,
        vocab,
        scratch,
    )


@lru_cache(maxsize=1)
def _shipped() -> dict[tuple, list[int]]:
    entries = json.loads(PRODUCT_TABLE.read_text())["entries"]
    return {
        (
            entry["device"],
            entry["cublaslt_version"],
            entry["m"],
            entry["n"],
            entry["k"],
            entry["scratch_bytes"],
        ): [entry["algorithm"][field] for field in PRODUCT_CONFIG_FIELDS]
        for entry in entries
    }


@cache
def _preferred(
    device: int, positions: int, hidden: int, vocab: int, scratch: int
) -> list[int]:
    key = product_table_key(
        torch.device("cuda", device), positions, hidden, vocab, scratch
    )
    return _shipped().get(key, [])


def product(
    weights: torch.Tensor,
    table: torch.Tensor,
    output: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    """Write ``weights @ table`` into ``output``, accumulating in FP32.

    ``weights`` is BF16 ``[positions, vocab]``, ``table`` the BF16 ``[vocab,
    hidden]`` embedding rows and ``output`` FP32 ``[positions, hidden]``, all
    with unit column stride; ``scratch`` is the contiguous uint8 cuBLASLt
    workspace (see :func:`product_scratch_bytes`), which concurrent products
    must not share.

    The first product of each shape and workspace size chooses the cuBLASLt
    algorithm, which synchronizes the device; later products launch it
    without synchronizing, and CUDA graphs capture them, so run one product
    of every shape before capturing a graph. The algorithms differ only in
    FP32 summation order, and so in the last bits of the product. When
    :data:`PRODUCT_TABLE` names an algorithm for the device model, cuBLASLt
    version and shape, the product uses it and every process rounds the
    same way. Otherwise the first product times cuBLASLt's proposals and
    keeps the fastest: the same performance, but another process may choose
    another algorithm, so results are not reproducible across restarts.
    :func:`product_algorithm` reports which case applies.
    """
    preferred = _preferred(
        weights.device.index,
        weights.shape[0],
        table.shape[1],
        weights.shape[1],
        scratch.numel(),
    )
    _extension().product(weights, table, output, scratch, preferred)


def product_algorithm(
    weights: torch.Tensor,
    table: torch.Tensor,
    output: torch.Tensor,
    scratch: torch.Tensor,
) -> dict[str, int | str] | None:
    """Return the cuBLASLt algorithm :func:`product` uses for these operands.

    ``None`` before the first product of the operands' shape and workspace
    size. Otherwise ``source`` is ``"table"`` when :data:`PRODUCT_TABLE`
    chose the algorithm (reproducible across restarts) or ``"measured"``
    when the first product timed the proposals (not reproducible), followed
    by the cuBLASLt version and the configuration fields of
    :data:`PRODUCT_CONFIG_FIELDS`.
    """
    values = _extension().product_algorithm(weights, table, output, scratch)
    if not values:
        return None
    return {
        "source": "table" if values[0] else "measured",
        "cublaslt_version": _extension().cublaslt_version(),
        **dict(zip(PRODUCT_CONFIG_FIELDS, values[1:], strict=True)),
    }


def condition(
    product: torch.Tensor,
    normalizer: torch.Tensor,
    output: torch.Tensor,
    *,
    scale: float,
) -> None:
    """Write ``bf16(product * scale / normalizer)`` row by row into ``output``.

    ``product`` is contiguous FP32 ``[positions, hidden]`` (hidden divisible
    by four), ``normalizer`` FP32 ``[positions]`` and ``output`` contiguous
    BF16 shaped like ``product``.
    """
    _extension().condition(product, normalizer, float(scale), output)


def start(
    seed: torch.Tensor,
    block: torch.Tensor,
    step: torch.Tensor,
    canvas: torch.Tensor,
    history: torch.Tensor,
    self_conditioning: torch.Tensor,
    *,
    vocab_size: int,
) -> None:
    """Begin a block on every row whose ``step`` is zero; leave others.

    Such a row receives its initial canvas, a history of -1 and zero
    self-conditioning rows (``[rows * canvas, hidden]``, 16-byte rows).
    """
    _extension().start(
        seed, block, step, canvas, history, self_conditioning, int(vocab_size)
    )
