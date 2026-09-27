"""BF16 grouped expert GEMMs over FlashInfer's MoE sort metadata (SM100).

:func:`gather_gemm` is FC1 of routed BF16 experts: it gathers every
permuted (token, route) row's hidden state, multiplies it by its expert's
up and gate rows and stores the gated product in BF16 at the
permuted row (``_gather_gemm_kernel.GatherGroupedGemmKernel``).
:func:`route_gemm` is FC2: it multiplies each permuted row by its expert's
down projection and stores the unweighted result at the row's expanded
index ``t * top_k + k`` (``_route_gemm_kernel.RouteGroupedGemmKernel``), so
the output is every token's uncombined routes. The metadata comes from
``flashinfer.fused_moe.cute_dsl.moe_utils.moe_sort`` with a routing tile
equal to both tactics' MMA tile M.

Compiled kernels are cached per tactic, activation and top-k; problem
sizes and pointers are runtime arguments, so one executor serves every
token count. Launches run on the caller's current stream and allocate
nothing, so CUDA graphs capture them.
"""

from __future__ import annotations

from typing import Any

import torch

#: (MMA tile M, N), cluster (M, N): the FC1 tactics. M 256 runs two-CTA MMA
#: over a cluster pair; an N tile is 64 up rows and their 64 gate rows. The
#: gathered A rows are loaded per CTA, so clusters never span N.
GATHER_TACTICS = (
    ((128, 128), (1, 1)),
    ((256, 128), (2, 1)),
)
DEFAULT_GATHER_TACTIC = GATHER_TACTICS[0]

#: (MMA tile M, N), cluster (M, N): the FC2 tactics. Tile M must equal the
#: routing tile, which FC1 shares.
ROUTE_TACTICS = (
    ((128, 128), (1, 1)),
    ((128, 128), (1, 2)),
    ((128, 256), (1, 1)),
    ((128, 256), (1, 2)),
    ((256, 128), (2, 1)),
    ((256, 128), (2, 2)),
    ((256, 256), (2, 1)),
    ((256, 256), (2, 2)),
)
DEFAULT_ROUTE_TACTIC = ROUTE_TACTICS[0]

_EXECUTORS: dict[tuple, Any] = {}


def unsupported(
    hidden: torch.Tensor, weights: torch.Tensor, activation: str
) -> str | None:
    """Return why :func:`gather_gemm` cannot take these operands, or None.

    ``hidden`` is contiguous BF16 ``[T, K]`` and ``weights`` contiguous BF16
    ``[E, N, K]`` on one SM100 device, with ``K`` a multiple of 64 and ``N``
    of 128 (each N tile is 64 up rows and the matching 64 gate rows).
    """
    if not (hidden.is_cuda and weights.device == hidden.device):
        return "operands must reside on one CUDA device"
    if torch.cuda.get_device_capability(hidden.device)[0] != 10:
        return "the kernel is built for SM100-class devices"
    if hidden.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16:
        return "the kernel reads BF16 hidden states and weights"
    if hidden.ndim != 2 or weights.ndim != 3:
        return "hidden states are [T, K] and weights [E, N, K]"
    if weights.shape[2] != hidden.shape[1] or hidden.shape[1] % 64:
        return "K must match and be a multiple of 64"
    if weights.shape[1] % 128:
        return "N must be a multiple of 128"
    if not (hidden.is_contiguous() and weights.is_contiguous()):
        return "operands must be contiguous"
    if activation not in ("silu", "gelu_tanh"):
        return f"unsupported gated activation {activation!r}"
    return None


def _executor(key: tuple, build, tactic, arguments: tuple):
    """Compile, or return the cached executor of, one kernel specialization.

    ``build()`` constructs the kernel object; ``arguments`` are the wrapper's
    runtime arguments with the stream last.
    """
    executor = _EXECUTORS.get(key)
    if executor is None:
        import cutlass.cute as cute
        from flashinfer.cute_dsl.utils import get_max_active_clusters

        (mma_tiler_mn, cluster_shape_mn) = tactic
        clusters = get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )
        executor = cute.compile(
            build().wrapper,
            *arguments[:-1],
            tile_size=mma_tiler_mn[0],
            max_active_clusters=clusters,
            stream=arguments[-1],
        )
        _EXECUTORS[key] = executor
    return executor


def gather_gemm(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    out: torch.Tensor,
    *,
    top_k: int,
    activation: str,
    tactic=DEFAULT_GATHER_TACTIC,
) -> None:
    """Store ``act(gate) * up`` of every permuted row into ``out``.

    ``hidden`` is BF16 ``[T, K]`` and ``weights`` BF16 ``[E, N, K]`` with
    each expert's ``N / 2`` up rows followed by its ``N / 2`` gate rows. The
    metadata tensors are ``moe_sort``'s outputs for routing tile M of
    ``tactic``; ``out`` is BF16 ``[M_perm, N / 2]`` with ``M_perm`` the
    length of ``permuted_idx_to_expanded_idx``, a multiple of the tile.
    Padding rows of written tiles receive unspecified values. Callers first
    check :func:`unsupported`.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from flashinfer.cute_dsl.utils import make_ptr

    gmem = cute.AddressSpace.gmem
    stream = cuda.CUstream(torch.cuda.current_stream(hidden.device).cuda_stream)
    arguments = (
        make_ptr(cutlass.BFloat16, hidden.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.BFloat16, weights.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.BFloat16, out.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), gmem),
        make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), gmem),
        make_ptr(cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), gmem),
        make_ptr(cutlass.Int32, num_non_exiting_tiles.data_ptr(), gmem),
        hidden.shape[0],
        permuted_idx_to_expanded_idx.shape[0],
        weights.shape[1],
        weights.shape[2],
        weights.shape[0],
        stream,
    )
    (mma_tiler_mn, cluster_shape_mn) = tactic

    def build():
        from ._gather_gemm_kernel import GatherGroupedGemmKernel

        return GatherGroupedGemmKernel(
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            topk=top_k,
            activation=activation,
        )

    key = ("gather", mma_tiler_mn, cluster_shape_mn, activation, top_k)
    executor = _executor(key, build, tactic, arguments)
    executor(*arguments[:-1], stream=stream)


def route_gemm(
    rows: torch.Tensor,
    weights: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    out: torch.Tensor,
    *,
    tactic=DEFAULT_ROUTE_TACTIC,
) -> None:
    """Store every permuted row's expert product at its expanded row of ``out``.

    ``rows`` is BF16 ``[M_perm, K]`` (FC1's output), ``weights`` BF16
    ``[E, N, K]`` and ``out`` BF16 ``[T * top_k, N]``: row ``t * top_k + k``
    receives route ``k`` of token ``t`` unweighted. The metadata tensors are
    ``moe_sort``'s outputs for routing tile M of ``tactic``. Every expanded
    row is written exactly once when all experts are resident; callers own
    rows of routes to non-resident experts. Callers first check
    :func:`unsupported` with ``rows`` as the hidden states.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from flashinfer.cute_dsl.utils import make_ptr

    gmem = cute.AddressSpace.gmem
    stream = cuda.CUstream(torch.cuda.current_stream(rows.device).cuda_stream)
    arguments = (
        make_ptr(cutlass.BFloat16, rows.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.BFloat16, weights.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.BFloat16, out.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), gmem),
        make_ptr(cutlass.Int32, num_non_exiting_tiles.data_ptr(), gmem),
        make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), gmem),
        make_ptr(cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), gmem),
        permuted_idx_to_expanded_idx.shape[0],
        weights.shape[1],
        weights.shape[2],
        weights.shape[0],
        out.shape[0],
        stream,
    )
    (mma_tiler_mn, cluster_shape_mn) = tactic

    def build():
        from ._route_gemm_kernel import RouteGroupedGemmKernel

        return RouteGroupedGemmKernel(
            mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn
        )

    key = ("route", mma_tiler_mn, cluster_shape_mn)
    executor = _executor(key, build, tactic, arguments)
    executor(*arguments[:-1], stream=stream)
