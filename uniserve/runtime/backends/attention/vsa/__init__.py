"""Factories and independently bound operators for Video Sparse Attention."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

import torch

from uniserve.nn.attention.vsa.inputs import BlockInput, Pattern


class Operator:
    """One numerical signature's plans and scratch.

    Serialized call sites may share an operator; block IDs remain live inputs.
    A concrete operator supplies ``kernel``, the numerical call
    ``kernel(q, k, v, out, q2k_index, q2k_num, variable_block_sizes,
    scale=...)`` that evaluates complete calls over the pattern's tiles
    (``Pattern.tile`` rows). K and V rows past a tile's valid size must be
    finite.
    """

    kernel: Callable[..., object]

    def __init__(
        self,
        pattern,
        *,
        num_heads,
        head_dim,
        dtype,
        workspace,
        transient=None,
    ):
        if min(num_heads, head_dim) < 1 or len(pattern.row_counts) not in (
            1,
            num_heads,
        ):
            raise ValueError(
                "VSA operator requires compatible positive head dimensions"
            )

        self.pattern, self.num_heads, self.head_dim, self.dtype = (
            pattern,
            num_heads,
            head_dim,
            dtype,
        )
        self.workspace, self._closed = workspace, False
        # ``transient(role, requirements, device)`` lends per-call work areas
        # that every operator of the owning context shares; without it the
        # operator allocates its own.
        self.transient = transient

    def bind(self, batch: BlockInput) -> None:
        """Check a live batch against the prepared pattern before use."""
        if self._closed:
            raise RuntimeError("VSA operator is closed")
        if batch.pattern != self.pattern:
            raise ValueError(
                "VSA block cardinalities differ from the prepared pattern"
            )

    def _validate(self, q, k, v, batch, out):
        """Bind the batch and check every tensor.

        Bind the batch and check every tensor against the prepared
        dimensions.
        """
        self.bind(batch)
        tile = self.pattern.tile
        if (
            q.shape
            != (
                len(self.pattern.row_counts[0]) * tile,
                self.num_heads,
                self.head_dim,
            )
            or k.shape != v.shape
            or k.ndim != 3
            or k.shape[1:] != q.shape[1:]
            or k.shape[0] != batch.variable_block_sizes.numel() * tile
            or out.shape != q.shape
            or out.dtype != q.dtype
            or any(value.dtype != self.dtype for value in (q, k, v))
            or any(
                value.device != q.device
                for value in (
                    k,
                    v,
                    out,
                    batch.q2k_index,
                    batch.q2k_num,
                    batch.variable_block_sizes,
                )
            )
            or batch.q2k_index.shape[:2] != (self.num_heads, q.shape[0] // tile)
        ):
            raise ValueError(
                "VSA tensors disagree with the prepared numerical dimensions"
            )

    def __call__(self, q, k, v, batch, *, scale, out):
        """Evaluate one attention call into ``out``."""
        self._validate(q, k, v, batch, out)
        self.kernel(
            q,
            k,
            v,
            out,
            batch.q2k_index,
            batch.q2k_num,
            batch.variable_block_sizes,
            scale=scale,
        )
        return out

    def close(self):
        """Release plans and borrowed workspace; the operator is not reused."""
        self._closed = True
        self.workspace = {}


class Backend:
    """Construct operators and their workspace for one concrete VSA provider."""

    name: str
    operator_class: type[Operator]

    def workspace_buffers(
        self,
        pattern: Pattern,
        *,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        """Describe extra device buffers.

        Describe extra device buffers the backend needs beyond caller
        tensors.
        """
        return {}

    def prepare(
        self,
        pattern: Pattern,
        *,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        workspace,
        transient=None,
    ):
        """Build an operator bound to one pattern.

        Build an operator bound to one pattern, dtype, and borrowed
        workspace. ``transient`` lends the owning context's shared per-call
        work areas (see ``ExecutionContext.scratch``).
        """
        return self.operator_class(
            pattern,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            workspace=workspace,
            transient=transient,
        )


# Provider modules in the order ``"auto"`` probes them.
_PROVIDERS = ("sm100", "flashinfer", "triton")


def resolve(backend, *, device, tile):
    """Return the Backend for a name or instance that serves ``tile``.

    ``backend`` is a name, an existing instance, or ``"auto"``, which probes
    the device for the first available provider of ``tile``-row tiles. A
    provider module declares the tile sizes it serves in ``TILES``.

    Raises:
        RuntimeError: No installed provider (or not the named one) serves
            ``tile``-row tiles on ``device``.
        ValueError: An unknown backend name.
    """
    if isinstance(backend, Backend):
        return backend
    # Every provider runs CUDA kernels; other devices have none to probe.
    cuda = torch.device(device).type == "cuda"
    if backend == "auto":
        for name in _PROVIDERS:
            candidate = import_module(f"{__name__}.{name}")
            if cuda and tile in candidate.TILES and candidate.available(device):
                return candidate.Backend()
        raise RuntimeError(
            f"no installed VSA backend serves {tile}-row tiles on {device}"
        )

    if backend not in _PROVIDERS:
        raise ValueError(f"unknown VSA backend {backend!r}")
    module = import_module(f"{__name__}.{backend}")
    if not cuda or tile not in module.TILES or not module.available(device):
        raise RuntimeError(
            f"VSA backend {backend!r} does not serve {tile}-row tiles on "
            f"{device}"
        )
    return module.Backend()
