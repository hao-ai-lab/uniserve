"""Grouped route dispatch and weight-overlay application for target roots.

Stage 5 of ``specs/unified_forward_execution.md``: route and overlay
variation is device data inside one packed layer traversal. This module owns
the two shared primitives the family roots build on:

* :class:`GroupedLinear` — one loop-free invocation computing every token's
  projection through its route's weights (``SegmentTable.route_id`` expanded
  to a per-token route column) plus its overlay slot's low-rank delta. No
  host loop invokes a route-specific module; all active routes are handled
  by one operator expression, and canonical token order is preserved (no
  regrouping).
* :class:`WeightOverlayBank` — the pointer-stable per-request overlay bank
  (LoRA-style low-rank deltas). Fixed startup capacity, stable addresses,
  slot generations, and reference counts; slot ``0`` is the designated
  neutral slot whose contribution is exactly zero. Load and unload are
  transaction-boundary commands; unload cannot reclaim a pinned slot.

The reference formulation favors capture safety and provability over speed:
``per-route one-hot masking + one batched matmul`` is a single fused
expression with static shapes, so it is CUDA-graph capturable at any bucket
capacity. Production kernels (grouped GEMMs, expert dispatch) replace the
inner expression behind the same interface without changing callers.
"""
from __future__ import annotations

import torch

__all__ = [
    "GroupedLinear",
    "OverlayBankError",
    "WeightOverlayBank",
]


class OverlayBankError(RuntimeError):
    """An overlay lifecycle rule was violated."""


class WeightOverlayBank:
    """Preallocated pointer-stable low-rank overlay bank.

    Storage: ``down[slots, rank, in_features]`` and
    ``up[slots, out_features, rank]``; a token's delta is
    ``x @ down[slot].T @ up[slot].T``. Slot 0 stays zero forever.
    """

    def __init__(
        self,
        *,
        slots: int,
        rank: int,
        in_features: int,
        out_features: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if slots < 1:
            raise OverlayBankError("the bank needs at least the neutral slot")
        self.down = torch.zeros((slots, rank, in_features), device=device, dtype=dtype)
        self.up = torch.zeros((slots, out_features, rank), device=device, dtype=dtype)
        self.generations = [0] * slots
        self._pins = [0] * slots
        self._down_ptr = self.down.data_ptr()
        self._up_ptr = self.up.data_ptr()

    def load(self, slot: int, down: torch.Tensor, up: torch.Tensor) -> int:
        """Load one overlay at a transaction boundary; returns the generation."""

        if slot == 0:
            raise OverlayBankError("slot 0 is the neutral overlay and stays zero")
        if not 0 < slot < self.down.shape[0]:
            raise OverlayBankError(f"slot {slot} is outside the bank capacity")
        if self._pins[slot]:
            raise OverlayBankError(f"slot {slot} is pinned by an accepted transaction")
        self.down[slot].copy_(down)
        self.up[slot].copy_(up)
        self.generations[slot] += 1
        self._assert_stable()
        return self.generations[slot]

    def unload(self, slot: int) -> None:
        if slot == 0:
            raise OverlayBankError("slot 0 cannot be unloaded")
        if self._pins[slot]:
            raise OverlayBankError(f"slot {slot} is pinned by an accepted transaction")
        self.down[slot].zero_()
        self.up[slot].zero_()
        self.generations[slot] += 1

    def pin(self, slot: int) -> None:
        self._pins[slot] += 1

    def unpin(self, slot: int) -> None:
        if self._pins[slot] <= 0:
            raise OverlayBankError(f"slot {slot} is not pinned")
        self._pins[slot] -= 1

    def _assert_stable(self) -> None:
        if (
            self.down.data_ptr() != self._down_ptr
            or self.up.data_ptr() != self._up_ptr
        ):
            raise OverlayBankError("overlay bank storage drifted")


class GroupedLinear:
    """One projection over route-indexed weights plus overlay deltas.

    ``weights[routes, out_features, in_features]``; disjoint parameter sets
    per route are presented as grouped route experts under one traversal
    rather than separate top-level forwards.
    """

    def __init__(self, weights: torch.Tensor, bank: WeightOverlayBank | None = None) -> None:
        if weights.dim() != 3:
            raise ValueError("grouped weights are [routes, out, in]")
        self.weights = weights
        self.bank = bank

    def forward(
        self,
        x: torch.Tensor,
        route_of_token: torch.Tensor,
        overlay_slot_of_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Loop-free grouped projection preserving canonical token order.

        ``x``: ``[tokens, in]``; ``route_of_token``: int64 ``[tokens]``;
        ``overlay_slot_of_token``: int64 ``[tokens]`` or None for all-base.
        """

        routes = self.weights.shape[0]
        # One-hot route mixing: every active route participates in one fused
        # expression; inactive routes contribute exactly zero.
        one_hot = torch.nn.functional.one_hot(
            route_of_token.clamp(min=0, max=routes - 1), num_classes=routes
        ).to(x.dtype)
        projected = torch.einsum("ti,roi->tro", x, self.weights)
        y = torch.einsum("tro,tr->to", projected, one_hot)
        if self.bank is not None and overlay_slot_of_token is not None:
            down = self.bank.down.index_select(0, overlay_slot_of_token)
            up = self.bank.up.index_select(0, overlay_slot_of_token)
            hidden = torch.einsum("ti,tri->tr", x, down.to(x.dtype))
            y = y + torch.einsum("tr,tor->to", hidden, up.to(x.dtype))
        return y
