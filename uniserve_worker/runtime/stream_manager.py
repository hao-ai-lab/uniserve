"""CUDA stream groups backed by Green Contexts.

The :class:`StreamManager` owns N stream groups, each a ``(prefill, decode)``
stream pair partitioned across the GPU's SMs by a CUDA Green Context, plus two
full-SM endpoints (prefill-only and decode-only) on plain torch streams. The
worker selects a group by the running decode batch size so a small decode batch
leaves most SMs for prefill while a large one shifts SMs toward decode — letting
prefill and decode run concurrently with hardware-enforced isolation.

SM partition counts come from :func:`divide_sm`, ported from sglang's
``pdmux_context.py`` with the architecture-specific granularity constraints
extended to SM10 (Blackwell / GB200). When green contexts are unavailable the
manager degrades to full-SM torch streams for every group, so selection still
works (with no real partitioning) and the caller needs no special-casing.

The feature is gated by worker runtime config; nothing here runs unless a
:class:`StreamManager` is constructed.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from ..nn.cuda.green_context import (
    GreenContextError,
    create_greenctx_streams,
    get_sm_available,
    green_contexts_available,
)

__all__ = [
    "get_arch_constraints",
    "divide_sm",
    "PerStreamGraphCache",
    "StreamManager",
]

logger = logging.getLogger(__name__)

# Decode must keep at least this many SMs to be worth a partition (sglang's floor).
_MIN_DECODE_SMS = 16


def get_arch_constraints(compute_capability: tuple[int, int]) -> tuple[int, int]:
    """``(min_sms_per_partition, sm_multiple)`` granularity for an SM arch.

    Ported from sglang ``pdmux_context.get_arch_constraints`` and extended to
    SM10 (Blackwell). SM9+ all share the 8/8 granularity; the driver's
    ``cuDevSmResourceSplitByCount`` enforces the true granularity regardless, so
    this only bounds which split *counts* are attempted.
    """
    major, _minor = compute_capability
    if major == 6:
        return 1, 1
    if major == 7:
        return 2, 2
    if major == 8:
        return 4, 2
    if major >= 9:
        # Hopper (9), Blackwell (10), and newer: 8-SM granularity.
        return 8, 8
    raise ValueError(f"Unsupported compute capability for SM split: {major}.{_minor}")


def divide_sm(
    total_sms: int,
    compute_capability: tuple[int, int],
    groups: int,
) -> list[tuple[int, int]]:
    """Partition SMs into ``groups`` ``(prefill_sm, decode_sm)`` divisions.

    Ported from sglang ``pdmux_context.divide_sm``: candidate prefill counts are
    multiples of the arch granularity with ``prefill >= decode`` and
    ``decode >= 16``; ``groups`` of them are sampled evenly and returned
    larger-prefill-first. Returns ``[]`` when ``groups <= 0`` or no valid
    partition exists.
    """
    if groups <= 0:
        return []
    min_per_part, multiple = get_arch_constraints(compute_capability)
    possible = [
        x
        for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
        if x >= total_sms - x and total_sms - x >= _MIN_DECODE_SMS
    ]
    if not possible:
        return []
    if len(possible) >= groups:
        step = max(1, len(possible) // groups)
        selected = possible[::step][:groups]
    else:
        selected = possible
    divisions = [(prefill, total_sms - prefill) for prefill in selected]
    divisions.reverse()  # larger prefill first
    return divisions


class PerStreamGraphCache:
    """Per-stream-group CUDA graph cache.

    Graphs captured on one green-context stream cannot replay on another, so each
    stream group keeps its own cache keyed by an arbitrary graph key. Capture is
    lazy (first use per group), amortizing cold-start cost.
    """

    def __init__(self, stream_idx: int) -> None:
        self.stream_idx = int(stream_idx)
        self.graphs: dict[Any, Any] = {}

    def get_or_capture(self, key: Any, capture_fn: Callable[[], Any]) -> Any:
        graph = self.graphs.get(key)
        if graph is None:
            graph = capture_fn()
            self.graphs[key] = graph
        return graph


class StreamManager:
    """Owns SM-partitioned CUDA stream groups backed by Green Contexts."""

    def __init__(
        self,
        gpu_id: int,
        *,
        sm_groups: int = 8,
        decode_bs_divisor: int = 36,
    ) -> None:
        self.gpu_id = int(gpu_id)
        self.decode_bs_divisor = max(1, int(decode_bs_divisor))
        self.current_idx = 0
        self._keepalive: list[Any] = []
        self.using_green_contexts = False
        try:
            self.total_sms = get_sm_available(self.gpu_id)
            capability = torch.cuda.get_device_capability(self.gpu_id)
        except Exception as exc:  # noqa: BLE001 - capability probe is best-effort.
            logger.warning("StreamManager: SM probe failed (%s); single full-SM group", exc)
            self.total_sms = 0
            self.stream_groups = [self._full_pair()]
            self.sm_counts = [(0, 0)]
            self.divisions: list[tuple[int, int]] = []
            self.graph_caches = [PerStreamGraphCache(0)]
            return

        self.divisions = divide_sm(self.total_sms, capability, max(0, sm_groups - 2))
        # SM_COUNTS layout (sglang): full-SM prefill-only, the green divisions,
        # then full-SM decode-only.
        self.sm_counts = [(self.total_sms, 0)]
        self.sm_counts.extend(self.divisions)
        self.sm_counts.append((0, self.total_sms))
        self.stream_groups = self._create_stream_groups()
        self.graph_caches = [PerStreamGraphCache(i) for i in range(len(self.stream_groups))]

    def _full_pair(self) -> tuple["torch.cuda.Stream", "torch.cuda.Stream"]:
        """A non-partitioned (full-SM) prefill/decode stream pair."""
        return (
            torch.cuda.Stream(device=self.gpu_id),
            torch.cuda.Stream(device=self.gpu_id),
        )

    def _create_stream_groups(self) -> list[tuple["torch.cuda.Stream", "torch.cuda.Stream"]]:
        groups: list[tuple[torch.cuda.Stream, torch.cuda.Stream]] = [self._full_pair()]
        can_partition = green_contexts_available() and self.divisions
        for prefill_sm, decode_sm in self.divisions:
            if can_partition:
                try:
                    gc = create_greenctx_streams(prefill_sm, decode_sm, self.gpu_id)
                    self._keepalive.append(gc)
                    groups.append((gc.prefill, gc.decode))
                    self.using_green_contexts = True
                    continue
                except GreenContextError as exc:
                    logger.warning(
                        "green-context split %d/%d SMs failed (%s); using full-SM streams",
                        prefill_sm, decode_sm, exc,
                    )
            groups.append(self._full_pair())
        groups.append(self._full_pair())
        return groups

    def select_idx(self, decode_batch_size: int) -> int:
        """Stream-group index for a given running decode batch size.

        0 decode → the prefill-only full-SM group; otherwise a green group whose
        decode partition grows with ``decode_batch_size`` (the last group is the
        decode-only full-SM endpoint, reserved for the heaviest decode).
        """
        n = len(self.stream_groups)
        if decode_batch_size <= 0 or n <= 2:
            return 0
        inner = n - 2  # number of green/partition groups between the endpoints
        bucket = decode_batch_size // self.decode_bs_divisor
        return 1 + min(inner, bucket)

    def select_streams(self, decode_batch_size: int) -> tuple["torch.cuda.Stream", "torch.cuda.Stream"]:
        """Select the ``(prefill, decode)`` stream pair for a decode batch size."""
        self.current_idx = self.select_idx(decode_batch_size)
        return self.stream_groups[self.current_idx]

    def graph_cache(self, idx: int | None = None) -> PerStreamGraphCache:
        return self.graph_caches[self.current_idx if idx is None else idx]

    def default_stream(self) -> "torch.cuda.Stream":
        """Full-SM stream for non-multiplexed (mixed / fused) execution."""
        return torch.cuda.default_stream(device=self.gpu_id)
