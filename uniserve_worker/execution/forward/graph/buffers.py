"""Declarative graph buffer registry for stable tensor identity and refresh."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

import torch

from ....foundation.errors import invalid_descriptor

__all__ = [
    "ForwardGraphBufferRegistry",
    "ForwardGraphSlot",
    "PaddingPolicy",
    "SlotAxis",
]


class SlotAxis(StrEnum):
    TOKENS = "tokens"
    ROWS = "rows"
    SEGMENTS = "segments"
    BRANCHES = "branches"
    SCALAR = "scalar"
    OPAQUE = "opaque"


class PaddingPolicy(StrEnum):
    KEEP = "keep"
    ZERO = "zero"
    SENTINEL = "sentinel"
    COPY_HEAD = "copy_head"
    FILL_ONCE = "fill_once"
    CUSTOM = "custom"


@dataclass
class ForwardGraphSlot:
    name: str
    axis: SlotAxis
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    padding: PaddingPolicy = PaddingPolicy.KEEP
    sentinel: int | float = 0
    refresh: Callable[[torch.Tensor, torch.Tensor, int], None] | None = None
    pin_cpu: bool = False
    tensor: torch.Tensor | None = None
    cpu_staging: torch.Tensor | None = None

    def allocate(self) -> None:
        if self.tensor is None:
            self.tensor = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        if self.pin_cpu:
            try:
                self.cpu_staging = torch.empty(self.shape, dtype=self.dtype, device="cpu", pin_memory=True)
            except RuntimeError:
                self.cpu_staging = torch.empty(self.shape, dtype=self.dtype, device="cpu")


class ForwardGraphBufferRegistry:
    def __init__(self) -> None:
        self._slots: dict[str, ForwardGraphSlot] = {}
        self.copy_bytes: dict[str, int] = {}

    def register_slot(
        self,
        name: str,
        *,
        axis: SlotAxis | str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str,
        padding: PaddingPolicy | str = PaddingPolicy.KEEP,
        sentinel: int | float = 0,
        refresh: Callable[[torch.Tensor, torch.Tensor, int], None] | None = None,
        pin_cpu: bool = False,
    ) -> ForwardGraphSlot:
        if name in self._slots:
            raise invalid_descriptor(f"forward graph slot {name!r} already exists")
        slot = ForwardGraphSlot(
            name=name,
            axis=SlotAxis(axis),
            shape=tuple(int(v) for v in shape),
            dtype=dtype,
            device=torch.device(device),
            padding=PaddingPolicy(padding),
            sentinel=sentinel,
            refresh=refresh,
            pin_cpu=pin_cpu,
        )
        slot.allocate()
        self._slots[name] = slot
        return slot

    def slot(self, name: str) -> ForwardGraphSlot:
        try:
            return self._slots[name]
        except KeyError as exc:
            raise invalid_descriptor(f"unknown forward graph slot {name!r}") from exc

    def tensor(self, name: str) -> torch.Tensor:
        tensor = self.slot(name).tensor
        if tensor is None:
            raise invalid_descriptor(f"forward graph slot {name!r} is not allocated")
        return tensor

    def refresh_slot(self, name: str, value: torch.Tensor, *, raw_length: int | None = None) -> torch.Tensor:
        slot = self.slot(name)
        target = self.tensor(name)
        if value.dtype != slot.dtype:
            value = value.to(dtype=slot.dtype)
        if value.device != target.device:
            value = value.to(device=target.device, non_blocking=True)
        raw = int(raw_length if raw_length is not None else _head_length(value, target))
        if raw < 0 or raw > target.shape[0]:
            raise invalid_descriptor("forward graph slot refresh length exceeds slot capacity")
        if slot.refresh is not None:
            slot.refresh(target, value, raw)
        else:
            _copy_head(target, value, raw)
            _pad_tail(slot, target, raw)
        self.copy_bytes[name] = self.copy_bytes.get(name, 0) + int(value.element_size() * value.numel())
        return target

    def validate_geometry(self, name: str, value: torch.Tensor) -> None:
        slot = self.slot(name)
        if value.ndim != len(slot.shape):
            raise invalid_descriptor("forward graph slot rank mismatch")
        for actual, expected in zip(value.shape[1:], slot.shape[1:]):
            if int(actual) != int(expected):
                raise invalid_descriptor("forward graph slot static geometry mismatch")

    def batch_view(self) -> dict[str, torch.Tensor]:
        return {name: self.tensor(name) for name in self._slots}


def _head_length(value: torch.Tensor, target: torch.Tensor) -> int:
    if value.ndim == 0:
        return 1
    if target.ndim == 0:
        return 1
    return int(value.shape[0])


def _copy_head(target: torch.Tensor, value: torch.Tensor, raw: int) -> None:
    if target.ndim == 0:
        target.copy_(value.reshape(()))
        return
    if raw == 0:
        return
    target[:raw].copy_(value[:raw])


def _pad_tail(slot: ForwardGraphSlot, target: torch.Tensor, raw: int) -> None:
    if target.ndim == 0 or raw >= target.shape[0]:
        return
    tail = target[raw:]
    if slot.padding is PaddingPolicy.KEEP:
        return
    if slot.padding is PaddingPolicy.ZERO:
        tail.zero_()
        return
    if slot.padding is PaddingPolicy.SENTINEL:
        tail.fill_(slot.sentinel)
        return
    if slot.padding is PaddingPolicy.COPY_HEAD:
        if raw > 0:
            tail.copy_(target[raw - 1].expand_as(tail))
        return
    if slot.padding is PaddingPolicy.FILL_ONCE:
        return
    if slot.padding is PaddingPolicy.CUSTOM:
        if slot.refresh is None:
            raise invalid_descriptor("custom forward graph slot padding requires a refresh hook")
        return
