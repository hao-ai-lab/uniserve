"""Device-tensor staging engine for text forward batches.

This is the H2D staging seam: a small ring of pinned host staging buffers
(:class:`TextTensorStager`) plus the function that stages the text core of a
:class:`~uniserve_worker.contracts.forward_batch.ForwardBatch` (input_ids/
positions plus the extend/seq-len offset tensors) from an already-parsed
:class:`~uniserve_worker.contracts.batches.TextBatch`. The
:class:`~uniserve_worker.runtime.forward_batch_builder.ForwardBatchBuilder`
owns this ring and resolves the KV-residency indices on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ..contracts.batches import TextBatch
from ..contracts.forward_batch import ForwardBatch
from ..foundation.errors import invalid_descriptor
from .host_staging import (
    canonical_device as _canonical_device,
    copy_cpu_to_device,
    cpu_int_staging_buffer,
    fill_cpu_ints as _fill_cpu_long,
    is_pinned as _is_pinned,
)

__all__ = ["TextTensorStager", "TextTensorStagingSlot", "stage_text_forward_batch"]


class TextTensorStager:
    """Small ring of host staging buffers for non-blocking text H2D copies."""

    def __init__(self, *, ring_depth: int = 3) -> None:
        self.ring_depth = max(1, int(ring_depth))
        self._slots: list[dict[str, torch.Tensor]] = [{} for _ in range(self.ring_depth)]
        self._cursor = 0
        self._pin_memory_supported = True

    def next_slot(self) -> "TextTensorStagingSlot":
        slot = self._slots[self._cursor]
        self._cursor = (self._cursor + 1) % self.ring_depth
        return TextTensorStagingSlot(self, slot)

    def stage_text(
        self,
        text: "TextBatch",
        device: torch.device | str,
        *,
        stage_slot: "TextTensorStagingSlot | None" = None,
        input_ids_override: torch.Tensor | None = None,
        input_ids_replacements: Mapping[int, torch.Tensor] | None = None,
        positions_override: torch.Tensor | None = None,
        padded_num_tokens: int | None = None,
    ) -> "ForwardBatch":
        """Stage a parsed text view into device tensors using this ring's buffers."""
        return stage_text_forward_batch(
            text,
            device,
            stager=self,
            stage_slot=stage_slot,
            input_ids_override=input_ids_override,
            input_ids_replacements=input_ids_replacements,
            positions_override=positions_override,
            padded_num_tokens=padded_num_tokens,
        )

    def _long_buffer(
        self,
        slot: dict[str, torch.Tensor],
        name: str,
        numel: int,
        *,
        pin: bool,
    ) -> torch.Tensor:
        want_pin = bool(pin and self._pin_memory_supported)
        buf = slot.get(name)
        if buf is None or int(buf.numel()) < int(numel) or (_is_pinned(buf) != want_pin):
            try:
                buf = torch.empty(int(numel), dtype=torch.long, pin_memory=want_pin)
            except RuntimeError:
                self._pin_memory_supported = False
                buf = torch.empty(int(numel), dtype=torch.long)
            slot[name] = buf
        return buf[: int(numel)]

    def _int_buffer(
        self,
        slot: dict[str, torch.Tensor],
        name: str,
        numel: int,
        *,
        pin: bool,
    ) -> torch.Tensor:
        want_pin = bool(pin and self._pin_memory_supported)
        buf = slot.get(name)
        if (
            buf is None
            or int(buf.numel()) < int(numel)
            or buf.dtype != torch.int32
            or (_is_pinned(buf) != want_pin)
        ):
            try:
                buf = torch.empty(int(numel), dtype=torch.int32, pin_memory=want_pin)
            except RuntimeError:
                self._pin_memory_supported = False
                buf = torch.empty(int(numel), dtype=torch.int32)
            slot[name] = buf
        return buf[: int(numel)]

    @staticmethod
    def _device_key(name: str, dtype: torch.dtype, device: torch.device | str) -> str:
        dev = _canonical_device(device)
        return f"device:{dev.type}:{dev.index}:{str(dtype)}:{name}"

    def _device_buffer(
        self,
        slot: dict[str, torch.Tensor],
        name: str,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        device = _canonical_device(device)
        key = self._device_key(name, dtype, device)
        buf = slot.get(key)
        if (
            buf is None
            or int(buf.numel()) < int(numel)
            or buf.dtype != dtype
            or torch.device(buf.device) != device
        ):
            buf = torch.empty(int(numel), dtype=dtype, device=device)
            slot[key] = buf
        return buf[: int(numel)]


@dataclass(frozen=True)
class TextTensorStagingSlot:
    stager: TextTensorStager
    buffers: dict[str, torch.Tensor]

    def long_buffer(self, name: str, numel: int, *, pin: bool) -> torch.Tensor:
        return self.stager._long_buffer(self.buffers, name, numel, pin=pin)

    def int_buffer(self, name: str, numel: int, *, pin: bool) -> torch.Tensor:
        return self.stager._int_buffer(self.buffers, name, numel, pin=pin)

    def device_buffer(
        self,
        name: str,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        return self.stager._device_buffer(
            self.buffers,
            name,
            numel,
            dtype=dtype,
            device=device,
        )


def stage_text_forward_batch(
    text: TextBatch,
    device: torch.device | str,
    *,
    stager: TextTensorStager | None = None,
    stage_slot: TextTensorStagingSlot | None = None,
    input_ids_override: torch.Tensor | None = None,
    input_ids_replacements: Mapping[int, torch.Tensor] | None = None,
    positions_override: torch.Tensor | None = None,
    padded_num_tokens: int | None = None,
) -> ForwardBatch:
    """Stage one parsed :class:`TextBatch` into the text core of a ``ForwardBatch``.

    Builds the device tensors (input_ids/positions plus the extend/seq-len
    offset tensors) from an already-parsed text view. The host-side staging
    buffers are owned by :class:`TextTensorStager`; this is the H2D staging seam,
    kept separate from the model-neutral op views on :class:`UniForwardBatch`.
    The KV-residency indices (``block_table``/``out_cache_loc``/``cache_seqlens``)
    are resolved on top of this by the ``ForwardBatchBuilder``.
    """
    device = torch.device(device)
    lengths, starts, total_tokens = _text_lengths_and_starts(text)
    padded_total_tokens = total_tokens if padded_num_tokens is None else int(padded_num_tokens)
    if padded_total_tokens < total_tokens:
        raise invalid_descriptor(
            "padded_num_tokens must be greater than or equal to the non-padded text token count"
        )
    if stage_slot is None and stager is not None and device.type == "cuda":
        stage_slot = stager.next_slot()
    input_ids = _stage_input_ids(
        text,
        device,
        stage_slot=stage_slot,
        override=input_ids_override,
        total_tokens=total_tokens,
        padded_total_tokens=padded_total_tokens,
    )
    if input_ids_replacements:
        _apply_input_id_replacements(
            input_ids,
            input_ids_replacements,
            total_tokens=total_tokens,
            device=device,
        )
    positions = _stage_positions(
        text,
        lengths,
        device,
        stage_slot=stage_slot,
        override=positions_override,
        total_tokens=total_tokens,
        padded_total_tokens=padded_total_tokens,
    )
    index_tensors = _stage_text_index_tensors(text, lengths, starts, device, stage_slot)
    return ForwardBatch(
        forward_mode=text.mode,
        req_ids=text.req_ids,
        ops=text.ops,
        spec_token_ids=text.spec_token_ids,
        input_ids=input_ids,
        positions=positions,
        seq_lens=index_tensors["seq_lens"],
        extend_seq_lens=index_tensors["extend_seq_lens"],
        extend_start_loc=index_tensors["extend_start_loc"],
        extend_prefix_lens=index_tensors["extend_prefix_lens"],
        last_token_indices=index_tensors["last_token_indices"],
        num_token_non_padded=total_tokens,
        padded_num_tokens=padded_total_tokens,
    )


def _text_lengths_and_starts(text: TextBatch) -> tuple[list[int], list[int], int]:
    lengths = [len(tokens) for tokens in text.token_ids]
    if any(length <= 0 for length in lengths):
        raise invalid_descriptor("text forward ops must contain at least one token")
    starts: list[int] = []
    running = 0
    for length in lengths:
        starts.append(running)
        running += int(length)
    return lengths, starts, running


def _stage_input_ids(
    text: TextBatch,
    device: torch.device,
    *,
    stage_slot: TextTensorStagingSlot | None,
    override: torch.Tensor | None,
    total_tokens: int,
    padded_total_tokens: int,
) -> torch.Tensor:
    if override is not None:
        return _padded_override(
            override,
            total_tokens=total_tokens,
            padded_total_tokens=padded_total_tokens,
            device=device,
            slot=stage_slot,
            name="input_ids",
        )
    input_ids_cpu = _cpu_long_buffer(
        padded_total_tokens,
        pin=device.type == "cuda",
        slot=stage_slot,
        name="input_ids",
    )
    _fill_cpu_long(input_ids_cpu[:total_tokens], [int(token) for tokens in text.token_ids for token in tokens])
    if padded_total_tokens > total_tokens:
        input_ids_cpu[total_tokens:].zero_()
    return _copy_cpu_long_to_device(
        input_ids_cpu,
        device=device,
        non_blocking=device.type == "cuda" and _is_pinned(input_ids_cpu),
        slot=stage_slot,
        name="input_ids",
    )


def _stage_positions(
    text: TextBatch,
    lengths: list[int],
    device: torch.device,
    *,
    stage_slot: TextTensorStagingSlot | None,
    override: torch.Tensor | None,
    total_tokens: int,
    padded_total_tokens: int,
) -> torch.Tensor:
    if override is not None:
        return _padded_override(
            override,
            total_tokens=total_tokens,
            padded_total_tokens=padded_total_tokens,
            device=device,
            slot=stage_slot,
            name="positions",
        )
    positions_cpu = _cpu_long_buffer(
        padded_total_tokens,
        pin=device.type == "cuda",
        slot=stage_slot,
        name="positions",
    )
    flat_positions = [
        pos
        for pos_range, length in zip(text.pos_ranges, lengths)
        for pos in range(int(pos_range[0]), int(pos_range[0]) + int(length))
    ]
    _fill_cpu_long(positions_cpu[:total_tokens], flat_positions)
    if padded_total_tokens > total_tokens:
        positions_cpu[total_tokens:].zero_()
    return _copy_cpu_long_to_device(
        positions_cpu,
        device=device,
        non_blocking=device.type == "cuda" and _is_pinned(positions_cpu),
        slot=stage_slot,
        name="positions",
    )


def _stage_text_index_tensors(
    text: TextBatch,
    lengths: list[int],
    starts: list[int],
    device: torch.device,
    stage_slot: TextTensorStagingSlot | None,
) -> dict[str, torch.Tensor]:
    return {
        "extend_seq_lens": _tensor_from_ints(lengths, device=device, slot=stage_slot, name="extend_seq_lens"),
        "extend_start_loc": _tensor_from_ints(starts, device=device, slot=stage_slot, name="extend_start_loc"),
        "extend_prefix_lens": _tensor_from_ints(
            [pos[0] for pos in text.pos_ranges],
            device=device,
            slot=stage_slot,
            name="extend_prefix_lens",
        ),
        "seq_lens": _tensor_from_ints(
            [pos[1] for pos in text.pos_ranges], device=device, slot=stage_slot, name="seq_lens"
        ),
        "last_token_indices": _tensor_from_ints(
            [start + length - 1 for start, length in zip(starts, lengths)],
            device=device,
            slot=stage_slot,
            name="last_token_indices",
        ),
    }


def _cpu_long_buffer(
    numel: int,
    *,
    pin: bool,
    slot: TextTensorStagingSlot | None = None,
    name: str = "buffer",
) -> torch.Tensor:
    return cpu_int_staging_buffer(numel, dtype=torch.long, pin=pin, slot=slot, name=name)


def _tensor_from_ints(
    values: Sequence[int],
    *,
    device: torch.device | str,
    slot: TextTensorStagingSlot | None = None,
    name: str = "buffer",
) -> torch.Tensor:
    device = torch.device(device)
    cpu = _cpu_long_buffer(len(values), pin=device.type == "cuda", slot=slot, name=name)
    _fill_cpu_long(cpu, values)
    non_blocking = device.type == "cuda" and _is_pinned(cpu)
    return _copy_cpu_long_to_device(
        cpu,
        device=device,
        non_blocking=non_blocking,
        slot=slot,
        name=name,
    )


def _device_long_buffer(
    numel: int,
    *,
    device: torch.device,
    slot: TextTensorStagingSlot | None,
    name: str,
) -> torch.Tensor:
    if slot is not None and device.type == "cuda":
        return slot.device_buffer(name, numel, dtype=torch.long, device=device)
    return torch.empty(int(numel), dtype=torch.long, device=device)


def _padded_override(
    raw: torch.Tensor,
    *,
    total_tokens: int,
    padded_total_tokens: int,
    device: torch.device,
    slot: TextTensorStagingSlot | None,
    name: str,
) -> torch.Tensor:
    """Validate a caller override and pad its tail to ``padded_total_tokens``.

    Shared by the ``input_ids`` and ``positions`` override branches of
    ``build_text``: validate the override holds exactly ``total_tokens`` device
    tokens, then either return it as-is (no padding) or copy the real tokens into
    ``[:total_tokens]`` of a padded buffer and zero the ``[total_tokens:]`` tail.
    """
    validated = _validate_input_ids_override(
        raw,
        total_tokens=total_tokens,
        device=device,
        name=f"{name}_override",
    )
    if padded_total_tokens == total_tokens:
        return validated
    padded = _device_long_buffer(
        padded_total_tokens,
        device=device,
        slot=slot,
        name=name,
    )
    padded[:total_tokens].copy_(validated, non_blocking=True)
    padded[total_tokens:].zero_()
    return padded


_copy_cpu_long_to_device = copy_cpu_to_device


def _validate_input_ids_override(
    tensor: torch.Tensor,
    *,
    total_tokens: int,
    device: torch.device,
    name: str = "input_ids_override",
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise invalid_descriptor(f"{name} must be a tensor")
    if tensor.dtype != torch.long:
        raise invalid_descriptor(f"{name} must use torch.long dtype")
    if torch.device(tensor.device) != device:
        raise invalid_descriptor(
            f"{name} must be on {device}, got {tensor.device}"
        )
    if int(tensor.numel()) != int(total_tokens):
        raise invalid_descriptor(
            f"{name} has {int(tensor.numel())} tokens, expected {int(total_tokens)}"
        )
    return tensor.reshape(-1)


def _apply_input_id_replacements(
    input_ids: torch.Tensor,
    replacements: Mapping[int, torch.Tensor],
    *,
    total_tokens: int,
    device: torch.device,
) -> None:
    flat = input_ids.reshape(-1)
    for raw_idx, value in replacements.items():
        idx = int(raw_idx)
        if idx < 0 or idx >= int(total_tokens):
            raise invalid_descriptor(
                f"input_ids replacement index {idx} is outside 0..{int(total_tokens)}"
            )
        if not isinstance(value, torch.Tensor):
            raise invalid_descriptor("input_ids replacement values must be tensors")
        if value.dtype != torch.long:
            raise invalid_descriptor("input_ids replacement tensors must use torch.long dtype")
        if torch.device(value.device) != device:
            raise invalid_descriptor(
                f"input_ids replacement tensor must be on {device}, got {value.device}"
            )
        if int(value.numel()) != 1:
            raise invalid_descriptor("input_ids replacement tensors must contain one token")
        flat[idx:idx + 1].copy_(value.reshape(1), non_blocking=True)
