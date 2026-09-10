"""Tensor copies and neural calls on an explicitly bound in-process device."""

from __future__ import annotations

import torch


def tensor_to_device(value: torch.Tensor, device: torch.device | None) -> torch.Tensor:
    """Borrow a local tensor or copy it on the caller's stream to its declared device."""

    if device is None or value.device == device:
        return value
    # Host consumers can read the returned tensor immediately. A D2H copy must
    # finish before handing that storage to CPU operators or request assembly.
    return value.to(device, non_blocking=device.type != "cpu")


def call_on_device(
    module: torch.nn.Module,
    value: torch.Tensor,
    *,
    device: torch.device | None,
    target: torch.device | None = None,
) -> torch.Tensor:
    """Execute a unary neural module and restore its consumer's device."""

    result = module(tensor_to_device(value, device))
    if not isinstance(result, torch.Tensor):
        raise TypeError("neural sublayer must return a tensor")
    return tensor_to_device(result, value.device if target is None else target)
