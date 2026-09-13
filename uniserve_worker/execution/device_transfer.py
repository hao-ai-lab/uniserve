"""Tensor copies on an explicitly bound in-process device."""

from __future__ import annotations

import torch


def tensor_to_device(value: torch.Tensor, device: torch.device | None) -> torch.Tensor:
    """Borrow a local tensor or copy it on the caller's stream to its declared device."""

    if device is None or value.device == device:
        return value
    # Host consumers can read the returned tensor immediately. A D2H copy must
    # finish before handing that storage to CPU operators or request assembly.
    return value.to(device, non_blocking=device.type != "cpu")
