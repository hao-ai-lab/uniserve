"""Numerical views borrowed from a process-local export."""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniserve import _slices
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import Locator
from uniserve_worker.transport.layout import read_destination, region_view

if TYPE_CHECKING:
    import torch


def _read_views(
    tensor: torch.Tensor | tuple[torch.Tensor, ...],
    locator: Locator,
    device: torch.device,
    destination: torch.Tensor | tuple[torch.Tensor, ...] | None,
    region: tuple[slice, ...] | None,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, ...],
    torch.Tensor | tuple[torch.Tensor, ...] | None,
]:
    """Borrow source spans and validate an optional copy destination."""
    first = tensor[0] if isinstance(tensor, tuple) else tensor
    if device != first.device:
        raise invalid_descriptor("local binding requires the source device")
    if region is not None:
        if not _slices.within(region, locator.shape):
            raise invalid_descriptor("read region exceeds the exported view")
        tensor = region_view(tensor, region)

    target = (
        None
        if destination is None
        else read_destination(locator, device, destination, region)
    )
    return tensor, target
