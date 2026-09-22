"""Publish physical tensors through the mechanisms their consumers need."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from uniserve_worker.errors import unsupported_setup
from uniserve_worker.protocol.transfer import Locator
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.vmm_pool import PoolExhaustedError

#: Mechanisms that carry a product where it lies on a device.
DEVICE_MECHANISMS = ("local", "cuda_vmm")
#: Mechanisms that carry a product as host bytes.
HOST_MECHANISMS = ("local", "shm", "channel")


if TYPE_CHECKING:
    import torch


def publish_tensor(
    transports: Mapping[str, Transport],
    source: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    retain: Callable[[concurrent.futures.Future[None]], None],
    offset: tuple[int, ...] | None = None,
    consumers: Sequence[int] = (),
    host: bool = False,
) -> tuple[Locator, ...]:
    """Publish one representation through the mechanisms its location needs.

    A device product is published where it lies, over the device mechanism
    of the rank's edges; a host product, or a device product the caller marks
    `host`, is published as host bytes over each host mechanism that reaches
    one of its named consumers: shared storage for a consumer on this host,
    the rank channel for one elsewhere. A device product that neither exports
    in place nor fits its device's pool falls back to host bytes for that
    product. A partial failure revokes all preceding locations. Each backend
    continues to retain the source until its submitted device work and
    readers retire.
    """
    if not transports:
        raise unsupported_setup(
            "tensor publication requires a configured transport"
        )
    first = source[0] if isinstance(source, tuple) else source
    device_product = first.is_cuda and not host
    names = DEVICE_MECHANISMS if device_product else HOST_MECHANISMS
    # A rank whose edges carry no device mechanism sends its device products
    # as bytes, the crossing any consumer off this device makes anyway.
    if device_product and "cuda_vmm" not in transports:
        names = HOST_MECHANISMS
    selected = [
        transports[name]
        for name in names
        if name in transports and transports[name].serves(consumers)
    ]
    locations: list[Locator] = []
    try:
        for transport in selected:
            try:
                location = transport.publish(
                    source, offset=offset, consumers=consumers
                )
            except PoolExhaustedError:
                # The product does not fit its device's pool: it travels as
                # host bytes instead, over every host mechanism this rank
                # publishes on.
                for fallback in HOST_MECHANISMS:
                    if (
                        fallback in transports
                        and fallback != "local"
                        and transports[fallback].serves(consumers)
                    ):
                        location = transports[fallback].publish(
                            source, offset=offset, consumers=consumers
                        )
                        locations.append(location)
                        retain(
                            transports[fallback].publication_retirement(
                                location
                            )
                        )
                continue
            locations.append(location)
            retain(transport.publication_retirement(location))
    except BaseException:
        for location in locations:
            transports[location.backend].release(location)
        raise
    return tuple(locations)
