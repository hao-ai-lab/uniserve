"""Export physical tensors through the mechanisms their consumers need.

`export_tensor` is how storage owners (the KV cache, and the latent,
tensor-store and encoder-feature exports of `execution.transfer` and
`execution.image`) expose a product to the ranks that read it. It
selects backends from the rank's configured transports by the product's
location and by which consumers each backend reaches, and returns one
`Locator` per export made.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from uniserve_worker._uniserve_ipc import Completion
from uniserve_worker._uniserve_ipc import release_exports as release_exports
from uniserve_worker.errors import resource_error, unsupported_setup
from uniserve_worker.protocol.transfer import Locator
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.vmm_pool import PoolExhaustedError

ExportLocations = tuple[tuple[Transport, Locator], ...]

#: Mechanisms that carry a product where it lies on a device.
DEVICE_MECHANISMS = ("local", "cuda_vmm")
#: Mechanisms that carry a product as host bytes.
HOST_MECHANISMS = ("local", "shm", "channel")


if TYPE_CHECKING:
    import torch


def export_tensor(
    transports: Mapping[str, Transport],
    source: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    retain: Callable[[Completion], None],
    offset: tuple[int, ...] | None = None,
    consumers: Sequence[int] = (),
    host: bool = False,
) -> tuple[Locator, ...]:
    """Export one representation through the mechanisms its location needs.

    A device product is exported where it lies, over the device mechanism
    of the rank's edges; a host product, or a device product the caller marks
    `host`, is exported as host bytes over each host mechanism that reaches
    one of its named consumers: shared storage for a consumer on this host,
    the rank channel for one elsewhere. A device product that neither exports
    in place nor fits its device's pool falls back to host bytes for that
    product. Whether a location reaches a consumer over that consumer's
    transfer edge is decided by the engine when it binds the consuming call,
    which fails the requests reading a product left without one. A partial
    failure revokes all preceding locations. Each backend continues to retain
    the source until its submitted device work and readers retire.

    Args:
        transports: The rank's configured backends, keyed by mechanism name.
        source: The product, as one tensor or as ordered first-axis spans.
        retain: Receives each export's retirement signal, which
            completes once that backend has handed its storage back.
        offset: The product's offset within the logical tensor it belongs
            to, zero on every axis when omitted.
        consumers: Acknowledgment slots of the ranks that read the product;
            when empty, every bound mechanism for the product's location
            serves.
        host: Export a device product as host bytes.

    Returns:
        One locator per export, in the order they were made. It is
        empty when no configured mechanism for the product's location serves
        `consumers`.

    Raises:
        WorkerError: `unsupported_setup` when `transports` is empty, and
            `resource_error` when the product does not fit its device's pool
            and no other mechanism exported it. A backend's export
            error propagates after the locations already made are released.
    """
    if not transports:
        raise unsupported_setup("tensor export requires a configured transport")
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
                location = transport.export(
                    source, offset=offset, consumers=consumers
                )
            except PoolExhaustedError as exhausted:
                # The product does not fit its device's pool: it travels as
                # host bytes instead, over every host mechanism that serves
                # its consumers. `local` is skipped because it is also a
                # device mechanism, listed before `cuda_vmm`, so when bound it
                # has already exported this product where it lies.
                for fallback in HOST_MECHANISMS:
                    if (
                        fallback in transports
                        and fallback != "local"
                        and transports[fallback].serves(consumers)
                    ):
                        location = transports[fallback].export(
                            source, offset=offset, consumers=consumers
                        )
                        locations.append(location)
                        retain(transports[fallback].retirement(location))
                # A product with no location at all is unreadable by any
                # consumer, so the exhaustion is reported as the resource
                # failure it is rather than as an empty export.
                if not locations:
                    raise resource_error(
                        "device product does not fit its VMM pool and the "
                        "rank binds no other mechanism that exports it"
                    ) from exhausted
                continue
            locations.append(location)
            retain(transport.retirement(location))
    except BaseException:
        for location in locations:
            transports[location.backend].release(location)
        raise
    return tuple(locations)
