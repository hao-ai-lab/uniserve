"""Persistent per-request tensor banks and borrowed slot views.

``RequestSlots`` backs the per-request state fields a model declares
(``state_buffers``) for every scheduler request slot until ``close``.
``RequestPool`` owns one instance; request execution borrows a slot's views
through ``tensors``, and ``DiffusionRunner`` borrows the device banks to
gather a slot's state by a device slot index.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig
from uniserve_worker.errors import invalid_descriptor


class RequestSlots:
    """Own persistent backing for scheduler-assigned request slots."""

    def __init__(
        self,
        capacity: int,
        *,
        state_buffers: Mapping[str, BufferConfig] | None,
        device: torch.device | str,
    ) -> None:
        """Allocate every declared field for ``capacity`` request slots.

        Args:
            capacity: Number of request slots; slot ids are one-based.
            state_buffers: Declared per-slot fields by name. ``None`` or an
                empty mapping allocates nothing, and ``tensors`` then raises.
            device: Device of the non-host fields. Host fields are pinned when
                it is a CUDA device.

        Raises:
            ValueError: ``capacity`` is below one.
        """
        size = int(capacity)
        if size < 1:
            raise ValueError("request storage capacity must be positive")
        self.capacity = size
        self._closed = False
        # Device state fields live in one bank per field with a leading slot
        # axis, so a captured graph can index a request's state, such as the
        # denoiser's tables and conditioning, through a device slot tensor
        # instead of baking a slot's addresses in. Row ``slot - 1`` of each
        # bank, at the field's capacity shape, belongs to request slot
        # ``slot``. Host fields stay one allocation per slot.
        self._bank: TensorBuffers | None = None
        self._host_slots: tuple[TensorBuffers, ...] = ()
        self.bank: Mapping[str, torch.Tensor] = {}
        self.tensor_slots: tuple[TensorBuffers, ...] = ()
        if state_buffers:
            pin_storage = torch.device(device).type == "cuda"
            device_fields = {
                name: config
                for name, config in state_buffers.items()
                if not config.host
            }
            host_fields = {
                name: config
                for name, config in state_buffers.items()
                if config.host
            }
            self._bank = TensorBuffers.allocate(
                {
                    name: BufferConfig(
                        (size, *config.shape),
                        config.dtype,
                        capacity_shape=(
                            size,
                            *(
                                config.shape
                                if config.capacity_shape is None
                                else config.capacity_shape
                            ),
                        ),
                    )
                    for name, config in device_fields.items()
                },
                device=device,
            )
            self.bank = {
                name: self._bank.backing(name) for name in device_fields
            }
            self._host_slots = tuple(
                TensorBuffers.allocate(
                    host_fields, device=device, pin_storage=pin_storage
                )
                for _ in range(size)
            )
            self.tensor_slots = tuple(
                TensorBuffers.from_tensors(
                    {
                        **{
                            name: self.bank[name][index]
                            for name in device_fields
                        },
                        **{
                            name: self._host_slots[index].backing(name)
                            for name in host_fields
                        },
                    }
                )
                for index in range(size)
            )

    def tensors(self, request_pool_idx: int) -> TensorBuffers:
        """Borrow a slot's tensors until its execution lease is retired.

        Raises ``RuntimeError`` after close, and ``invalid_descriptor`` when
        the one-based slot is out of range or no fields were declared.
        """
        if self._closed:
            raise RuntimeError("request storage is closed")
        slot = int(request_pool_idx)
        if not 1 <= slot <= self.capacity:
            raise invalid_descriptor("request storage slot exceeds capacity")
        if not self.tensor_slots:
            raise invalid_descriptor(
                "request has no declared persistent tensor storage"
            )
        return self.tensor_slots[slot - 1]

    def close(self) -> None:
        """Release slot views before their backing after all consumers drain."""
        if self._closed:
            return
        self._closed = True
        for slot in self.tensor_slots:
            slot.close()
        for slot in self._host_slots:
            slot.close()
        if self._bank is not None:
            self._bank.close()
        self.tensor_slots = ()
        self._host_slots = ()
        self.bank = {}
        self._bank = None
