"""Mutable batch execution state borrowing one bound numerical call."""

from uniserve.runtime.resources import close_resources

from .graphs import Execution


class Lane(Execution):
    """Own batch staging and graphs on a borrowed lane stream."""

    def __init__(
        self,
        name,
        call,
        device,
        kinds,
        stream,
        context,
        inputs,
        *,
        memory,
        devices,
    ):
        super().__init__(context, memory=memory, devices=devices)
        self.name, self.call, self.device = name, call, device
        self.call_kinds, self.cuda_stream = tuple(kinds), stream
        self.input_buffers = inputs

    def close(self):
        close_resources(super().close, self.input_buffers.close)
