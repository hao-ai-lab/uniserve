"""Encoder numerical calls and reusable text input storage."""

from __future__ import annotations

import torch

from uniserve.model import PatchEncoder
from uniserve.nn.vae import PatchAutoencoder
from uniserve.runtime.device import fill_cpu_ints
from uniserve.runtime.resources import close_resources
from uniserve_worker.errors import InputError
from uniserve_worker.storage.host_buffers import HostBuffers

from .model_runner import ModelRunner
from .output import ExecutionOutput


class EncoderRunner(ModelRunner):
    """Evaluate encoder inputs and retain their transfer backing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokens: torch.Tensor | None = None
        self._host: HostBuffers | None = None

    def prepare_tokens(self, tokens, *, capacity):
        """Copy a flat token sequence while retaining its host source fence."""
        if not 1 <= len(tokens) <= capacity:
            raise InputError("text encoder input exceeds its token capacity")
        if self._tokens is None or self._host is None:
            self._tokens = torch.empty(
                capacity, dtype=torch.int64, device=self.device
            )
            self._host = HostBuffers(
                (capacity,), dtype=torch.int64, depth=2, device=self.device
            )
        index, host = self._host.acquire()
        fill_cpu_ints(host, tokens)
        target = self._tokens[: len(tokens)]
        target.copy_(host[: len(tokens)], non_blocking=self._tokens.is_cuda)
        self._host.record_copy(index)
        return target.view(1, -1)

    def batch_forward(self, batch, *, padded=False):
        """Encode a homogeneous image batch through its declared capability."""
        if isinstance(self.model, PatchEncoder):
            return ExecutionOutput(self.model.encode(batch.inputs))
        if isinstance(self.model, PatchAutoencoder):
            return ExecutionOutput(
                tuple(
                    self.model.encode(torch.stack(batch.inputs.images)).unbind(
                        0
                    )
                )
            )
        raise InputError("encoder has no batched image capability")

    def close(self):
        close_resources(
            super().close, *(() if self._host is None else (self._host.close,))
        )
        self._tokens = self._host = None
