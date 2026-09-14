"""Bind neural branch input/output delivery to configured local devices."""

from __future__ import annotations

from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_map_only

from uniserve.attention.metadata import ExpertRoute
from uniserve.nn.branch import branch_role
from uniserve.runtime.device import canonical_device


def branch_device(
    module: nn.Module,
    path: str,
    *,
    device: torch.device | str,
    flow_device: torch.device | str | None = None,
) -> torch.device:
    """Resolve a module's inherited logical role against caller-owned placement."""

    selected = branch_role(module)
    owner = module
    for part in path.split(".") if path else ():
        owner = owner.get_submodule(part)
        selected = branch_role(owner) or selected
    return torch.device(flow_device if selected is ExpertRoute.FLOW and flow_device else device)


@dataclass(frozen=True)
class BranchCall:
    """One invocation's return destination and borrowed CUDA stream scopes."""

    target: torch.device | None
    scope: ExitStack
    origin: torch.cuda.Stream | None
    stream: torch.cuda.Stream | None


@dataclass(frozen=True)
class BranchBinding:
    """Bind device execution and ordered delivery for an ordinary module call.

    Each caller stream receives a separate destination stream, shared by the
    model's branches. Fork/join events preserve eager ordering and join all
    branch work into capture. The runtime retains streams through model use;
    invocation scopes contain no request state or persistent tensor addresses.
    """

    device: torch.device
    streams: dict[tuple[torch.device, int, torch.device], torch.cuda.Stream]
    _calls: ContextVar[tuple[BranchCall, ...]] = field(
        default_factory=lambda: ContextVar("branch_calls", default=())
    )

    def prepare(self, module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        values, _ = tree_flatten((args, kwargs))
        target = next((value.device for value in values if isinstance(value, torch.Tensor)), None)
        origin = torch.cuda.current_stream() if self.device.type == "cuda" else None
        stream = origin
        if origin is not None and origin.device != self.device:
            key = (origin.device, origin.cuda_stream, self.device)
            stream = self.streams.get(key)
            if stream is None:
                stream = torch.cuda.Stream(device=self.device)
                self.streams[key] = stream
                # A nested branch returning to its caller's GPU reuses the
                # enclosing stream, preserving the same execution ownership.
                self.streams[(stream.device, stream.cuda_stream, origin.device)] = origin
        scope = ExitStack()
        call = BranchCall(target, scope, origin, stream)
        self._calls.set((*self._calls.get(), call))
        if stream is not None and origin is not None:
            if stream != origin:
                stream.wait_stream(origin)
            scope.enter_context(torch.cuda.device(self.device))
            scope.enter_context(torch.cuda.stream(stream))

        def copy(value: torch.Tensor) -> torch.Tensor:
            return value.to(self.device, non_blocking=self.device.type != "cpu")

        return tree_map_only(torch.Tensor, copy, (args, kwargs))

    def finish(self, module: nn.Module, args: tuple, kwargs: dict, result: Any) -> Any:
        calls = self._calls.get()
        if not calls:
            return result
        call = calls[-1]
        self._calls.set(calls[:-1])
        try:
            target = call.target
            if target is None or result is None:
                return result

            def copy(value: torch.Tensor) -> torch.Tensor:
                return value.to(target, non_blocking=target.type != "cpu")

            return tree_map_only(torch.Tensor, copy, result)
        finally:
            try:
                if call.origin is not None and call.stream != call.origin:
                    assert call.stream is not None
                    call.origin.wait_stream(call.stream)
            finally:
                call.scope.close()


def bind_branches(
    module: nn.Module,
    *,
    device: torch.device | str,
    flow_device: torch.device | str | None = None,
) -> None:
    """Bind declared branches before numerical calls or graph capture.

    This does not move weights: checkpoint materialization establishes their
    placement. A model's parameter names and mathematical module graph remain
    unchanged. Bind once, while no caller can execute the module.
    """

    streams: dict[tuple[torch.device, int, torch.device], torch.cuda.Stream] = {}
    for child in module.modules():
        role = branch_role(child)
        if role is None:
            continue
        destination = flow_device if role is ExpertRoute.FLOW and flow_device else device
        binding = BranchBinding(canonical_device(destination), streams)
        child.register_forward_pre_hook(binding.prepare, prepend=True, with_kwargs=True)
        child.register_forward_hook(binding.finish, with_kwargs=True, always_call=True)
