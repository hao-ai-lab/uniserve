"""Execution-owned delivery at numerical modules placed on another device."""

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace

import torch

_ACTIVE = ContextVar("uniserve_transfers", default=None)


def _map(value, function):
    """Apply ``function`` to every tensor inside a nested argument structure."""

    if isinstance(value, torch.Tensor):
        return function(value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _map(getattr(value, field.name), function)
                for field in fields(value)
                if field.init
            },
        )
    if isinstance(value, Mapping):
        return {key: _map(member, function) for key, member in value.items()}
    if isinstance(value, tuple):
        mapped = tuple(_map(member, function) for member in value)
        return type(value)(*mapped) if hasattr(value, "_fields") else mapped
    if isinstance(value, list):
        return [_map(member, function) for member in value]
    return value


def _copy_out(value, out):
    """Return caller output storage after delivery from a remote module."""

    if isinstance(out, torch.Tensor):
        return out.copy_(value, non_blocking=out.device.type != "cpu")
    if isinstance(out, Mapping) and isinstance(value, Mapping):
        for name, destination in out.items():
            _copy_out(value[name], destination)
        return out
    raise TypeError("cross-device output storage must be a tensor or named tensors")


@dataclass
class _Call:
    target: torch.device | None
    scope: ExitStack
    origin: torch.cuda.Stream | None
    stream: torch.cuda.Stream | None
    out: object


class _Binding:
    def __init__(self, device, streams):
        self.device, self.streams = device, streams
        self.calls = ContextVar("uniserve_transfer_calls", default=())

    def prepare(self, args, kwargs):
        """Record the caller's devices, select the delivery stream, and move inputs."""

        devices = []

        def locate(value):
            devices.append(value.device)
            return value

        _map((args, kwargs), locate)
        target = devices[0] if devices else None

        origin = torch.cuda.current_stream() if self.device.type == "cuda" else None
        stream = origin
        if origin is not None and origin.device != self.device:
            key = (origin.device, origin.cuda_stream, self.device)
            stream = self.streams.get(key)
            if stream is None:
                # A graph can use a different root stream from eager warmup.
                # Its context still owns the same serialized destination stream.
                stream = next(
                    (value for value in self.streams.values() if value.device == self.device), None
                )
            if stream is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm cross-device numerical calls before capture")
                stream = torch.cuda.Stream(device=self.device)
            self.streams[key] = stream
            self.streams[(stream.device, stream.cuda_stream, origin.device)] = origin

        scope = ExitStack()
        self.calls.set((*self.calls.get(), _Call(target, scope, origin, stream, kwargs.get("out"))))
        if stream is not None:
            if stream != origin:
                stream.wait_stream(origin)
            scope.enter_context(torch.cuda.device(self.device))
            scope.enter_context(torch.cuda.stream(stream))

        return _map(
            (args, kwargs),
            lambda value: value.to(self.device, non_blocking=self.device.type != "cpu"),
        )

    def finish(self, result):
        """Return outputs to the caller's device and rejoin its origin stream."""

        calls = self.calls.get()
        if not calls:
            return result
        call = calls[-1]
        self.calls.set(calls[:-1])

        try:
            if call.target is None or result is None:
                return result
            if call.out is not None:
                return _copy_out(result, call.out)
            return _map(
                result, lambda value: value.to(call.target, non_blocking=call.target.type != "cpu")
            )
        finally:
            try:
                if call.origin is not None and call.stream != call.origin:
                    call.origin.wait_stream(call.stream)
            finally:
                call.scope.close()


class _Transfers:
    """Retain fork/join streams while models borrow only scoped delivery hooks.

    Hook closures contain an activation key, not an execution owner. The
    active context supplies their bindings; nested and independent contexts
    therefore use their own streams even when they share a module object.
    """

    def __init__(self, module, device):
        self.modules, self.bindings, self.streams = {}, {}, {}

        # A submodule inherits its parent's placement unless every parameter
        # (or, failing that, every buffer) agrees on one different device.
        inherited = {"": device}
        for path, child in module.named_modules(remove_duplicate=False):
            parent = inherited[path.rpartition(".")[0]]
            devices = {value.device for value in child.parameters() if not value.is_meta}
            if not devices:
                devices = {value.device for value in child.buffers() if not value.is_meta}
            target = next(iter(devices)) if len(devices) == 1 else parent
            inherited[path] = target
            if path and target != parent:
                self.modules[id(child)] = child
                self.bindings[id(child)] = _Binding(target, self.streams)

    @contextmanager
    def activate(self):
        if not self.bindings:
            yield
            return
        key = object()

        def prepare(module, args, kwargs):
            active = _ACTIVE.get()
            if active is not None and active[0] is key:
                return active[1][id(module)].prepare(args, kwargs)

        def finish(module, args, kwargs, result):
            active = _ACTIVE.get()
            if active is not None and active[0] is key:
                return active[1][id(module)].finish(result)

        with ExitStack() as scope:
            token = _ACTIVE.set((key, self.bindings))
            scope.callback(_ACTIVE.reset, token)
            for module in self.modules.values():
                scope.callback(
                    module.register_forward_pre_hook(prepare, prepend=True, with_kwargs=True).remove
                )
                scope.callback(
                    module.register_forward_hook(finish, with_kwargs=True, always_call=True).remove
                )
            yield

    def reset(self):
        """Release retired streams without replacing active binding identities."""
        self.streams.clear()

    def close(self):
        # Readers and graphs retire before their context. Stream destruction
        # introduces no additional synchronization in the numerical call path.
        self.bindings.clear()
        self.modules.clear()
        self.reset()
