"""Own prepared denoising calls.

graph residency and numerical resource lifetime.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Generic, TypeVar

import torch

from uniserve.diffusion import DenoisingStep, Schedule
from uniserve.distributed import Communicator
from uniserve.model import Denoiser, DenoiserInput
from uniserve.runtime import CUDAGraph, ExecutionContext, PrefixCache

from .graph_inputs import clone_inputs, copy_inputs, input_signature
from .model_entry import capture_required

InputT = TypeVar("InputT", bound=DenoiserInput)
SizeT = TypeVar("SizeT")


def _numerical_signature(value: object) -> Hashable:
    """Identify static values and tensor backing independently of tensor.

    contents.

    Immutable numerical records may be rebuilt between calls. Their values and
    borrowed tensor addresses determine graph reuse; mutable device contents do
    not, so this function never reads tensors back to the host.
    """
    if isinstance(value, torch.Tensor):
        return (
            value.device,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
            value.data_ptr(),
        )
    if isinstance(value, Mapping):
        return tuple(
            (key, _numerical_signature(item)) for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return tuple(_numerical_signature(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple(
            (field.name, _numerical_signature(getattr(value, field.name)))
            for field in fields(value)
        )
    if isinstance(value, Hashable):
        return value
    raise TypeError(
        f"unsupported numerical graph value: {type(value).__name__}"
    )


def restore_samples(inputs: DenoiserInput):
    """Snapshot solver samples while keeping disposable prediction storage.

    separate.
    """
    # Keying by identity snapshots each backing tensor once, even when several
    # latent views alias the same storage.
    samples = {
        id(value.tensor): value.tensor
        for values in inputs.latents.values()
        for value in values
    }
    snapshots = tuple((value, value.clone()) for value in samples.values())

    def restore():
        for value, saved in snapshots:
            value.copy_(saved)

    return restore


class DenoisingRunner(Generic[InputT, SizeT]):
    """Execute one denoiser capability over caller-supplied typed inputs.

    Prepared contexts own plans, constants, communication resources and scratch.
    Graphs borrow those contexts and request-slot samples, but own their
    schedule and timestep inputs. New requests can supply new schedule tensors
    without replacing a slot's captured sample addresses. Retirement releases
    graphs before their contexts or sample backing can be reused.
    """

    def __init__(
        self,
        model: Denoiser[InputT, SizeT],
        *,
        device: torch.device,
        capture_stream: torch.cuda.Stream | None,
        groups: tuple[Communicator, ...],
        capacity: int,
        shapes: int = 1,
        cache: PrefixCache | None = None,
        attention="auto",
        matmul="auto",
        additional_devices: tuple[torch.device, ...] = (),
    ):
        if capacity < 1:
            raise ValueError(
                "denoising requires a positive resident request capacity"
            )
        if shapes < 1:
            raise ValueError(
                "denoising requires a positive resident size capacity"
            )

        self.model, self.device = model, device
        self.capture_stream, self.groups, self.capacity = (
            capture_stream,
            groups,
            capacity,
        )
        self.shapes = shapes
        self.cache, self.attention, self.matmul = cache, attention, matmul

        # Captured graphs draw their workspace from per-device memory pools.
        self.device_pools = {}
        if capture_stream is not None:
            for target in dict.fromkeys((device, *additional_devices)):
                with torch.cuda.device(target):
                    self.device_pools[target] = torch.cuda.MemPool()

        self.graphs: dict[
            tuple[Hashable, Hashable, Hashable], tuple[CUDAGraph, tuple]
        ] = {}
        self.prepared_inputs: OrderedDict[Hashable, ExecutionContext] = (
            OrderedDict()
        )
        # One entry per resident ladder: a request slot's sample addresses at
        # one numerical signature, mapped to the prepared size it borrows.
        # Least recently used ladders retire first.
        self._resident: OrderedDict[tuple[Hashable, Hashable], Hashable] = (
            OrderedDict()
        )
        self._closed = False

    @property
    def captures(self) -> bool:
        """Whether this runner replays captured graphs.

        A runner without a capture stream evaluates every step eagerly.
        """
        return self.capture_stream is not None

    def prepare_inputs(
        self, key: Hashable, size: SizeT
    ) -> ExecutionContext[SizeT]:
        """Prepare one exact numerical size.

        retaining it through dependent graphs.
        """
        if self._closed:
            raise RuntimeError("denoising runner is closed")

        if key not in self.prepared_inputs:
            if len(self.prepared_inputs) >= self.shapes:
                self.release_inputs(next(iter(self.prepared_inputs)))

            context = ExecutionContext(
                self.model,
                cache=self.cache,
                attention=self.attention,
                matmul=self.matmul,
                stream=self.capture_stream,
            )
            try:
                if context.stream is not None:
                    context.stream.wait_stream(
                        torch.cuda.current_stream(self.device)
                    )
                context.prepare(size)
            except BaseException:
                context.close()
                raise
            self.prepared_inputs[key] = context

        self.prepared_inputs.move_to_end(key)
        return self.prepared_inputs[key]

    def _call(self, inputs, schedules, state, key):
        if self._closed:
            raise RuntimeError("denoising runner is closed")
        context = self.prepared_inputs[key]
        return context, DenoisingStep(
            self.model,
            inputs,
            schedules,
            state,
            context.constants,
            context.workspace,
        )

    @torch.inference_mode()
    def warmup(
        self,
        inputs: InputT,
        schedules: Mapping[str, Schedule],
        *,
        state: Mapping[str, torch.Tensor],
        input_key: Hashable,
    ) -> None:
        """Prepare the actual call's kernel specializations without advancing.

        its samples.
        """
        context, call = self._call(inputs, schedules, state, input_key)
        if context.stream is not None:
            context.stream.wait_stream(torch.cuda.current_stream(self.device))

        with context.activate():
            restore = restore_samples(inputs)
            try:
                call()
            finally:
                restore()

        if self.device.type == "cuda":
            (
                context.stream or torch.cuda.current_stream(self.device)
            ).synchronize()

    def _resident_ladder(
        self,
        inputs: InputT,
        schedules: Mapping[str, Schedule],
        *,
        state: Mapping[str, torch.Tensor],
        slot: Hashable,
        input_key: Hashable,
    ):
        """Return the graph one call replays, capturing it when it is missing.

        Yields the graph, its graph-owned temporal storage, the caller's
        temporal values, the prepared context and whether this call captured.
        Capture never advances the caller's samples.
        """
        if self._closed:
            raise RuntimeError("denoising runner is closed")

        context = self.prepared_inputs[input_key]

        # Request slots retain sample and conditioning backing. Schedules are
        # rebuilt per trajectory, so their values are copied into graph-owned
        # storage on every invocation instead of making addresses part of reuse.
        signature = (
            _numerical_signature(
                (
                    tuple(
                        (field.name, getattr(inputs, field.name))
                        for field in fields(inputs)
                        if field.name not in {"latents", "step_index"}
                    ),
                    {
                        name: tuple(value.tensor for value in values)
                        for name, values in inputs.latents.items()
                    },
                    state,
                    context.constants,
                    context.workspace,
                )
            ),
            input_signature(schedules),
        )
        timesteps = {
            name: tuple(value.timestep for value in values)
            for name, values in inputs.latents.items()
        }
        temporal = schedules, timesteps
        variant = inputs.step_index, input_signature(timesteps)
        key = (slot, signature, variant)

        missing = capture_required(
            key not in self.graphs, self.groups, self.device
        )
        if missing:
            self._admit((slot, signature), input_key)
            self._discard_graph(key)
            self.warmup(inputs, schedules, state=state, input_key=input_key)

            graph = CUDAGraph(context=context, pools=self.device_pools)
            try:
                with context.activate():
                    # Bind graph-owned schedule and timestep copies so replay
                    # only needs their values refreshed, never new addresses.
                    staged = clone_inputs(temporal)
                    bound = replace(
                        inputs,
                        latents={
                            name: tuple(
                                replace(value, timestep=timestep)
                                for value, timestep in zip(
                                    values, staged[1][name], strict=True
                                )
                            )
                            for name, values in inputs.latents.items()
                        },
                    )
                    call = DenoisingStep(
                        self.model,
                        bound,
                        staged[0],
                        state,
                        context.constants,
                        context.workspace,
                    )
                    restore = restore_samples(inputs)
                graph.capture(call, restore=restore)
            except BaseException:
                graph.close()
                raise
            self.graphs[key] = graph, staged

        self._resident.move_to_end((slot, signature))
        graph, staged = self.graphs[key]
        return graph, staged, temporal, context, missing

    @torch.inference_mode()
    def capture(
        self,
        inputs: InputT,
        schedules: Mapping[str, Schedule],
        *,
        state: Mapping[str, torch.Tensor],
        slot: Hashable,
        input_key: Hashable,
    ) -> None:
        """Make one ladder step's graph resident for a request slot.

        Warmup calls this for every step of the production ladder on every slot
        so no request pays capture on its own path. The caller's samples are
        unchanged when this returns.
        """
        if self.capture_stream is None:
            raise RuntimeError(
                "denoising graph capture requires a capture stream"
            )
        self._resident_ladder(
            inputs, schedules, state=state, slot=slot, input_key=input_key
        )

    @torch.inference_mode()
    def step(
        self,
        inputs: InputT,
        schedules: Mapping[str, Schedule],
        *,
        state: Mapping[str, torch.Tensor],
        slot: Hashable,
        input_key: Hashable,
    ) -> tuple[Mapping[str, tuple[torch.Tensor, ...]], str]:
        """Run one numerical update.

        the worker separately commits request progress.
        """
        if self.capture_stream is None:
            context, call = self._call(inputs, schedules, state, input_key)
            with context.activate():
                return call(), "eager"

        graph, staged, temporal, context, captured = self._resident_ladder(
            inputs, schedules, state=state, slot=slot, input_key=input_key
        )
        current = torch.cuda.current_stream(self.device)
        context.stream.wait_stream(current)

        with context.activate():
            copy_inputs(staged, temporal)
            result = graph.replay()
        current.wait_stream(context.stream)
        return result, "graph_capture" if captured else "graph_replay"

    def _admit(
        self, ladder: tuple[Hashable, Hashable], input_key: Hashable
    ) -> None:
        """Reserve residency for one slot's ladder at one numerical signature.

        Residency spans every slot at every prepared size, so a ladder captured
        at startup survives until its slot or its prepared size retires.
        """
        if ladder not in self._resident and len(self._resident) >= (
            self.capacity * self.shapes
        ):
            self._release_ladder(next(iter(self._resident)))
        self._resident[ladder] = input_key

    def _release_ladder(self, ladder: tuple[Hashable, Hashable]) -> None:
        """Retire every captured step of one slot's ladder."""
        slot, signature = ladder
        for key in tuple(self.graphs):
            if key[0] == slot and key[1] == signature:
                self._discard_graph(key)
        self._resident.pop(ladder, None)

    def _discard_graph(self, key):
        resident = self.graphs.pop(key, None)
        if resident is not None:
            graph, _staged = resident
            # Automatic residency eviction may follow an asynchronous replay.
            # Finish this producer before resetting its captured allocations.
            (
                graph.context.stream or torch.cuda.current_stream(self.device)
            ).synchronize()
            graph.close()

    def release_slot(self, slot: Hashable) -> None:
        """Retire a drained slot's graphs before its sample backing is.

        reused.
        """
        for ladder in tuple(self._resident):
            if ladder[0] == slot:
                self._release_ladder(ladder)

    def release_inputs(self, key: Hashable) -> None:
        """Retire dependent calls before releasing their execution context."""
        for ladder, resident_key in tuple(self._resident.items()):
            if resident_key == key:
                self._release_ladder(ladder)

        context = self.prepared_inputs.pop(key, None)
        if context is not None:
            if self.device.type == "cuda":
                (
                    context.stream or torch.cuda.current_stream(self.device)
                ).synchronize()
            context.close()

    def close(self) -> None:
        if self._closed:
            return

        for key in tuple(self.prepared_inputs):
            self.release_inputs(key)
        self.device_pools.clear()
        self._closed = True
