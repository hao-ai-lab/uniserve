"""Bind denoising trajectories once and own their prepared graph residency."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

import torch

from uniserve.diffusion import DenoisingStep, Schedule
from uniserve.distributed import Communicator
from uniserve.model import Denoiser, DenoiserInput
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve_worker.model_executor.component_binding import capture_required
from uniserve_worker.model_executor.cuda_graph import (
    CUDAGraphRunner,
    Execution,
    GraphBucket,
    Inputs,
    clone_inputs,
    input_signature,
    map_tensors,
)
from uniserve_worker.model_executor.graph_storage import GraphStorage

from .model_runner import ModelRunner
from .output import ExecutionOutput

InputT = TypeVar("InputT", bound=DenoiserInput)
SizeT = TypeVar("SizeT")


def restore_samples(inputs: DenoiserInput):
    """Snapshot mutable solver samples without retaining prediction scratch."""
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


@dataclass(frozen=True)
class Trajectory:
    """One request's fixed numerical calls and named spans of its slot storage.

    Tensor contents remain live. Changing static parameters or tensor layouts
    requires a new binding; each solver step already has its own typed input.
    """

    input_key: Hashable
    inputs: tuple[DenoiserInput, ...]
    schedules: Mapping[str, Schedule]
    state: Mapping[str, torch.Tensor]
    slot: int
    signature: Hashable
    # Tensor identity maps to an explicitly named field, never to a bank search.
    fields: Mapping[int, str]
    spans: Mapping[str, tuple[int, tuple[int, ...]]]
    temporal: tuple[tuple[torch.Tensor, ...], ...]


@dataclass
class DenoisingBucket(GraphBucket):
    """Shared solver stages and graph variants for one numerical trajectory."""

    state: Mapping[str, torch.Tensor] = field(default_factory=dict)
    gathers: tuple = ()
    scatters: tuple = ()


class TrajectoryRunner(Generic[InputT, SizeT]):
    """Own prepared contexts and their denoising graph buckets.

    A bound trajectory states its input correspondence once. Replays select a
    step and a slot, update live temporal values and gather/scatter that slot.
    Prepared sizes and graph buckets each have the configured ``shapes`` limit;
    evicting a bucket leaves its context usable for eager work or recapture.
    """

    def __init__(
        self,
        model: Denoiser[InputT, SizeT],
        *,
        device: torch.device,
        stream: torch.cuda.Stream | None,
        groups: tuple[Communicator, ...],
        capacity: int,
        capture: bool = True,
        graph_storage: GraphStorage | None = None,
        shapes: int = 1,
        cache: PrefixCache | None = None,
        attention="auto",
        matmul="auto",
        additional_devices=(),
    ):
        if capacity < 1 or shapes < 1:
            raise ValueError(
                "denoising requires positive slot and size capacities"
            )
        self.model, self.device, self.stream = model, device, stream
        self.groups, self.capacity, self.shapes = groups, capacity, shapes
        self._captures = capture and stream is not None
        self.graph_storage = (
            graph_storage if graph_storage is not None else GraphStorage()
        )
        self._devices = (device, *additional_devices) if self.captures else ()
        self.cache, self.attention, self.matmul = cache, attention, matmul
        self.prepared: OrderedDict[Hashable, Execution] = OrderedDict()
        self._bank: dict[str, torch.Tensor] = {}
        self._slot_index = (
            torch.zeros(1, dtype=torch.int64, device=device)
            if self.captures
            else None
        )
        self._slot_values: dict[int, torch.Tensor] = {}
        self._closed = False

    @property
    def captures(self):
        """Graph policy is independent of the granted execution stream."""
        return self._captures

    def prepare_inputs(self, key: Hashable, size: SizeT) -> ExecutionContext:
        """Prepare and retain one exact size through its dependent graphs."""
        if self._closed:
            raise RuntimeError("denoising runner is closed")
        if key not in self.prepared:
            if len(self.prepared) >= self.shapes:
                self.release_inputs(next(iter(self.prepared)))
            context = ExecutionContext(
                self.model,
                cache=self.cache,
                attention=self.attention,
                matmul=self.matmul,
                stream=self.stream,
                groups=self.groups,
            )
            entry = Execution(
                context, storage=self.graph_storage, devices=self._devices
            )
            try:
                if self.stream is not None:
                    self.stream.wait_stream(
                        torch.cuda.current_stream(self.device)
                    )
                with self.graph_storage.allocate(entry):
                    context.prepare(size)
                self.graph_storage.check()
            except BaseException:
                entry.close()
                raise
            self.prepared[key] = entry
        self.prepared.move_to_end(key)
        return self.prepared[key].context

    def _call(self, inputs, schedules, state, key):
        if self._closed:
            raise RuntimeError("denoising runner is closed")
        context = self.prepared[key].context
        return context, DenoisingStep(
            self.model,
            inputs,
            schedules,
            state,
            context.constants,
            context.workspace,
        )

    @torch.inference_mode()
    def warmup(self, inputs, schedules, *, state, input_key):
        """Prepare kernel specializations without advancing the samples."""
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

    def bind_bank(self, bank: Mapping[str, torch.Tensor]):
        """Borrow named request fields with a leading slot axis."""
        if self.prepared:
            raise RuntimeError(
                "bind request storage before preparing execution"
            )
        for name, value in bank.items():
            if (
                value.ndim < 1
                or value.shape[0] != self.capacity
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"bank {name!r} requires {self.capacity} "
                    "contiguous slot rows"
                )
        self._bank = dict(bank)

    def bind_inputs(self, input_key, inputs, schedules, *, state, slot):
        """Bind the complete typed solver trajectory to named request storage.

        State field names come from RequestPool/MediaBuilder. For each known
        field, record its row offset once and validate that the supplied view
        lies in the requested slot. Models receive only the numerical views.
        """
        if not 1 <= slot <= self.capacity or not inputs:
            raise ValueError(
                "a trajectory requires a valid slot and solver steps"
            )
        inputs = tuple(inputs)
        if any(value.step_index != index for index, value in enumerate(inputs)):
            raise ValueError(
                "trajectory inputs must enumerate solver steps in order"
            )
        spans, names = {}, {}
        if self.captures:
            for name, tensor in state.items():
                bank = self._bank.get(name)
                if bank is None:
                    continue
                row = bank[slot - 1]
                start = tensor.storage_offset() - row.storage_offset()
                if (
                    tensor.dtype != bank.dtype
                    or tensor.device != bank.device
                    or tensor.untyped_storage().data_ptr()
                    != bank.untyped_storage().data_ptr()
                    or not tensor.is_contiguous()
                    or start < 0
                    or start + tensor.numel() > row.numel()
                ):
                    raise ValueError(
                        f"state field {name!r} is outside its named slot"
                    )
                spans[name] = start, tuple(tensor.shape)
                names[id(tensor)] = name
            if any(
                id(value.tensor) not in names
                for inputs_at_step in inputs
                for values in inputs_at_step.latents.values()
                for value in values
            ):
                raise ValueError(
                    "every latent must name a supplied bank state field"
                )
        signature = (
            input_signature((inputs, schedules)),
            tuple(sorted(spans.items())),
        )
        temporal = tuple(
            tuple(
                value
                for _, value in Inputs(step).tensors
                if id(value) not in names
            )
            for step in inputs
        )
        return Trajectory(
            input_key,
            inputs,
            schedules,
            state,
            slot,
            signature,
            names,
            spans,
            temporal,
        )

    def _bucket(self, trajectory):
        entry = self.prepared[trajectory.input_key]
        bucket = entry.buckets.get(trajectory.signature)
        if bucket is None:
            # Context capacity and captured trajectory capacity are independent.
            # Entries own their stages and graphs; evict the oldest bucket.
            residents = [
                (bucket.last_used, owner, key)
                for owner in self.prepared.values()
                for key, bucket in owner.buckets.items()
            ]
            if len(residents) >= self.shapes:
                _, owner, key = min(residents, key=lambda value: value[0])
                owner.close_bucket(key)
            stages, gathers, scatters = {}, [], []
            mutable = {
                trajectory.fields[id(value.tensor)]
                for step in trajectory.inputs
                for values in step.latents.values()
                for value in values
            }
            for name, (start, shape) in trajectory.spans.items():
                with self.graph_storage.allocate(entry):
                    stage = torch.empty_like(trajectory.state[name])
                stages[name] = stage
                if stage.numel():
                    rows = self._bank[name].view(self.capacity, -1)
                    pair = (
                        rows[:, start : start + stage.numel()],
                        stage.view(1, -1),
                    )
                    gathers.append(pair)
                    if name in mutable:
                        scatters.append(pair)
            bucket = DenoisingBucket(
                state=stages,
                gathers=tuple(gathers),
                scatters=tuple(scatters),
            )
            entry.buckets[trajectory.signature] = bucket
        bucket.touch()
        return entry, bucket

    def _slot_value(self, slot):
        if slot not in self._slot_values:
            self._slot_values[slot] = torch.tensor(
                [slot - 1], dtype=torch.int64
            ).pin_memory()
        return self._slot_values[slot]

    def _graph(self, trajectory, index):
        if self._closed:
            raise RuntimeError("denoising runner is closed")
        entry, bucket = self._bucket(trajectory)
        context = entry.context
        slot_index = cast(torch.Tensor, self._slot_index)
        graph = bucket.graphs.get(index)
        missing = capture_required(graph is None, self.groups, self.device)
        if missing:
            if graph is not None:
                graph.close()
            live = trajectory.inputs[index]
            self.warmup(
                live,
                trajectory.schedules,
                state=trajectory.state,
                input_key=trajectory.input_key,
            )
            try:
                with context.activate():
                    # Bank tensors are gathered in the graph. Other numerical
                    # tensors, including timesteps, have graph-owned copies.
                    sources = trajectory.temporal[index]
                    with self.graph_storage.allocate(entry):
                        temporal = clone_inputs((trajectory.schedules, sources))
                    replacements = {
                        id(value): staged
                        for value, staged in zip(
                            sources, temporal[1], strict=True
                        )
                    }
                    stepped = map_tensors(
                        live,
                        lambda value: (
                            bucket.state[trajectory.fields[id(value)]]
                            if id(value) in trajectory.fields
                            else replacements[id(value)]
                        ),
                    )
                    call = DenoisingStep(
                        self.model,
                        stepped,
                        temporal[0],
                        bucket.state,
                        context.constants,
                        context.workspace,
                    )
                    slot_index.copy_(
                        self._slot_value(trajectory.slot), non_blocking=True
                    )

                    def compute(_):
                        for rows, stage in bucket.gathers:
                            torch.index_select(rows, 0, slot_index, out=stage)
                        samples = call()
                        for rows, stage in bucket.scatters:
                            rows.index_copy_(0, slot_index, stage)
                        return samples

                    graph = CUDAGraphRunner.capture(
                        context,
                        temporal,
                        compute,
                        pools=entry.pools,
                        restore=restore_samples(live),
                    )
                    bucket.graphs[index] = graph
                    self.graph_storage.check()
            except BaseException:
                self.release_inputs(trajectory.input_key)
                raise
        return graph, context, missing

    @torch.inference_mode()
    def capture(self, trajectory: Trajectory, index: int):
        """Capture one bound step without advancing the request's samples."""
        if not self.captures:
            raise RuntimeError("denoising graph capture requires a stream")
        self._graph(trajectory, index)

    @torch.inference_mode()
    def step(self, trajectory: Trajectory, index: int):
        """Advance a bound step; the worker commits request progress."""
        live = trajectory.inputs[index]
        if not self.captures:
            context, call = self._call(
                live,
                trajectory.schedules,
                trajectory.state,
                trajectory.input_key,
            )
            current = (
                torch.cuda.current_stream(self.device)
                if context.stream is not None
                else None
            )
            if context.stream is not None:
                context.stream.wait_stream(current)
            try:
                with context.activate():
                    return call(), "eager"
            finally:
                if context.stream is not None:
                    current.wait_stream(context.stream)
        graph, context, captured = self._graph(trajectory, index)
        current = torch.cuda.current_stream(self.device)
        context.stream.wait_stream(current)
        with context.activate():
            # The trajectory's non-bank tensors have known positions. Keep
            # their correspondence per binding, including new schedule objects.
            temporal = trajectory.schedules, trajectory.temporal[index]
            cast(torch.Tensor, self._slot_index).copy_(
                self._slot_value(trajectory.slot), non_blocking=True
            )
            graph.replay(temporal)
        current.wait_stream(context.stream)
        return {
            name: tuple(value.tensor for value in values)
            for name, values in live.latents.items()
        }, "graph_capture" if captured else "graph_replay"

    def release_inputs(self, key):
        """Retire one prepared size and every graph that borrows it."""
        entry = self.prepared.pop(key, None)
        if entry is not None:
            if self.device.type == "cuda":
                (
                    self.stream or torch.cuda.current_stream(self.device)
                ).synchronize()
            entry.close()

    def close(self, *, aborted: bool = False):
        if self._closed:
            return
        if aborted:
            from uniserve.runtime.execution import close_stream_collectives
            from uniserve.runtime.resources import retain_until_exit

            self._closed = True
            retain_until_exit(self)
            if self.stream is not None:
                close_stream_collectives(self.stream, aborted=True)
            return
        for key in tuple(self.prepared):
            self.release_inputs(key)
        # Pinned copies retain their destination stream; release them before
        # the caller destroys that stream's execution partition.
        self._slot_values.clear()
        self._slot_index = None
        self._bank.clear()
        self._closed = True


class DiffusionRunner(ModelRunner):
    """Evaluate image denoising inputs and distribute pipeline predictions."""

    def batch_forward(self, batch, *, padded=False):
        module, inputs = self.model, batch.inputs
        result = module(
            inputs,
            state={},
            constants=self.context.constants,
            workspace=self.context.workspace,
        )["image"]

        # Predictions come from the last pipeline stage; other stages
        # broadcast placeholder storage that the broadcast overwrites, so
        # every rank returns the same per-image values.
        pipeline = module.mesh.get_group(
            "pp" if "pp" in module.mesh.axes else ()
        )
        values = []
        for prediction, size in zip(result, inputs.sizes, strict=True):
            value = (
                prediction.tensor
                if prediction is not None
                else torch.empty(
                    module.latent_shape("image", size),
                    dtype=module.prediction_dtype,
                    device=self.device,
                )
            )
            pipeline.broadcast(value, src=pipeline.size - 1)
            values.append(value)
        return ExecutionOutput(tuple(values))
