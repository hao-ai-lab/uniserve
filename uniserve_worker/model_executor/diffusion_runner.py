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
from uniserve.runtime import CUDAStream, ExecutionContext, PrefixCache
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

    Graphs are captured only through ``capture``, at startup, on pinned
    contexts, and stay resident until the runner closes. A step replays when
    its trajectory's ladder is resident and otherwise runs eagerly, so serving
    never captures: a request pays no capture latency and claims no graph
    storage, and every rank of the component takes the same path because the
    resident set is fixed before serving begins. Unpinned contexts serve
    eager work; at most ``shapes`` of them stay prepared, retired least
    recently used first.
    """

    def __init__(
        self,
        model: Denoiser[InputT, SizeT],
        *,
        device: torch.device,
        stream: CUDAStream | None,
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
        self._pinned: set[Hashable] = set()
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

    def prepare_inputs(
        self, key: Hashable, size: SizeT, *, pin: bool = False
    ) -> ExecutionContext:
        """Prepare one size's context and retain it under ``key``.

        A pinned context stays prepared, with the ladders ``capture`` makes
        resident on it, until the runner closes; pinning an unpinned context
        keeps it. Preparing an unpinned size beyond ``shapes`` retires the
        least recently used unpinned context.
        """
        if self._closed:
            raise RuntimeError("denoising runner is closed")
        if key not in self.prepared:
            unpinned = [
                name for name in self.prepared if name not in self._pinned
            ]
            if not pin and len(unpinned) >= self.shapes:
                self.release_inputs(unpinned[0])
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
                    self.stream.wait(torch.cuda.current_stream(self.device))
                with self.graph_storage.allocate(entry):
                    context.prepare(size)
                self.graph_storage.check()
            except BaseException:
                entry.close()
                raise
            self.prepared[key] = entry
        if pin:
            self._pinned.add(key)
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
            context.stream.wait(torch.cuda.current_stream(self.device))
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

    def _resident(self, trajectory, index):
        """Return the step's captured graph, or None when it has none."""
        entry = self.prepared.get(trajectory.input_key)
        bucket = (
            None if entry is None else entry.buckets.get(trajectory.signature)
        )
        return None if bucket is None else bucket.graphs.get(index)

    def _graph(self, trajectory, index):
        if self._closed:
            raise RuntimeError("denoising runner is closed")
        entry, bucket = self._bucket(trajectory)
        context = entry.context
        slot_index = cast(torch.Tensor, self._slot_index)
        graph = bucket.graphs.get(index)
        if graph is None:
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

    @torch.inference_mode()
    def capture(self, trajectory: Trajectory, index: int):
        """Make one bound step's graph resident without advancing samples.

        The graph serves every slot of the trajectory's pinned context: it
        gathers the slot the device slot index names, so any request whose
        trajectory has the same signature replays it. Capture is collective
        across the component's ranks and belongs to startup.
        """
        if not self.captures:
            raise RuntimeError("denoising graph capture requires a stream")
        if trajectory.input_key not in self._pinned:
            raise RuntimeError(
                "denoising graphs are captured on pinned contexts only"
            )
        self._graph(trajectory, index)

    @torch.inference_mode()
    def step(self, trajectory: Trajectory, index: int):
        """Advance a bound step; the worker commits request progress.

        Returns the samples and the execution path, ``"graph_replay"`` when
        the step's graph is resident and ``"eager"`` otherwise.
        """
        if self._closed:
            raise RuntimeError("denoising runner is closed")
        live = trajectory.inputs[index]
        graph = self._resident(trajectory, index) if self.captures else None
        if graph is None:
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
                context.stream.wait(current)
            try:
                with context.activate():
                    return call(), "eager"
            finally:
                if context.stream is not None:
                    current.wait_stream(context.stream.stream)
        context = self.prepared[trajectory.input_key].context
        current = torch.cuda.current_stream(self.device)
        context.stream.wait(current)
        with context.activate():
            # The trajectory's non-bank tensors have known positions. Keep
            # their correspondence per binding, including new schedule objects.
            temporal = trajectory.schedules, trajectory.temporal[index]
            cast(torch.Tensor, self._slot_index).copy_(
                self._slot_value(trajectory.slot), non_blocking=True
            )
            graph.replay(temporal)
        current.wait_stream(context.stream.stream)
        return {
            name: tuple(value.tensor for value in values)
            for name, values in live.latents.items()
        }, "graph_replay"

    def release_inputs(self, key):
        """Retire one prepared size and every graph that borrows it."""
        self._pinned.discard(key)
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
            from uniserve.runtime.resources import retain_until_exit

            # The borrowed stream's owner retires its communicators.
            self._closed = True
            retain_until_exit(self)
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
