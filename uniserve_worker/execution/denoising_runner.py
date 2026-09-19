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
        # One entry per resident ladder: one numerical signature, mapped to
        # the prepared size it borrows. A ladder serves every request slot,
        # because its graph reads and writes slot storage through the slot
        # index below rather than through a slot's addresses. Least recently
        # used ladders retire first.
        self._resident: OrderedDict[Hashable, Hashable] = OrderedDict()
        # Request slot storage: one bank per field with a leading slot axis,
        # bound by the owner of the request slots. Graph-owned stage tensors
        # per prepared size hold the slot a replay gathers, and the slot index
        # is the one device word a replay reads to find its rows.
        self._bank: dict[str, torch.Tensor] = {}
        self._stages: dict[Hashable, dict[Hashable, torch.Tensor]] = {}
        self._slot_index = (
            torch.zeros(1, dtype=torch.int64, device=device)
            if capture_stream is not None
            else None
        )
        self._slot_values: dict[int, torch.Tensor] = {}
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

    def bind_bank(self, bank: Mapping[str, torch.Tensor]) -> None:
        """Name the request slot storage captured graphs index into.

        Each value is one field's bank with a leading axis of `capacity` slot
        rows; a request's tensors are prefixes of its row. A graph gathers the
        rows the slot index names into its stage, runs the step there, and
        scatters the samples back, so one captured ladder serves every slot.
        """
        for name, value in bank.items():
            if value.ndim < 1 or value.shape[0] != self.capacity:
                raise ValueError(
                    f"slot bank {name!r} must hold {self.capacity} slot rows"
                )
            if not value.is_contiguous():
                raise ValueError(f"slot bank {name!r} must be contiguous")
        self._bank = dict(bank)

    def _locate(
        self, tensor: torch.Tensor
    ) -> tuple[str, int | None, torch.Tensor | None] | None:
        """Find the bank field and slot row a tensor is a span of.

        A slot tensor is one contiguous span of its row: a size's leading
        prefix, or a sequence-parallel rank's shard inside it. Returns the
        field, the slot row and every row's matching span as a 2-D view, or
        None for a tensor outside every bank.
        """
        pointer = tensor.untyped_storage().data_ptr()
        for name, bank in self._bank.items():
            if bank.untyped_storage().data_ptr() != pointer:
                continue
            if tensor.numel() == 0:
                # A rank whose shard of this field is empty holds no span of
                # any row: there is nothing to gather and nothing names a slot.
                return name, None, None
            rows = bank.view(bank.shape[0], -1)
            row_bytes = rows.shape[1] * bank.element_size()
            offset = tensor.data_ptr() - bank.data_ptr()
            row, within = divmod(offset, row_bytes)
            start = within // bank.element_size()
            if (
                within % bank.element_size()
                or not tensor.is_contiguous()
                or not 0 <= row < rows.shape[0]
                or start + tensor.numel() > rows.shape[1]
            ):
                raise ValueError(
                    f"slot tensor is not a contiguous span of a {name!r} row: "
                    f"tensor {tuple(tensor.shape)} {tensor.dtype} on "
                    f"{tensor.device} at {tensor.data_ptr():#x} with storage "
                    f"{pointer:#x}, bank {tuple(bank.shape)} at "
                    f"{bank.data_ptr():#x} with storage "
                    f"{bank.untyped_storage().data_ptr():#x}"
                )
            return name, row, rows[:, start : start + tensor.numel()]
        return None

    def _stage(
        self, input_key: Hashable, name: str, template: torch.Tensor
    ) -> torch.Tensor:
        """Return the graph-owned stage for one span of a field at one size.

        A field is staged once per span shape: a rank's latent shard and the
        state's view of the whole row are different spans of the same field,
        and each keeps its own stage so a ladder's addresses stay fixed.
        """
        stages = self._stages.setdefault(input_key, {})
        key = (name, tuple(template.shape), template.dtype)
        stage = stages.get(key)
        if stage is None:
            stage = torch.empty_like(template)
            stages[key] = stage
        return stage

    def _slot_value(self, slot: int) -> torch.Tensor:
        """Return the pinned host word naming one slot's bank row."""
        value = self._slot_values.get(slot)
        if value is None:
            if not 1 <= slot <= self.capacity:
                raise ValueError(f"request slot {slot} is outside the bank")
            value = torch.tensor([slot - 1], dtype=torch.int64).pin_memory()
            self._slot_values[slot] = value
        return value

    def _stage_inputs(self, inputs, state, input_key):
        """Rebind a call's slot tensors to the stage its graph runs on.

        Returns the rebound inputs and state, the slot row the inputs name,
        the gathers a replay performs before the step and the scatters it
        performs after: pairs of a bank's rows and the stage that holds one.
        """
        slots: dict[int, list[str]] = {}
        gathers = []
        scatters = []

        def staged(tensor, *, writeback):
            located = self._locate(tensor)
            if located is None:
                raise ValueError(
                    "denoising graph capture requires request slot storage "
                    "bound as a bank"
                )
            name, slot, span = located
            stage = self._stage(input_key, name, tensor)
            if span is None:
                return stage
            slots.setdefault(slot, []).append(name)
            pair = (span, stage.view(1, -1))
            gathers.append(pair)
            if writeback:
                scatters.append(pair)
            return stage

        def rebind(value):
            """Replace every bank-resident tensor in a field by its stage."""
            if isinstance(value, torch.Tensor):
                if self._locate(value) is None:
                    return value
                return staged(value, writeback=False)
            if isinstance(value, tuple):
                return tuple(rebind(item) for item in value)
            if isinstance(value, Mapping):
                return {name: rebind(item) for name, item in value.items()}
            return value

        latents = {
            name: tuple(
                replace(value, tensor=staged(value.tensor, writeback=True))
                for value in values
            )
            for name, values in inputs.latents.items()
        }
        # Conditioning and any other slot-resident input is gathered but not
        # written back; the latents are the state a step advances.
        others = {
            field.name: rebind(getattr(inputs, field.name))
            for field in fields(inputs)
            if field.name not in {"latents", "step_index", "sizes"}
        }
        bound = replace(inputs, latents=latents, **others)
        # State names the same slot storage the latents do; a graph reads the
        # stage that holds it, never a slot's own addresses.
        bound_state = {}
        for name, tensor in state.items():
            located = self._locate(tensor)
            if located is None:
                bound_state[name] = tensor
            else:
                bound_state[name] = staged(tensor, writeback=False)
        if len(slots) != 1:
            raise ValueError(
                "a denoising call binds the storage of exactly one slot, not "
                f"{slots}"
            )
        return bound, bound_state, next(iter(slots)) + 1, gathers, scatters

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
        bound, bound_state, bound_slot, gathers, scatters = self._stage_inputs(
            inputs, state, input_key
        )
        if slot != bound_slot:
            raise ValueError(
                f"denoising call names slot {slot} but binds slot {bound_slot}"
            )

        # The signature is taken over the stage the graph runs on, so it holds
        # for every slot. Schedules are rebuilt per trajectory, so their
        # values are copied into graph-owned storage on every invocation
        # instead of making addresses part of reuse.
        signature = (
            _numerical_signature(
                (
                    tuple(
                        (field.name, getattr(bound, field.name))
                        for field in fields(bound)
                        if field.name not in {"latents", "step_index"}
                    ),
                    tuple(
                        (name, tuple(value.tensor for value in values))
                        for name, values in bound.latents.items()
                    ),
                    bound_state,
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
        key = (signature, variant)

        missing = capture_required(
            key not in self.graphs, self.groups, self.device
        )
        if missing:
            self._admit(signature, input_key)
            self._discard_graph(key)
            self.warmup(inputs, schedules, state=state, input_key=input_key)

            graph = CUDAGraph(context=context, pools=self.device_pools)
            try:
                with context.activate():
                    # Bind graph-owned schedule and timestep copies so replay
                    # only needs their values refreshed, never new addresses.
                    staged = clone_inputs(temporal)
                    stepped = replace(
                        bound,
                        latents={
                            name: tuple(
                                replace(value, timestep=timestep)
                                for value, timestep in zip(
                                    values, staged[1][name], strict=True
                                )
                            )
                            for name, values in bound.latents.items()
                        },
                    )
                    step = DenoisingStep(
                        self.model,
                        stepped,
                        staged[0],
                        bound_state,
                        context.constants,
                        context.workspace,
                    )
                    slot_index = self._slot_index
                    assert slot_index is not None
                    slot_index.copy_(self._slot_value(slot), non_blocking=True)

                    def call():
                        # The slot index names the rows: gather them into the
                        # stage, run the step there, and scatter the samples
                        # back, all inside the graph.
                        for rows, stage in gathers:
                            torch.index_select(rows, 0, slot_index, out=stage)
                        samples = step()
                        for rows, stage in scatters:
                            rows.index_copy_(0, slot_index, stage)
                        return samples

                    restore = restore_samples(inputs)
                graph.capture(call, restore=restore)
            except BaseException:
                graph.close()
                raise
            self.graphs[key] = graph, staged

        self._resident.move_to_end(signature)
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
        """Make one ladder step's graph resident.

        Warmup calls this for every step of the production ladder at every
        declared size so no request pays capture on its own path; the graph
        serves every request slot. The caller's samples are unchanged when
        this returns.
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
            assert self._slot_index is not None
            self._slot_index.copy_(self._slot_value(slot), non_blocking=True)
            graph.replay()
        current.wait_stream(context.stream)
        # The samples live in the slot's rows, where the graph scattered them.
        samples = {
            name: tuple(value.tensor for value in values)
            for name, values in inputs.latents.items()
        }
        return samples, "graph_capture" if captured else "graph_replay"

    def _admit(self, signature: Hashable, input_key: Hashable) -> None:
        """Reserve residency for one ladder at one numerical signature.

        Residency spans every prepared size, so a ladder captured at startup
        survives until its prepared size retires.
        """
        if signature not in self._resident and len(self._resident) >= (
            self.shapes
        ):
            self._release_ladder(next(iter(self._resident)))
        self._resident[signature] = input_key

    def _release_ladder(self, signature: Hashable) -> None:
        """Retire every captured step of one ladder."""
        for key in tuple(self.graphs):
            if key[0] == signature:
                self._discard_graph(key)
        self._resident.pop(signature, None)

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

    def release_inputs(self, key: Hashable) -> None:
        """Retire dependent calls before releasing their execution context."""
        for ladder, resident_key in tuple(self._resident.items()):
            if resident_key == key:
                self._release_ladder(ladder)
        self._stages.pop(key, None)

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
