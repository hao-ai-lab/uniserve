"""Run the Denoiser capability: predictions, solver updates and ladders."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from uniserve.diffusion import Branch, DenoisingStep, Schedule, advance_
from uniserve.model import DenoiserInput, LatentInput
from uniserve.runtime import ExecutionContext
from uniserve_worker.model_executor.cuda_graph import (
    CUDAGraphRunner,
    GraphBucket,
    Inputs,
    clone_inputs,
    input_signature,
    map_tensors,
)
from uniserve_worker.protocol.call import MediaCall

from .model_runner import ModelRunner
from .output import ExecutionOutput

if TYPE_CHECKING:
    from uniserve_worker.storage.latent_pool import LatentPool


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
class Ladder:
    """One request's bound denoising steps over its slot and its pages.

    The latents view the runner's samples, which each step fills from the
    request's committed pages and writes back to the pages' other bank. The
    other state fields are named spans of the request's slot. Tensor
    contents remain live. Changing static parameters or tensor layouts
    requires a new binding; each solver step already has its own typed input.
    """

    inputs: tuple[DenoiserInput, ...]
    schedules: Mapping[str, Schedule]
    state: Mapping[str, torch.Tensor]
    slot: int
    # The request's leading pages that hold the layout's samples.
    pages: tuple[int, ...]
    signature: Hashable
    # Tensor identity maps to an explicitly named field, never to a bank search.
    fields: Mapping[int, str]
    spans: Mapping[str, tuple[int, tuple[int, ...]]]
    temporal: tuple[tuple[torch.Tensor, ...], ...]
    # The runner storage the latents view; a ladder serves only its runner.
    samples: torch.Tensor


@dataclass
class LadderBucket(GraphBucket):
    """Slot staging and the captured steps shared by one ladder signature."""

    state: Mapping[str, torch.Tensor] = field(default_factory=dict)
    gathers: tuple = ()


class DiffusionRunner(ModelRunner):
    """Run one denoiser binding's diffusion computation.

    The denoiser's conditioning decides the form, and execution dispatches by
    that capability rather than by model:

    - A KV-conditioned denoiser, the text backbone attending to the request's
      KV prefixes, is its binding's batched entry. ``batch_forward``
      evaluates prediction rows of several requests and guidance branches,
      and ``integrate`` combines one request's branch predictions and applies
      the solver.
    - A standalone denoiser owns explicit constants and workspace, and one
      runner serves each prepared layout. The runner's samples hold one
      request's samples of the layout at a time: a step gathers them from
      the request's committed pages of the latent pool, evaluates the fused
      prediction, solver update and pipeline feedback on them, and scatters
      the successor to the pages' other bank. ``bind`` states a request's
      ladder over its slot of the request bank and its pages once; ``step``
      advances it, replaying the step's graph when ``capture`` made it
      resident and evaluating the same computation eagerly otherwise. A
      captured graph addresses the pages and the slot through device
      indices, so it serves every slot and every request of the layout.

    Both forms apply the solver through ``uniserve.diffusion.advance_``.
    """

    def __init__(
        self,
        *args,
        bank: Mapping[str, torch.Tensor] | None = None,
        slots: int = 0,
        pool: LatentPool | None = None,
        pages: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        bank = dict(bank or {})
        for name, value in bank.items():
            if value.ndim < 1 or value.shape[0] != slots:
                raise ValueError(
                    f"bank {name!r} requires {slots} contiguous slot rows"
                )
            if not value.is_contiguous():
                raise ValueError(f"bank {name!r} requires contiguous rows")
        self.bank, self.slots = bank, slots
        self._slot_index = (
            torch.zeros(1, dtype=torch.int64, device=self.device)
            if self.captures
            else None
        )
        self._slot_values: dict[int, torch.Tensor] = {}

        # A standalone denoiser's samples: ``pages`` pool pages, gathered
        # from the source rows and scattered to the target rows ``_rows``
        # names, [2, pages] int64. Both are allocated with the context.
        self.pool, self.pages = pool, int(pages)
        self.samples: torch.Tensor | None = None
        self._rows: torch.Tensor | None = None
        self._row_values: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}

    @classmethod
    def for_layout(
        cls,
        name,
        call,
        layout,
        *,
        device,
        stream,
        storage,
        devices=(),
        bank: Mapping[str, torch.Tensor] | None = None,
        slots: int = 0,
        pool: LatentPool,
        pages: int,
        attention="auto",
    ) -> DiffusionRunner:
        """Prepare a standalone denoiser's runner for one layout.

        ``devices`` lists the devices graphs may allocate on; without them,
        or without a ``stream``, the runner evaluates every step eagerly.
        Capturing runners borrow ``bank``, the request bank of ``slots`` rows
        their ladders gather state from. Every runner borrows ``pool``, whose
        leading ``pages`` pages of a request hold the layout's samples.
        Preparation, the runner's samples included, is charged to
        ``storage``, whose budget it must fit.
        """
        # The worker holds the denoiser's layout as an opaque size value.
        context: ExecutionContext[object] = ExecutionContext(
            call.module, attention=attention, stream=stream, groups=call.groups
        )
        runner = cls(
            name,
            call,
            device,
            (MediaCall.LATENT_PREPARATION, MediaCall.DENOISING),
            stream,
            context,
            storage=storage,
            devices=devices,
            bank=bank,
            slots=slots,
            pool=pool,
            pages=pages,
        )
        try:
            if pool.device != runner.device or not 0 < pages < pool.num_pages:
                raise ValueError(
                    "a denoiser runner requires layout pages of a pool on "
                    "its device"
                )
            if stream is not None:
                stream.wait(torch.cuda.current_stream(device))
            with storage.allocate(runner):
                context.prepare(layout)
                runner.samples = torch.empty(
                    (pages * pool.page_units, pool.latent_width),
                    dtype=pool.dtype,
                    device=runner.device,
                )
                runner._rows = torch.empty(
                    (2, pages), dtype=torch.int64, device=runner.device
                )
            storage.check()
        except BaseException:
            runner.close()
            raise
        return runner

    @property
    def captures(self) -> bool:
        """Whether this runner may capture: it has a stream and graph pools."""
        return self.cuda_stream is not None and bool(self.pools)

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

    def integrate(
        self,
        state,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        predictions: tuple[torch.Tensor, ...],
        index: int,
    ) -> None:
        """Advance one KV-conditioned request's sample by one step.

        ``predictions`` align with the guidance branches the request's
        ``DiffusionState`` selects for evaluation ``index``; they are combined
        into the guided prediction and ``sample``, at ``timestep``, is updated
        in place on the caller's stream.
        """
        (name,) = self.model.modalities
        schedule, guidance = state.schedules[name], state.guidance
        branches: tuple[Branch, ...] = guidance.branches(schedule, index)
        # Predictions may come from another device of the binding; a device
        # consumer keeps the copy ordered on the caller's stream.
        guided = guidance.combine(
            {
                branch: prediction.to(
                    sample.device, non_blocking=sample.device.type != "cpu"
                )
                for branch, prediction in zip(
                    branches, predictions, strict=True
                )
            },
            schedule,
            index,
        )
        advance_(
            self.model,
            {name: (LatentInput(sample, timestep),)},
            {name: (guided,)},
            state.schedules,
            index,
        )

    def _call(self, inputs, schedules, state):
        context = self.context
        return DenoisingStep(
            self.model,
            inputs,
            schedules,
            state,
            context.constants,
            context.workspace,
        )

    @torch.inference_mode()
    def warmup(self, inputs, schedules, *, state):
        """Prepare kernel specializations without advancing the samples."""
        context = self.context
        call = self._call(inputs, schedules, state)
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

    def bind(
        self, inputs, schedules, *, state, slot, pages: Sequence[int]
    ) -> Ladder:
        """Bind a request's complete ladder to its slot and its pages.

        Every latent must view the runner's samples. State field names come
        from the request bank: for each banked field, record its offset in
        the slot's row once and validate that the view lies in that row.
        ``pages`` is the request's page table, whose leading pages hold the
        layout's samples. Models receive only the numerical views.
        """
        samples = self.samples
        if samples is None or self.pool is None:
            raise ValueError("a ladder requires the runner's sample pages")
        if not 1 <= slot <= self.slots or not inputs or len(pages) < self.pages:
            raise ValueError(
                "a ladder requires a valid slot, its layout's pages and "
                "solver steps"
            )
        inputs = tuple(inputs)
        if any(value.step_index != index for index, value in enumerate(inputs)):
            raise ValueError(
                "ladder inputs must enumerate solver steps in order"
            )

        latents = {
            id(value.tensor): value.tensor
            for step in inputs
            for values in step.latents.values()
            for value in values
        }
        for tensor in latents.values():
            start = tensor.storage_offset() - samples.storage_offset()
            if (
                tensor.untyped_storage().data_ptr()
                != samples.untyped_storage().data_ptr()
                or tensor.dtype != samples.dtype
                or not tensor.is_contiguous()
                or start < 0
                or start + tensor.numel() > samples.numel()
            ):
                raise ValueError("every latent must view the runner's samples")

        spans, names = {}, {}
        if self.captures:
            for name, tensor in state.items():
                bank = self.bank.get(name)
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
        signature = (
            input_signature((inputs, schedules)),
            tuple(sorted(spans.items())),
        )
        # The samples and banked fields have fixed storage; the remaining
        # tensors, timesteps among them, are copied into graph inputs.
        temporal = tuple(
            tuple(
                value
                for _, value in Inputs(step).tensors
                if id(value) not in names and id(value) not in latents
            )
            for step in inputs
        )
        return Ladder(
            inputs,
            schedules,
            state,
            slot,
            tuple(int(page) for page in pages[: self.pages]),
            signature,
            names,
            spans,
            temporal,
            samples,
        )

    def binds(self, ladder: Ladder) -> bool:
        """Whether ``ladder`` was bound over this runner's samples."""
        return ladder.samples is self.samples

    def _bucket(self, ladder: Ladder) -> LadderBucket:
        bucket = self.buckets.get(ladder.signature)
        if bucket is None:
            stages, gathers = {}, []
            for name, (start, shape) in ladder.spans.items():
                with self.graph_storage.allocate(self):
                    stage = torch.empty_like(ladder.state[name])
                stages[name] = stage
                if stage.numel():
                    rows = self.bank[name].view(self.slots, -1)
                    gathers.append(
                        (
                            rows[:, start : start + stage.numel()],
                            stage.view(1, -1),
                        )
                    )
            bucket = LadderBucket(state=stages, gathers=tuple(gathers))
            self.buckets[ladder.signature] = bucket
        return cast(LadderBucket, bucket)

    def _row_value(self, bank: int, pages: tuple[int, ...]) -> torch.Tensor:
        """Return the pool rows one step reads and writes, [2, pages] int64.

        Row ``b * num_pages + page`` is ``page`` of bank ``b``: the first row
        of the pair reads the committed ``bank`` and the second writes the
        other one. Values are immutable and retained, so an asynchronous copy
        never outlives its source.
        """
        key = int(bank), pages
        value = self._row_values.get(key)
        if value is None:
            pool = cast("LatentPool", self.pool)
            ids = torch.tensor(pages, dtype=torch.int64)
            value = torch.stack(
                (bank * pool.num_pages + ids, (1 - bank) * pool.num_pages + ids)
            )
            if self.device.type == "cuda":
                value = value.pin_memory()
            self._row_values[key] = value
        return value

    def _advance(self, call, gathers=(), slot_index=None):
        """Evaluate one step over the pages ``_rows`` names.

        Gathers the committed samples, and with ``gathers`` the slot's banked
        state the device ``slot_index`` names, evaluates ``call`` and
        scatters the successor samples. Eager steps and captured graphs run
        this same computation.
        """
        rows = cast("LatentPool", self.pool).page_rows
        indices = cast(torch.Tensor, self._rows)
        staged = cast(torch.Tensor, self.samples).view(self.pages, -1)
        torch.index_select(rows, 0, indices[0], out=staged)
        for bank_rows, stage in gathers:
            torch.index_select(bank_rows, 0, slot_index, out=stage)
        samples = call()
        rows.index_copy_(0, indices[1], staged)
        return samples

    def _slot_value(self, slot):
        if slot not in self._slot_values:
            self._slot_values[slot] = torch.tensor(
                [slot - 1], dtype=torch.int64
            ).pin_memory()
        return self._slot_values[slot]

    @torch.inference_mode()
    def capture(self, ladder: Ladder, index: int) -> None:
        """Make one bound step's graph resident without advancing samples.

        The graph serves every slot: it gathers the slot the device slot
        index names and the pages the device rows name, so any request whose
        ladder has the same signature replays it. Capture is collective
        across the component's ranks and belongs to startup, while no request
        owns the ladder's pages; the owner retains a capturing runner for the
        worker's lifetime.
        """
        if not self.captures:
            raise RuntimeError("denoising graph capture requires a stream")
        bucket = self._bucket(ladder)
        if index in bucket.graphs:
            return
        live = ladder.inputs[index]
        self.warmup(live, ladder.schedules, state=ladder.state)
        slot_index = cast(torch.Tensor, self._slot_index)
        context = self.context
        with context.activate():
            # Bank tensors are gathered in the graph. Other numerical
            # tensors, including timesteps, have graph-owned copies.
            sources = ladder.temporal[index]
            with self.graph_storage.allocate(self):
                temporal = clone_inputs((ladder.schedules, sources))
            replacements = {
                id(value): staged
                for value, staged in zip(sources, temporal[1], strict=True)
            }
            # Banked fields read the bucket's staging and the samples stay
            # the runner's own.
            samples = {
                id(value.tensor)
                for values in live.latents.values()
                for value in values
            }
            stepped = map_tensors(
                live,
                lambda value: (
                    bucket.state[ladder.fields[id(value)]]
                    if id(value) in ladder.fields
                    else value
                    if id(value) in samples
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
            # Capture addresses the ladder's pages as a fresh trajectory's,
            # committed in bank one; replay sets the request's own rows.
            cast(torch.Tensor, self._rows).copy_(
                self._row_value(1, ladder.pages), non_blocking=True
            )
            slot_index.copy_(self._slot_value(ladder.slot), non_blocking=True)

            def compute(_):
                return self._advance(call, bucket.gathers, slot_index)

            bucket.graphs[index] = CUDAGraphRunner.capture(
                context,
                temporal,
                compute,
                pools=self.pools,
                restore=restore_samples(live),
            )
            self.graph_storage.check()

    @torch.inference_mode()
    def step(self, ladder: Ladder, index: int, bank: int):
        """Advance a bound step; the worker commits request progress.

        ``bank`` holds the request's committed samples; the successor is
        written to the other bank of its pages. Returns the successor
        samples, which the runner's samples hold until its next step, and
        the execution path, ``"graph_replay"`` when the step's graph is
        resident and ``"eager"`` otherwise.
        """
        if not self.binds(ladder):
            raise ValueError("the ladder was bound by another runner")
        live = ladder.inputs[index]
        bucket = self.buckets.get(ladder.signature)
        graph = None if bucket is None else bucket.graphs.get(index)
        context = self.context
        stream = context.stream
        if stream is not None:
            stream.wait(torch.cuda.current_stream(self.device))
        try:
            with context.activate():
                cast(torch.Tensor, self._rows).copy_(
                    self._row_value(bank, ladder.pages), non_blocking=True
                )
                if graph is None:
                    return self._advance(
                        self._call(live, ladder.schedules, ladder.state)
                    ), "eager"

                # The ladder's non-bank tensors have known positions. Keep
                # their correspondence per binding, including new schedule
                # objects.
                temporal = ladder.schedules, ladder.temporal[index]
                cast(torch.Tensor, self._slot_index).copy_(
                    self._slot_value(ladder.slot), non_blocking=True
                )
                graph.replay(temporal)
            return {
                name: tuple(value.tensor for value in values)
                for name, values in live.latents.items()
            }, "graph_replay"
        finally:
            # Activation has restored the caller's stream on this device.
            if stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(
                    stream.stream
                )

    def close(self):
        # Pinned copies retain their destination stream; release them before
        # the owner destroys that stream's execution partition.
        try:
            super().close()
        finally:
            self._slot_values.clear()
            self._row_values.clear()
            self._slot_index = None
            self.samples = self._rows = self.pool = None
            self.bank = {}
