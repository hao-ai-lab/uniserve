"""Run the Denoiser capability: predictions, solver updates and ladders."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from uniserve.diffusion import Branch, DenoisingStep, Schedule, advance_
from uniserve.model import DenoiserInput, LatentInput
from uniserve.runtime import ExecutionContext, TensorBuffers
from uniserve.tensors import BufferConfig
from uniserve_worker._uniserve_ipc import DenoisingBuffers
from uniserve_worker.model_executor.cuda_graph import (
    CUDAGraphRunner,
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

    # The layout every step evaluates, one the runner has prepared.
    layout: Hashable
    # One typed input per solver step; ``inputs[i].step`` views
    # ``Schedule.step(i)`` of the ladder's schedules.
    inputs: tuple[DenoiserInput, ...]
    schedules: Mapping[str, Schedule]
    state: Mapping[str, torch.Tensor]
    # The request slot, 1-based; its bank row is ``slot - 1``.
    slot: int
    # The request's leading pages that hold the layout's samples.
    pages: tuple[int, ...]
    signature: Hashable
    # Tensor identity maps to an explicitly named field, never to a bank search.
    fields: Mapping[int, str]
    # Banked field name -> (element offset in the slot's bank row, shape).
    # Empty unless the runner captures.
    spans: Mapping[str, tuple[int, tuple[int, ...]]]
    # Per step, the tensors that are neither samples nor named in ``fields``
    # (the step index and timesteps among them), in one order for every
    # step; a replay copies the step's tensors into the graph's inputs.
    temporal: tuple[tuple[torch.Tensor, ...], ...]
    # The runner storage the latents view; a ladder serves only its runner.
    samples: torch.Tensor


class DiffusionRunner(ModelRunner):
    """Run one denoiser binding's diffusion computation.

    The denoiser's conditioning decides the form, and execution dispatches by
    that capability rather than by model:

    - A KV-conditioned denoiser, the text backbone attending to the request's
      KV prefixes, is its binding's batched entry. ``batch_forward``
      evaluates prediction rows of several requests and guidance branches,
      and ``integrate`` combines one request's branch predictions and applies
      the solver.
    - A standalone denoiser has one runner, built by ``for_layouts``, that
      serves every layout its worker admits. The runner's context is
      prepared for the largest layout. Smaller layouts borrow its workspace,
      samples, state buffers and shared scratch as compact leading views:
      the layouts' steps run one at a time on the runner's stream, and no
      step reads what another left behind. Each layout keeps only its own
      constants, sample page count and captured steps. A step gathers one
      request's samples from its committed pages of the latent pool,
      evaluates the fused prediction, solver update and pipeline feedback on
      them, and scatters the successor to the pages' other bank. ``bind``
      states a request's ladder over its slot of the request bank and its
      pages once; ``step`` advances it, replaying the layout's graph when
      ``capture`` made it resident and evaluating the same computation
      eagerly when the runner does not capture. A captured graph addresses
      the step through a device index and the pages and the slot through
      device indices, so one graph serves every step, slot and request of
      the layout, and every layout's graph allocates from the runner's one
      private pool.

    Both forms apply the solver through ``uniserve.diffusion.advance_``.
    """

    def __init__(
        self,
        *args,
        bank: Mapping[str, torch.Tensor] | None = None,
        slots: int = 0,
        pool: LatentPool | None = None,
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
        # [1] int64 device bank row a captured step gathers, set before each
        # capture and replay from pinned sources retained by Execution.
        self._slot_index = (
            torch.zeros(1, dtype=torch.int64, device=self.device)
            if self.captures
            else None
        )

        # A standalone denoiser's prepared layouts and the storage they
        # share: the samples, [pages * page_units, latent_width] in the
        # pool's dtype for the largest layout's pages, the workspace and the
        # captured steps' copies of banked state. ``for_layouts`` allocates
        # them with the context.
        self.pool = pool
        self.samples: torch.Tensor | None = None
        self._workspace: TensorBuffers | None = None
        self._state_buffers: TensorBuffers | None = None

    @classmethod
    def for_layouts(
        cls,
        name,
        call,
        maximum,
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
        """Prepare a standalone denoiser's runner for layouts up to a bound.

        ``maximum`` is the largest layout the runner will serve: every
        other layout's workspace and state must fit its extents dimension by
        dimension, and ``prepare`` must prepare it first; banked state
        fits the bank's slot rows. ``pages`` is the
        pool pages the runner's samples span, which covers the samples of
        every layout. ``devices`` lists the devices graphs may allocate on;
        without them, or without a ``stream``, the runner evaluates every
        step eagerly. Capturing runners borrow ``bank``, the request bank of
        ``slots`` rows their ladders gather state from. Every runner borrows
        ``pool``, whose leading pages of a request hold a layout's samples.
        Preparation, the shared storage included, is charged to ``storage``,
        whose budget it must fit; ``prepare`` adds each layout.
        """
        # The worker holds the denoiser's layout as an opaque size value. As
        # in every serving context, attention planning uses only the host
        # sequence lengths a layout's inputs carry and never copies them from
        # the device.
        context: ExecutionContext[object] = ExecutionContext(
            call.module,
            attention=attention,
            stream=stream,
            groups=call.groups,
            derive_host_lengths=False,
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
        )
        try:
            if pool.device != runner.device or not 0 < pages < pool.num_pages:
                raise ValueError(
                    "a denoiser runner requires layout pages of a pool on "
                    "its device"
                )
            # Preparation on the runner's stream follows work the caller has
            # already queued.
            if stream is not None:
                stream.wait(torch.cuda.current_stream(device))
            module = call.module
            query = getattr(module, "workspace_buffers", None)
            with storage.allocate(runner.execution):
                runner._workspace = TensorBuffers.allocate(
                    {} if query is None else query(maximum), device=device
                )
                runner.samples = torch.empty(
                    (pages * pool.page_units, pool.latent_width),
                    dtype=pool.dtype,
                    device=runner.device,
                )
                if runner.captures:
                    # One input buffer per banked field at the bank's slot row
                    # shape, the capacity every layout's field fits.
                    runner._state_buffers = TensorBuffers.allocate(
                        {
                            name: BufferConfig(
                                tuple(value.shape[1:]), value.dtype
                            )
                            for name, value in runner.bank.items()
                        },
                        device=device,
                    )
            runner.execution.configure_denoising(maximum)
            storage.check()
        except BaseException:
            runner.close()
            raise
        return runner

    @property
    def captures(self) -> bool:
        """Whether this runner may capture: it has a stream and graph pools."""
        return self.cuda_stream is not None and bool(self.execution.pools)

    def prepare(self, layout, *, pages: int) -> DenoisingBuffers:
        """Prepare one layout whose samples span ``pages`` pool pages.

        The first layout prepared must be the maximum ``for_layouts`` names;
        it prepares the context. Every later layout only fills its own
        constants and views the shared storage. A layout already prepared is
        returned as it is. Collective across the component's ranks, which
        prepare the same layouts in the same order.

        Raises:
            ValueError: The first layout is not the maximum, the layout's
                workspace does not fit the maximum's, or its pages exceed
                the runner's samples.
        """
        return self.execution.prepare_denoising(self, layout, pages)

    def _prepare_layout(self, layout, pages, first) -> DenoisingBuffers:
        """Allocate constants and borrow shared numerical buffers."""
        samples, pool = self.samples, self.pool
        # Empty sequence shards need no sample pages.
        if (
            samples is None
            or pool is None
            or self._workspace is None
            or not 0 <= pages * pool.page_units <= samples.shape[0]
        ):
            raise ValueError("a layout's samples exceed the runner's pages")

        module, context, device = (
            self.model,
            self.execution.context,
            self.device,
        )
        constants = getattr(module, "constant_buffers", None)
        workspace = getattr(module, "workspace_buffers", None)
        requirements = {} if constants is None else constants(layout)
        backing = TensorBuffers.allocate(requirements, device=device)
        rows = torch.empty((2, pages), dtype=torch.int64, device=device)
        try:
            constants = backing.view(requirements)
            workspace = self._workspace.view(
                {} if workspace is None else workspace(layout)
            )
            if first:
                # The maximum prepares the context's bindings; its constants
                # and workspace are this layout's own views.
                context.prepare(
                    layout, constants=backing, workspace=self._workspace
                )
            elif requirements:
                with context.activate():
                    module.prepare_constants(layout, out=constants)
        except BaseException:
            backing.close()
            raise
        return DenoisingBuffers(backing, constants, workspace, pages, rows)

    def retire(self, layout) -> None:
        """Release a prepared layout that has no captured step.

        Work queued on the runner's stream may still read the layout's
        constants, so the stream drains first. The maximum, which every
        layout's workspace views, and captured layouts stay prepared.

        Raises:
            ValueError: ``layout`` is the maximum or has a captured step.
        """
        self.execution.retire_denoising(layout)

    def layout(self, layout) -> DenoisingBuffers:
        """Return a prepared layout.

        Raises:
            KeyError: The runner has not prepared ``layout``.
        """
        return self.execution.denoising_layout(layout)

    def batch_forward(self, batch, *, padded=False):
        module, inputs = self.model, batch.inputs
        result = module(
            inputs,
            state={},
            constants=self.execution.context.constants,
            workspace=self.execution.context.workspace,
        )["image"]

        # Only the last pipeline stage returns predictions. Other stages
        # allocate placeholder storage that the broadcast from the last stage
        # overwrites, so every rank returns the same per-image values. A mesh
        # without a ``pp`` axis selects the rank-local group, whose broadcast
        # copies nothing.
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
            schedule.step(index),
        )

    def _call(self, entry: DenoisingBuffers, inputs, schedules, state):
        return DenoisingStep(
            self.model,
            inputs,
            schedules,
            state,
            entry.constants,
            entry.workspace,
        )

    def warmup(self, ladder: Ladder):
        """Prepare a layout's kernels, plans and scratch for its steps.

        Runs the ladder's first step eagerly, restores the samples it
        mutated, and synchronizes the runner's stream (or the device's
        current stream) on CUDA before returning; a layout already warm is
        left as it is. Every step of a layout evaluates the same shapes, so
        one warm step prepares all of them for capture.

        Every layout warms before any captures. The warm step's persistent
        allocations come from the runner's pool, whose free blocks every
        captured graph uses as intermediates; one made after a capture could
        land where that graph writes on each replay.

        Raises:
            RuntimeError: A graph is already resident and this layout is not
                warm.
        """
        self.execution.warm_denoising(self, ladder)

    def _warm_step(self, entry, ladder):
        """Warm numerical plans without changing the request's samples."""
        live = ladder.inputs[0]
        restore = restore_samples(live)
        try:
            self._call(entry, live, ladder.schedules, ladder.state)()
        finally:
            restore()

    def bind(
        self, layout, inputs, schedules, *, state, slot, pages: Sequence[int]
    ) -> Ladder:
        """Bind a request's complete ladder to its slot and its pages.

        ``layout`` is the prepared layout every step evaluates. Every latent
        must view the runner's samples. State field names come from the
        request bank: for each banked field, record its offset in the slot's
        row once and validate that the view lies in that row. ``pages`` is
        the request's page table, whose leading pages hold the layout's
        samples. Models receive only the numerical views.
        """
        samples, entry = self.samples, self.layout(layout)
        if samples is None or self.pool is None:
            raise ValueError("a ladder requires the runner's sample pages")
        if (
            not 1 <= slot <= self.slots
            or not inputs
            or len(pages) < entry.pages
        ):
            raise ValueError(
                "a ladder requires a valid slot, its layout's pages and "
                "solver steps"
            )
        inputs = tuple(inputs)
        # Each input names its step by a view of a schedule's step indices
        # (``Schedule.step``); the views must enumerate every solver step in
        # order.
        if any(
            len(inputs) != schedule.num_steps for schedule in schedules.values()
        ) or any(
            value.step.data_ptr()
            not in {
                schedule.step(index).data_ptr()
                for schedule in schedules.values()
            }
            for index, value in enumerate(inputs)
        ):
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
            layout,
            inputs,
            schedules,
            state,
            slot,
            tuple(int(page) for page in pages[: entry.pages]),
            signature,
            names,
            spans,
            temporal,
            samples,
        )

    def binds(self, ladder: Ladder) -> bool:
        """Whether ``ladder`` was bound over this runner's samples."""
        return self.execution.binds_denoising(self, ladder)

    def _state_views(self, ladder):
        """Bind compact input views for the fields gathered by a step graph."""
        state_buffers = cast(TensorBuffers, self._state_buffers)
        state_views = state_buffers.view(
            {
                name: BufferConfig(shape, self.bank[name].dtype)
                for name, (_, shape) in ladder.spans.items()
            }
        )
        gathers = []
        for name, (start, _) in ladder.spans.items():
            buffer = state_views[name]
            if buffer.numel():
                rows = self.bank[name].view(self.slots, -1)
                gathers.append(
                    (
                        rows[:, start : start + buffer.numel()],
                        buffer.view(1, -1),
                    )
                )
        return state_views, tuple(gathers)

    def _row_source(self, bank: int, pages: tuple[int, ...]) -> torch.Tensor:
        """Build [2, pages] indices for committed and successor rows."""
        pool = cast("LatentPool", self.pool)
        ids = torch.tensor(pages, dtype=torch.int64)
        value = torch.stack(
            (bank * pool.num_pages + ids, (1 - bank) * pool.num_pages + ids)
        )
        return value.pin_memory() if self.device.type == "cuda" else value

    def _copy_indices(self, entry, rows, slot):
        """Enqueue retained host indices on the active numerical stream."""
        entry.rows.copy_(rows, non_blocking=True)
        if slot is not None:
            cast(torch.Tensor, self._slot_index).copy_(slot, non_blocking=True)

    def _slot_source(self, slot):
        """Build the pinned [1] int64 source holding the zero-based bank row."""
        return torch.tensor([slot - 1], dtype=torch.int64).pin_memory()

    def _advance(
        self, entry: DenoisingBuffers, call, gathers=(), slot_index=None
    ):
        """Evaluate one step over the pages ``entry.rows`` names.

        Gathers the committed samples, and with ``gathers`` the slot's banked
        state the device ``slot_index`` names, evaluates ``call`` and
        scatters the successor samples. Eager steps and captured graphs run
        this same computation.
        """
        pool = cast("LatentPool", self.pool)
        rows, indices = pool.page_rows, entry.rows
        workspace = cast(torch.Tensor, self.samples)[
            : entry.pages * pool.page_units
        ].view(entry.pages, pool.page_units * pool.latent_width)
        if entry.pages:
            torch.index_select(rows, 0, indices[0], out=workspace)
        for bank_rows, buffer in gathers:
            torch.index_select(bank_rows, 0, slot_index, out=buffer)
        samples = call()
        if entry.pages:
            rows.index_copy_(0, indices[1], workspace)
        return samples

    def capture(self, ladder: Ladder) -> None:
        """Make the layout's step graph resident without advancing samples.

        The ladder's layout must be warmed (``warmup``). One graph serves
        every solver step, slot and request of the layout: the step index,
        timesteps and schedules are its inputs, which a replay copies from
        the ladder's step, and it gathers the slot the device slot index
        names and the pages the device rows name, so any request of the
        layout whose ladder has the same structure replays it. Capture is
        collective across the component's ranks and belongs to startup,
        while no request owns the ladder's pages; the owner retains a
        capturing runner for the worker's lifetime and checks the graph
        storage budget once its captures end.
        """
        self.execution.capture_denoising(self, ladder)

    def _capture_step(self, entry, ladder):
        """Construct fixed numerical inputs and capture their solver step."""
        live = ladder.inputs[0]
        slot_index = cast(torch.Tensor, self._slot_index)
        context = self.execution.context
        sources = ladder.temporal[0]
        with self.execution.storage.allocate(self.execution):
            temporal = clone_inputs((ladder.schedules, sources))
        replacements = {
            id(value): buffer
            for value, buffer in zip(sources, temporal[1], strict=True)
        }
        # Banked fields read the bucket's state buffers and the samples stay
        # the runner's own.
        samples = {
            id(value.tensor)
            for values in live.latents.values()
            for value in values
        }
        stepped = map_tensors(
            live,
            lambda value: (
                entry.state[ladder.fields[id(value)]]
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
            entry.state,
            entry.constants,
            entry.workspace,
        )

        def compute(_):
            return self._advance(entry, call, entry.gathers, slot_index)

        return CUDAGraphRunner.capture(
            context,
            temporal,
            compute,
            pools=self.execution.pools,
            restore=restore_samples(live),
            warm=False,
        )

    def step(self, ladder: Ladder, index: int, bank: int):
        """Advance a bound step; the worker commits request progress.

        ``bank`` holds the request's committed samples; the successor is
        written to the other bank of its pages. Returns the successor
        samples, which the runner's samples hold until its next step, and
        the execution path: ``"graph_replay"`` for a layout whose step
        ``capture`` made resident, and ``"eager"`` for any other layout,
        such as one prepared while serving, or for a runner without graphs.
        """
        return self.execution.step_denoising(self, ladder, index, bank)

    def _eager_step(self, entry, ladder, index):
        return self._advance(
            entry,
            self._call(
                entry, ladder.inputs[index], ladder.schedules, ladder.state
            ),
        )

    def _step_values(self, ladder, index):
        """Borrow the successor samples held until the runner's next call."""
        return {
            name: tuple(value.tensor for value in values)
            for name, values in ladder.inputs[index].latents.items()
        }

    def close(self):
        # Pinned copy sources retain their destination stream; release them
        # before the owner destroys that stream's execution partition. The
        # layouts' constants and the shared storage outlive the graphs and
        # the context that read them.
        try:
            super().close()
        finally:
            for backing in (self._workspace, self._state_buffers):
                if backing is not None:
                    backing.close()
            self._slot_index = None
            self.samples = self.pool = None
            self._workspace = self._state_buffers = None
            self.bank = {}
