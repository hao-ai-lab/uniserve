"""Numerical prediction, solver updates and denoiser graph inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from uniserve.diffusion import Branch, DenoisingStep, advance_
from uniserve.model import DenoiserInput, LatentInput
from uniserve_worker._uniserve_ipc import DenoisingBuffers
from uniserve_worker._uniserve_ipc import DiffusionRunner as _DiffusionRunner
from uniserve_worker.model_executor.cuda_graph import map_tensors

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


class DiffusionRunner(_DiffusionRunner):
    """Evaluate KV-conditioned predictions or standalone denoising steps.

    Rust owns layouts, request sequences, storage and graph dispatch. These
    methods bind numerical views and run the model with the same solver in
    eager execution and CUDA graphs.
    """

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

    def _call(self, buffers: DenoisingBuffers, inputs, schedules, state):
        return DenoisingStep(
            self.model,
            inputs,
            schedules,
            state,
            buffers.constants,
            buffers.workspace,
        )

    def _warm_step(self, buffers, sequence):
        """Warm numerical plans without changing the request's samples."""
        live = sequence.inputs[0]
        restore = restore_samples(live)
        try:
            self._call(buffers, live, sequence.schedules, sequence.state)()
        finally:
            restore()

    def _advance(
        self, buffers: DenoisingBuffers, call, gathers=(), slot_index=None
    ):
        """Evaluate one step over the pages ``buffers.rows`` names.

        Gathers the committed samples, and with ``gathers`` the slot's banked
        state the device ``slot_index`` names, evaluates ``call`` and
        scatters the successor samples. Eager steps and captured graphs run
        this same computation.
        """
        pool = cast("LatentPool", self.pool)
        rows, indices = pool.page_rows, buffers.rows
        workspace = cast(torch.Tensor, self.samples)[
            : buffers.pages * pool.page_units
        ].view(buffers.pages, pool.page_units * pool.latent_width)
        if buffers.pages:
            torch.index_select(rows, 0, indices[0], out=workspace)
        for bank_rows, buffer in gathers:
            torch.index_select(bank_rows, 0, slot_index, out=buffer)
        samples = call()
        if buffers.pages:
            rows.index_copy_(0, indices[1], workspace)
        return samples

    def _graph_forward(self, buffers, sequence, temporal):
        """Bind solver tensors and snapshot samples restored after capture."""
        live = sequence.inputs[0]
        slot_index = cast(torch.Tensor, self._slot_index)
        sources = sequence.temporal[0]
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
        fields = sequence.fields
        stepped = map_tensors(
            live,
            lambda value: (
                buffers.state[fields[id(value)]]
                if id(value) in fields
                else value
                if id(value) in samples
                else replacements[id(value)]
            ),
        )
        call = DenoisingStep(
            self.model,
            stepped,
            temporal[0],
            buffers.state,
            buffers.constants,
            buffers.workspace,
        )

        def compute(_):
            return self._advance(buffers, call, buffers.gathers, slot_index)

        return compute, restore_samples(live)

    def _eager_step(self, buffers, sequence, index):
        return self._advance(
            buffers,
            self._call(
                buffers,
                sequence.inputs[index],
                sequence.schedules,
                sequence.state,
            ),
        )
