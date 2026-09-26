"""Canvas readout around the public token-denoiser capability.

``CanvasRunner`` evaluates a ``TokenDenoiser`` once over staged canvas rows
and reads the log-probabilities of each answer slot's candidate tokens. The
vocabulary head projects only the slot rows, so no full-canvas logits are
materialized. With pipeline parallelism the last stage computes the
readout and broadcasts it, so every stage returns the same rows.
"""

from __future__ import annotations

import torch

from uniserve.diffusion.tokens import candidate_logprobs

from .model_runner import ModelRunner
from .output import ExecutionOutput


class CanvasRunner(ModelRunner):
    """Run token-denoising passes and their slot readout on one lane.

    Canvas passes run eagerly: a readout's slot and candidate counts vary
    per call, and the head that consumes them follows the backbone within
    one numerical call.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mesh = self.model.backbone.mesh
        self.pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())

    def batch_forward(self, batch, *, padded=False):
        """Denoise the staged canvases once and read their candidates.

        Returns one FP32 ``[candidates]`` value per canvas row: the natural
        log-probability of each candidate under the log-softmax over the
        full vocabulary of the soft-capped logits at its slot, in slot then
        candidate order.
        """
        inputs = batch.inputs
        hidden = self.model(inputs.canvas)

        count = int(inputs.selection.numel())
        if self.pipeline.rank == self.pipeline.size - 1:
            logits = self.model.compute_logits(
                hidden, token_indices=inputs.slot_tokens
            )
            # The softmax normalizes over the whole vocabulary, so tensor
            # shards gather their columns for the slot rows alone.
            values = (
                candidate_logprobs(logits.gather(), inputs.candidates)
                .reshape(-1)
                .index_select(0, inputs.selection)
            )
        else:
            values = torch.empty(
                (count,), dtype=torch.float32, device=hidden.device
            )
        self.pipeline.broadcast(values, src=self.pipeline.size - 1)
        return ExecutionOutput(tuple(values.split(inputs.row_candidates)))

    def select_graph_shape(self, batch, *, eligible):
        """Run every canvas pass eagerly; see the class description."""
        return None
