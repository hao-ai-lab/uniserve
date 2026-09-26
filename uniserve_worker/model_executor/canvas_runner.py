"""Canvas passes around the public token-denoiser capability.

``CanvasRunner`` evaluates a ``TokenDenoiser`` once over staged canvas rows.
A readout reads the log-probabilities of each answer slot's candidate tokens:
the vocabulary head projects only the slot rows, so no full-canvas logits are
materialized. With pipeline parallelism the last stage computes the readout
and broadcasts it, so every stage returns the same rows.

A generating canvas step denoises each row's resident canvas by one step of
block diffusion with the sampler of ``uniserve.diffusion.canvas``: rows at
step zero start their canvas before the pass reads it, and after the pass
the sampler scores the full-canvas logits, decides each row's next canvas
and stop, and writes the next pass's self-conditioning embedding. The
stepped state returns to the rows' request slots (``CanvasSlots``). Each row
reports its stop flag and its end-of-sequence truncated argmax canvas as one
int64 ``[1 + canvas]`` vector, which the batch copies to the host once.
"""

from __future__ import annotations

import torch

from uniserve.diffusion import canvas as sampler
from uniserve.diffusion.tokens import candidate_logprobs

from .input_batch import CanvasStepInput
from .model_runner import ModelRunner
from .output import ExecutionOutput


class CanvasRunner(ModelRunner):
    """Run token-denoising passes, their slot readout, and canvas steps.

    Canvas passes run eagerly: a readout's slot and candidate counts vary
    per call, and the head that consumes them follows the backbone within
    one numerical call; a canvas step runs the head and the sampler in
    chunks of whole canvases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mesh = self.model.backbone.mesh
        self.pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
        self.canvas_slots = None

    def bind_canvas_slots(self, slots) -> None:
        """Borrow the resident sampler state of generating canvases.

        The runner's denoiser must generate canvases on this rank
        (``storage.canvas_slots.generating_denoiser``).
        """
        self.canvas_slots = slots
        self.input_buffers.bind_canvas_slots(slots)

    def batch_forward(self, batch, *, padded=False):
        """Denoise the staged canvases once and read or step them.

        For a readout, returns one FP32 ``[candidates]`` value per canvas
        row: the natural log-probability of each candidate under the
        log-softmax over the full vocabulary of the soft-capped logits at its
        slot, in slot then candidate order. For canvas steps, returns one
        int64 ``[1 + canvas]`` value per row, as ``step`` describes.
        """
        inputs = batch.inputs
        if isinstance(inputs, CanvasStepInput):
            return ExecutionOutput(self.step(inputs))

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

    def step(self, inputs: CanvasStepInput) -> tuple[torch.Tensor, ...]:
        """Run one denoising step of every staged canvas.

        Rows at step zero start their canvas before the pass reads it. The
        head then projects the full canvases of at most
        ``CanvasSlots.step_rows`` rows at a time, all sharing their sampler
        constants, and the sampler steps them. The stepped state returns to
        the rows' slots.

        Returns one int64 ``[1 + canvas]`` vector per row: 1 when the step
        finished the row's block and 0 otherwise, followed by the block's
        argmax tokens, padded after its first end-of-sequence token.
        """
        slots = self.canvas_slots
        if slots is None:
            raise ValueError("canvas steps require bound sampler state")
        state = inputs.state
        rows, length = state.canvas.shape
        vocab = slots.vocab_size
        embedding = self.model.backbone.embedding.weight
        scale = self.model.backbone.embedding_scale

        sampler.start_canvas(state, vocab_size=vocab)
        hidden = self.model(inputs.canvas)

        results = torch.empty(
            (rows, 1 + length), dtype=torch.int64, device=hidden.device
        )
        for start, stop in _runs(inputs.sampling, slots.step_rows):
            count = stop - start
            positions = count * length
            # The full canvases of this run, FP32 [count * canvas, vocab].
            logits = self.model.compute_logits(
                hidden,
                token_indices=torch.arange(
                    start * length,
                    stop * length,
                    dtype=torch.int64,
                    device=hidden.device,
                ),
            ).gather()
            decision = sampler.CanvasDecision.empty(
                count, length, device=hidden.device
            )
            sampler.denoise_canvas(
                logits.view(count, length, vocab),
                embedding,
                scale,
                _rows(state, start, stop),
                inputs.sampling[start],
                scores=sampler.CanvasScores.empty(
                    count, length, device=hidden.device
                ),
                decision=decision,
                workspace=_chunk_workspace(slots.workspace, positions),
            )
            results[start:stop, 0] = decision.finished[:, 0]
            results[start:stop, 1:] = decision.tokens

        slots.commit(inputs.slots, inputs.views)
        return tuple(results.unbind(0))

    def kernels(self) -> list[dict[str, object]]:
        """The model's kernel records and the sampler's product algorithms.

        Generating canvases on CUDA adds one ``product`` record per step
        chunk shape the self-conditioning product has run: its positions
        (canvases times canvas length), vocabulary and hidden sizes, and the
        cuBLASLt algorithm it uses (``source`` ``table`` when a reproducible
        table chose it, ``measured`` when its first product timed the
        candidates, then the cuBLASLt version and the configuration).
        """
        records = super().kernels()
        slots = self.canvas_slots
        if slots is None or slots.workspace is None:
            return records
        workspace = slots.workspace
        if workspace.weights.device.type != "cuda":
            return records

        from uniserve_kernels.diffusion import canvas as kernels

        table = self.model.backbone.embedding.weight[: slots.vocab_size]
        for rows in range(1, slots.step_rows + 1):
            positions = rows * slots.canvas_length
            chunk = _chunk_workspace(workspace, positions)
            algorithm = kernels.product_algorithm(
                chunk.weights, table, chunk.product, chunk.scratch
            )
            if algorithm is not None:
                records.append(
                    {
                        "path": "canvas_sampler.self_conditioning",
                        "op": "product",
                        "positions": positions,
                        "vocab": slots.vocab_size,
                        "hidden": slots.hidden_size,
                        "provider": "cublaslt",
                        **algorithm,
                    }
                )
        return records

    def select_graph_shape(self, batch, *, eligible):
        """Run every canvas pass eagerly; see the class description."""
        return None


def _chunk_workspace(
    workspace: sampler.CanvasWorkspace, positions: int
) -> sampler.CanvasWorkspace:
    """The leading ``positions`` rows of a step workspace, for one chunk.

    The scratch keeps the size the CUDA product expects for the chunk's own
    shape, which keys its reproducible algorithm table, rather than the
    size of the largest chunk the workspace holds.
    """
    from uniserve_kernels.diffusion.canvas import product_scratch_bytes

    hidden = workspace.product.shape[1]
    return sampler.CanvasWorkspace(
        workspace.weights[:positions],
        workspace.normalizer[:positions],
        workspace.product[:positions],
        workspace.scratch[: product_scratch_bytes(positions, hidden)],
    )


def _rows(state: sampler.CanvasState, start: int, stop: int):
    """The sampler state of rows ``start:stop``, as contiguous views."""
    length = state.canvas.shape[1]
    return sampler.CanvasState(
        seed=state.seed[start:stop],
        block=state.block[start:stop],
        step=state.step[start:stop],
        canvas=state.canvas[start:stop],
        history=state.history[start:stop],
        self_conditioning=state.self_conditioning[
            start * length : stop * length
        ],
    )


def _runs(sampling, limit):
    """Split rows into runs of at most ``limit`` rows of equal sampling.

    The sampler takes one set of constants per call, so each run shares its
    rows' constants. Yields ``(start, stop)`` row ranges in row order.
    """
    start = 0
    for index in range(1, len(sampling) + 1):
        if (
            index == len(sampling)
            or index - start == limit
            or sampling[index] != sampling[start]
        ):
            yield start, index
            start = index
