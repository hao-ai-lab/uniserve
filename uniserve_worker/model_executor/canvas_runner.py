"""Canvas passes around the public token-denoiser capability.

``CanvasRunner`` evaluates a ``TokenDenoiser`` once over a batch of canvas rows.
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
reports its outcome (``STEP_CONTINUED`` or ``STEP_STOPPED``) and its
end-of-sequence truncated argmax canvas as one int64 ``[1 + canvas]``
vector, which the batch copies to the host once. A step queued behind the
one that stopped its block runs as a no-op, decided on the device from its
slot's continuation flag: its state stays as the stopping step left it,
and it reports ``STEP_SKIPPED``.

On CUDA, startup captures graphs by canvas row count and call kind. Readout
graphs also bucket the canvas length, so shorter canvases do not evaluate
a full model canvas of padding per row. Every call replays the smallest
bucket that holds its canvases; after startup a call no captured bucket
holds fails. A canvas step graph holds the whole step: the pass, the head
and the sampler of
every chunk, and the commit of the stepped state, for the one sampling the
deployment serves; a call whose rows all start their canvas replays its own
graph, whose pass skips the self-conditioning signal, zero at step zero. A
readout graph holds the pass up to the final layer's attention output
(``TokenDenoiser.attend``), over the bucket's canvases' worth of tokens in
up to twice as many sequences, so it serves canvases of any length up to
the model's. The rest of the final layer, the output norm, the head and the
log-softmax over the vocabulary evaluate the slot rows alone
(``TokenDenoiser.finish``) in a readout tail graph per slot bucket
(``SLOT_BUCKETS``), which startup captures with the first readout bucket:
the slot rows of the attend state are gathered into the tail's fixed rows,
a call with more slots than the largest bucket replays it in chunks, and
one gather after each replay reads the call's candidates, whose number
varies from call to call. A tail that exchanges tokens with other ranks,
an expert-parallel final layer, runs eagerly instead, once over every slot
(``tail_exchanges``). Every sequence of a canvas graph, padding included,
is at most one canvas long, which bounds its attention launch.
Padding sequences read no prefix, and a padding step row starts a canvas in
the sentinel slot zero.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import torch

from uniserve.diffusion import canvas as sampler
from uniserve.diffusion.tokens import candidate_logprobs
from uniserve.model import CanvasInput, Logits
from uniserve.nn.attention import (
    AttentionBatch,
    BlockTable,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.nn.linear import VocabParallelEmbedding
from uniserve.nn.moe import FusedMoE
from uniserve_worker._uniserve_ipc import SLOT_BUCKETS
from uniserve_worker._uniserve_ipc import CanvasRunner as _CanvasRunner
from uniserve_worker.storage.canvas_slots import (
    STEP_CONTINUED,
    STEP_SKIPPED,
    CanvasSlots,
)

from .graph_inputs import (
    _copy_offsets,
    _fixed_view,
)
from .input_batch import CanvasStepInput, InputBatch, ReadoutInput
from .input_buffers import clear_padding, sampler_buffers
from .output import ExecutionOutput


class CanvasRunner(_CanvasRunner):
    """Run token-denoising passes, their slot readout, and canvas steps.

    Calls replay the graphs startup captures per row bucket (see the module
    description); without graph pools every call runs eagerly.
    """

    @staticmethod
    def sampler_bytes(denoiser, *, max_rows, history_depth, device_type):
        """Per-execution canvas input and sampler workspace bytes."""
        length = denoiser.canvas.length
        vocab = denoiser.lm_head.vocab.size
        embedding = denoiser.backbone.embedding
        buffers = sampler_buffers(
            max_rows=max_rows,
            canvas_length=length,
            history_depth=history_depth,
            hidden_size=embedding.embedding_dim,
            dtype=embedding.weight.dtype,
        )
        return sum(
            config.nbytes for config in buffers.values()
        ) + sampler.CanvasWorkspace.nbytes(
            CanvasSlots.step_rows(
                canvas_length=length, vocab_size=vocab, max_rows=max_rows
            ),
            length,
            vocab,
            embedding.embedding_dim,
            dtype=embedding.weight.dtype,
            device_type=device_type,
        )

    def batch_forward(self, batch, *, padded=False):
        """Denoise the input canvases once and read or step them.

        For a readout, returns one FP32 ``[candidates]`` value per canvas
        row: the natural log-probability of each candidate under the
        log-softmax over the full vocabulary of the soft-capped logits at its
        slot, in slot then candidate order. For canvas steps, returns one
        int64 ``[1 + canvas]`` value per row, as ``step`` describes.
        """
        inputs = batch.inputs
        if isinstance(inputs, CanvasStepInput):
            return ExecutionOutput(self.step(inputs))
        return self.readout(self.model.attend(inputs.canvas), inputs)

    def readout(self, state: tuple[torch.Tensor, ...], inputs: ReadoutInput):
        """Read the slots of ``inputs`` from the pass's ``attend`` state.

        ``state`` holds ``TokenDenoiser.attend``'s per-token state of at
        least the input canvas tokens, in the packed order
        ``inputs.slot_tokens`` indexes. The final layer's remainder, the
        output norm and the head evaluate the slot rows alone, gathered once
        for every slot of the call.
        """
        values = None
        if self.pipeline.rank == self.pipeline.size - 1:
            slots = self.model.finish(state, inputs.slot_tokens)
            logits = self.model.compute_logits(
                slots,
                token_indices=torch.arange(slots.shape[0], device=slots.device),
            )
            # The softmax normalizes over the whole vocabulary, so tensor
            # shards gather their columns for the slot rows alone.
            values = candidate_logprobs(
                cast(Logits, logits).gather(), inputs.candidates
            )
        return self._broadcast_readout(values, state, inputs)

    def _broadcast_readout(self, values, state, inputs: ReadoutInput):
        """Select the real candidates and share them across pipeline stages.

        ``values`` is the last stage's FP32 ``[slots, width]`` candidate
        log-probabilities, None on other stages, which receive them.
        """
        if values is None:
            values = torch.empty(
                (int(inputs.selection.numel()),),
                dtype=torch.float32,
                device=state[0].device,
            )
        else:
            values = values.reshape(-1).index_select(0, inputs.selection)
        self.pipeline.broadcast(values, src=self.pipeline.size - 1)
        return ExecutionOutput(tuple(values.split(inputs.row_candidates)))

    def step(self, inputs: CanvasStepInput) -> tuple[torch.Tensor, ...]:
        """Run one denoising step of every input canvas.

        Rows at step zero start their canvas before the pass reads it. The
        head then projects the full canvases of at most
        ``step_rows`` rows at a time, all sharing their sampler
        constants, and the sampler steps them. The stepped state returns to
        the rows' slots.

        Returns one int64 ``[1 + canvas]`` vector per row: ``STEP_STOPPED``
        when the step finished the row's block and ``STEP_CONTINUED``
        otherwise, followed by the block's argmax tokens, padded after its
        first end-of-sequence token. A row whose block an earlier step
        stopped (its slot's ``CanvasSlots.live`` flag is clear and it is not
        at step zero) is computed but not committed, and reports
        ``STEP_SKIPPED``.
        """
        slots = self.canvas_slots
        if slots is None:
            raise ValueError("canvas steps require bound sampler state")
        state = inputs.state
        rows, length = state.canvas.shape
        vocab = slots.vocab_size
        embedding = cast(
            VocabParallelEmbedding, self.model.backbone.embedding
        ).weight
        scale = self.model.backbone.embedding_scale

        # A queued step runs only while its block continues; step zero
        # starts a block and always runs. Decided on the device, so the call
        # never waits for the host to observe the step before it.
        active = (state.step == 0) | (
            slots.live.index_select(0, inputs.slots) != 0
        )

        sampler.start_canvas(state, vocab_size=vocab)
        # A canvas's first pass has a zero self-conditioning signal, whose
        # mixing is exactly the unweighted norm (``SelfConditioning``).
        hidden = self.model(
            replace(inputs.canvas, self_conditioning=None)
            if inputs.first
            else inputs.canvas
        )

        results = torch.empty(
            (rows, 1 + length), dtype=torch.int64, device=hidden.device
        )
        for start, stop in _runs(inputs.sampling, self.step_rows):
            count = stop - start
            positions = count * length
            # The full canvases of this run, FP32 [count * canvas, vocab].
            projection = self.model.compute_logits(
                hidden,
                token_indices=torch.arange(
                    start * length,
                    stop * length,
                    dtype=torch.int64,
                    device=hidden.device,
                ),
            )
            logits = cast(Logits, projection).gather()
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
                workspace=_chunk_workspace(self.sampler_workspace, positions),
            )
            results[start:stop, 0] = decision.finished[:, 0]
            results[start:stop, 1:] = decision.tokens

        # The commit masks zero targets, keeping skipped request rows and
        # shared padding storage untouched across concurrent microbatches.
        results[:, 0].masked_fill_(~active, STEP_SKIPPED)
        targets = torch.where(
            active, inputs.slots, torch.zeros_like(inputs.slots)
        )
        slots.commit(
            targets,
            inputs.views,
            live=(results[:, 0] == STEP_CONTINUED).to(slots.live.dtype),
        )
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

        # With graphs, the last pipeline stage's readout tails replay graphs
        # per slot bucket, or run eagerly when the tail exchanges tokens
        # with other ranks (``local_tail``).
        if (
            self.execution.pools
            and self.pipeline.rank == self.pipeline.size - 1
        ):
            records.append(
                {
                    "path": "readout_tail",
                    "op": "graph",
                    "provider": "cuda_graph",
                    "slot_buckets": list(SLOT_BUCKETS),
                }
                if self.local_tail
                else {
                    "path": "readout_tail",
                    "op": "graph",
                    "provider": "eager",
                    "reason": "expert exchange in the final layer",
                }
            )

        slots = self.canvas_slots
        if slots is None or self.sampler_workspace is None:
            return records
        workspace = self.sampler_workspace
        if workspace.weights.device.type != "cuda":
            return records

        from uniserve_kernels.diffusion import canvas as kernels

        embedding = cast(VocabParallelEmbedding, self.model.backbone.embedding)
        table = embedding.weight[: slots.vocab_size]
        for rows in range(1, self.step_rows + 1):
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

    def _tail(self, inputs):
        """Full-vocabulary log-probabilities of a tail's fixed slot rows.

        ``inputs`` holds the slot rows of the attend state and the identity
        row indices; rows past a call's live slots hold earlier values and
        their results are not read.
        """
        state, rows = inputs
        hidden = self.model.finish(state, rows)
        logits = self.model.compute_logits(hidden, token_indices=rows)
        return torch.log_softmax(cast(Logits, logits).gather().float(), dim=-1)


def tail_inputs(state, slots, device):
    """Allocate fixed slot rows outside other graphs' transient pools."""
    return (
        tuple(value.new_zeros((slots, *value.shape[1:])) for value in state),
        torch.arange(slots, dtype=torch.int64, device=device),
    )


def gather_tail(state, static, inputs, start, live):
    """Gather this chunk's selected attend rows into the tail's fixed inputs."""
    rows = inputs.slot_tokens[start : start + live]
    for value, buffer in zip(state, static[0], strict=True):
        torch.index_select(value, 0, rows, out=buffer[:live])


def tail_candidates(normalized, inputs, start, live):
    """Read candidate log probabilities from a replayed tail chunk."""
    return normalized[:live].gather(-1, inputs.candidates[start : start + live])


def tail_exchanges(model, exchange) -> bool:
    """Whether a token denoiser's readout tail exchanges tokens across ranks.

    The tail completes the final layer, the output norm and the head for a
    readout's slot rows (``TokenDenoiser.finish``). An expert-parallel
    ``FusedMoE`` in the final layer exchanges tokens with its expert group,
    whose ranks read different calls and must join each exchange once per
    expert step at the step's agreed capacity; a captured tail would carry
    its capture capacity, and chunked replays would add exchanges the other
    ranks never join. Tensor-parallel collectives in the tail are joined
    alike by ranks that read the same slots, so they keep the tail local.
    """
    if exchange is None:
        return False
    final = next(reversed(model.backbone.layers.values()))
    return any(
        isinstance(module, FusedMoE)
        and (exchange.attention_ranks or module.expert_group.size > 1)
        for module in final.modules()
    )


def _pad_attention(
    attention: AttentionBatch,
    padding: tuple[int, ...],
    width: int,
    widths: tuple[int, ...],
    buffers,
):
    """Append padding sequences to canvas attention inputs in place.

    ``padding`` holds the query tokens of each padding sequence, which reads
    no prefix, table unit zero from page zero and its whole own sequence;
    every table shares the query and prefix columns, which are padded once.
    ``width``, the fixed extent of the current-sequence visibility, bounds
    every sequence of the call, live or padding, whenever it replays; it
    sizes the captured attention launch. Table ``t`` is viewed at its graph
    width ``widths[t]``; live rows read only the pages their prefixes
    cover. Offsets derive from the host lengths, and every table's padding
    rows clear with one launch per column of ``buffers``, whose views the
    attention is.
    """
    queries = attention.queries
    live = queries.batch_size
    extra = len(padding)
    rows = live + extra

    values = _fixed_view(queries.values, (rows,))
    if extra:
        values[live:].copy_(
            torch.tensor(padding, dtype=values.dtype), non_blocking=True
        )
    offsets = _fixed_view(queries.offsets, (rows + 1,))
    _copy_offsets(offsets, queries.host + padding)
    shared = SequenceLengths(
        host=queries.host + padding, values=values, offsets=offsets
    )

    first = next(iter(attention.entries.values())).prefixes
    prefix_values = _fixed_view(first.values, (rows,))
    prefix_values[live:].zero_()
    prefix_offsets = _fixed_view(first.offsets, (rows + 1,))
    host_prefixes = None
    if first.host is None:
        # Device-resident prefix lengths have no host mirror to sum.
        torch.cumsum(prefix_values, dim=0, out=prefix_offsets[1:])
    else:
        host_prefixes = first.host + (0,) * extra
        _copy_offsets(prefix_offsets, host_prefixes)
    prefixes = SequenceLengths(
        host=host_prefixes, values=prefix_values, offsets=prefix_offsets
    )
    clear_padding(buffers, live_rows=live, rows=rows, live_tokens=0, tokens=0)

    # Every canvas token sees its whole canvas.
    visible = values[:, None].expand(-1, width)
    entries = {}
    for number, entry in attention.entries.items():
        blocks = entry.block_table
        if blocks.indices.shape[1] > widths[number]:
            raise ValueError("prefix table exceeds its configured graph width")
        table = _fixed_view(blocks.indices, (rows, widths[number]))
        start, start_host = blocks.start_page, blocks.start_page_host
        if start is not None:
            start = _fixed_view(start, (rows,))
            if start_host is not None:
                start_host = start_host + (0,) * extra
        entries[number] = SegmentedInput(
            shared,
            prefixes,
            BlockTable(table, blocks.block_size, start, start_host),
            None,
            visible,
            True,
        )
    return AttentionBatch(entries, shared)


def _pad_rows(tensor: torch.Tensor, rows: int, live: int) -> torch.Tensor:
    """Widen a row-major input tensor to ``rows`` rows; padding is zero."""
    # A view with an empty row (a history of zero depth) holds no elements
    # whose backing could be widened; its widened form is empty as well.
    padded = (
        tensor.new_zeros((rows, *tensor.shape[1:]))
        if not tensor.numel() and not all(tensor.shape[1:])
        else _fixed_view(tensor, (rows, *tensor.shape[1:]))
    )
    padded[live:].zero_()
    return padded


def _pad_readout(
    batch: InputBatch,
    rows: int,
    length: int,
    widths: tuple[int, ...],
    buffers,
) -> InputBatch:
    """The canvas pass of a readout call as the ``rows`` bucket represents it.

    The bucket holds ``rows * length`` tokens in ``2 * rows`` sequences
    of at most ``length`` tokens: the live canvases, padding sequences of
    ``length`` tokens that make up what the live canvases leave (the last
    shorter when they are shorter than the model's), then empty ones.
    Padding tokens are token zero at position zero. The slots and their
    candidates stay with the live call, which the head reads after the
    replay.
    """
    live = batch.row_count
    canvas = batch.inputs.canvas
    tokens = canvas.input_ids.numel()
    total = rows * length
    remaining = total - tokens
    padding = [length] * (remaining // length)
    if remaining % length:
        padding.append(remaining % length)
    padding += [0] * (2 * rows - live - len(padding))

    ids = _fixed_view(canvas.input_ids, (total,))
    ids[tokens:].zero_()
    positions = _fixed_view(canvas.positions, (total,))
    positions[tokens:].zero_()
    return InputBatch(
        batch.forward_mode,
        CanvasInput(
            ids,
            positions,
            _pad_attention(
                canvas.attention, tuple(padding), length, widths, buffers
            ),
        ),
        _pad_rows(batch.request_pool_indices, 2 * rows, live),
    )


def _pad_steps(
    batch: InputBatch,
    rows: int,
    length: int,
    widths: tuple[int, ...],
    buffers,
) -> InputBatch:
    """A canvas step call widened to ``rows`` canvases.

    A padding row starts a canvas (step zero) of seed and block zero in
    slot zero, the sentinel, whose state its commit overwrites; its canvas
    reads no prefix at position zero.
    """
    live = batch.row_count
    inputs = batch.inputs
    state = inputs.state
    views = {
        name: _pad_rows(view, rows, live) for name, view in inputs.views.items()
    }
    self_conditioning = views["self_conditioning"].view(
        rows * length, state.self_conditioning.shape[1]
    )
    positions = _fixed_view(inputs.canvas.positions, (rows * length,))
    positions[live * length :].zero_()
    padded_state = sampler.CanvasState(
        seed=_pad_rows(state.seed, rows, live),
        block=_pad_rows(state.block, rows, live),
        step=_pad_rows(state.step, rows, live),
        canvas=views["canvas"],
        history=views["history"],
        self_conditioning=self_conditioning,
    )
    return batch.replace(
        inputs=CanvasStepInput(
            CanvasInput(
                views["canvas"].view(-1),
                positions,
                _pad_attention(
                    inputs.canvas.attention,
                    (length,) * (rows - live),
                    length,
                    widths,
                    buffers,
                ),
                self_conditioning=self_conditioning,
            ),
            padded_state,
            views,
            _pad_rows(inputs.slots, rows, live),
            inputs.sampling + (inputs.sampling[0],) * (rows - live),
        ),
        request_pool_indices=_pad_rows(batch.request_pool_indices, rows, live),
    )


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
