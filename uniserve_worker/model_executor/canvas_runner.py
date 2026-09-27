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

On CUDA, startup captures one graph per canvas row bucket (``canvas_rows``)
and call kind, and every call replays the smallest bucket that holds its
canvases; after startup a call no captured bucket holds fails. A canvas
step graph holds the whole step: the pass, the head and the sampler of
every chunk, and the commit of the stepped state, for the one sampling the
deployment serves; a call whose rows all start their canvas replays its own
graph, whose pass skips the self-conditioning signal, zero at step zero. A
readout graph holds the pass up to the final layer's attention output
(``TokenDenoiser.attend``), over the bucket's canvases' worth of tokens in
up to twice as many sequences, so it serves canvases of any length up to
the model's; the rest of the final layer, the output norm and the head run
after the replay on the live slot rows alone (``TokenDenoiser.finish``),
whose number and candidates vary from call to call. Every sequence of a
canvas graph, padding included, is at most one canvas long, which bounds
its attention launch. Padding sequences read no prefix, and a padding step
row starts a canvas in the sentinel slot zero.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from uniserve.diffusion import canvas as sampler
from uniserve.diffusion.tokens import candidate_logprobs
from uniserve.model import CanvasInput
from uniserve.nn.attention import (
    AttentionBatch,
    BlockTable,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.runtime.cuda_graph import CUDAGraphError

from .graph_inputs import _fixed_view, capture_hidden, replay_hidden
from .input_batch import CanvasStepInput, InputBatch, ReadoutInput
from .model_runner import ModelRunner
from .output import ExecutionOutput

# Canvas rows a captured graph holds: every count up to four, then steps of
# at most half the previous count, so padding stays under a third of a
# call's rows. ``CanvasRunner.canvas_rows`` keeps the counts a call can
# stage and adds that limit itself.
CANVAS_ROW_BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


class CanvasRunner(ModelRunner):
    """Run token-denoising passes, their slot readout, and canvas steps.

    Calls replay the graphs startup captures per row bucket (see the module
    description); without graph pools every call runs eagerly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mesh = self.model.backbone.mesh
        self.pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
        self.canvas_slots = None
        # Rows of one page of every cache group the KV unit pool holds, set
        # by ``ModelExecutor.bind``; None leaves canvases unbounded by it.
        self.pool_rows: int | None = None

    @property
    def canvas_length(self) -> int:
        """Tokens of one canvas the model declares."""
        return self.model.canvas.length

    @property
    def max_canvases(self) -> int:
        """The most canvases one call stages.

        With graph pools a readout bucket stages up to twice its canvases
        as sequences, so staging holds twice the canvases (see
        ``canvas_staging_rows``). The scheduler reuses cached prompt pages
        only up to a whole page before a prompt's last token, which the
        request computes into a page of every cache group of its own, so no
        call reads more canvases than the KV unit pool has ``pool_rows``.
        """
        buffers = self.input_buffers
        rows = buffers.max_rows // 2 if self.pools else buffers.max_rows
        if self.pool_rows is not None:
            rows = min(rows, self.pool_rows)
        return min(rows, buffers.max_tokens // self.canvas_length)

    @property
    def canvas_rows(self) -> tuple[int, ...]:
        """Row buckets of the canvas graphs, in increasing order.

        A bucket holds at most ``max_canvases`` canvases, and a canvas
        step's rows fit the bound canvas state; the largest bucket is that
        limit.
        """
        limit = self.max_canvases
        if self.canvas_slots is not None:
            limit = min(
                limit,
                self.canvas_slots.max_rows,
                self.canvas_slots.request_pool_size,
            )
        return tuple(rows for rows in CANVAS_ROW_BUCKETS if rows < limit) + (
            (limit,) if limit > 0 else ()
        )

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
        return self.readout(self.model.attend(inputs.canvas), inputs)

    def readout(self, state: tuple[torch.Tensor, ...], inputs: ReadoutInput):
        """Read the slots of ``inputs`` from the pass's ``attend`` state.

        ``state`` holds ``TokenDenoiser.attend``'s per-token state of at
        least the staged canvas tokens, in the packed order
        ``inputs.slot_tokens`` indexes. The final layer's remainder, the
        output norm and the head evaluate the slot rows alone, gathered once
        for every slot of the call.
        """
        count = int(inputs.selection.numel())
        device = state[0].device
        if self.pipeline.rank == self.pipeline.size - 1:
            slots = self.model.finish(state, inputs.slot_tokens)
            logits = self.model.compute_logits(
                slots,
                token_indices=torch.arange(slots.shape[0], device=slots.device),
            )
            # The softmax normalizes over the whole vocabulary, so tensor
            # shards gather their columns for the slot rows alone.
            values = (
                candidate_logprobs(logits.gather(), inputs.candidates)
                .reshape(-1)
                .index_select(0, inputs.selection)
            )
        else:
            values = torch.empty((count,), dtype=torch.float32, device=device)
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
        """Choose the canvas graph bucket of a call and pad the call to it.

        Returns ``None`` for eager execution, without graph pools or for an
        ineligible call. Otherwise returns ``(key, padded_batch, True)``: a
        readout key is ``("canvas", rows)`` and its padded batch holds the
        canvas pass's input alone; a step key is ``("canvas_step", rows,
        sampling, first)``, where ``first`` is whether every row starts its
        canvas; padding rows start one too.

        Raises:
            CUDAGraphError: The call holds more canvases than every bucket,
                or a canvas longer than the model's.
        """
        if not eligible or not self.pools:
            return None
        inputs = batch.inputs
        length = self.canvas_length
        buckets = self.canvas_rows
        rows = next(
            (value for value in buckets if value >= batch.row_count), None
        )
        if rows is None or max(inputs.attention.queries.host) > length:
            raise CUDAGraphError(
                f"{self.name} has no canvas graph for a call of "
                f"{batch.row_count} canvases of up to "
                f"{max(inputs.attention.queries.host)} tokens; its canvas "
                f"graphs hold up to {buckets[-1] if buckets else 0} canvases "
                f"of {length} tokens"
            )

        widths = self.input_buffers.table_widths
        if isinstance(inputs, CanvasStepInput):
            padded = _pad_steps(batch, rows, length, widths)
            key: tuple[object, ...] = (
                "canvas_step",
                rows,
                padded.inputs.sampling[0],
                padded.inputs.first,
            )
        else:
            padded = _pad_readout(batch, rows, length, widths)
            key = ("canvas", rows)
        return key, padded, True

    def capture_graph(self, key, execution, forward):
        """Capture a readout bucket's pass, or a whole canvas step.

        A readout graph's output is ``TokenDenoiser.attend``'s state of its
        bucket's canvas tokens, which the graph retains.
        """
        if key[0] != "canvas":
            return super().capture_graph(key, execution, forward)
        return capture_hidden(
            self.context,
            execution,
            lambda static: self.model.attend(static.inputs),
            pools=self.pools,
            cache=self.cache,
        )

    def replay_graph(self, key, execution, batch, *, borrow):
        """Replay a canvas graph; a readout then reads the live slots.

        The readout's values are new tensors, so the result owns them
        whether or not ``borrow`` is set.
        """
        if key[0] != "canvas":
            return super().replay_graph(key, execution, batch, borrow=borrow)
        state = replay_hidden(self.buckets[key].graphs[None], execution)
        with self.context.activate():
            return self.readout(state, batch.inputs)


def _pad_attention(
    attention: AttentionBatch,
    padding: tuple[int, ...],
    width: int,
    widths: tuple[int, ...],
):
    """Append padding sequences to staged canvas attention in place.

    ``padding`` holds the query tokens of each padding sequence, which reads
    no prefix, table unit zero from page zero and its whole own sequence;
    every table shares the query and prefix columns, which are padded once.
    ``width``, the fixed extent of the current-sequence visibility, bounds
    every sequence of the call, live or padding, whenever it replays; it
    sizes the captured attention launch. Table ``t`` is viewed at its graph
    width ``widths[t]``; live rows read only the pages their prefixes
    cover.
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
    torch.cumsum(values, dim=0, out=offsets[1:])
    shared = SequenceLengths(
        host=queries.host + padding, values=values, offsets=offsets
    )

    first = next(iter(attention.entries.values())).prefixes
    prefix_values = _fixed_view(first.values, (rows,))
    prefix_values[live:].zero_()
    prefix_offsets = _fixed_view(first.offsets, (rows + 1,))
    torch.cumsum(prefix_values, dim=0, out=prefix_offsets[1:])
    prefixes = SequenceLengths(
        host=None if first.host is None else first.host + (0,) * extra,
        values=prefix_values,
        offsets=prefix_offsets,
    )

    # Every canvas token sees its whole canvas.
    visible = values[:, None].expand(-1, width)
    entries = {}
    for number, entry in attention.entries.items():
        blocks = entry.block_table
        if blocks.indices.shape[1] > widths[number]:
            raise ValueError("prefix table exceeds its configured graph width")
        table = _fixed_view(blocks.indices, (rows, widths[number]))
        table[live:].zero_()
        start, start_host = blocks.start_page, blocks.start_page_host
        if start is not None:
            start = _fixed_view(start, (rows,))
            start[live:].zero_()
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
    """Widen a row-major staged tensor to ``rows`` rows; padding is zero."""
    # A view with an empty row (a history of zero depth) holds no elements
    # whose backing could be widened; its widened form is empty as well.
    padded = (
        tensor.new_zeros((rows, *tensor.shape[1:]))
        if not tensor.numel() and not all(tensor.shape[1:])
        else _fixed_view(tensor, (rows, *tensor.shape[1:]))
    )
    padded[live:].zero_()
    return padded


def canvas_staging_rows(max_rows: int) -> int:
    """Rows the canvas staging of a call bound of ``max_rows`` holds.

    A readout graph of ``rows`` canvases stages up to ``2 * rows``
    sequences (``CanvasRunner.select_graph_shape``); the extra rows consume
    staging, but no scheduler request slot.
    """
    return 2 * max_rows


def _pad_readout(
    batch: InputBatch, rows: int, length: int, widths: tuple[int, ...]
) -> InputBatch:
    """The canvas pass of a readout call as the ``rows`` bucket stages it.

    The bucket stages ``rows * length`` tokens in ``2 * rows`` sequences
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
            _pad_attention(canvas.attention, tuple(padding), length, widths),
        ),
        _pad_rows(batch.request_pool_indices, 2 * rows, live),
    )


def _pad_steps(
    batch: InputBatch, rows: int, length: int, widths: tuple[int, ...]
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
    return replace(
        batch,
        inputs=CanvasStepInput(
            CanvasInput(
                views["canvas"].view(-1),
                positions,
                _pad_attention(
                    inputs.canvas.attention,
                    (length,) * (rows - live),
                    length,
                    widths,
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
