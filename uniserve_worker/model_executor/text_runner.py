"""Worker output selection around the public causal language-model capability.

``TextRunner`` evaluates a ``CausalLM`` backbone once per staged call and
returns, per row, final-token logits, all-token logits or hidden states, or
an empty value for a row that selects ``TokenSelection.CACHE``. A call whose
rows all select ``CACHE`` only writes the K/V cache (``CausalLM.fill_cache``).
With pipeline parallelism, only the last stage projects vocabulary columns,
and it broadcasts the selected results so every stage returns the same rows.

Graphs serve two kinds of text calls. A decode bucket captures the
final-position logits of every row together with greedy decoding. A prefill
bucket captures the backbone's hidden states alone, copied into one output
backing that every prefill bucket of the runner shares; after the replay,
``select_outputs`` projects the rows' logits and slices their hidden states
from that backing with the live lengths, so one bucket serves any mix of
output selections. A cache-only prefill bucket (``PrefillShape.outputs``)
captures ``fill_cache`` and serves calls whose rows all select ``CACHE``.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import accumulate

import torch

from uniserve.model import TextInput, VocabShard
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve_worker.model_executor.output import ExecutionOutput
from uniserve_worker.protocol.call import ForwardMode
from uniserve_worker.sampling.metadata import TokenSelection

from .graph_inputs import (
    capture_hidden,
    pad_text,
    replay_hidden,
    text_shape,
)
from .model_runner import ModelRunner


def _host_indices(ranges, device):
    """Stage the token rows of ``ranges`` as one int64 index vector.

    The rows are known on the host from staged lengths, so one host-to-device
    copy replaces a device operation per row.
    """
    values = [index for span in ranges for index in span]
    return torch.tensor(values, dtype=torch.int64).to(device, non_blocking=True)


class TextRunner(ModelRunner):
    """Bind vocabulary metadata once and distribute selected pipeline results.

    The execution context supplies layer resources around each call. Selection
    belongs to the caller: one backbone evaluation serves LAST/ALL/HIDDEN rows,
    and the final pipeline stage alone projects vocabulary columns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hidden states of the latest prefill replay: [tokens, hidden] rows
        # every prefill graph writes, sized by the first (largest) capture.
        self._prefill_output: torch.Tensor | None = None
        try:
            self._bind_vocabulary()
        except BaseException:
            self.close()
            raise

    def _bind_vocabulary(self):
        """Share vocabulary descriptors across the bound pipeline stages."""
        model = self.model
        mesh = model.backbone.mesh
        self.pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
        tensor_group = mesh.get_group("tp" if "tp" in mesh.axes else ())
        last = self.pipeline.rank == self.pipeline.size - 1

        # The vocabulary head may be nonresident on this pipeline stage. Its
        # actual descriptor is communicated rather than reproducing the head's
        # numerical padding/partition rule in execution.
        descriptor = torch.empty(
            4, dtype=torch.int64, device=self.pipeline.device
        )
        if last:
            vocab = model.lm_head.vocab
            descriptor.copy_(
                torch.tensor(
                    (
                        vocab.size,
                        vocab.padded_size,
                        vocab.local_slice.start,
                        vocab.local_slice.stop,
                    ),
                    dtype=torch.int64,
                    device=descriptor.device,
                )
            )
        self.pipeline.broadcast(descriptor, src=self.pipeline.size - 1)

        size, padded, start, stop = descriptor.cpu().tolist()
        self.vocab = VocabShard(size, slice(start, stop), padded, tensor_group)

    def last_logits(self, inputs: TextInput) -> ExecutionOutput:
        """Project one final position per sequence, padding rows included.

        A graph bucket can contain empty padding sequences. Their placeholder
        result is discarded by execution; live rows must each contain a token.
        Device offsets select positions on every replay, and the output has
        one row per sequence regardless of host query lengths.
        ``batch_forward`` uses this path for padded last-logits buckets.
        """
        hidden = self.model(inputs)
        queries = inputs.attention.queries
        if queries is None:
            raise ValueError("final-position logits require sequence offsets")
        offsets = queries.offsets
        indices = (offsets[1:].to(torch.int64) - 1).clamp_min(0)

        if self.pipeline.rank == self.pipeline.size - 1:
            values = self.model.compute_logits(
                hidden, token_indices=indices
            ).values
        else:
            values = hidden.new_empty(
                (
                    inputs.batch_size,
                    self.vocab.local_slice.stop - self.vocab.local_slice.start,
                )
            )
        self.pipeline.broadcast(values, src=self.pipeline.size - 1)
        return ExecutionOutput(
            tuple(values.split(1)), (self.vocab,) * inputs.batch_size
        )

    def __call__(
        self, inputs: TextInput, selections: tuple[TokenSelection, ...]
    ) -> ExecutionOutput:
        """Run one backbone pass and return the per-row selection it requests.

        Each row independently selects final-token logits, all-token logits,
        raw hidden states, or nothing (``CACHE``). Logit and hidden columns
        are computed once on the last pipeline stage and broadcast so every
        stage returns the same rows. A call whose rows all select ``CACHE``
        only writes the K/V cache.
        """
        if all(selection is TokenSelection.CACHE for selection in selections):
            self.model.fill_cache(inputs)
            return self._cache_rows(len(selections))
        return self.select_outputs(self.model(inputs), inputs, selections)

    def _cache_rows(self, rows: int) -> ExecutionOutput:
        """Empty ``[0, hidden]`` values of ``rows`` cache-only rows."""
        empty = torch.empty(
            (0, self.model.backbone.hidden_size), device=self.device
        )
        return ExecutionOutput((empty,) * rows, (None,) * rows)

    def hidden_states(self, inputs: TextInput) -> torch.Tensor:
        """Evaluate the backbone into the runner's prefill output backing.

        This is the call a prefill bucket captures. The backbone's
        ``[tokens, ...]`` output is copied into leading rows of one backing
        every prefill graph of the runner shares, so their outputs do not
        each retain graph storage; the first call, the largest bucket's warm
        call, allocates it from the runner's graph storage.

        Raises:
            CUDAGraphError: The backing is missing during capture, or a later
                output does not fit it.
        """
        hidden = self.model(inputs)
        backing = self._prefill_output
        if backing is None:
            if torch.cuda.is_current_stream_capturing():
                raise CUDAGraphError(
                    "prefill output backing must exist before capture"
                )
            # Later captures reuse free blocks of the shared pools, so this
            # persistent allocation precedes the first prefill capture.
            with self.graph_storage.allocate(self):
                backing = torch.empty_like(hidden)
            self._prefill_output = backing
        if (
            hidden.shape[0] > backing.shape[0]
            or hidden.shape[1:] != backing.shape[1:]
            or hidden.dtype != backing.dtype
            or hidden.device != backing.device
        ):
            raise CUDAGraphError(
                "prefill buckets capture largest first; a later bucket's "
                "hidden states exceed the output backing"
            )
        output = backing[: hidden.shape[0]]
        output.copy_(hidden)
        return output

    def select_outputs(
        self,
        hidden: torch.Tensor,
        inputs: TextInput,
        selections: tuple[TokenSelection, ...],
        *,
        copy_hidden: bool = False,
    ) -> ExecutionOutput:
        """Select each row's logits or hidden states from backbone output.

        ``hidden`` holds the packed ``[tokens, hidden]`` rows of ``inputs``,
        possibly followed by padding rows, and ``selections`` has one entry
        per row of ``inputs``. Every row selecting logits is projected in one
        gather and one head call: its final token for ``LAST_LOGITS``, every
        token for ``ALL_LOGITS``. Hidden rows are views of ``hidden``, or
        copies with ``copy_hidden``, which a caller whose ``hidden`` is
        reused storage requests. A ``CACHE`` row receives an empty
        ``[0, hidden]`` value. On other pipeline stages the last stage's
        results are received by broadcast.

        Raises:
            ValueError: Host query lengths are missing or the selections do
                not align with the input sequences.
        """
        queries = inputs.attention.queries
        if queries is None:
            count = inputs.input_ids.numel() // inputs.batch_size
            lengths = (count,) * inputs.batch_size
        else:
            if queries.host is None:
                raise ValueError(
                    "text output selection requires host query lengths"
                )
            lengths = queries.host
        if len(selections) != len(lengths) or any(
            not isinstance(item, TokenSelection) for item in selections
        ):
            raise ValueError(
                "text output selections must align with the input sequences"
            )

        last = self.pipeline.rank == self.pipeline.size - 1
        width = self.vocab.local_slice.stop - self.vocab.local_slice.start
        device = hidden.device
        # Host token offsets: row r spans tokens [starts[r], starts[r + 1]).
        starts = tuple(accumulate(lengths, initial=0))
        rows = tuple(zip(starts[:-1], starts[1:], selections, strict=True))

        # Logit rows: the final token of a final-token row (none for an empty
        # row) and every token of an all-token row.
        projected = (TokenSelection.LAST_LOGITS, TokenSelection.ALL_LOGITS)
        logit_counts = tuple(
            stop - start
            if selection is TokenSelection.ALL_LOGITS
            else min(1, stop - start)
            for start, stop, selection in rows
            if selection in projected
        )
        logits = hidden.new_empty((0, width))
        if sum(logit_counts):
            if queries is not None and all(
                selection is TokenSelection.LAST_LOGITS and stop > start
                for start, stop, selection in rows
            ):
                # Every row's final token, from the staged device offsets.
                indices = queries.offsets[1 : len(rows) + 1] - 1
            else:
                indices = _host_indices(
                    (
                        range(start, stop)
                        if selection is TokenSelection.ALL_LOGITS
                        else range(stop - min(1, stop - start), stop)
                        for start, stop, selection in rows
                        if selection in projected
                    ),
                    device,
                )
            if last:
                logits = self.model.compute_logits(
                    hidden, token_indices=indices
                ).values
            else:
                # Earlier stages allocate the receive buffer for the
                # broadcast below.
                logits = hidden.new_empty((sum(logit_counts), width))
            self.pipeline.broadcast(logits, src=self.pipeline.size - 1)

        hidden_lengths = tuple(
            stop - start
            for start, stop, selection in rows
            if selection is TokenSelection.HIDDEN
        )
        selected = None
        if hidden_lengths:
            if not last:
                selected = hidden.new_empty(
                    (sum(hidden_lengths), self.model.backbone.hidden_size)
                )
            elif len(hidden_lengths) == len(rows):
                # Every token participates: the packed live rows.
                selected = hidden[: starts[-1]]
                if copy_hidden:
                    selected = selected.clone()
            else:
                selected = hidden.index_select(
                    0,
                    _host_indices(
                        (
                            range(start, stop)
                            for start, stop, selection in rows
                            if selection is TokenSelection.HIDDEN
                        ),
                        device,
                    ),
                )
            if selected.numel():
                self.pipeline.broadcast(selected, src=self.pipeline.size - 1)

        # Interleave both runs back into row order; cache-only rows receive
        # an empty value.
        logit_rows = iter(logits.split(logit_counts))
        hidden_rows = iter(
            () if selected is None else selected.split(hidden_lengths)
        )
        empty = hidden.new_empty((0, hidden.shape[1]))
        outputs, vocabularies = [], []
        for selection in selections:
            if selection in projected:
                outputs.append(next(logit_rows))
                vocabularies.append(self.vocab)
            elif selection is TokenSelection.HIDDEN:
                outputs.append(next(hidden_rows))
                vocabularies.append(None)
            else:
                outputs.append(empty)
                vocabularies.append(None)
        return ExecutionOutput(tuple(outputs), tuple(vocabularies))

    def batch_forward(self, batch, *, padded=False):
        """Select numerical language outputs from one prepared input batch.

        A padded batch is a decode bucket, whose rows all select final-token
        logits (``pad_text``) and use ``last_logits``; any other batch
        selects each row's outputs.
        """
        return (
            self.last_logits(batch.inputs)
            if padded
            and batch.token_selections[0] is TokenSelection.LAST_LOGITS
            else self(batch.inputs, batch.token_selections)
        )

    def select_graph_shape(self, batch, *, eligible):
        """Choose a captured text graph bucket and pad the batch to it.

        Only buckets that startup captures are candidates. A decode batch
        (one query token per row) must fit a decode bucket. With prefill
        graphs enabled, every other batch must fit a prefill bucket of its
        causality and embedding replacement that evaluates hidden states,
        or, when every row selects ``CACHE``, a cache-only bucket if the
        runner has any. Returns ``None`` for eager
        execution (graphs disabled, the batch ineligible, or prefill graphs
        disabled for a non-decode batch), otherwise ``(key, padded_batch,
        True)``; a decode key starts with ``"text"`` and a prefill key with
        ``"prefill"``, and both carry the ``text_shape`` tuple second.

        Raises:
            CUDAGraphError: No decode bucket holds a decode batch, or prefill
                graphs are enabled and no prefill bucket holds another batch.
        """
        if not eligible or not self.pools:
            return None

        decode = (
            batch.forward_mode is ForwardMode.DECODE
            and batch.inputs.attention.queries.host == (1,) * batch.row_count
        )
        prefill_shapes = self._prefill_family(batch)
        if not decode and not prefill_shapes:
            return None

        shape = text_shape(
            batch,
            decode_sizes=self.decode_shapes,
            prefill_shapes=prefill_shapes,
            table_widths=self.table_widths,
        )
        if shape is None:
            raise CUDAGraphError(self._unserved(batch, decode=decode))

        if shape[-1]:
            if batch.decode_force_finish is None:
                # Direct token inputs and device continuations use the same
                # numerical decode graph. Only the latter consumes its greedy
                # continuation result; the former supplies an inert finish
                # mask.
                finish = self.input_buffers.decode_force_finish[
                    : batch.row_count
                ]
                finish.zero_()
                batch = replace(batch, decode_force_finish=finish)
            padded = pad_text(batch, *shape, staging=self.input_buffers)
            inputs = padded.inputs
            key: tuple[object, ...] = (
                "text",
                shape,
                inputs.input_ids.dtype,
                inputs.positions.ndim,
                inputs.embeddings is not None,
                next(iter(inputs.attention.entries.values())).causal[0],
                padded.token_selections[0],
            )
            return key, padded, True

        # A prefill graph computes hidden states, or only the K/V cache;
        # the force-finish column and the rows' selections stay outside it.
        padded = pad_text(batch, *shape, staging=self.input_buffers)
        inputs = padded.inputs
        key = (
            "prefill",
            shape,
            inputs.input_ids.dtype,
            inputs.positions.ndim,
            inputs.embeddings is not None,
            next(iter(inputs.attention.entries.values())).causal[0],
            prefill_shapes[0].outputs,
        )
        return key, padded, True

    def _prefill_family(self, batch):
        """The prefill buckets a batch may replay: one outputs kind.

        A batch whose rows all select ``CACHE`` replays a cache-only bucket
        when the runner has any; every other batch, one that evaluates
        hidden states.
        """
        if not self.prefill_graph:
            return ()
        cache_only = all(
            selection is TokenSelection.CACHE
            for selection in batch.token_selections
        )
        family = tuple(
            shape for shape in self.prefill_shapes if not shape.outputs
        )
        if not (cache_only and family):
            family = tuple(
                shape for shape in self.prefill_shapes if shape.outputs
            )
        return family

    def _unserved(self, batch, *, decode):
        """Describe a batch no captured bucket holds and the captured range."""
        inputs = batch.inputs
        if decode:
            captured = (
                f"its captured decode graphs hold up to "
                f"{max(self.decode_shapes)} rows"
                if self.decode_shapes
                else "no decode graph is captured"
            )
            return (
                f"{self.name} has no decode graph for a call of "
                f"{batch.row_count} rows; {captured}"
            )
        causality = {
            flag
            for entry in inputs.attention.entries.values()
            for flag in entry.causal
        }
        kind = (
            "mixed-causality"
            if len(causality) > 1
            else "causal"
            if True in causality
            else "non-causal"
        )
        embeddings = inputs.embeddings is not None
        shapes = tuple(
            shape
            for shape in self._prefill_family(batch)
            if causality == {shape.causal} and shape.embeddings == embeddings
        )
        captured = (
            "its captured prefill graphs hold up to "
            f"{max(shape.row_bucket for shape in shapes) - 1} rows and "
            f"{max(shape.token_bucket for shape in shapes)} tokens"
            if shapes
            else "no prefill graph of that kind is captured"
        )
        return (
            f"{self.name} has no prefill graph for a {kind} "
            f"{batch.forward_mode.value} call of {batch.row_count} rows and "
            f"{inputs.input_ids.numel()} tokens"
            + (" with embedding replacement" if embeddings else "")
            + f"; {captured}"
        )

    def capture_graph(self, key, execution, forward):
        """Capture a prefill bucket's hidden states or its K/V cache writes.

        Other buckets defer to the base.
        """
        if key[0] != "prefill":
            return super().capture_graph(key, execution, forward)
        outputs = key[-1]
        return capture_hidden(
            self.context,
            execution,
            (
                lambda static: (
                    self.hidden_states(static.inputs)
                    if outputs
                    else self.model.fill_cache(static.inputs)
                )
            ),
            pools=self.pools,
            cache=self.cache,
        )

    def replay_graph(self, key, execution, batch, *, borrow):
        """Replay a prefill bucket and select the live rows' outputs.

        The selection projects logits into new tensors and copies hidden
        rows out of the shared backing, so the result owns its values
        whether or not ``borrow`` is set. A cache-only bucket's rows receive
        empty values.
        """
        if key[0] != "prefill":
            return super().replay_graph(key, execution, batch, borrow=borrow)
        hidden = replay_hidden(self.buckets[key].graphs[None], execution)
        if not key[-1]:
            return self._cache_rows(batch.row_count)
        with self.context.activate():
            return self.select_outputs(
                hidden,
                batch.inputs,
                batch.token_selections,
                copy_hidden=True,
            )
