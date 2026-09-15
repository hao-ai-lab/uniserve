"""Worker output selection around the public causal language-model.

capability.
"""

from __future__ import annotations

import torch

from uniserve.model import CausalLM, TextInput, VocabShard
from uniserve.nn.attention import DenseInput

from .batch import ExecutionOutput
from .sampling import TokenSelection


class TextCall:
    """Bind vocabulary metadata once and distribute selected pipeline results.

    The execution context supplies layer resources around each call. Selection
    belongs to the caller: one backbone evaluation serves LAST/ALL/HIDDEN rows,
    and the final pipeline stage alone projects vocabulary columns.
    """

    def __init__(self, model: CausalLM):
        self.model = model
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
        """Project one final position per sequence.

        including discarded padding rows.

        A graph bucket can contain empty padding sequences. Their placeholder
        result is discarded by execution; live rows must each contain a token.
        Device offsets select positions on every replay.
        """
        hidden = self.model(inputs)
        indices = (
            inputs.attention.queries.offsets[1:].to(torch.int64) - 1
        ).clamp_min(0)

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

        Each row independently selects final-token logits, all-token logits, or
        raw hidden states. Logit and hidden columns are computed once on the
        last pipeline stage and broadcast so every stage returns the same rows.
        """
        if isinstance(inputs.attention, DenseInput):
            count = inputs.input_ids.numel() // inputs.batch_size
            lengths = (count,) * inputs.batch_size
            offsets = (
                torch.arange(
                    inputs.batch_size + 1, device=inputs.input_ids.device
                )
                * count
            )
        else:
            lengths = inputs.attention.queries.host
            offsets = inputs.attention.queries.offsets

        if len(selections) != len(lengths) or any(
            not isinstance(item, TokenSelection) for item in selections
        ):
            raise ValueError(
                "text output selections must align with the input sequences"
            )

        hidden = self.model(inputs)
        last = self.pipeline.rank == self.pipeline.size - 1
        indices = []
        logit_lengths, hidden_lengths = [], []
        for index, (length, selection) in enumerate(
            zip(lengths, selections, strict=True)
        ):
            if selection is TokenSelection.HIDDEN:
                hidden_lengths.append(length)
            else:
                selected = (
                    length
                    if selection is TokenSelection.ALL_LOGITS
                    else min(1, length)
                )
                logit_lengths.append(selected)
                if selected:
                    # LAST follows live device offsets when a prefill graph
                    # replays with another distribution of sequence lengths.
                    indices.append(
                        offsets[index + 1 : index + 2].to(torch.int64) - 1
                        if selection is TokenSelection.LAST_LOGITS
                        else torch.arange(length, device=hidden.device)
                        + offsets[index]
                    )

        num_logits = sum(logit_lengths)
        logits = None
        if num_logits:
            if last:
                token_indices = (
                    indices[0] if len(indices) == 1 else torch.cat(indices)
                )
                logits = self.model.compute_logits(
                    hidden, token_indices=token_indices
                ).values
            else:
                logits = hidden.new_empty(
                    (
                        num_logits,
                        self.vocab.local_slice.stop
                        - self.vocab.local_slice.start,
                    )
                )
            self.pipeline.broadcast(logits, src=self.pipeline.size - 1)
        else:
            logits = hidden.new_empty(
                (0, self.vocab.local_slice.stop - self.vocab.local_slice.start)
            )

        selected_hidden = None
        if hidden_lengths:
            if last and len(hidden_lengths) == len(lengths):
                # Every token participates, so the backbone already supplies
                # the packed result needed by the pipeline and caller views.
                selected_hidden = hidden
            elif last:
                start = 0
                parts = []
                for length, selection in zip(lengths, selections, strict=True):
                    if selection is TokenSelection.HIDDEN:
                        parts.append(hidden[start : start + length])
                    start += length
                selected_hidden = (
                    parts[0] if len(parts) == 1 else torch.cat(parts)
                )
            else:
                selected_hidden = hidden.new_empty(
                    (sum(hidden_lengths), self.model.backbone.hidden_size)
                )
            if selected_hidden.numel():
                self.pipeline.broadcast(
                    selected_hidden, src=self.pipeline.size - 1
                )

        outputs, vocabularies = [], []
        logits_rows = iter(logits.split(logit_lengths))
        hidden_rows = (
            iter(())
            if selected_hidden is None
            else iter(selected_hidden.split(hidden_lengths))
        )
        for selection in selections:
            selected = selection is not TokenSelection.HIDDEN
            outputs.append(next(logits_rows if selected else hidden_rows))
            vocabularies.append(self.vocab if selected else None)
        return ExecutionOutput(tuple(outputs), tuple(vocabularies))
