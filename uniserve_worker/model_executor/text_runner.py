"""Worker output selection around the public causal language-model capability.

``TextRunner`` evaluates a ``CausalLM`` backbone once per batch and
returns, per row, final-token logits, all-token logits or hidden states, or
an empty value for a row that selects ``TokenSelection.CACHE``. A call whose
rows all select ``CACHE`` only writes the K/V cache (``CausalLM.fill_cache``).
With pipeline parallelism, only the last stage projects vocabulary columns,
and it broadcasts the selected results so every stage returns the same rows.

Graphs serve two kinds of text calls. Decode buckets share final-position
logit backing and capture greedy decoding with it. A prefill
bucket captures the backbone's hidden states alone, copied into one output
backing that every prefill bucket of the runner shares; after the replay,
``select_outputs`` projects the rows' logits and slices their hidden states
from that backing with the live lengths, so one bucket serves any mix of
output selections. A cache-only prefill bucket (``PrefillShape.outputs``)
captures ``fill_cache`` and serves calls whose rows all select ``CACHE``.
"""

from __future__ import annotations

from typing import cast

import torch

from uniserve.model import Logits, TextInput, VocabShard
from uniserve_worker._uniserve_ipc import TextRunner as _TextRunner
from uniserve_worker.model_executor.output import ExecutionOutput


def _indices(values, device):
    """Copy host-selected token positions into a numerical index vector."""
    return torch.tensor(values, dtype=torch.int64, device="cpu").to(
        device, non_blocking=True
    )


def bind_vocabulary(model):
    """Share vocabulary descriptors across the bound pipeline stages."""
    mesh = model.backbone.mesh
    pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
    tensor_group = mesh.get_group("tp" if "tp" in mesh.axes else ())
    last = pipeline.rank == pipeline.size - 1

    # The vocabulary head may be nonresident on this pipeline stage. Its
    # actual descriptor is communicated rather than reproducing the head's
    # numerical padding/partition rule in execution.
    descriptor = torch.empty(4, dtype=torch.int64, device=pipeline.device)
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
    pipeline.broadcast(descriptor, src=pipeline.size - 1)

    size, padded, start, stop = descriptor.cpu().tolist()
    return pipeline, VocabShard(size, slice(start, stop), padded, tensor_group)


class TextRunner(_TextRunner):
    """Bind vocabulary metadata once and distribute selected pipeline results.

    The execution context supplies layer resources around each call. Selection
    belongs to the caller: one backbone evaluation serves LAST/ALL/HIDDEN rows,
    and the final pipeline stage alone projects vocabulary columns.
    """

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

        values = self._project_logits(hidden, indices, inputs.batch_size)
        values = self.decode_output(values)
        return ExecutionOutput(
            tuple(values.split(1)), (self.vocab,) * inputs.batch_size
        )

    def _project_logits(self, hidden, indices, count):
        """Project selected tokens on the final stage and share its logits."""
        if self.pipeline.rank == self.pipeline.size - 1:
            # The last pipeline stage owns the vocabulary projection.
            logits = cast(
                Logits, self.model.compute_logits(hidden, token_indices=indices)
            )
            values = logits.values
        else:
            width = self.vocab.local_slice.stop - self.vocab.local_slice.start
            values = hidden.new_empty((count, width))
        self.pipeline.broadcast(values, src=self.pipeline.size - 1)
        return values

    def _select_hidden(self, hidden, indices, count, copy_hidden):
        """Gather live hidden rows and preserve borrowed graph output."""
        if self.pipeline.rank != self.pipeline.size - 1:
            selected = hidden.new_empty(
                (count, self.model.backbone.hidden_size)
            )
        elif indices is None:
            selected = hidden[:count]
            if copy_hidden:
                selected = selected.clone()
        else:
            selected = hidden.index_select(0, indices)
        if selected.numel():
            self.pipeline.broadcast(selected, src=self.pipeline.size - 1)
        return selected
