"""H3's learned Q/K normalization, partial RoPE and attention kinds.

Every H3 layer projects its normalized rows to 56 heads of 128 channels,
normalizes each query and key head with a learned RMSNorm, rotates the
leading 96 channels of each head with neox RoPE over the (time, height,
width) coordinates (16 frequencies per axis; the trailing 32 channels stay
unrotated), attends, and projects back. The attention itself is one of:

* ``Dense``: full bidirectional attention over the packed sequence
  (``packing.DensePacking``); under Ulysses each rank exchanges its token
  shard for a head shard of the whole sequence.
* ``Sparse``: video sparse attention over the segment packing
  (``packing.SegmentPacking``): text, audio and condition query tiles attend
  every valid key tile, and video query tiles attend the dense tiles plus a
  selected subset of the key tiles of every video segment, each selected
  independently (``uniserve.nn.attention.vsa.SegmentAttention``). A gated
  attention over mean-pooled tiles is added to that fine result.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import torch
from torch import nn

from uniserve.nn import (
    MergedColumnParallelLinear,
    RMSNorm,
    RowParallelLinear,
    functional,
)
from uniserve.nn.attention import Attention, AttentionBatch, vsa

from .config import TransformerConfig
from .inputs import SegmentInput, SequenceInput


class Dense(nn.Module):
    """Project heads, normalize and rotate Q/K, and attend every used row.

    The Q/K/V projections keep every head for this rank's token rows; the
    attention layer's Ulysses exchange trades the token shard for a head
    shard of the whole sequence and back.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head_dim = config.head_dim
        inner = config.num_attention_heads * config.head_dim
        self.projection = MergedColumnParallelLinear(
            config.hidden_size,
            dict.fromkeys(("q", "k", "v"), inner),
            bias=False,
        )
        self.output = RowParallelLinear(inner, config.hidden_size, bias=False)
        self.query_norm = RMSNorm(config.head_dim, config.qk_norm_eps)
        self.key_norm = RMSNorm(config.head_dim, config.qk_norm_eps)
        self.attention = Attention(
            config.num_attention_heads,
            config.num_attention_heads,
            config.head_dim,
        )

    @torch.inference_mode()
    def forward_chunks(
        self,
        hidden: torch.Tensor | Iterator[tuple[slice, torch.Tensor]],
        cos: torch.Tensor,
        sin: torch.Tensor,
        inputs: SequenceInput,
    ) -> Iterator[tuple[slice, torch.Tensor]]:
        """Attend this rank's token shard and yield its projected output.

        ``hidden`` is the normalized shard, whole or as ``(global row slice,
        rows)`` chunks in row order. ``cos`` and ``sin`` hold the compact
        rotary factors ``[shard rows, 3 * rope_frequency_dim]`` of the
        shard's rows: one per rotated channel pair of the three axes' halves.
        Yields one chunk covering the shard.
        """
        if not isinstance(hidden, torch.Tensor):
            chunks = tuple(value for _, value in hidden)
            hidden = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        values = self.projection(hidden)
        q, k, v = (
            values[name].view(
                -1, values[name].shape[-1] // self.head_dim, self.head_dim
            )
            for name in ("q", "k", "v")
        )
        # One RMS domain and one rotary axis span each whole head; the compact
        # factors rotate its leading 96 channels split-half and the trailing
        # 32 channels pass through.
        q, k = functional.qk_norm_rope(
            q,
            k,
            (self.query_norm.weight,),
            (self.key_norm.weight,),
            (cos,),
            (sin,),
            eps=self.query_norm.eps,
            axis_dims=(self.head_dim,),
            rounding=functional.Rounding.STEPWISE,
        )
        attended = self.attention(
            q, k, v, AttentionBatch.single(inputs.visible)
        )
        yield inputs.token_slice, self.output(attended.flatten(1))

    def forward(self, hidden, cos, sin, inputs):
        return next(self.forward_chunks(hidden, cos, sin, inputs))[1]


class Sparse(nn.Module):
    """Project heads, normalize and rotate Q/K, and attend every segment.

    The merged projection produces four branches: query, key, value, and
    ``gate``, which weights the compressed-tile attention elementwise before
    it is added to the selected fine attention. It gathers this rank's
    context interval of the sequence and projects its head shard of it;
    ``vsa.SegmentAttention`` selects each video query's key tiles of every
    segment, adds the gated tile compression and returns the rows of this
    rank's token shard, which the output projection maps back. Q/K
    normalization and rotation round after every operation, as the released
    network evaluates them.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.head_dim = config.head_dim
        inner = config.num_attention_heads * config.head_dim
        self.projection = MergedColumnParallelLinear(
            config.hidden_size,
            dict.fromkeys(("q", "k", "v", "gate"), inner),
            branch_width=config.head_dim,
            bias=False,
        )
        self.output = RowParallelLinear(inner, config.hidden_size, bias=False)
        self.query_norm = RMSNorm(config.head_dim, config.qk_norm_eps)
        self.key_norm = RMSNorm(config.head_dim, config.qk_norm_eps)
        self.vsa = vsa.SegmentAttention(
            vsa.BlockAttention(config.head_dim**-0.5)
        )

    @torch.inference_mode()
    def forward_chunks(
        self,
        hidden: torch.Tensor | Iterator[tuple[slice, torch.Tensor]],
        cos: torch.Tensor,
        sin: torch.Tensor,
        inputs: SegmentInput,
    ) -> Iterator[tuple[slice, torch.Tensor]]:
        """Attend this rank's token shard and yield its projected output.

        ``hidden`` is the normalized shard, whole or as ``(global row slice,
        rows)`` chunks. ``cos`` and ``sin`` hold the compact rotary factors
        ``[padded rows, 3 * rope_frequency_dim]`` of every packed row: one
        per rotated channel pair of the three axes' halves. Yields the
        shard's output in row intervals as their attention exchange
        completes, or as one chunk when the output projection's input
        quantizer needs dynamic tensor-wide statistics.
        """
        rows = inputs.segments.padded_tokens
        interval = self.vsa.query_rows(rows)
        count = interval.stop - interval.start
        # Every projected interval of this rank's context interval lands in
        # its rows of one [count, q | k | v | gate] matrix, whose branches
        # attention reads as strided views. Intervals arrive in any order,
        # and attention prepares each as it lands.
        if isinstance(hidden, torch.Tensor):
            source = hidden
        else:
            first = next(hidden)
            source = first[1]
            hidden = itertools.chain((first,), hidden)
        projected = self.projection.output_buffers(
            count, dtype=source.dtype, device=source.device
        )
        heads = {
            name: value.view(count, -1, self.head_dim)
            for name, value in projected.items()
        }
        ready = (
            chunk
            for chunk, _ in self.projection.forward_chunks(
                hidden,
                token_slice=inputs.token_slice,
                num_tokens=rows,
                out=projected,
            )
        )
        # One RMS domain and one rotary axis span each whole head; the
        # compact factors rotate its leading 96 channels split-half and the
        # trailing 32 channels pass through.
        norm = vsa.NormRope(
            self.query_norm.weight,
            self.key_norm.weight,
            self.query_norm.eps,
            cos,
            sin,
        )
        attended = self.vsa.forward_chunks(
            heads["q"],
            heads["k"],
            heads["v"],
            heads["gate"],
            inputs.segments,
            norm=norm,
            ready=ready,
        )
        del projected, heads

        # Dynamic tensor-wide statistics (uncalibrated NVFP4, tensor-wide FP8)
        # span the whole shard, so such an output projection encodes it at
        # once; every other projection consumes intervals as they arrive.
        quantizer = self.output.input_quantizer
        if quantizer is not None and quantizer.requires_complete_source:
            parts = [values.flatten(1) for _, values in attended]
            output = self.output(
                parts[0] if len(parts) == 1 else torch.cat(parts)
            )
            del parts
            yield inputs.token_slice, output
            return
        for interval_rows, values in attended:
            output = self.output(values.flatten(1))
            del values
            yield interval_rows, output

    def forward(self, hidden, cos, sin, inputs):
        outputs = [
            value for _, value in self.forward_chunks(hidden, cos, sin, inputs)
        ]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)
