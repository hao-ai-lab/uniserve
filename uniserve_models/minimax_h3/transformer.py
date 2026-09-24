"""H3's modulated multimodal transformer and separate latent output heads.

On the first pipeline stage, ``Denoiser.forward`` scatters refined text and
projected latent rows into this rank's packed hidden shard; every stage then
calls ``Transformer.forward`` once per solver step. Each
``TransformerLayer.forward_chunks`` yields ``(global row slice, rows)``
chunks, so a layer's completed rows feed the next layer's attention
projection before the whole shard is finished. Timestep conditioning comes
from ``Modulation`` products that ``weights._prepare_modulation`` precomputes
at load time for the fixed solver ladder; no timestep embedding runs per
request.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import chain
from typing import cast

import torch
from torch import nn

from uniserve.distributed import DeviceMesh
from uniserve.nn import (
    ColumnParallelLinear,
    GatedMLP,
    Linear,
    Modulation,
    RMSNorm,
    functional,
)
from uniserve.quantization import Quantizer

from .attention import Attention
from .config import TransformerConfig
from .inputs import AttentionInput
from .modulation import OutputNorm


class TransformerLayer(nn.Module):
    """Apply modality-indexed attention and feed-forward residual updates."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = nn.ModuleList(
            (
                RMSNorm(config.hidden_size, config.norm_eps),
                RMSNorm(config.hidden_size, config.norm_eps),
            )
        )
        self.attention = Attention(config)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)

    @torch.inference_mode()
    def forward_chunks(
        self, hidden, modulation, inputs: AttentionInput, *, workspace
    ):
        """Connect token-local residual updates to the following layer's projection.

        Args:
            hidden: The layer input rows of ``inputs.token_slice``: the whole
                shard as one tensor, or an iterable of ``(global row slice,
                rows)`` chunks, such as a preceding layer's output.
                ``Transformer.forward`` passes the first layer one chunk
                covering the shard.
            modulation: This layer's products at the current step,
                ``[2, 18 * hidden_size]``; see ``Transformer.modulation``.
            inputs: The attention input of this rank's token shard.
            workspace: ``modulation_indices``, rotary ``cos``/``sin`` and
                this layer's attention scratch set.

        Yields:
            ``(global row slice, rows)`` output chunks covering the shard, or
            the whole shard as one chunk when a feed-forward input quantizer
            needs dynamic tensor-wide statistics.
        """  # noqa: E501
        if isinstance(hidden, torch.Tensor):
            source: Iterable[tuple[slice, torch.Tensor]] = (
                (inputs.token_slice, hidden),
            )
        else:
            # Chunked input arrives as separate per-interval tensors. Allocate
            # this layer's contiguous residual for the shard; the
            # normalization pass fills it through ``retain``.
            chunks = iter(hidden)
            first = next(chunks)
            hidden = first[1].new_empty(
                (
                    inputs.token_slice.stop - inputs.token_slice.start,
                    self.hidden_size,
                )
            )
            source = chain((first,), chunks)
        indices = workspace["modulation_indices"]

        # Module containers hold one class each: the pre-attention and pre-MLP
        # RMSNorms, and the column-parallel gate/up branches.
        attention_norm = cast(RMSNorm, self.norm[0])
        mlp_norm = cast(RMSNorm, self.norm[1])
        gate = cast(ColumnParallelLinear, self.mlp.gate_up.projections["gate"])
        up = cast(ColumnParallelLinear, self.mlp.gate_up.projections["up"])

        # Six affine vectors: shift/scale/gate for attention and MLP, each
        # [6, hidden] after the reshape. The six rows are the (video, text,
        # audio) token groups under the video timestep, then under the audio
        # timestep; ``indices`` selects one row per token.
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            value.to(hidden.dtype)
            for value in modulation.reshape(-1, 6 * self.hidden_size).chunk(
                6, dim=-1
            )
        )

        def normalize():
            for interval, value in source:
                local = slice(
                    interval.start - inputs.token_slice.start,
                    interval.stop - inputs.token_slice.start,
                )
                # Retain this layer's residual while its projected values pass
                # through the next numerical stage's borrowed scratch; the
                # normalization pass writes the copy as it reads the rows.
                yield (
                    interval,
                    functional.modulated_rms_norm(
                        value,
                        attention_norm.weight,
                        shift_attn,
                        scale_attn,
                        indices[local],
                        eps=attention_norm.eps,
                        retain=hidden[local],
                    ),
                )

        attended = self.attention.forward_chunks(
            normalize(),
            workspace["cos"],
            workspace["sin"],
            inputs,
            workspace=workspace,
        )

        def finish(interval, update):
            local = slice(
                interval.start - inputs.token_slice.start,
                interval.stop - inputs.token_slice.start,
            )
            selected = indices[local]
            # gated_residual_rms_norm_fp8 returns E4M3 rows with [rows, 1]
            # FP32 scales, which is the row-wise FP8 encoding. Both gate and
            # up must consume exactly that encoding to borrow it without
            # requantizing.
            quantizer = gate.input_quantizer
            if (
                quantizer is not None
                and quantizer == Quantizer("fp8", axis=0)
                and up.input_quantizer == quantizer
            ):
                residual, values, scales = (
                    functional.gated_residual_rms_norm_fp8(
                        hidden[local],
                        update,
                        gate_attn,
                        mlp_norm.weight,
                        shift_mlp,
                        scale_mlp,
                        selected,
                        eps=mlp_norm.eps,
                    )
                )
                normalized: torch.Tensor = quantizer.from_tensors(
                    {"values": values, "scale": scales},
                    shape=tuple(values.shape),
                    dtype=hidden.dtype,
                )
            else:
                residual, normalized = functional.gated_residual_rms_norm(
                    hidden[local],
                    update,
                    gate_attn,
                    mlp_norm.weight,
                    shift_mlp,
                    scale_mlp,
                    selected,
                    eps=mlp_norm.eps,
                )
            return functional.gated_residual(
                residual, self.mlp(normalized), gate_mlp, selected
            )

        # Dynamic tensor-wide statistics (uncalibrated NVFP4, tensor-wide FP8)
        # require the complete shard before any feed-forward input is encoded.
        # Row-wise FP8, MXFP8 and calibrated NVFP4 consume intervals directly.
        quantizers = (
            gate.input_quantizer,
            up.input_quantizer,
            self.mlp.down.input_quantizer,
        )
        if any(
            value is not None and value.requires_complete_source
            for value in quantizers
        ):
            outputs = tuple(value for _, value in attended)
            update = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
            yield inputs.token_slice, finish(inputs.token_slice, update)
            return
        for interval, value in attended:
            yield interval, finish(interval, value)

    def forward(self, hidden, modulation, inputs: AttentionInput, *, workspace):
        outputs = tuple(
            value
            for _, value in self.forward_chunks(
                hidden, modulation, inputs, workspace=workspace
            )
        )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


class Transformer(nn.Module):
    """Evaluate resident H3 blocks and publish FP32 video/audio predictions.

    Latent input heads belong to the first pipeline stage. The final stage
    normalizes and projects each modality independently. The caller supplies
    contiguous activations and owns every state, constant, and workspace tensor.

    Construction builds every layer and both head pairs on each rank;
    ``weights._resident_layers`` then keeps only this pipeline stage's share.
    ``layers`` keys remain the global checkpoint layer numbers, while
    ``modulation`` is indexed by resident position.
    """

    def __init__(self, config: TransformerConfig, *, num_steps: int = 4):
        super().__init__()
        if type(num_steps) is not int or num_steps <= 0:
            raise ValueError("H3 transformer requires a positive step count")
        self.config = config
        self.mesh = DeviceMesh(ranks=(0,), shape=(1,), axes=("tp",), rank=0)
        self.video_input = Linear(
            config.video_channels * 4, config.hidden_size, dtype=torch.float32
        )
        self.audio_input = Linear(
            config.audio_channels, config.hidden_size, dtype=torch.float32
        )
        self.layers = nn.ModuleDict(
            {
                str(index): TransformerLayer(config)
                for index in range(config.num_hidden_layers)
            }
        )
        # Precomputed affine products of the checkpoint evaluation ladder:
        # [step, layer, timestep (video, audio), 3 token groups x 6 vectors x
        # hidden] per transformer layer, and [step, timestep, shift + scale]
        # for the final output norm. The empty tensors are placeholders that
        # the post-load hook weights._prepare_modulation replaces.
        self.modulation = Modulation(
            torch.empty(
                num_steps,
                config.num_hidden_layers,
                2,
                18 * config.hidden_size,
                dtype=torch.bfloat16,
            ),
            torch.empty(
                num_steps, 2, 2 * config.hidden_size, dtype=torch.bfloat16
            ),
        )
        self.output_norm = OutputNorm(config)
        self.video_output = Linear(
            config.hidden_size, config.video_channels * 4, dtype=torch.float32
        )
        self.audio_output = Linear(
            config.hidden_size, config.audio_channels, dtype=torch.float32
        )

    @torch.inference_mode()
    def forward(
        self,
        hidden: torch.Tensor,
        inputs: AttentionInput,
        *,
        step_index: int,
        tables: Mapping[str, torch.Tensor],
        workspace: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Evaluate the layers over one token shard.

        ``tables`` holds the per-row modulation indices and the rotary
        ``cos``/``sin`` of every packed row.

        A stage after the first receives its input into ``hidden`` from the
        preceding stage. A stage before the last sends its output onward and
        returns an empty tuple; the last stage returns the FP32 video and
        audio predictions for this shard's local video and audio rows.
        """
        pipeline = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        if pipeline.rank:
            pipeline.recv(src=pipeline.rank - 1, out=hidden)

        buffers = {
            **workspace,
            "modulation_indices": tables["modulation_indices"],
            "cos": tables["cos"],
            "sin": tables["sin"],
        }
        chunks = ((inputs.token_slice, hidden),)
        for index, layer in enumerate(self.layers.values()):
            layer = cast(TransformerLayer, layer)
            # Adjacent layers alternate between the two attention scratch sets
            # that Denoiser.workspace_buffers declares, so a completed chunk
            # can feed the next projection in place. The loop only composes
            # generators; the layers run when ``chunks`` is consumed below.
            prefix = f"attention.{index % 2}."
            layer_buffers = {
                **buffers,
                **{
                    name.removeprefix(prefix): value
                    for name, value in workspace.items()
                    if name.startswith(prefix)
                },
            }
            chunks = layer.forward_chunks(
                chunks,
                self.modulation(step_index, index),
                inputs,
                workspace=layer_buffers,
            )

        outputs = tuple(value for _, value in chunks)
        hidden = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        if pipeline.rank + 1 < pipeline.size:
            pipeline.send(hidden, dst=pipeline.rank + 1)
            return ()

        # The final pipeline stage projects each modality with its own head.
        modulation = self.modulation.output(step_index)
        results = []
        for index, (indices, projection) in enumerate(
            (
                (inputs.local_video_indices, self.video_output),
                (inputs.local_audio_indices, self.audio_output),
            )
        ):
            selected = hidden.index_select(0, indices)
            selected = self.output_norm(selected, modulation[index : index + 1])
            results.append(
                projection(selected.float(), output_dtype=torch.float32)
            )
        return tuple(results)
