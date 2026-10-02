"""H3's modulated multimodal transformer and separate latent output heads.

On the first pipeline stage, ``Denoiser.forward`` fills this rank's packed
hidden shard with its prefix rows and projected latent rows; every stage then
calls ``Transformer.forward`` once per solver step. Each
``TransformerLayer.forward_chunks`` yields ``(global row slice, rows)``
chunks, so a layer's completed rows feed the next layer's attention
projection before the whole shard is finished. Timestep conditioning comes
from ``Modulation`` products that ``weights._prepare_modulation`` precomputes
at load time for the fixed schedule; no timestep embedding runs per request.
A parallel-decoding student's output projections are likewise fused per
evaluation at load time (``StepProjection``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import chain
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

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

from . import config as configs
from .attention import Dense, Sparse
from .config import TransformerConfig
from .inputs import AttentionInput, SequenceInput
from .modulation import OutputNorm


class TransformerLayer(nn.Module):
    """Apply modality-indexed attention and feed-forward residual updates."""

    def __init__(self, config: TransformerConfig, attention: Dense | Sparse):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.rounding = config.rounding
        self.norm = nn.ModuleList(
            (
                RMSNorm(config.hidden_size, config.norm_eps),
                RMSNorm(config.hidden_size, config.norm_eps),
            )
        )
        self.attention = attention
        self.mlp = GatedMLP(
            config.hidden_size,
            config.intermediate_size,
            rounding=config.rounding,
        )

    @torch.inference_mode()
    def forward_chunks(
        self,
        hidden,
        modulation,
        inputs: AttentionInput | SequenceInput,
        *,
        workspace,
    ):
        """Connect token-local residual updates to the following layer's projection.

        Args:
            hidden: The layer input rows of ``inputs.token_slice``: the whole
                shard as one tensor, or an iterable of ``(global row slice,
                rows)`` chunks, such as a preceding layer's output.
                ``Transformer.forward`` passes the first layer one chunk
                covering the shard.
            modulation: This layer's products at the current step,
                ``[groups, 18 * hidden_size]``; see ``Transformer.modulation``.
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
        # [3 * groups, hidden] after the reshape. Row ``3 * group + tag`` is
        # the (video, text, audio) tag's products under the timestep group's
        # value; ``indices`` selects one row per token.
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
                        rounding=self.rounding,
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
                        rounding=self.rounding,
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
                    rounding=self.rounding,
                )
            return functional.gated_residual(
                residual,
                self.mlp(normalized),
                gate_mlp,
                selected,
                rounding=self.rounding,
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

    def forward(
        self,
        hidden,
        modulation,
        inputs: AttentionInput | SequenceInput,
        *,
        workspace,
    ):
        outputs = tuple(
            value
            for _, value in self.forward_chunks(
                hidden, modulation, inputs, workspace=workspace
            )
        )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


class StepProjection(nn.Module):
    """Project rows with one fused output head per solver evaluation.

    A parallel-decoding student predicts one head per fine-grid interval; an
    evaluation applies its block's heads fused with their normalized
    integration weights (``uniserve.diffusion.fuse_heads``). Loading fuses
    every block once (``weights._prepare_heads``) into ``weight`` [steps,
    out, in] and ``bias`` [steps, out], and a call gathers the step's head
    on the device, so one captured evaluation serves every step.
    """

    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, steps: int, in_features: int, out_features: int):
        super().__init__()
        # Placeholders the post-load hook replaces with the fused heads.
        self.register_buffer(
            "weight",
            torch.empty(
                (steps, out_features, in_features), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "bias", torch.empty((steps, out_features), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Project FP32 ``x`` [rows, in] with the [1] int64 ``step``'s head."""
        return F.linear(
            x,
            self.weight.index_select(0, step)[0],
            self.bias.index_select(0, step)[0],
        )


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

    def __init__(
        self,
        config: TransformerConfig,
        *,
        attention: configs.Attention,
        entries: int,
        steps: int,
        groups: int,
    ):
        """Build the network.

        Args:
            config: The network dimensions.
            attention: Dense or sparse attention for every layer.
            entries: Distinct timestep values the schedule evaluates.
            steps: Network evaluations of the schedule.
            groups: Timestep groups a step reads (see ``Modulation``).
        """
        super().__init__()
        if any(
            type(value) is not int or value <= 0
            for value in (entries, steps, groups)
        ):
            raise ValueError(
                "H3 transformer requires positive timestep entries, steps "
                "and groups"
            )
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
                str(index): TransformerLayer(
                    config,
                    Dense(config)
                    if isinstance(attention, configs.DenseAttention)
                    else Sparse(config, sparsity=attention.sparsity),
                )
                for index in range(config.num_hidden_layers)
            }
        )
        # Precomputed affine products of the schedule's distinct timesteps:
        # [layer, entry, 3 token tags x 6 vectors x hidden] per transformer
        # layer and [entry, shift + scale] for the final output norm, with
        # the [step, group] entries each step reads. The empty tensors are
        # placeholders that the post-load hook weights._prepare_modulation
        # replaces.
        self.modulation = Modulation(
            torch.empty(
                config.num_hidden_layers,
                entries,
                18 * config.hidden_size,
                dtype=torch.bfloat16,
            ),
            torch.empty(entries, 2 * config.hidden_size, dtype=torch.bfloat16),
            torch.zeros((steps, groups), dtype=torch.int64),
        )
        self.output_norm = OutputNorm(config)
        self.video_output: Linear | StepProjection
        self.audio_output: Linear | StepProjection
        if config.output_heads == 1:
            self.video_output = Linear(
                config.hidden_size,
                config.video_channels * 4,
                dtype=torch.float32,
            )
            self.audio_output = Linear(
                config.hidden_size, config.audio_channels, dtype=torch.float32
            )
        else:
            self.video_output = StepProjection(
                steps, config.hidden_size, config.video_channels * 4
            )
            self.audio_output = StepProjection(
                steps, config.hidden_size, config.audio_channels
            )

    @torch.inference_mode()
    def forward(
        self,
        hidden: torch.Tensor,
        inputs: AttentionInput | SequenceInput,
        *,
        step: torch.Tensor,
        tables: Mapping[str, torch.Tensor],
        workspace: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Evaluate the layers over one token shard.

        ``step`` is the solver step's [1] int64 device index; the step's
        modulation products are gathered through it, so one captured
        evaluation serves every step. ``tables`` holds the per-row modulation
        indices and the rotary ``cos``/``sin`` of the rows the attention kind
        reads (every packed row for sparse attention, the shard's rows for
        dense attention).

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
        # [resident layer, group, 18 * hidden] products of this step.
        modulation = self.modulation(step)
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
                modulation[index],
                inputs,
                workspace=layer_buffers,
            )

        outputs = tuple(value for _, value in chunks)
        hidden = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        if pipeline.rank + 1 < pipeline.size:
            pipeline.send(hidden, dst=pipeline.rank + 1)
            return ()

        # The final pipeline stage projects each modality with its own head;
        # the output norm reads the generated video and audio timesteps,
        # groups 0 and 1.
        modulation = self.modulation.output(step)
        results = []
        for index, (indices, projection) in enumerate(
            (
                (inputs.local_video_indices, self.video_output),
                (inputs.local_audio_indices, self.audio_output),
            )
        ):
            selected = hidden.index_select(0, indices)
            selected = self.output_norm(selected, modulation[index : index + 1])
            if isinstance(projection, StepProjection):
                results.append(projection(selected.float(), step))
            else:
                results.append(
                    projection(selected.float(), output_dtype=torch.float32)
                )
        return tuple(results)
