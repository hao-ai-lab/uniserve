"""H3's modulated multimodal transformer and separate latent output heads."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import chain

import torch
from torch import nn

from uniserve import ops
from uniserve.distributed import DeviceMesh
from uniserve.nn import GatedMLP, Linear, Modulation, RMSNorm
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
        """Connect token-local residual updates to the following layer's projection."""  # noqa: E501
        if isinstance(hidden, torch.Tensor):
            source = ((inputs.token_slice, hidden),)
        else:
            source = iter(hidden)
            first = next(source)
            hidden = first[1].new_empty(
                (
                    inputs.token_slice.stop - inputs.token_slice.start,
                    self.hidden_size,
                )
            )
            source = chain((first,), source)
        indices = workspace["modulation_indices"]
        # Six per-token affine vectors: shift/scale/gate for attention and MLP.
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
                # through the next numerical stage's borrowed scratch.
                hidden[local].copy_(value)
                yield (
                    interval,
                    ops.modulated_rms_norm(
                        value,
                        self.norm[0].weight,
                        shift_attn,
                        scale_attn,
                        indices[local],
                        eps=self.norm[0].eps,
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
            quantizer = self.mlp.gate_up.projections["gate"].input_quantizer
            if (
                quantizer == Quantizer("fp8", axis=0)
                and self.mlp.gate_up.projections["up"].input_quantizer
                == quantizer
            ):
                residual, values, scales = ops.gated_residual_rms_norm_fp8(
                    hidden[local],
                    update,
                    gate_attn,
                    self.norm[1].weight,
                    shift_mlp,
                    scale_mlp,
                    selected,
                    eps=self.norm[1].eps,
                )
                normalized = quantizer.from_tensors(
                    {"values": values, "scale": scales},
                    shape=tuple(values.shape),
                    dtype=hidden.dtype,
                )
            else:
                residual, normalized = ops.gated_residual_rms_norm(
                    hidden[local],
                    update,
                    gate_attn,
                    self.norm[1].weight,
                    shift_mlp,
                    scale_mlp,
                    selected,
                    eps=self.norm[1].eps,
                )
            return ops.gated_residual(
                residual, self.mlp(normalized), gate_mlp, selected
            )

        # Tensor-wide activation statistics require the complete source domain.
        # Row/block quantizers retain interval consumption and transfer overlap.
        quantizers = (
            self.mlp.gate_up.projections["gate"].input_quantizer,
            self.mlp.gate_up.projections["up"].input_quantizer,
            self.mlp.down.input_quantizer,
        )
        if any(
            value is not None
            and (
                value.format == "nvfp4"
                or (value.format == "fp8" and value.axis is None)
            )
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
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
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
        # Precomputed affine products of the fixed four-evaluation ladder:
        # [step, layer, modality, packed shift/scale/gate] per transformer
        # layer, and [step, modality, shift + scale] for the final output
        # norm.
        self.modulation = Modulation(
            torch.empty(
                4,
                config.num_hidden_layers,
                2,
                18 * config.hidden_size,
                dtype=torch.bfloat16,
            ),
            torch.empty(4, 2, 2 * config.hidden_size, dtype=torch.bfloat16),
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
        constants: Mapping[str, torch.Tensor],
        workspace: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        pipeline = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        if pipeline.rank:
            pipeline.recv(src=pipeline.rank - 1, out=hidden)

        buffers = {
            **workspace,
            "modulation_indices": constants["modulation_indices"],
            "cos": constants["cos"],
            "sin": constants["sin"],
        }
        chunks = ((inputs.token_slice, hidden),)
        for index, layer in enumerate(self.layers.values()):
            # Adjacent layers alternate between the two attention scratch sets
            # so a completed chunk can feed the next projection in place.
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
