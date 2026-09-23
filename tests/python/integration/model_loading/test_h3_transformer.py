"""H3's loaded sparse transformer preserves its numerical equations.

Each preserved equation uses a fixed step.
"""

import pytest
import torch
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
)
from safetensors.torch import save_file
from torch.nn import functional as F

from tests.python.fixtures.vsa import reference_attention
from uniserve import loading
from uniserve.distributed import DeviceMesh
from uniserve.loading import checkpoint, weights
from uniserve.nn.attention import vsa
from uniserve.runtime import CUDAGraph, ExecutionContext, TensorBuffers
from uniserve_models.minimax_h3 import (
    AttentionInput,
    DiffusionConfig,
    Packing,
    Transformer,
    TransformerConfig,
)
from uniserve_models.minimax_h3.config import TRANSFORMER_FIELDS
from uniserve_models.minimax_h3.weights import transformer_component

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _reference(hidden, source, config, step, tags, cosine, sine, valid):
    def parameter(name, dtype=torch.bfloat16):
        return source[name].to(device=hidden.device, dtype=dtype)

    def linear(x, prefix, dtype=torch.bfloat16):
        bias = (
            parameter(prefix + ".bias", dtype)
            if prefix + ".bias" in source
            else None
        )
        return F.linear(x.to(dtype), parameter(prefix + ".weight", dtype), bias)

    # The Preview export's DMD rungs on the 1000-step clock, shifted by the
    # video (12) and audio (3) scheduler shifts.
    rung = (999, 749, 500, 250)[step] / 1000
    sigma = torch.tensor(
        [
            12.0 * rung / (1 + 11.0 * rung),
            3.0 * rung / (1 + 2.0 * rung),
        ],
        device=hidden.device,
    )
    features = get_timestep_embedding(
        1.0 - sigma,
        config.frequency_dim,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
    )
    times = F.silu(
        linear(
            F.silu(linear(features, "time_embedder.linear_1", torch.float32)),
            "time_embedder.linear_2",
            torch.float32,
        )
    )
    indices = (tags == 2).long() * 3 + tags
    for index in range(config.num_hidden_layers):
        prefix = f"transformer_blocks.{index}"
        vectors = (
            linear(times, prefix + ".adaln_proj.linear")
            .reshape(6, 6 * config.hidden_size)
            .index_select(0, indices)
            .float()
        )
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = vectors.chunk(
            6, dim=-1
        )
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + config.norm_eps
        )
        normalized = (
            normalized
            * parameter(prefix + ".norm1.weight").float()
            * (1 + scale_a)
            + shift_a
        ).bfloat16()
        projections = [
            linear(normalized, prefix + ".attn.to_" + name).view(
                256, config.num_attention_heads, 128
            )
            for name in ("q", "k", "v", "gate_compress")
        ]
        for branch, name in ((0, "q"), (1, "k")):
            value = projections[branch].float()
            value = (
                value
                * torch.rsqrt(
                    value.square().mean(-1, keepdim=True) + config.qk_norm_eps
                )
                * parameter(prefix + ".attn.norm_" + name + ".weight").float()
            )
            width = cosine.shape[-1]
            first, second = (
                value[..., :width].clone(),
                value[..., width : 2 * width].clone(),
            )
            value[..., :width] = (
                first * cosine[:, None] - second * sine[:, None]
            )
            value[..., width : 2 * width] = (
                second * cosine[:, None] + first * sine[:, None]
            )
            projections[branch] = value.bfloat16()
        attended, live = reference_attention(*projections, valid)
        update = linear(attended.flatten(1), prefix + ".attn.to_out.0")
        residual = hidden.float() + gate_a * update.float()
        normalized = residual * torch.rsqrt(
            residual.square().mean(-1, keepdim=True) + config.norm_eps
        )
        normalized = (
            normalized
            * parameter(prefix + ".norm2.weight").float()
            * (1 + scale_f)
            + shift_f
        ).bfloat16()
        value, gate = linear(normalized, prefix + ".ff.net.0.proj").chunk(
            2, dim=-1
        )
        mlp = linear(
            (F.silu(gate.float()) * value.float()).bfloat16(),
            prefix + ".ff.net.2",
        )
        hidden = (residual.bfloat16().float() + gate_f * mlp.float()).bfloat16()
    shift, scale = linear(times, "norm_out.linear").chunk(2, dim=-1)
    normalized = (
        hidden.float()
        * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + config.norm_eps
        )
        * parameter("norm_out.norm.weight").float()
    ).bfloat16()
    results = []
    for tag, row, prefix in ((0, 0, "proj_out"), (2, 1, "audio_proj_out")):
        selected = normalized[(tags == tag) & live]
        selected = selected * (1.0 + scale[row]) + shift[row]
        results.append(linear(selected, prefix, torch.float32))
    return tuple(results)


@torch.inference_mode()
def test_loaded_modulated_sparse_transformer_and_graph(tmp_path):
    config = TransformerConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_refiner_layers=1,
        intermediate_size=128,
        text_dim=40,
        frequency_dim=16,
        time_hidden_dim=64,
        time_dim=32,
        rope_frequency_dim=4,
    )
    torch.manual_seed(181)
    native = MiniMaxH3Transformer3DModel(
        **{
            source: getattr(config, target)
            for source, target in TRANSFORMER_FIELDS.items()
        },
        patch_size=(1, 2, 2),
        final_norm_eps=config.norm_eps,
    )
    source = native.state_dict()
    for index in range(config.num_hidden_layers):
        source[f"transformer_blocks.{index}.attn.to_gate_compress.weight"] = (
            torch.randn(
                config.num_attention_heads * config.head_dim, config.hidden_size
            )
            * 0.04
        )
    save_file(source, tmp_path / "model.safetensors")
    model = loading.load_model(
        Transformer,
        config,
        checkpoint=(
            checkpoint.Config("denoiser").resolve(
                tmp_path, io=loading.Config()
            ),
        ),
        mapping=lambda model: (
            transformer_component(model, DiffusionConfig()),
        ),
        device="cuda",
        weights=weights.Config(
            dtypes=dict.fromkeys(
                (
                    "video_input",
                    "audio_input",
                    "video_output",
                    "audio_output",
                ),
                torch.float32,
            )
        ),
    ).model
    valid = torch.tensor([64, 17, 64, 0], dtype=torch.int32, device="cuda")
    tags = torch.zeros(256, dtype=torch.int64, device="cuda")
    tags[:32], tags[32:64] = 1, 2
    live = torch.arange(256, device="cuda") % 64 < valid.repeat_interleave(64)
    selections = {
        name: torch.where((tags == tag) & live)[0]
        for name, tag in (("text", 1), ("audio", 2), ("video", 0))
    }
    positions = (
        torch.arange(768, device="cuda", dtype=torch.float32).reshape(256, 3)
        / 17
    )
    frequencies = config.rope_theta ** (
        -torch.arange(
            config.rope_frequency_dim, device="cuda", dtype=torch.float32
        )
        / config.rope_frequency_dim
    )
    angles = (positions[..., None] * frequencies).flatten(1)
    cosine, sine = angles.cos(), angles.sin()
    packing = Packing(
        num_tokens=int(live.sum()),
        padded_tokens=256,
        position_ids=positions.cpu(),
        token_tags=tags.cpu(),
        text_indices=selections["text"].cpu(),
        audio_indices=selections["audio"].cpu(),
        video_indices=selections["video"].cpu(),
        video_raster_indices=torch.arange(selections["video"].numel()),
        video_untile_indices=selections["video"].cpu(),
        tile_valid_sizes=valid.cpu(),
        prefix_tiles=1,
        video_tiles=2,
        video_frames=1,
        latent_height=1,
        latent_width=1,
        audio_frames=16,
    )
    group = DeviceMesh(ranks=(0,), shape=(1,), axes=("tp",), rank=0).get_group(
        ()
    )
    sparse = vsa.Input(
        256,
        1,
        2,
        3,
        valid,
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.arange(3, device="cuda", dtype=torch.int32),
        torch.tensor(1, device="cuda", dtype=torch.int32),
    )
    inputs = AttentionInput(
        packing,
        slice(0, 256),
        group,
        sparse,
        selections["text"],
        selections["video"],
        selections["audio"],
    )
    tables = {
        "cos": cosine,
        "sin": sine,
        "modulation_indices": (tags == 2).long() * 3 + tags,
    }
    requirements = {
        f"attention.{slot}.{name}": config
        for slot in range(2)
        for name, config in model.layers["0"]
        .attention.workspace_buffers(256, 256, dtype=torch.bfloat16)
        .items()
    }
    hidden = torch.randn(
        256, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    with TensorBuffers.allocate(requirements, device="cuda") as owner:
        workspace = owner.view(requirements)
        with ExecutionContext(model, vsa="cute") as context:
            context.prepare(None)
            for step in range(4):
                expected = _reference(
                    hidden, source, config, step, tags, cosine, sine, valid
                )
                actual = model(
                    hidden,
                    inputs,
                    step_index=step,
                    tables=tables,
                    workspace=workspace,
                )
                for result, reference in zip(actual, expected, strict=True):
                    torch.testing.assert_close(
                        result, reference, rtol=2e-2, atol=2e-2
                    )
            with CUDAGraph(context=context) as graph:
                graph.capture(
                    lambda: model(
                        hidden,
                        inputs,
                        step_index=3,
                        tables=tables,
                        workspace=workspace,
                    )
                )
                hidden.mul_(0.5)
                actual = graph.replay()
                expected = _reference(
                    hidden, source, config, 3, tags, cosine, sine, valid
                )
                for result, reference in zip(actual, expected, strict=True):
                    torch.testing.assert_close(
                        result, reference, rtol=2e-2, atol=2e-2
                    )
