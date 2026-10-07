"""Tiny checkpoint builders shared by model loading and runtime tests.

Each builder writes a seeded safetensors checkpoint under ``root`` whose
dimensions are small enough for CPU equation references.
"""

from __future__ import annotations

import torch
from safetensors.torch import save_file
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from uniserve import loading
from uniserve.diffusion import NoiseScale
from uniserve.loading import checkpoint, weights
from uniserve_models import bagel, siglip
from uniserve_models import sensenova_u1 as u1
from uniserve_models.bagel import vae
from uniserve_models.sensenova_u1 import flow, vision


def qwen_checkpoint(root, tied=False, theta=1_000_000.0):
    """Save a two-layer Qwen3 checkpoint and return its reference model."""
    config = Qwen3Config(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=tied,
        attention_bias=True,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=theta,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(481)
    model = Qwen3ForCausalLM(config).eval()
    model.save_pretrained(root)
    return model


def qwen_moe_checkpoint(root, *, norm_topk_prob=True, **overrides):
    """Save a two-layer Qwen3-MoE checkpoint and return its reference model.

    Transformers writes each expert's gate, up and down projections as
    separate tensors, the layout of released Qwen3-MoE checkpoints.
    ``overrides`` replace any ``Qwen3MoeConfig`` field.
    """
    fields = {
        "vocab_size": 37,
        "hidden_size": 32,
        "intermediate_size": 48,
        "moe_intermediate_size": 16,
        "num_experts": 6,
        "num_experts_per_tok": 2,
        "norm_topk_prob": norm_topk_prob,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
    }
    config = Qwen3MoeConfig(**(fields | overrides))
    config._attn_implementation = "eager"
    torch.manual_seed(907)
    model = Qwen3MoeForCausalLM(config).eval()
    model.save_pretrained(root)
    return model


def bagel_checkpoint(root):
    """Write a two-layer BAGEL checkpoint with a zero-update flow expert.

    Returns the Hugging Face Qwen reference for the text expert, the saved
    state and the matching typed config.
    """
    torch.manual_seed(662)
    config = Qwen3Config(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
        attention_bias=True,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(config).bfloat16().eval()
    # BAGEL has Q/K/V biases and an unbiased attention output contraction.
    for layer in reference.model.layers:
        layer.self_attn.o_proj.register_parameter("bias", None)
    state = {}
    for name, value in reference.state_dict().items():
        state["language_model." + name] = value
        if name == "model.norm.weight":
            state["language_model.model.norm_moe_gen.weight"] = value.clone()
        elif name.startswith("model.layers."):
            parts = name.split(".")
            if parts[3] == "self_attn":
                parts[4] += "_moe_gen"
            else:
                parts[3] += "_moe_gen"
            # A zero-update flow expert makes its residual identity observable
            # while text markers still traverse a nonzero independent expert.
            state["language_model." + ".".join(parts)] = torch.zeros_like(value)
    for prefix, out_features, in_features in (
        ("vae2llm", 32, 8),
        ("llm2vae", 8, 32),
        ("time_embedder.mlp.0", 32, 256),
        ("time_embedder.mlp.2", 32, 32),
    ):
        state[prefix + ".weight"] = (
            torch.randn(out_features, in_features) * 0.02
        ).bfloat16()
        state[prefix + ".bias"] = (torch.randn(out_features) * 0.01).bfloat16()
    state["latent_pos_embed.pos_embed"] = (
        torch.randn(16, 32) * 0.02
    ).bfloat16()
    # Declared, unselected vision fields can coexist in the same primary file.
    state["vit_pos_embed.pos_embed"] = torch.randn(16, 32).bfloat16()
    save_file(state, root / "ema.safetensors")
    config = bagel.Config(
        bagel.TransformerConfig(
            32, 48, 2, 4, 2, 37, 1e-6, 1_000_000.0, 8, True, 64
        ),
        siglip.Config(2, 8, 3, siglip.TransformerConfig(32, 4, 48, 1, 1e-6)),
        vae.Config(8, 3, 2, 32, 3, (1, 1), 1, 2, 0.5, 0.25),
        35,
        36,
        2,
        4,
        1.0,
        "gelu_pytorch_tanh",
    )
    return reference, state, config


def load_bagel(root, config):
    """Load the text and denoiser capabilities of :func:`bagel_checkpoint`."""
    return loading.load_model(
        bagel.Model,
        config,
        checkpoint=(
            checkpoint.Config(
                "primary", filenames=("ema.safetensors",)
            ).resolve(root, io=loading.Config()),
        ),
        mapping=bagel.checkpoint_mappings,
        device="cpu",
        weights=weights.Config(),
        modules=frozenset(("text", "denoiser")),
    ).model


def sensenova_checkpoint(root, dtype):
    """Write a two-layer SenseNova U1 checkpoint and load it on CPU.

    Generation-expert projections are zero so the image branch residual is
    observable. Returns the loaded model and the saved state.
    """
    torch.manual_seed(221)
    config = u1.Config(
        u1.TransformerConfig(37, 32, 48, 2, 4, 2, 8, ("full_attention",) * 2),
        vision.Config(16, 32, 0.5, 2, 3, 10000.0),
        flow.Config(
            flow.HeadConfig(32, 2, 1.0),
            False,
            True,
            NoiseScale(1.0, "constant", 1, 8),
        ),
        64,
    )
    state = {}

    def matrix(name, shape, *, bias=False):
        state[name + ".weight"] = (torch.randn(shape) * 0.02).to(dtype)
        if bias:
            state[name + ".bias"] = (torch.randn(shape[0]) * 0.01).to(dtype)

    matrix("language_model.model.embed_tokens", (37, 32))
    matrix("language_model.lm_head", (37, 32))
    for suffix in ("", "_mot_gen"):
        state[f"language_model.model.norm{suffix}.weight"] = torch.ones(
            32, dtype=dtype
        )
        for layer in range(2):
            prefix = f"language_model.model.layers.{layer}."
            for norm in ("input_layernorm", "post_attention_layernorm"):
                state[prefix + norm + suffix + ".weight"] = torch.ones(
                    32, dtype=dtype
                )
            for name, width in (("q", 32), ("k", 16), ("v", 16), ("o", 32)):
                matrix(prefix + f"self_attn.{name}_proj{suffix}", (width, 32))
                if suffix:
                    state[
                        prefix + f"self_attn.{name}_proj{suffix}.weight"
                    ].zero_()
            for name in ("q_norm", "q_norm_hw", "k_norm", "k_norm_hw"):
                state[prefix + f"self_attn.{name}{suffix}.weight"] = (
                    1 + torch.randn(4) * 0.1
                ).to(dtype)
            for name, shape in (
                ("gate", (48, 32)),
                ("up", (48, 32)),
                ("down", (32, 48)),
            ):
                matrix(prefix + f"mlp{suffix}.{name}_proj", shape)
                if suffix:
                    state[prefix + f"mlp{suffix}.{name}_proj.weight"].zero_()
    for prefix in ("vision_model", "fm_modules.vision_model_mot_gen"):
        matrix(prefix + ".embeddings.patch_embedding", (16, 3, 2, 2), bias=True)
        matrix(
            prefix + ".embeddings.dense_embedding", (32, 16, 2, 2), bias=True
        )
    for prefix in (
        "fm_modules.timestep_embedder",
        "fm_modules.noise_scale_embedder",
    ):
        matrix(prefix + ".mlp.0", (32, 256), bias=True)
        matrix(prefix + ".mlp.2", (32, 32), bias=True)
    matrix("fm_modules.fm_head.0", (32, 32), bias=True)
    matrix("fm_modules.fm_head.2", (48, 32), bias=True)
    save_file(state, root / "model.safetensors")
    model = loading.load_model(
        u1.Model,
        config,
        checkpoint=(
            checkpoint.Config("primary").resolve(root, io=loading.Config()),
        ),
        mapping=u1.checkpoint_mappings,
        weights=weights.Config(dtype=dtype),
        device="cpu",
    ).model
    return model, state
