"""Synthetic checkpoint metadata for model configuration tests."""

from __future__ import annotations

import json

from uniserve import loading
from uniserve_models import diffusion_gemma


def neo_metadata():
    """Return minimal SenseNova U1 checkpoint metadata."""
    return {
        "llm_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "vocab_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rope_theta": 10000.0,
        },
        "vision_config": {
            "hidden_size": 8,
            "llm_hidden_size": [16],
            "downsample_ratio": [0.5],
        },
        "pad_token_id": 3,
    }


def diffusion_gemma_metadata():
    """Return DiffusionGemma checkpoint metadata.

    The text stack is reduced to one sliding and one full layer; the
    vision tower and image tokens are those of the released checkpoints.
    """
    return {
        "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "model_type": "diffusion_gemma",
        "canvas_length": 256,
        "image_token_id": 258880,
        "boi_token_id": 255999,
        "eoi_token_id": 258882,
        "eos_token_id": [1, 106],
        "tie_word_embeddings": True,
        "vision_soft_tokens_per_image": 280,
        "text_config": {
            "vocab_size": 262144,
            "hidden_size": 2816,
            "intermediate_size": 2112,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_global_key_value_heads": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 1024,
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
            "hidden_activation": "gelu_pytorch_tanh",
            "attention_bias": False,
            "use_bidirectional_attention": "vision",
            "final_logit_softcapping": 30.0,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
            },
        },
        "vision_config": {
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 72,
            "hidden_activation": "gelu_pytorch_tanh",
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "position_embedding_size": 10240,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
            "standardize": True,
            "use_clipped_linears": False,
        },
    }


def read_diffusion_gemma(root, metadata):
    """Write DiffusionGemma metadata and tokenizer files, then read them."""
    (root / "config.json").write_text(json.dumps(metadata))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<pad>",
                "mask_token": "<mask>",
                "eot_token": "<turn|>",
            }
        )
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 0, "content": "<pad>"},
                    {"id": 4, "content": "<mask>"},
                    {"id": 106, "content": "<turn|>"},
                ]
            }
        )
    )
    return diffusion_gemma.read_config(root, loading.Config(), sources={})
