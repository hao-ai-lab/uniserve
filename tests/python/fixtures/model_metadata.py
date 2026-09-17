"""Synthetic checkpoint metadata for model configuration tests."""

from __future__ import annotations


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
