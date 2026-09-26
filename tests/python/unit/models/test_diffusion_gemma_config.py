"""DiffusionGemma metadata normalization and Gemma-4 image preprocessing.

Unsupported mathematical options fail at the loading boundary, and image
sizes, patches and positions follow the pinned Transformers processor.
"""

import copy
import json

import pytest
import torch
from transformers.models.gemma4.image_processing_gemma4 import (
    Gemma4ImageProcessor,
)

from uniserve import loading
from uniserve_models import diffusion_gemma
from uniserve_models.diffusion_gemma import processing

pytestmark = pytest.mark.unit

_METADATA = {
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


def _read(root, metadata):
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


def test_metadata_normalizes_layers_windows_and_canvas_tokens(tmp_path):
    config = _read(tmp_path, _METADATA)
    sliding, full = config.text.layers
    # The checkpoint's window counts the query itself.
    assert (sliding.kind, sliding.window, sliding.head_dim) == (
        "sliding",
        1023,
        256,
    )
    assert (full.kind, full.window, full.num_kv_heads) == ("full", None, 2)
    assert (full.rotary.kind, full.rotary.partial_rotary_factor) == (
        "proportional",
        0.25,
    )
    assert config.vision.max_patches == 2520
    diffusion = config.diffusion
    assert (
        diffusion.mask_token_id,
        diffusion.pad_token_id,
        diffusion.end_of_turn_id,
        diffusion.eos_token_ids,
    ) == (4, 0, 106, (1, 106))


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        (
            "text_config",
            "rope_parameters",
            {
                "full_attention": {"rope_theta": 1e6, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 1e4},
            },
            "full layers require proportional rope",
        ),
        (
            "text_config",
            "use_bidirectional_attention",
            "all",
            "bidirectional attention within images only",
        ),
        (
            "text_config",
            "use_bidirectional_attention",
            None,
            "bidirectional attention within images only",
        ),
        (
            "vision_config",
            "use_clipped_linears",
            True,
            "clipped vision linears are unsupported",
        ),
    ],
)
def test_unsupported_mathematics_is_rejected(
    tmp_path, section, field, value, message
):
    metadata = copy.deepcopy(_METADATA)
    metadata[section][field] = value
    with pytest.raises(ValueError, match=message):
        _read(tmp_path, metadata)


@pytest.mark.parametrize(
    "height,width",
    [(480, 640), (224, 224), (1080, 1920), (1600, 90), (10, 8000), (8000, 10)],
)
def test_image_sizes_patches_and_positions_follow_the_processor(
    tmp_path, height, width
):
    config = _read(tmp_path, _METADATA).vision
    reference = Gemma4ImageProcessor(
        patch_size=16, max_soft_tokens=280, pooling_kernel_size=3
    )
    generator = torch.Generator().manual_seed(height * 7 + width)
    image = torch.randint(
        0, 256, (3, height, width), dtype=torch.uint8, generator=generator
    )
    expected = reference(images=[image], return_tensors="pt")

    processed = processing.preprocess(image, config)
    rows, positions = processing.patches(processed, config)
    count = rows.shape[0]
    assert processing.soft_tokens(height, width, config) == int(
        expected["num_soft_tokens_per_image"][0]
    )
    assert tuple(processed.shape[1:]) == processing.resized_size(
        height, width, config
    )
    assert count * 256 == processed.shape[1] * processed.shape[2]
    torch.testing.assert_close(
        rows, expected["pixel_values"][0, :count], rtol=0, atol=0
    )
    assert torch.equal(positions, expected["image_position_ids"][0, :count])
    # Every patch beyond the image is padding.
    assert (expected["image_position_ids"][0, count:] == -1).all()
