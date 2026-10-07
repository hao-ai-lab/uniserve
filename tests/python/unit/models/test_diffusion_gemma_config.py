"""DiffusionGemma metadata normalization and its image token budget.

Unsupported mathematical options fail at the loading boundary, and the
declared image transform sizes images and counts their soft tokens as the
pinned Transformers processor and the server's prompt planner do.
"""

import json
from pathlib import Path

import pytest
from transformers.models.gemma4.image_processing_gemma4 import (
    get_aspect_ratio_preserving_size,
)

from tests.python.fixtures.model_metadata import (
    diffusion_gemma_metadata,
    read_diffusion_gemma,
)
from uniserve_models import diffusion_gemma

pytestmark = pytest.mark.unit

# Image sizes and soft-token counts; the server's prompt planner test reads
# the same table.
_IMAGE_TOKENS = json.loads(
    (
        Path(__file__).parents[2]
        / "fixtures"
        / "diffusion_gemma_image_tokens.json"
    ).read_text()
)


def test_metadata_normalizes_layers_windows_and_canvas_tokens(tmp_path):
    config = read_diffusion_gemma(tmp_path, diffusion_gemma_metadata())
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
    metadata = diffusion_gemma_metadata()
    metadata[section][field] = value
    with pytest.raises(ValueError, match=message):
        read_diffusion_gemma(tmp_path, metadata)


@pytest.mark.parametrize(
    "case",
    _IMAGE_TOKENS["cases"],
    ids=lambda case: f"{case['width']}x{case['height']}",
)
def test_image_soft_tokens_match_the_server_planner_table(tmp_path, case):
    config = read_diffusion_gemma(tmp_path, diffusion_gemma_metadata())
    transform = diffusion_gemma.image_processor(config).vit
    pooling = _IMAGE_TOKENS["pooling_kernel_size"]
    assert (
        transform.patch_size,
        transform.downsample,
        transform.resize.max_patches,
    ) == (
        _IMAGE_TOKENS["patch_size"],
        pooling,
        _IMAGE_TOKENS["max_soft_tokens"] * pooling**2,
    )

    height, width = case["height"], case["width"]
    assert transform.tokens(height, width) == case["soft_tokens"]
    assert transform.resized_size(
        height, width
    ) == get_aspect_ratio_preserving_size(
        height,
        width,
        patch_size=transform.patch_size,
        max_patches=transform.resize.max_patches,
        pooling_kernel_size=pooling,
    )
