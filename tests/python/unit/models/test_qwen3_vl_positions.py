"""Qwen3-VL presentations take the Transformers M-RoPE coordinates."""

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLModel

from uniserve_models.qwen3_vl import rope_index

pytestmark = pytest.mark.unit

IMAGE, VIDEO, START, END = 151_655, 151_656, 151_652, 151_653


def _block(token, count):
    return [START, *(token,) * count, END]


# Presentations as H3 builds them: labels, vision blocks and timestamped
# video blocks between ordinary text tokens.
_CASES = {
    "text": ([11, 12, 13, 14, 15], (), ()),
    "image": ([7, 8, *_block(IMAGE, 6), 9, 10, 11], ((1, 4, 6),), ()),
    "adjacent_images": (
        [*_block(IMAGE, 6), *_block(IMAGE, 10), 4],
        ((1, 4, 6), (1, 2, 20)),
        (),
    ),
    "video": (
        [5, *_block(VIDEO, 4), 6, *_block(VIDEO, 4), 7, *_block(VIDEO, 4), 8],
        (),
        ((3, 4, 4),),
    ),
    "mixed": (
        [
            1,
            *_block(IMAGE, 8),
            2,
            3,
            *_block(VIDEO, 12),
            4,
            *_block(VIDEO, 12),
            5,
            *_block(IMAGE, 2),
        ],
        ((1, 8, 4), (1, 2, 4)),
        ((2, 6, 8),),
    ),
}


@pytest.fixture(scope="module")
def reference():
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 16,
            "deepstack_visual_indexes": [],
        },
    )
    with torch.device("meta"):
        return Qwen3VLModel(config)


@pytest.mark.parametrize("case", _CASES)
def test_positions_follow_the_transformers_rope_index(reference, case):
    ids, images, videos = _CASES[case]
    input_ids = torch.tensor([ids])
    types = torch.zeros_like(input_ids)
    types[input_ids == IMAGE] = 1
    types[input_ids == VIDEO] = 2
    expected, _ = reference.get_rope_index(
        input_ids,
        types,
        torch.tensor(images) if images else None,
        torch.tensor(videos) if videos else None,
    )

    actual = rope_index(
        ids,
        image_grids=images,
        video_grids=videos,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        spatial_merge_size=2,
    )
    assert actual.dtype == torch.int64
    assert torch.equal(actual, expected[:, 0])


@pytest.mark.parametrize(
    ("ids", "images", "error"),
    [
        (_block(IMAGE, 6), ((1, 4, 4),), "merged tokens"),
        (_block(IMAGE, 6), (), "has no grid"),
        ([1, 2], ((1, 4, 6),), "requires a placeholder run"),
        (_block(IMAGE, 3), ((1, 3, 4),), "whole merge blocks"),
    ],
)
def test_positions_reject_placeholders_that_disagree_with_grids(
    ids, images, error
):
    with pytest.raises(ValueError, match=error):
        rope_index(
            ids,
            image_grids=images,
            image_token_id=IMAGE,
            video_token_id=VIDEO,
            spatial_merge_size=2,
        )
