from __future__ import annotations

import pytest
import torch

from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.models.sensenova.interleaved_image import TextImageDenoiseOps
from uniserve_worker.runtime.masks import build_commit_attention_mask

pytestmark = pytest.mark.unit


def test_commit_attention_mask_blocks_image_tokens_from_end_marker_only():
    mask = build_commit_attention_mask(num_image_tokens=3, past_len=5, device="cpu")

    assert mask.shape == (1, 1, 4, 9)
    assert torch.isneginf(mask[0, 0, :3, 8]).all()
    assert mask[0, 0, 3, 8] == 0
    visible = mask.clone()
    visible[0, 0, :3, 8] = 0
    assert torch.equal(visible, torch.zeros_like(visible))


def test_image_params_preserve_zero_cfg_values_and_reject_zero_steps():
    ops = TextImageDenoiseOps()

    params = ops._parse_image_params(
        {
            "height": 512,
            "width": 512,
            "steps": 1,
            "cfg_text_scale": 0.0,
            "cfg_img_scale": 0.0,
            "cfg_interval": [0.0, 1.0],
            "cfg_renorm_type": "none",
            "cfg_renorm_min": 0.0,
            "timestep_shift": 0.5,
        }
    )

    assert params.cfg_text == 0.0
    assert params.cfg_img == 0.0
    assert params.cfg_renorm_min == 0.0
    assert params.timestep_shift == 0.5

    with pytest.raises(WorkerError):
        ops._parse_image_params({"height": 512, "width": 512, "steps": 0})
    with pytest.raises(WorkerError):
        ops._parse_image_params({"steps": 1})
    with pytest.raises(WorkerError):
        ops._parse_image_params({"height": 512, "width": 512, "steps": 1})
