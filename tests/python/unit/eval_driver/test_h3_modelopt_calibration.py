"""Behavioral checks for checkpoint-owned FastH3 calibration schedules."""

import json

import pytest

from uniserve_eval.h3_modelopt_calibration import load_h3_inference_contract

pytestmark = pytest.mark.unit


def _checkpoint(tmp_path, *, forwards=8, steps=9, video_shift=10.0):
    contract = {
        "schema_version": "fasth3-inference-contract-v1",
        "transformer_forwards": forwards,
        "num_inference_steps": steps,
        "video_scheduler_shift": video_shift,
        "audio_scheduler_shift": 3.0,
        "vsa_sparsity": 0.8,
        "vsa_tile_size": 64,
    }
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps(contract))
    for name, shift in (("scheduler", 10.0), ("audio_scheduler", 3.0)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "scheduler_config.json").write_text(
            json.dumps({"shift": shift})
        )
    return tmp_path


def test_eight_forward_contract_is_loaded_from_checkpoint(tmp_path):
    contract = load_h3_inference_contract(_checkpoint(tmp_path))

    assert contract["transformer_forwards"] == 8
    assert contract["num_inference_steps"] == 9
    assert contract["vsa_sparsity"] == 0.8


@pytest.mark.parametrize(
    ("forwards", "steps", "video_shift", "message"),
    (
        (8, 8, 10.0, "inconsistent step counts"),
        (8, 9, 12.0, "scheduler shift 10.0 disagrees"),
    ),
)
def test_invalid_checkpoint_schedule_is_rejected(
    tmp_path, forwards, steps, video_shift, message
):
    model = _checkpoint(
        tmp_path,
        forwards=forwards,
        steps=steps,
        video_shift=video_shift,
    )

    with pytest.raises(ValueError, match=message):
        load_h3_inference_contract(model)
