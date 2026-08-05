"""One declared image operating point over canonical and external request schemas.

UniServe receives only ``image_config``. External backends may also receive the
request-root aliases their implementations require.
"""

from __future__ import annotations

import pytest

from uniserve_eval.harness.spec import BenchmarkSpec, TaskName
from uniserve_eval.harness.tasks.t2i import T2ITask

pytestmark = pytest.mark.unit

# Canonical ``image_config`` name -> the request-root names servers read it from.
ROOT_NAMES = {
    "steps": ("num_inference_steps",),
    "seed": ("seed",),
    "num_images": ("num_outputs_per_prompt",),
    "guidance_scale": ("guidance_scale", "cfg_scale", "cfg_text_scale"),
    "image_guidance_scale": ("image_guidance_scale", "img_cfg_scale", "cfg_img_scale"),
    "cfg_norm": ("cfg_norm", "cfg_renorm_type"),
    "cfg_interval": ("cfg_interval",),
    "timestep_shift": ("timestep_shift",),
    "think": ("think",),
    "t_eps": ("t_eps",),
}


def _spec(**overrides) -> BenchmarkSpec:
    values = {
        "task": TaskName.T2I,
        "model": "SenseNova-U1",
        "wire": "openai_chat_json",
        "width": 2048,
        "height": 1152,
        "steps": 50,
        "seed": 42,
        "max_images": 1,
        "guidance_scale": 4.0,
        "image_guidance_scale": 1.0,
        "cfg_norm": "none",
        "cfg_interval": (0.0, 1.0),
        "timestep_shift": 3.0,
        "image_think": False,
        "image_t_eps": 0.02,
        "num_prompts": 1,
    }
    values.update(overrides)
    return BenchmarkSpec(**values)


def _request(**overrides):
    return T2ITask(_spec(**overrides)).build_request({"prompt": "a red cube"})


def _payload(**overrides) -> dict:
    return _request(**overrides).payload


def test_image_only_chat_requests_target_the_chat_completions_wire():
    request = _request()

    assert request.endpoint == "/v1/chat/completions"
    assert request.kind == "openai_chat_json"
    assert request.payload["modalities"] == ["image"]
    assert request.payload["messages"] == [{"role": "user", "content": "a red cube"}]


def test_canonical_image_config_states_every_declared_parameter():
    image_config = _payload()["image_config"]

    assert image_config == {
        "width": 2048,
        "height": 1152,
        "steps": 50,
        "seed": 42,
        "num_images": 1,
        "guidance_scale": 4.0,
        "image_guidance_scale": 1.0,
        "cfg_norm": "none",
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 3.0,
    }


def test_root_only_parameters_stay_out_of_the_canonical_image_config():
    payload = _payload()

    assert "think" not in payload["image_config"]
    assert "t_eps" not in payload["image_config"]
    assert "think" not in payload
    assert "t_eps" not in payload


def test_vllm_omni_request_schema_receives_every_required_root_alias():
    payload = _payload(request_schema="vllm_omni")
    declared = {**payload["image_config"], "think": False, "t_eps": 0.02}

    for name, root_names in ROOT_NAMES.items():
        for root_name in root_names:
            assert payload[root_name] == declared[name], root_name


def test_canonical_output_size_stays_in_image_config():
    payload = _payload()

    assert payload["image_config"]["width"] == 2048
    assert payload["image_config"]["height"] == 1152
    assert "width" not in payload
    assert "height" not in payload
    assert "size" not in payload


def test_undeclared_parameters_stay_absent_from_the_request():
    payload = _payload(timestep_shift=None, image_t_eps=None)

    assert "timestep_shift" not in payload
    assert "t_eps" not in payload
    assert "timestep_shift" not in payload["image_config"]


def test_extra_request_body_resolves_the_final_request_value():
    payload = _payload(extra_request_body={"num_inference_steps": 8})

    assert payload["num_inference_steps"] == 8
