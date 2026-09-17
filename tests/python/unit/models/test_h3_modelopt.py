"""H3 calibrated manifests select only architecture-owned NVFP4 Linears."""

import json
import shutil
from pathlib import Path

import pytest
import torch

from tests.python.fixtures.launch import worker_args
from uniserve.loading import Config as IOConfig
from uniserve_models import loading as model_loading
from uniserve_models.minimax_h3 import Config, Model
from uniserve_models.minimax_h3.modelopt import (
    MODEL_OPT_COMMIT,
    calibrated_weight_config,
    denoiser_modules,
    text_modules,
    video_vae_modules,
)
from uniserve_worker.bootstrap.model_loader import prepare_worker_model

pytestmark = pytest.mark.unit


def _manifest(*, text=True, video_vae=True):
    groups = {
        "denoiser": denoiser_modules(),
        "text_encoder": text_modules() if text else (),
        "video_vae": video_vae_modules() if video_vae else (),
    }
    return {
        "schema_version": 1,
        "modelopt_commit": MODEL_OPT_COMMIT,
        "numerical_format": {
            "values": "e2m1",
            "block_size": 16,
            "block_scale": "fp8_e4m3",
            "tensor_scale": "fp32",
            "weight": "w4",
            "activation": "a4",
            "output": "bf16",
        },
        "components": {
            name: {
                "enabled": (
                    (name != "text_encoder" or text)
                    and (name != "video_vae" or video_vae)
                ),
                "modules": {path: {"activation_amax": 12.5} for path in paths},
            }
            for name, paths in groups.items()
        },
    }


@pytest.fixture
def packed_checkpoint(tmp_path):
    source = Path(__file__).parents[2] / "fixtures/models/fasth3"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}),
        encoding="utf-8",
    )
    (tmp_path / "modelopt_manifest.json").write_text(
        json.dumps(_manifest(text=False)), encoding="utf-8"
    )
    return tmp_path


def test_calibrated_manifest_has_exact_component_coverage():
    with torch.device("meta"):
        model = Model(Config())
    config = calibrated_weight_config(_manifest(text=False), model)

    assert len(config.quantization) == 150 + 252
    assert not any(
        path.startswith("text_encoder") for path in config.quantization
    )
    for choice in config.quantization.values():
        assert choice.weight.format == "nvfp4"
        assert choice.activation.calibrated_amax == 12.5


def test_calibrated_manifest_rejects_missing_target():
    with torch.device("meta"):
        model = Model(Config())
    manifest = _manifest()
    manifest["components"]["denoiser"]["modules"].pop(denoiser_modules()[0])
    with pytest.raises(ValueError, match="coverage mismatch"):
        calibrated_weight_config(manifest, model)


def test_calibrated_manifest_keeps_uncalibrated_vae_in_bf16():
    with torch.device("meta"):
        model = Model(Config())
    config = calibrated_weight_config(
        _manifest(text=False, video_vae=False), model
    )

    assert len(config.quantization) == 150
    assert not any(
        path.startswith(("text_encoder", "video_decoder"))
        for path in config.quantization
    )
    vae_root = "video_decoder.decoder.decoder"
    assert config.dtypes[f"{vae_root}.post_quant_conv"] == torch.bfloat16
    assert config.dtypes[f"{vae_root}.decoder.input"] == torch.bfloat16
    assert config.dtypes[f"{vae_root}.decoder.output"] == torch.bfloat16


def test_packed_checkpoint_loads_without_a_precision_selector(
    packed_checkpoint,
):
    source = model_loading.read_config(
        packed_checkpoint, io=IOConfig(), modules=frozenset()
    )

    assert source.checkpoint_format == "modelopt_nvfp4"
    assert not source.precisions
    assert len(source.weights.quantization) == 150 + 252


@pytest.mark.parametrize(
    "options",
    (
        '{"mode":"calibrated"}',
        '{"mode":"maximum"}',
        '{"components":{"mlp":"nvfp4"}}',
        '{"ignored_layers":["denoiser"]}',
    ),
)
def test_packed_checkpoint_rejects_runtime_numerical_overrides(
    packed_checkpoint, options, tmp_path
):
    config = worker_args(
        tmp_path,
        model=str(packed_checkpoint),
        entries={"output": {"ranks": [0], "parallel_config": {}}},
        max_batch_tokens=8192,
        quantization_config=json.loads(options),
    )

    with pytest.raises(ValueError, match="owns its numerical configuration"):
        prepare_worker_model(config)
