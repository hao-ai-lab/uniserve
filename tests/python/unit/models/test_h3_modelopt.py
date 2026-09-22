"""H3 calibrated manifests select only architecture-owned NVFP4 Linears."""

import json
import re
import shutil
from pathlib import Path

import pytest
import torch

from tests.python.fixtures.launch import worker_args
from uniserve.loading import Config as IOConfig
from uniserve.nn import ColumnParallelLinear, RowParallelLinear
from uniserve.quantization import QuantizationConfig, Quantizer
from uniserve_models import loading as model_loading
from uniserve_models.minimax_h3 import Config, Model
from uniserve_models.minimax_h3.modelopt import (
    MODEL_OPT_COMMIT,
    calibrated_weight_config,
)
from uniserve_worker.bootstrap.model_loader import prepare_worker_model

pytestmark = pytest.mark.unit

# Calibrated H3 targets: the three MLP projections of each of the 50 denoiser
# layers and all seven Linears of each of the 36 VAE decoder Transformer
# layers. Attention projections, conditioning and convolution stay BF16.
DENOISER_TARGETS = 50 * 3
VIDEO_VAE_TARGETS = 36 * 7
_TARGETS = {
    "denoiser": re.compile(r"denoiser\.transformer\.layers\.\d+\.mlp\."),
    "video_vae": re.compile(
        r"video_decoder\.decoder\.decoder\.decoder\.layers\.\d+\."
    ),
}


@pytest.fixture(scope="module")
def model():
    with torch.device("meta"):
        return Model(Config())


def _calibration(model, *, video_vae=True):
    """Return a ModelOpt export sample with one distinct amax per target.

    Targets come from the model's Linear leaves rather than the loader's own
    path tables, so a loader that selects the wrong owners fails coverage.
    """
    enabled = {"denoiser": True, "text_encoder": False, "video_vae": video_vae}
    linears = [
        path
        for path, module in model.named_modules()
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear))
    ]
    components = {}
    for name, on in enabled.items():
        pattern = _TARGETS.get(name)
        paths = [p for p in linears if on and pattern and pattern.match(p)]
        components[name] = {
            "enabled": on,
            # Distinct calibrated maxima expose a path/amax misassignment.
            "modules": {
                path: {"activation_amax": 0.5 + index / 64}
                for index, path in enumerate(paths)
            },
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
        "components": components,
    }


def _amax(manifest):
    return {
        path: declaration["activation_amax"]
        for component in manifest["components"].values()
        for path, declaration in component["modules"].items()
    }


@pytest.fixture
def packed_checkpoint(tmp_path, model):
    source = Path(__file__).parents[2] / "fixtures/models/fasth3"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}),
        encoding="utf-8",
    )
    (tmp_path / "modelopt_manifest.json").write_text(
        json.dumps(_calibration(model)), encoding="utf-8"
    )
    return tmp_path


def test_calibrated_manifest_quantizes_each_target_with_its_amax(model):
    manifest = _calibration(model)

    config = calibrated_weight_config(manifest, model)

    amax = _amax(manifest)
    assert len(amax) == DENOISER_TARGETS + VIDEO_VAE_TARGETS
    assert config.quantization == {
        path: QuantizationConfig(
            Quantizer("nvfp4"), Quantizer("nvfp4", calibrated_amax=value)
        )
        for path, value in amax.items()
    }


def test_calibrated_manifest_rejects_missing_calibration(model):
    manifest = _calibration(model)
    modules = manifest["components"]["denoiser"]["modules"]
    modules.pop(next(iter(modules)))

    with pytest.raises(ValueError, match="coverage mismatch"):
        calibrated_weight_config(manifest, model)


def test_calibrated_manifest_keeps_uncalibrated_vae_in_bf16(model):
    manifest = _calibration(model, video_vae=False)

    config = calibrated_weight_config(manifest, model)

    assert set(config.quantization) == set(_amax(manifest))
    assert len(config.quantization) == DENOISER_TARGETS
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
    assert len(source.weights.quantization) == (
        DENOISER_TARGETS + VIDEO_VAE_TARGETS
    )


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
        components={"muxer": {"ranks": [0], "parallel_config": {}}},
        max_batch_tokens=8192,
        quantization_config=json.loads(options),
    )

    with pytest.raises(ValueError, match="owns its numerical configuration"):
        prepare_worker_model(config)
