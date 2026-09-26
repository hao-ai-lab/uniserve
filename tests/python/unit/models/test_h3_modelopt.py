"""A calibrated H3 export owns its numerical representation."""

import json
import shutil
from pathlib import Path

import pytest
import torch

from tests.python.fixtures.launch import worker_args
from uniserve.loading import Config as IOConfig
from uniserve_models import loading as model_loading
from uniserve_worker.bootstrap.model_loader import prepare_worker_model

pytestmark = pytest.mark.unit

# ModelOpt's unified Hugging Face quantization_config for static NVFP4 W4A4,
# which its diffusers export writes into each quantized component's config.
MODELOPT_NVFP4 = {
    "quant_method": "modelopt",
    "quant_algo": "NVFP4",
    "config_groups": {
        "group_0": {
            role: {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            }
            for role in ("weights", "input_activations")
        }
    },
    "ignore": [],
}


@pytest.fixture(params=("unified", "manifest"))
def packed_checkpoint(tmp_path, request):
    source = Path(__file__).parents[2] / "fixtures/models/fasth3"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}),
        encoding="utf-8",
    )
    if request.param == "manifest":
        (tmp_path / "modelopt_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "numerical_format": {
                        "activation": "a4",
                        "block_scale": "fp8_e4m3",
                        "block_size": 16,
                        "output": "bf16",
                        "tensor_scale": "fp32",
                        "values": "e2m1",
                        "weight": "w4",
                    },
                    "components": {
                        "denoiser": {
                            "enabled": True,
                            "modules": {
                                "denoiser.transformer.layers.0.mlp.down": {
                                    "activation_amax": 2688
                                }
                            },
                        }
                    },
                }
            )
        )
        return tmp_path
    for component in ("transformer", "vae"):
        path = tmp_path / component / "config.json"
        config = json.loads(path.read_text())
        config["quantization_config"] = MODELOPT_NVFP4
        path.write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_packed_checkpoint_keeps_dense_modules_in_bf16(packed_checkpoint):
    source = model_loading.read_config(
        packed_checkpoint, io=IOConfig(), modules=frozenset()
    )

    assert source.checkpoint_format == "modelopt_nvfp4"
    assert not source.precisions
    # The quantized VAE layers produce BF16, so the VAE boundaries the export
    # leaves dense run in BF16 as well rather than the FP16 of dense presets.
    vae_root = "video_decoder.decoder.decoder"
    for path in (
        f"{vae_root}.post_quant_conv",
        f"{vae_root}.decoder.input",
        f"{vae_root}.decoder.output",
    ):
        assert source.weights.dtypes[path] == torch.bfloat16


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
