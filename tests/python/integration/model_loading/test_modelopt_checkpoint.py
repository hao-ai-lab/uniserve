"""ModelOpt unified checkpoints load through the public loading boundary."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.python.fixtures.checkpoints import qwen_checkpoint
from uniserve.nn.linear import Linear
from uniserve.quantization import QuantizedTensor
from uniserve_models import loading as models

pytestmark = pytest.mark.integration

# Checkpoint Linears the export quantizes, with distinct static input scales
# so a calibrated scale landing on the wrong module is observable.
PACKED = {
    "model.layers.0.mlp.gate_proj": 0.25,
    "model.layers.0.mlp.up_proj": 0.5,
    "model.layers.1.mlp.down_proj": 0.125,
}

# ModelOpt's unified Hugging Face quantization_config for static NVFP4 W4A4.
MODELOPT_NVFP4 = {
    "quant_method": "modelopt",
    "quant_algo": "NVFP4",
    "config_groups": {
        "group_0": {
            "weights": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            },
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            },
            "targets": ["Linear"],
        }
    },
    "ignore": ["lm_head"],
    "producer": {"name": "modelopt", "version": "0.37.0"},
}


def _export(root, quantization_config=MODELOPT_NVFP4):
    """Rewrite a dense checkpoint in ModelOpt's unified NVFP4 layout."""
    qwen_checkpoint(root)
    path = root / "model.safetensors"
    state = load_file(path)
    generator = torch.Generator().manual_seed(7)
    encoded = {}
    for name, input_scale in PACKED.items():
        rows, columns = state.pop(name + ".weight").shape
        # Loading preserves serialized encodings, so any E2M1 bytes and
        # finite E4M3 block scales exercise it.
        values = torch.randint(
            0, 256, (rows, columns // 2), dtype=torch.uint8, generator=generator
        )
        encoded[name] = values
        state[name + ".weight"] = values
        state[name + ".weight_scale"] = torch.ones(rows, columns // 16).to(
            torch.float8_e4m3fn
        )
        state[name + ".weight_scale_2"] = torch.tensor(
            0.01, dtype=torch.float32
        )
        state[name + ".input_scale"] = torch.tensor(
            input_scale, dtype=torch.float32
        )
    save_file(state, path)
    config = json.loads((root / "config.json").read_text())
    config["quantization_config"] = quantization_config
    (root / "config.json").write_text(json.dumps(config))
    return encoded


def test_packed_linears_run_with_their_weights_and_static_input_scales(
    tmp_path,
):
    encoded = _export(tmp_path)

    config = models.read_config(tmp_path)

    # The export owns its numerical representation: no runtime presets.
    assert config.checkpoint_format == "modelopt_nvfp4"
    assert not config.precisions
    # Exactly the stored-packed Linears quantize, each with the static
    # activation scale its input_scale records.
    quantized = config.weights.quantization
    assert sorted(
        setting.activation.calibrated_scale for setting in quantized.values()
    ) == sorted(PACKED.values())

    model = models.load_model(config, device="cpu").model
    packed = {
        path: module
        for path, module in model.named_modules()
        if isinstance(module, Linear)
        and isinstance(module.weight, QuantizedTensor)
    }
    assert set(packed) == set(quantized)
    stored = sorted(values.tolist() for values in encoded.values())
    assert (
        sorted(
            module.weight.buffers()["values"].tolist()
            for module in packed.values()
        )
        == stored
    )
    for path, module in packed.items():
        assert module.input_quantizer == quantized[path].activation


@pytest.mark.parametrize(
    "change",
    (
        {"quant_algo": "FP8"},
        {
            "config_groups": {
                "group_0": {
                    **MODELOPT_NVFP4["config_groups"]["group_0"],
                    "input_activations": {
                        "dynamic": True,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                }
            }
        },
    ),
)
def test_only_static_nvfp4_exports_are_accepted(tmp_path, change):
    _export(tmp_path, {**MODELOPT_NVFP4, **change})

    with pytest.raises(ValueError, match="static NVFP4"):
        models.read_config(tmp_path)
