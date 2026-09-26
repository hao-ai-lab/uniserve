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


def _export(root, quantization_config=MODELOPT_NVFP4, *, manifest=False):
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
        state[name + (".weight_packed" if manifest else ".weight")] = values
        state[name + ".weight_scale"] = torch.ones(rows, columns // 16).to(
            torch.float8_e4m3fn
        )
        state[
            name + (".weight_tensor_scale" if manifest else ".weight_scale_2")
        ] = torch.tensor(0.01, dtype=torch.float32)
        if not manifest:
            state[name + ".input_scale"] = torch.tensor(
                input_scale, dtype=torch.float32
            )
    save_file(state, path)
    config = json.loads((root / "config.json").read_text())
    if manifest:
        # The packed manifest names public numerical modules rather than
        # checkpoint fields, including separate branches of a gated MLP.
        modules = {}
        for name, scale in PACKED.items():
            path = name.replace("model.", "backbone.")
            path = path.replace("gate_proj", "gate_up.projections.gate")
            path = path.replace("up_proj", "gate_up.projections.up")
            path = path.replace("down_proj", "down")
            modules[path] = {"activation_amax": scale * 6 * 448}
        (root / "modelopt_manifest.json").write_text(
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
                        "decoder": {"enabled": True, "modules": modules}
                    },
                }
            )
        )
    else:
        config["quantization_config"] = quantization_config
    (root / "config.json").write_text(json.dumps(config))
    return encoded


@pytest.mark.parametrize("manifest", (False, True))
def test_packed_linears_run_with_their_weights_and_static_input_scales(
    tmp_path,
    manifest,
):
    encoded = _export(tmp_path, manifest=manifest)

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
        buffers = module.weight.buffers()
        assert torch.all(buffers["block_scale"] == 56)  # E4M3 encoding of 1
        torch.testing.assert_close(buffers["tensor_scale"], torch.tensor(0.01))


@pytest.mark.parametrize("mutation", ("format", "scale", "missing", "dense"))
def test_packed_manifest_rejects_inconsistent_calibration(tmp_path, mutation):
    _export(tmp_path, manifest=True)
    path = tmp_path / "modelopt_manifest.json"
    manifest = json.loads(path.read_text())
    modules = manifest["components"]["decoder"]["modules"]
    first = next(iter(modules))
    if mutation == "format":
        manifest["numerical_format"]["block_size"] = 32
    elif mutation == "scale":
        modules[first]["activation_amax"] = 0
    elif mutation == "missing":
        del modules[first]
    else:
        modules["backbone.layers.1.mlp.gate_up.projections.gate"] = {
            "activation_amax": 2688
        }
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="ModelOpt"):
        models.read_config(tmp_path)


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
