"""Stacked experts load checkpoint encodings without re-encoding them."""

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import QuantizationConfig, Quantizer, ScaleLayout

pytestmark = pytest.mark.integration

EXPERTS, HIDDEN, INTERMEDIATE = 3, 32, 16
E2M1 = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), dtype=torch.float32)


def _decode(values, block_scale, tensor_scale):
    """Independent NVFP4 decode: low nibble first, E4M3 per K16 block."""
    codes = torch.stack((values & 15, values >> 4), dim=-1).reshape(
        values.shape[0], -1
    )
    magnitudes = E2M1[(codes & 7).long()]
    signed = torch.where(codes & 8 != 0, -magnitudes, magnitudes)
    scales = block_scale.view(torch.float8_e4m3fn).float()
    return signed * scales.repeat_interleave(16, dim=1) * tensor_scale


def _projection(generator, rows, columns, tensor_scale, input_scale):
    return {
        "weight": torch.randint(
            0, 256, (rows, columns // 2), dtype=torch.uint8, generator=generator
        ),
        # E4M3 block scales between 1/8 and 2, stored as their byte pattern.
        "weight_scale": (
            torch.rand(rows, columns // 16, generator=generator) * 1.875 + 0.125
        ).to(torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(tensor_scale, dtype=torch.float32),
        "input_scale": torch.tensor(input_scale, dtype=torch.float32),
    }


def _checkpoint(root, *, mismatched_up=False, fp8_down=False):
    generator = torch.Generator().manual_seed(29)
    tensors = {}
    for expert in range(EXPERTS):
        scale = 0.01 * (expert + 1)
        projections = {
            "gate": _projection(
                generator, INTERMEDIATE, HIDDEN, scale, 0.02 + expert * 0.01
            ),
            "up": _projection(
                generator,
                INTERMEDIATE,
                HIDDEN,
                scale * (2 if mismatched_up and expert == 1 else 1),
                0.03,
            ),
            "down": _projection(
                generator, HIDDEN, INTERMEDIATE, 0.005, 0.04 + expert * 0.01
            ),
        }
        if fp8_down:
            projections["down"] = {
                "weight": torch.randn(
                    HIDDEN, INTERMEDIATE, generator=generator
                ).to(torch.float8_e4m3fn),
                "weight_scale": torch.tensor(0.5, dtype=torch.float32),
            }
        for name, fields in projections.items():
            for field, value in fields.items():
                tensors[f"experts.{expert}.{name}_proj.{field}"] = value
    save_file(tensors, root / "model.safetensors")
    return tensors


def _load(root, module):
    io = loading.Config()
    source = checkpoint.Config(name="primary").resolve(root, io=io)
    configs = {
        path: QuantizationConfig(
            Quantizer("nvfp4"), Quantizer("nvfp4", calibrated_scale=scale)
        )
        for path, scale in (("up_gate", 0.05), ("down", 0.06))
    }

    def assign(reader):
        result = []
        for expert in range(EXPERTS):
            prefix = f"experts.{expert}"
            result.extend(
                weights.expert_assignments(
                    module,
                    up=(reader.get(f"{prefix}.up_proj.weight"), 0),
                    gate=(reader.get(f"{prefix}.gate_proj.weight"), 0),
                    down=reader.get(f"{prefix}.down_proj.weight"),
                    expert=expert,
                )
            )
        return tuple(result)

    loading.load_weights(
        module,
        (source,),
        mapping=lambda model: (
            weights.ModuleMapping(
                model,
                "primary",
                assign,
                frozenset({"up_gate.weight", "down.weight"}),
            ),
        ),
        device="cpu",
        weights=weights.Config(dtype=torch.float32, quantization=configs),
    )


def _module():
    with torch.device("meta"):
        return FusedMoE(
            EXPERTS, HIDDEN, INTERMEDIATE, top_k=2, activation="gelu_tanh"
        )


def test_expert_encodings_keep_values_and_per_expert_scales(tmp_path):
    tensors = _checkpoint(tmp_path)
    module = _module()
    _load(tmp_path, module)

    up_gate = module.up_gate.weight
    assert up_gate.scale_layout is ScaleLayout.SWIZZLED_128X4
    torch.testing.assert_close(
        up_gate.buffers()["tensor_scale"],
        torch.tensor([0.01, 0.02, 0.03]),
    )
    decoded, down = up_gate.dequantize(), module.down.weight.dequantize()
    for expert in range(EXPERTS):

        def reference(name):
            prefix = f"experts.{expert}.{name}_proj"
            return _decode(
                tensors[f"{prefix}.weight"],
                tensors[f"{prefix}.weight_scale"].view(torch.uint8),
                tensors[f"{prefix}.weight_scale_2"],
            )

        # Resident rows hold each expert's up projection, then its gate.
        torch.testing.assert_close(
            decoded[expert, :INTERMEDIATE], reference("up")
        )
        torch.testing.assert_close(
            decoded[expert, INTERMEDIATE:], reference("gate")
        )
        torch.testing.assert_close(down[expert], reference("down"))

    # Experts share each projection's activation encoding at the largest
    # calibrated expert scale.
    assert module.up_gate.input_quantizer == Quantizer(
        "nvfp4", calibrated_scale=0.05
    )


def test_portable_experts_round_trip_activations_through_static_scales(
    tmp_path,
):
    _checkpoint(tmp_path)
    module = _module()
    _load(tmp_path, module)
    generator = torch.Generator().manual_seed(31)
    hidden = torch.randn(4, HIDDEN, generator=generator)
    ids = torch.tensor([[0, 2], [1, 0], [2, 1], [1, 2]], dtype=torch.int32)
    weights_ = torch.rand(4, 2, generator=generator)

    def encode(value, scale):
        return Quantizer("nvfp4", calibrated_scale=scale).round_trip(value)

    up_gate, down = module.up_gate.weight.dequantize(), module.down.weight
    down = down.dequantize()
    expected = torch.zeros_like(hidden)
    encoded = encode(hidden, 0.05)
    for token in range(4):
        for slot in range(2):
            expert = int(ids[token, slot])
            up, gate = (up_gate[expert] @ encoded[token]).split(INTERMEDIATE)
            # Activations are encoded as one row of K16 blocks.
            activated = encode(
                (F.gelu(gate, approximate="tanh") * up).unsqueeze(0), 0.06
            ).squeeze(0)
            expected[token] += weights_[token, slot] * (
                down[expert] @ activated
            )
    torch.testing.assert_close(module(hidden, ids, weights_), expected)


def test_disagreeing_expert_tensor_scales_are_rejected(tmp_path):
    _checkpoint(tmp_path, mismatched_up=True)
    with pytest.raises(ValueError, match="disagree on their tensor scale"):
        _load(tmp_path, _module())


def test_checkpoint_encoding_must_match_the_configured_format(tmp_path):
    _checkpoint(tmp_path, fp8_down=True)
    with pytest.raises(ValueError, match="not the configured nvfp4"):
        _load(tmp_path, _module())
