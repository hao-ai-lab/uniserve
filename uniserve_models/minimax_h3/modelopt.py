"""Validated loading contract for calibrated H3 ModelOpt checkpoints."""

from __future__ import annotations

import math
from dataclasses import replace

from torch import nn

from uniserve.loading import weights
from uniserve.quantization import QuantizationConfig, Quantizer

from .precision import weight_config

MODEL_OPT_COMMIT = "6a4b3f147e14a6fec690fedbced8df402344085d"


def denoiser_modules() -> tuple[str, ...]:
    """Return the architecture-owned H3 denoiser MLP projection paths."""
    root = "denoiser.transformer.layers"
    return tuple(
        path
        for index in range(50)
        for path in (
            f"{root}.{index}.mlp.gate_up.projections.gate",
            f"{root}.{index}.mlp.gate_up.projections.up",
            f"{root}.{index}.mlp.down",
        )
    )


def text_modules() -> tuple[str, ...]:
    """Return the retained Qwen MLP projection paths, excluding layers 50-63."""
    root = "text_encoder.network.layers"
    return tuple(
        path
        for index in range(50)
        for path in (
            f"{root}.{index}.mlp.gate_up.projections.gate",
            f"{root}.{index}.mlp.gate_up.projections.up",
            f"{root}.{index}.mlp.down",
        )
    )


def video_vae_modules() -> tuple[str, ...]:
    """Return only the decoder Transformer Linear paths authorized for PTQ."""
    root = "video_decoder.decoder.decoder.decoder.layers"
    return tuple(
        path
        for index in range(36)
        for path in (
            *(
                f"{root}.{index}.qkv.projections.{name}"
                for name in ("q", "k", "v")
            ),
            f"{root}.{index}.output",
            f"{root}.{index}.mlp.gate_up.projections.gate",
            f"{root}.{index}.mlp.gate_up.projections.up",
            f"{root}.{index}.mlp.down",
        )
    )


def calibrated_weight_config(
    manifest: dict, model: nn.Module
) -> weights.Config:
    """Build an execution config only after validating the complete manifest.

    The packed checkpoint owns weight scales. The manifest owns one frozen
    activation global amax per numerical Linear; runtime conversion computes
    only input-dependent K16 block encodings.
    """
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("H3 ModelOpt manifest requires schema_version 1")
    if manifest.get("modelopt_commit") != MODEL_OPT_COMMIT:
        raise ValueError(
            "H3 ModelOpt manifest uses an unapproved ModelOpt commit"
        )
    numerical = manifest.get("numerical_format")
    expected = {
        "values": "e2m1",
        "block_size": 16,
        "block_scale": "fp8_e4m3",
        "tensor_scale": "fp32",
        "weight": "w4",
        "activation": "a4",
        "output": "bf16",
    }
    if numerical != expected:
        raise ValueError(
            "H3 ModelOpt manifest numerical format is not NVFP4 W4A4"
        )

    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != {
        "denoiser",
        "text_encoder",
        "video_vae",
    }:
        raise ValueError(
            "H3 ModelOpt manifest must describe all three components"
        )
    expected_modules = {
        "denoiser": denoiser_modules(),
        "text_encoder": text_modules(),
        "video_vae": video_vae_modules(),
    }
    actual_paths = dict(model.named_modules())
    quantization = {}
    for component, required in expected_modules.items():
        declaration = components[component]
        if not isinstance(declaration, dict):
            raise ValueError(
                f"H3 ModelOpt {component} declaration must be an object"
            )
        enabled = declaration.get("enabled")
        modules = declaration.get("modules")
        if not isinstance(enabled, bool) or not isinstance(modules, dict):
            raise ValueError(
                f"H3 ModelOpt {component} requires enabled and modules"
            )
        required_set = set(required) if enabled else set()
        if set(modules) != required_set:
            raise ValueError(
                f"H3 ModelOpt {component} module coverage mismatch: "
                f"expected {len(required_set)}, found {len(modules)}"
            )
        for path, declaration in modules.items():
            if path not in actual_paths:
                raise ValueError(
                    f"H3 ModelOpt manifest names unknown module {path!r}"
                )
            if not isinstance(declaration, dict):
                raise ValueError(
                    f"H3 ModelOpt module {path!r} must be an object"
                )
            amax = declaration.get("activation_amax")
            if (
                not isinstance(amax, (int, float))
                or isinstance(amax, bool)
                or not math.isfinite(amax)
                or amax <= 0
            ):
                raise ValueError(
                    f"H3 ModelOpt module {path!r} has invalid activation amax"
                )
            quantization[path] = QuantizationConfig(
                Quantizer("nvfp4"),
                Quantizer("nvfp4", calibrated_amax=float(amax)),
            )

    # Calibrated VAE boundaries and unselected decoder operations are BF16;
    # convolutional/residual owners retain their declared FP32 accumulation.
    base = weight_config(preset="quality", video_vae="bf16")
    return replace(base, quantization=quantization)
