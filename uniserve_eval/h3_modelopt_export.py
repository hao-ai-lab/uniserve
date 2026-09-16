"""Export calibrated H3 NVFP4 components and deployment manifests.

The exporter quantizes the checkpoint's real architecture-owned tensors with
ModelOpt's NVFP4 MSE E4M3-scale sweep.  It preserves all non-target tensors,
replaces each target weight with packed E2M1 values and its two scale levels,
and builds complete candidates from hard-linked immutable source files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODELOPT_COMMIT = "6a4b3f147e14a6fec690fedbced8df402344085d"
NUMERICAL_FORMAT = {
    "values": "e2m1",
    "block_size": 16,
    "block_scale": "fp8_e4m3",
    "tensor_scale": "fp32",
    "weight": "w4",
    "activation": "a4",
    "output": "bf16",
}


@dataclass(frozen=True)
class Component:
    source_directory: str
    index_name: str
    source_pattern: re.Pattern
    source_count: int


COMPONENTS = {
    "denoiser": Component(
        "transformer",
        "diffusion_pytorch_model.safetensors.index.json",
        re.compile(
            r"^transformer_blocks\.(?P<layer>\d+)\.ff\."
            r"(?P<projection>net\.0\.proj|net\.2)\.weight$"
        ),
        100,
    ),
    "text_encoder": Component(
        "text_encoder",
        "model.safetensors.index.json",
        re.compile(
            r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\."
            r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
        ),
        150,
    ),
    "video_vae": Component(
        "vae",
        "diffusion_pytorch_model.safetensors.index.json",
        re.compile(
            r"^decoder\.transformer_blocks\.(?P<layer>\d+)\."
            r"(?P<projection>attn\.to_[qkv]|attn\.to_out\.0|"
            r"ff\.net\.0\.proj|ff\.net\.2)\.weight$"
        ),
        216,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _targeted(component: str, name: str) -> bool:
    match = COMPONENTS[component].source_pattern.fullmatch(name)
    if match is None:
        return False
    layer = int(match.group("layer"))
    return (
        layer < 50 if component in {"denoiser", "text_encoder"} else layer < 36
    )


@torch.inference_mode()
def quantize_weight(
    weight: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Run ModelOpt's static MSE FP8-scale sweep and pack one real weight."""
    if weight.ndim != 2 or weight.shape[1] % 16:
        raise ValueError(
            "NVFP4 Linear weights must be matrices with K divisible by 16"
        )
    if weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"unsupported NVFP4 source dtype {weight.dtype}")

    from modelopt.torch.quantization.calib.mse import NVFP4MSECalibrator
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    original = weight.cuda().contiguous()
    blocked = original.float().reshape(-1, 16)
    initial_amax = blocked.abs().amax(dim=1)
    global_amax = original.float().abs().amax()
    if not torch.isfinite(global_amax) or global_amax <= 0:
        raise ValueError("NVFP4 weight requires a positive finite global amax")
    calibrator = NVFP4MSECalibrator(initial_amax, global_amax)
    calibrator.collect(blocked)
    calibrated_amax = calibrator.compute_amax()
    if calibrated_amax is None or not torch.isfinite(calibrated_amax).all():
        raise RuntimeError(
            "ModelOpt MSE scale sweep produced invalid weight scales"
        )

    # This is ModelOpt's static NVFP4 export transform: the selected per-K16
    # amax is encoded in E4M3 relative to the FP32 tensor-wide scale.
    tensor_scale = global_amax / (6.0 * 448.0)
    block_scale = (
        (calibrated_amax / global_amax * 448.0)
        .clamp(min=2**-9, max=448.0)
        .to(torch.float8_e4m3fn)
    )
    block_scale = block_scale.view(original.shape[0], original.shape[1] // 16)
    encoded, exported_scale, exported_tensor_scale = NVFP4QTensor.quantize(
        original,
        16,
        weights_scaling_factor=block_scale,
        weights_scaling_factor_2=tensor_scale,
    )
    packed = encoded._quantized_data
    if not torch.equal(exported_scale, block_scale):
        raise RuntimeError(
            "ModelOpt changed the calibrated block scale during packing"
        )
    torch.testing.assert_close(
        exported_tensor_scale, tensor_scale, rtol=0, atol=0
    )

    low, high = packed & 0x0F, packed >> 4
    saturated = ((low == 7) | (low == 15)).sum() + (
        (high == 7) | (high == 15)
    ).sum()
    summary = {
        "shape": list(original.shape),
        "source_dtype": str(original.dtype).removeprefix("torch."),
        "weight_global_amax": float(global_amax),
        "weight_tensor_scale": float(tensor_scale),
        "block_scale_min": float(block_scale.float().min()),
        "block_scale_max": float(block_scale.float().max()),
        "block_scale_mean": float(block_scale.float().mean()),
        "weight_saturation_rate": float(saturated / original.numel()),
        "calibration": "static_mse_fp8_e4m3_scale_sweep_126_candidates",
    }
    tensors = {
        "weight_packed": packed.cpu(),
        "weight_scale": block_scale.cpu(),
        "weight_tensor_scale": tensor_scale.float().reshape(()).cpu(),
    }
    return tensors, summary


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def export_component(
    model: Path,
    destination: Path,
    component: str,
) -> dict:
    """Quantize every declared weight in one component, shard by shard."""
    definition = COMPONENTS[component]
    source = model / definition.source_directory
    index_path = source / definition.index_name
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    selected = sorted(name for name in weight_map if _targeted(component, name))
    if len(selected) != definition.source_count:
        raise RuntimeError(
            f"{component} expected {definition.source_count} target weights, "
            f"found {len(selected)}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    selected_set = set(selected)
    summaries = {}
    updated_map = dict(weight_map)
    shards = sorted(set(weight_map.values()))
    for shard_index, shard_name in enumerate(shards, 1):
        source_shard = source / shard_name
        output_shard = destination / shard_name
        shard_targets = sorted(
            selected_set.intersection(
                name
                for name, mapped in weight_map.items()
                if mapped == shard_name
            )
        )
        if not shard_targets:
            if not output_shard.exists():
                _link(source_shard, output_shard)
            continue
        if output_shard.exists():
            raise FileExistsError(
                f"refusing to reuse unverified partial export {output_shard}"
            )
        tensors = load_file(source_shard, device="cpu")
        with safe_open(source_shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        for name in shard_targets:
            weight = tensors.pop(name)
            encoded, summary = quantize_weight(weight)
            base = name.removesuffix(".weight")
            for field, value in encoded.items():
                encoded_name = f"{base}.{field}"
                tensors[encoded_name] = value
                updated_map[encoded_name] = shard_name
            del updated_map[name]
            summaries[name] = summary
        temporary = output_shard.with_suffix(output_shard.suffix + ".tmp")
        save_file(tensors, temporary, metadata=metadata)
        temporary.replace(output_shard)
        print(
            f"[{component}] shard {shard_index}/{len(shards)}: "
            f"{len(shard_targets)} weights",
            flush=True,
        )

    for path in source.iterdir():
        if (
            path.is_file()
            and path.name not in shards
            and path.name != definition.index_name
        ):
            target = destination / path.name
            if not target.exists():
                _link(path, target)
    exported_index = dict(index)
    exported_index["weight_map"] = dict(sorted(updated_map.items()))
    (destination / definition.index_name).write_text(
        json.dumps(exported_index, indent=2, sort_keys=True) + "\n"
    )
    report = {
        "component": component,
        "source": str(source),
        "target_weights": len(selected),
        "modelopt_commit": MODELOPT_COMMIT,
        "numerical_format": NUMERICAL_FORMAT,
        "weights": summaries,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(destination.iterdir())
            if path.is_file()
        },
    }
    (destination / "modelopt-component-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _runtime_modules(component: str, activation: dict) -> dict[str, dict]:
    modules = {}
    if component == "denoiser":
        for layer in range(50):
            root = f"denoiser.transformer.layers.{layer}.mlp"
            incoming = activation[
                f"denoiser.transformer_blocks.{layer}.ff.fc_in"
            ]
            outgoing = activation[
                f"denoiser.transformer_blocks.{layer}.ff.fc_out"
            ]
            for branch in ("gate", "up"):
                modules[f"{root}.gate_up.projections.{branch}"] = {
                    "activation_amax": incoming["activation_amax"]
                }
            modules[f"{root}.down"] = {
                "activation_amax": outgoing["activation_amax"]
            }
    elif component == "text_encoder":
        for layer in range(50):
            root = f"text_encoder.network.layers.{layer}.mlp"
            source = f"text_encoder.language_model.layers.{layer}.mlp"
            for projection, target in (
                ("gate_proj", "gate_up.projections.gate"),
                ("up_proj", "gate_up.projections.up"),
                ("down_proj", "down"),
            ):
                modules[f"{root}.{target}"] = {
                    "activation_amax": activation[f"{source}.{projection}"][
                        "activation_amax"
                    ]
                }
    else:
        for layer in range(36):
            root = f"video_decoder.decoder.decoder.decoder.layers.{layer}"
            source = f"video_vae.decoder.transformer_blocks.{layer}"
            for branch in ("q", "k", "v"):
                modules[f"{root}.qkv.projections.{branch}"] = {
                    "activation_amax": activation[f"{source}.attn.to_{branch}"][
                        "activation_amax"
                    ]
                }
            modules[f"{root}.output"] = {
                "activation_amax": activation[f"{source}.attn.to_out.0"][
                    "activation_amax"
                ]
            }
            fused = activation[f"{source}.ff.net.0.proj"]["activation_amax"]
            for branch in ("gate", "up"):
                modules[f"{root}.mlp.gate_up.projections.{branch}"] = {
                    "activation_amax": fused
                }
            modules[f"{root}.mlp.down"] = {
                "activation_amax": activation[f"{source}.ff.net.2"][
                    "activation_amax"
                ]
            }
    return modules


def build_manifest(
    *,
    phase_a_scales: Path,
    phase_b_scales: Path | None,
    calibration_sha256: str,
    text_nvfp4: bool,
    denoising_forwards_per_record: int,
) -> dict:
    """Build the runtime contract for the components that were calibrated.

    Phase A always owns the denoiser and optional text statistics.  A missing
    Phase-B artifact is represented as an explicitly disabled video-VAE
    component, so deployment retains its BF16 path instead of silently using
    dynamic quantization or incomplete calibration state.
    """
    phase_a = json.loads(phase_a_scales.read_text())
    phase_b = (
        json.loads(phase_b_scales.read_text())
        if phase_b_scales is not None
        else None
    )
    if denoising_forwards_per_record <= 0:
        raise ValueError("denoising forwards per record must be positive")
    return {
        "schema_version": 1,
        "modelopt_commit": MODELOPT_COMMIT,
        "numerical_format": NUMERICAL_FORMAT,
        "calibration": {
            "sample_sha256": calibration_sha256,
            "records": 1000,
            "denoising_forwards_per_record": denoising_forwards_per_record,
            "weight_scale": "static MSE, complete E4M3 scale sweep",
            "activation_global_scale": (
                "ModelOpt NVFP4 activation headroom calibration"
            ),
            "activation_block_scale": "runtime dynamic K16 encoding",
        },
        "components": {
            "denoiser": {
                "enabled": True,
                "modules": _runtime_modules("denoiser", phase_a),
                "quantized_patterns": [
                    "transformer_blocks.*.ff.net.0.proj",
                    "transformer_blocks.*.ff.net.2",
                ],
                "excluded_patterns": [
                    "attention",
                    "token_refiner",
                    "adaln",
                    "norm",
                    "latent_heads",
                ],
            },
            "text_encoder": {
                "enabled": text_nvfp4,
                "modules": _runtime_modules("text_encoder", phase_a)
                if text_nvfp4
                else {},
                "quantized_patterns": [
                    "retained_layers.0:50.mlp.{gate,up,down}"
                ],
                "excluded_patterns": [
                    "attention",
                    "embedding",
                    "norm",
                    "rope",
                    "softmax",
                    "checkpoint_layers.50:64",
                    "conditioning_output",
                ],
            },
            "video_vae": {
                "enabled": phase_b is not None,
                "modules": (
                    _runtime_modules("video_vae", phase_b)
                    if phase_b is not None
                    else {}
                ),
                "quantized_patterns": [
                    "decoder.transformer_blocks.*.{q,k,v,out,gate,up,down}"
                ],
                "excluded_patterns": [
                    "post_quant_conv",
                    "decoder_boundaries",
                    "conv",
                    "norm",
                    "rope",
                    "softmax",
                    "residual",
                    "affine",
                    "pixel_postprocess",
                ],
            },
        },
    }


def _hardlink_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _link(path, target)


def build_candidate(
    *,
    model: Path,
    components: Path,
    destination: Path,
    manifest: dict,
) -> None:
    """Assemble an immutable candidate with component-local manifests."""
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    _hardlink_tree(model, temporary)
    component_directories = {
        "denoiser": "transformer",
        "text_encoder": "text_encoder",
        "video_vae": "vae",
    }
    declarations = manifest.get("components")
    if not isinstance(declarations, dict):
        raise ValueError("candidate manifest requires component declarations")
    replacements = {
        directory: components / component
        for component, directory in component_directories.items()
        if declarations.get(component, {}).get("enabled") is True
    }
    for directory, source in replacements.items():
        target = temporary / directory
        shutil.rmtree(target)
        target.mkdir()
        _hardlink_tree(source, target)
    (temporary / "modelopt_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(destination)


def smoke_export(model: Path, output: Path) -> dict:
    """Pack a real VAE target and verify UniServe reads values and scales."""
    from uniserve.loading import Config as IOConfig
    from uniserve.loading import checkpoint

    definition = COMPONENTS["video_vae"]
    source = model / definition.source_directory
    index = json.loads((source / definition.index_name).read_text())[
        "weight_map"
    ]
    name = next(name for name in sorted(index) if _targeted("video_vae", name))
    shard = source / index[name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(name)
    encoded, summary = quantize_weight(weight)
    base = name.removesuffix(".weight")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "modelopt-export.safetensors"
    save_file(
        {f"{base}.{field}": value for field, value in encoded.items()}, path
    )
    with (
        checkpoint.Config()
        .resolve(output, io=IOConfig())
        .open(io=IOConfig()) as reader
    ):
        loaded = reader.get(base + ".weight").read()
        buffers = loaded.buffers()
    if not torch.equal(buffers["values"], encoded["weight_packed"]):
        raise RuntimeError("UniServe changed ModelOpt packed E2M1 values")
    expected_scale = encoded["weight_scale"].view(torch.uint8)
    if not torch.equal(buffers["block_scale"], expected_scale):
        raise RuntimeError("UniServe changed ModelOpt E4M3 block scales")
    torch.testing.assert_close(
        buffers["tensor_scale"], encoded["weight_tensor_scale"], rtol=0, atol=0
    )
    return {"source_weight": name, "file": str(path), "summary": summary}
