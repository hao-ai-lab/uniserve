"""Rewrite a packed FastH3 NVFP4 checkpoint in ModelOpt's unified layout.

The FastH3 NVFP4 checkpoints were first published with their calibration in
a root `modelopt_manifest.json` keyed by UniServe module paths and with
packed tensors named `weight_packed` and `weight_tensor_scale`. ModelOpt's
unified Hugging Face export, which every serving framework loads, stores the
same numbers differently:

- the packed E2M1 bytes under the ordinary `.weight` name;
- the K16 E4M3 block scales in `.weight_scale` (unchanged);
- the FP32 weight scale, `amax / (6 * 448)`, in `.weight_scale_2`;
- the static activation scale, `amax / (6 * 448)`, in `.input_scale` next to
  its weight;
- a `quantization_config` with `quant_method: modelopt` in each quantized
  component's `config.json`.

The per-component `modelopt-component-manifest.json` files hash the packed
layout's shards and are not carried over.

The rewrite is lossless: packed values and scales are copied bit for bit, and
each module's `input_scale` is the FP32 tensor scale the calibrated amax
produced at run time, `fp32(amax) / 2688` in FP32, so activations encode
exactly as before. Files outside the quantized components are hardlinked into
the output.

Usage:
    python scripts/convert_h3_modelopt_checkpoint.py SOURCE OUTPUT
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from uniserve.loading import Config as IOConfig
from uniserve_models import loading as model_loading
from uniserve_models import minimax_h3

# NVFP4 global scales map the largest E2M1 value (6) times the largest E4M3
# block scale (448) onto the calibrated amax.
SCALE_RANGE = 6.0 * 448.0

# Manifest component name -> checkpoint source name in the H3 package.
SOURCES = {
    "denoiser": "denoiser",
    "text_encoder": "text_encoder",
    "video_vae": "video_decoder",
}

RENAMES = {
    ".weight_packed": ".weight",
    ".weight_tensor_scale": ".weight_scale_2",
}

# The ModelOpt commit that calibrated the published checkpoints.
PRODUCER = {"name": "modelopt", "version": "0.47.0rc0-63-g6a4b3f14"}


def quantization_config(ignore: list[str]) -> dict:
    """Return ModelOpt's unified config for static NVFP4 W4A4, K16 blocks."""
    encoding = {
        "dynamic": False,
        "num_bits": 4,
        "type": "float",
        "group_size": 16,
    }
    return {
        "config_groups": {
            "group_0": {
                "input_activations": dict(encoding),
                "weights": dict(encoding),
                "targets": ["Linear"],
            }
        },
        "ignore": ignore,
        "quant_algo": "NVFP4",
        "producer": PRODUCER,
        "quant_method": "modelopt",
    }


@dataclass(frozen=True)
class _Header:
    """A checkpoint tensor's name and logical shape, without its data."""

    name: str
    shape: tuple[int, ...]


class _RenamedHeaders:
    """Present a source's tensors under their unified names.

    The architecture's checkpoint mappings only inspect names and shapes, so
    headers suffice to resolve which module each packed tensor feeds. A
    packed weight's logical width is twice its byte width.
    """

    def __init__(self, files: list[Path]):
        self._headers = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    tensor = handle.get_slice(name)
                    shape = tuple(tensor.get_shape())
                    target = _renamed(name)
                    if name.endswith(".weight_packed"):
                        shape = (shape[0], shape[1] * 2)
                    self._headers[target] = _Header(target, shape)

    def names(self) -> tuple[str, ...]:
        return tuple(self._headers)

    def get(self, name: str) -> _Header:
        return self._headers[name]


def _renamed(name: str) -> str:
    for suffix, target in RENAMES.items():
        if name.endswith(suffix):
            return name.removesuffix(suffix) + target
    return name


def _shards(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.safetensors"))


def _input_scales(source: Path, manifest: dict) -> dict[str, dict[str, float]]:
    """Map each packed tensor prefix to its calibrated `input_scale`.

    The manifest keys activation amax by module path; the architecture's own
    checkpoint mappings resolve the tensor each module loads from.
    """
    config = model_loading.read_config(
        source, io=IOConfig(), modules=frozenset()
    )
    with torch.device("meta"):
        model = minimax_h3.Model(config.model)
    paths = {
        id(parameter): path
        for path, module in model.named_modules()
        for parameter in module.parameters(recurse=False)
    }
    declared = {
        declaration.name: declaration.directory
        for declaration in minimax_h3.checkpoint_sources
    }

    scales: dict[str, dict[str, float]] = {}
    for component, entry in manifest["components"].items():
        if not entry["enabled"]:
            continue
        source_name = SOURCES[component]
        directory = declared[source_name]
        reader = _RenamedHeaders(_shards(source / directory))
        modules = entry["modules"]
        resolved = set()
        scales[directory] = {}
        for mapping in minimax_h3.checkpoint_mappings(model):
            if mapping.source != source_name:
                continue
            for assignment in mapping.map_weights(reader):
                path = paths[id(assignment.target)]
                # Only a module's weight is packed; its bias stays dense.
                if path not in modules or not assignment.source.name.endswith(
                    ".weight"
                ):
                    continue
                prefix = assignment.source.name.removesuffix(".weight")
                amax = modules[path]["activation_amax"]
                scale = float(
                    torch.tensor(amax, dtype=torch.float32) / SCALE_RANGE
                )
                previous = scales[directory].setdefault(prefix, scale)
                if previous != scale:
                    raise ValueError(
                        f"{prefix} feeds modules with different calibrated "
                        "activation amax"
                    )
                resolved.add(path)
        if resolved != set(modules):
            raise ValueError(
                f"{component}: no checkpoint tensor resolves "
                f"{sorted(set(modules) - resolved)}"
            )
    return scales


def _rewrite(directory: Path, target: Path, scales: dict[str, float]) -> None:
    """Rewrite one quantized component's shards, index and config."""
    target.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    ignore = set()
    written = set()
    for shard in _shards(directory):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        state = {}
        for name, value in load_file(shard).items():
            renamed = _renamed(name)
            state[renamed] = value
            prefix = renamed.removesuffix(".weight")
            if (
                renamed.endswith(".weight")
                and value.dim() == 2
                and value.is_floating_point()
            ):
                ignore.add(prefix)
        for prefix, scale in scales.items():
            if prefix + ".weight_scale_2" in state:
                state[prefix + ".input_scale"] = torch.tensor(
                    scale, dtype=torch.float32
                )
                written.add(prefix)
        save_file(state, target / shard.name, metadata=metadata)
        weight_map.update(dict.fromkeys(state, shard.name))
    if written != set(scales):
        raise ValueError(
            f"packed tensors missing for {sorted(set(scales) - written)}"
        )

    for index in directory.glob("*.safetensors.index.json"):
        value = json.loads(index.read_text())
        value["weight_map"] = dict(sorted(weight_map.items()))
        (target / index.name).write_text(json.dumps(value, indent=2) + "\n")

    config = json.loads((directory / "config.json").read_text())
    config["quantization_config"] = quantization_config(sorted(ignore))
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for path in directory.iterdir():
        rewritten = (
            path.suffix == ".safetensors"
            or path.name.endswith(".index.json")
            or path.name in {"config.json", "modelopt-component-manifest.json"}
        )
        if not rewritten:
            _link(path, target / path.name)


def _link(source: Path, target: Path) -> None:
    """Hardlink an unchanged file, copying across filesystems."""
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _link(child, target / child.name)
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def convert(source: Path, output: Path) -> None:
    """Write `source` in ModelOpt's unified layout under `output`."""
    manifest = json.loads((source / "modelopt_manifest.json").read_text())
    scales = _input_scales(source, manifest)
    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.name == "modelopt_manifest.json":
            continue
        if path.name in scales:
            _rewrite(path, output / path.name, scales[path.name])
        else:
            _link(path, output / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
