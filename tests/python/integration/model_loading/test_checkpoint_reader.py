"""Checkpoint readers expose source values and preserve pre-encoded scales."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from uniserve.loading import Config, checkpoint

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("format", ["safetensors", "pt"])
@pytest.mark.parametrize(
    "mode,mmap",
    [("eager", True), ("eager", False), ("layered", True), ("layered", False)],
)
def test_reader_slices_follow_declared_sources(tmp_path, format, mode, mmap):
    values = torch.arange(24).view(4, 6).float()
    suffix = ".safetensors" if format == "safetensors" else ".pt"
    first, second = (
        tmp_path / ("first" + suffix),
        tmp_path / ("second" + suffix),
    )
    if format == "safetensors":
        save_file({"weight": values}, first)
        save_file({"scalar": torch.tensor(3.0)}, second)
    else:
        torch.save({"state_dict": {"weight": values}}, first)
        torch.save({"scalar": torch.tensor(3.0)}, second)
    io = Config(format=format, mode=mode, mmap=mmap, num_threads=2)
    source = checkpoint.Config(prefix="network.").resolve(tmp_path, io=io)
    with source.open(io=io) as reader:
        assert reader.names() == ("network.scalar", "network.weight")
        weight = reader.get("network.weight")
        assert weight.shape == (4, 6)
        torch.testing.assert_close(
            weight.read((slice(1, 3), slice(2, 5))), values[1:3, 2:5]
        )
        torch.testing.assert_close(
            reader.get("network.scalar").read(), torch.tensor(3.0)
        )
        torch.testing.assert_close(weight.read(), values)
        assert weight.read((slice(4, 4), slice(0, 6))).shape == (0, 6)
        with pytest.raises(ValueError, match="slice"):
            weight.read((slice(0, 5), slice(0, 6)))
    with pytest.raises(RuntimeError, match="closed"):
        weight.read()


@pytest.mark.parametrize("axis", [None, 0])
def test_fp8_source_slices_keep_original_scales(tmp_path, axis):
    values = torch.arange(24).view(4, 6).to(torch.float8_e4m3fn)
    scale = (
        torch.tensor(0.25)
        if axis is None
        else torch.tensor([[0.25], [0.5], [1.0], [2.0]])
    )
    save_file(
        {"values": values, "scale": scale}, tmp_path / "model.safetensors"
    )
    io = Config()
    with checkpoint.Config().resolve(tmp_path, io=io).open(io=io) as reader:
        weight = checkpoint.FP8Weight(
            reader.get("values"), reader.get("scale"), axis, torch.float32
        )
        result = weight.read((slice(1, 3), slice(2, 5)))
        torch.testing.assert_close(
            result.dequantize(), (values.float() * scale)[1:3, 2:5]
        )
        torch.testing.assert_close(
            result.buffers()["scale"], scale if axis is None else scale[1:3]
        )


def test_modelopt_nvfp4_source_preserves_packed_values_and_scales(tmp_path):
    values = torch.tensor(
        [
            [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ],
        dtype=torch.uint8,
    )
    scale = torch.tensor([[2.0], [1.0]], dtype=torch.float8_e4m3fn)
    tensor_scale = torch.tensor(0.25, dtype=torch.float32)
    input_scale = torch.tensor(0.5, dtype=torch.float32)
    # ModelOpt's unified export layout: packed values under `.weight`, K16
    # block scales, the FP32 `weight_scale_2` and the static `input_scale`.
    save_file(
        {
            "layer.weight": values,
            "layer.weight_scale": scale,
            "layer.weight_scale_2": tensor_scale,
            "layer.input_scale": input_scale,
        },
        tmp_path / "model.safetensors",
    )

    io = Config()
    with checkpoint.Config().resolve(tmp_path, io=io).open(io=io) as reader:
        weight = reader.get("layer.weight")
        assert isinstance(weight, checkpoint.NVFP4Weight)
        encoded = weight.read((slice(1, 2), slice(0, 16)))
        assert torch.equal(encoded.buffers()["values"], values[1:2])
        assert torch.equal(
            encoded.buffers()["block_scale"], scale[1:2].view(torch.uint8)
        )
        torch.testing.assert_close(
            encoded.buffers()["tensor_scale"], tensor_scale
        )
        assert weight.shape == (2, 16)
        assert weight.input_scale() == 0.5


def test_index_and_checksum_enforce_declared_file_set(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2)}, path)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": path.name}}))
    manifest = tmp_path / "sha256.json"
    manifest.write_text(
        json.dumps({path.name: hashlib.sha256(path.read_bytes()).hexdigest()})
    )
    io = Config(checksum_manifest=manifest)
    source = checkpoint.Config().resolve(tmp_path, io=io)
    with source.open(io=io) as reader:
        torch.testing.assert_close(reader.get("weight").read(), torch.ones(2))
    manifest.write_text(json.dumps({path.name: "0" * 64}))
    with pytest.raises(ValueError, match="checksum mismatch"):
        source.open(io=io)
    index.write_text(
        json.dumps({"weight_map": {"weight": "absent.safetensors"}})
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        checkpoint.Config().resolve(tmp_path, io=Config())


def test_header_only_source_serves_metadata_to_dummy_reads(tmp_path):
    # Header entries describe tensors whose values are not present, so only
    # dummy reads, which synthesize values from metadata, accept them.
    source = checkpoint.Source(
        "primary",
        tmp_path,
        (),
        "network.",
        headers=(("weight", (4, 6), "BF16"),),
    )
    with source.open(io=Config(mode="dummy")) as reader:
        assert reader.names() == ("network.weight",)
        weight = reader.get("network.weight")
        assert (weight.shape, weight.dtype) == ((4, 6), torch.bfloat16)
        assert weight.read().shape == (4, 6)
    for mode in ("eager", "layered"):
        with pytest.raises(ValueError, match="only dummy reads"):
            source.open(io=Config(mode=mode))
