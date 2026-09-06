"""Checkpoint-to-Linear numerical contracts across physical tensor partitions."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from uniserve_worker.loader.handles import SafetensorFileWeightHandle
from uniserve_worker.loader.weight_loaders import attach_parameter_loaders, load_parameter_weight
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.linear import (
    InterleavedMergedColumnParallelLinear,
    LinearBase,
    QKVParallelLinear,
    RowParallelLinear,
)
from uniserve_worker.nn.mesh import TensorParallel
from uniserve_worker.nn.parallel import ParallelConfig
from uniserve_worker.nn.quant import (
    DynamicW4A4NvFp4LinearMethod,
    DynamicW8A8Fp8LinearMethod,
    DynamicW8A8MxFp8LinearMethod,
)
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _method(format):
    return {
        "fp8": DynamicW8A8Fp8LinearMethod,
        "nvfp4": DynamicW4A4NvFp4LinearMethod,
        "mxfp8": DynamicW8A8MxFp8LinearMethod,
    }[format]()


def _load(module, path, name, shape, shard=None):
    attach_parameter_loaders(module, device=module.weight.device, dtype=torch.bfloat16)
    load_parameter_weight(
        module.weight, SafetensorFileWeightHandle(name, Path(path), shape, torch.bfloat16), shard
    )


def _run(rank, rendezvous, checkpoint):
    device = f"cuda:{rank}"
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    )
    meshes = initialize_model_parallel(
        environment,
        {
            "four": ((0, 1, 2, 3), ParallelConfig(4)),
            "two": ((3, 1), ParallelConfig(2)),
            "one": ((2,), ParallelConfig()),
        },
    )
    local = LayerConfig(TensorParallel(0, 1), None)
    torch.manual_seed(107)
    x = (torch.rand(128, 512, device=device) + 0.25).bfloat16()
    # Input-column groups have distinct magnitudes, so local scale domains do
    # not accidentally coincide with the full logical activation domain.
    x *= torch.tensor([0.03, 0.3, 3, 30], device=device).repeat_interleave(128)
    for name in ("four", "two", "one"):
        if name not in meshes:
            continue
        group = meshes[name].get_group("tp")
        config = LayerConfig(TensorParallel(group.rank_in_group, group.world_size), None, group)
        for format in ("fp8", "nvfp4", "mxfp8"):
            with torch.device(device):
                reference = LinearBase(
                    512, 256, layer_config=local, bias=False, quant_method=_method(format)
                )
                sharded = RowParallelLinear(
                    512, 256, layer_config=config, bias=False, quant_method=_method(format)
                )
            for module in (reference, sharded):
                _load(module, checkpoint, "row", (256, 512))
                module.finalize_weights()
                module.logical_input_row_partitions = 4
            expected = reference(x)
            actual = sharded(x.chunk(group.world_size, dim=-1)[group.rank_in_group].contiguous())
            # Positive operands avoid cancellation. Each BF16 partial and its
            # reduction add at most one unit roundoff; gamma bounds their sum.
            operations = group.world_size + 1
            gamma = operations * 2**-8 / (1 - operations * 2**-8)
            torch.testing.assert_close(
                actual, expected, rtol=gamma, atol=0, msg=f"{name}/{format}: logical row projection"
            )

        for format in ("fp8", "nvfp4"):
            qkv_projections = []
            for layout in (local, config):
                with torch.device(device):
                    qkv = QKVParallelLinear(
                        512,
                        32,
                        8,
                        4,
                        layer_config=layout,
                        quant_method=_method(format),
                        bias=False,
                    )
                for branch, width in (("q", 256), ("k", 128), ("v", 128)):
                    _load(qkv, checkpoint, f"qkv_{branch}", (width, 512), branch)
                qkv.finalize_weights()
                qkv_projections.append(qkv)
            full_branches = qkv_projections[0](x).split((256, 128, 128), dim=-1)
            expected_qkv = torch.cat(
                [
                    branch.chunk(group.world_size, dim=-1)[group.rank_in_group]
                    for branch in full_branches
                ],
                dim=-1,
            )
            torch.testing.assert_close(
                qkv_projections[1](x),
                expected_qkv,
                rtol=2**-7,
                atol=0,
                msg=f"{name}/{format}: grouped-query checkpoint projection",
            )

            projections = []
            for layout in (local, config):
                method = (
                    DynamicW8A8Fp8LinearMethod(tensorwise=True)
                    if format == "fp8"
                    else _method(format)
                )
                with torch.device(device):
                    projection = InterleavedMergedColumnParallelLinear(
                        512,
                        256,
                        4,
                        16,
                        layer_config=layout,
                        quant_method=method,
                        bias=False,
                        weight_scale_partition_size=256,
                        sequence_group=layout.tensor_group(),
                    )
                for branch in range(4):
                    _load(projection, checkpoint, f"branch{branch}", (256, 512), branch)
                projection.finalize_weights()
                projections.append(projection)
            expected = projections[0](x).chunk(group.world_size, dim=-1)[group.rank_in_group]
            actual = projections[1](x)
            torch.testing.assert_close(
                actual,
                expected,
                rtol=2**-7,
                atol=0,
                msg=f"{name}/{format}: logical output scale partitions",
            )
            local_rows = x.chunk(group.world_size, dim=0)[group.rank_in_group].contiguous()
            workspace = torch.empty(x.numel() * x.element_size(), dtype=torch.uint8, device=device)
            exchanged = projections[1].forward_sequence_parallel(local_rows, workspace)
            torch.testing.assert_close(
                exchanged,
                expected,
                rtol=2**-7,
                atol=0,
                msg=f"{name}/{format}: prepared value/scale sequence transport",
            )
    environment.close()
    dist.destroy_process_group()


def test_checkpoint_quantization_preserves_logical_domains_across_tp(tmp_path):
    torch.manual_seed(113)
    tensors = {"row": (torch.rand(256, 512) + 0.125).bfloat16()}
    tensors["row"] *= torch.tensor([0.03, 0.3, 3, 30]).repeat_interleave(128)
    for branch in range(4):
        value = (torch.rand(256, 512) + 0.125).bfloat16()
        value *= torch.tensor([0.03, 0.3, 3, 30]).repeat_interleave(64).reshape(-1, 1)
        tensors[f"branch{branch}"] = value
    tensors["qkv_q"] = tensors["branch0"].clone()
    tensors["qkv_k"] = tensors["branch1"][:128].clone()
    tensors["qkv_v"] = tensors["branch2"][:128].clone()
    checkpoint = tmp_path / "projection.safetensors"
    save_file(tensors, checkpoint)
    mp.spawn(_run, args=((tmp_path / "rendezvous").as_uri(), str(checkpoint)), nprocs=4, join=True)
