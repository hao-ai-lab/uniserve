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
from uniserve_worker.nn.mesh import Communicator
from uniserve_worker.nn.parallel import ParallelConfig
from uniserve_worker.nn.quant import (
    DynamicW8A8Fp8LinearMethod,
)
from uniserve_worker.nn.quant.config import QuantizationConfig
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _method(format):
    return QuantizationConfig.from_model_config(
        {"quantization_config": {"quant_method": format}}
    ).get_quant_method()


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
    local = LayerConfig(Communicator(), None)
    torch.manual_seed(107)
    x = (torch.rand(128, 512, device=device) + 0.25).bfloat16()
    # Input-column groups have distinct magnitudes, so local scale domains do
    # not accidentally coincide with the full logical activation domain.
    x *= torch.tensor([0.03, 0.3, 3, 30], device=device).repeat_interleave(128)
    for name in ("four", "two", "one"):
        if name not in meshes:
            continue
        group = meshes[name].get_group("tp")
        config = LayerConfig(group, None)
        for format in ("fp8", "nvfp4", "mxfp8"):
            with torch.device(device):
                reference = LinearBase(
                    512,
                    256,
                    layer_config=local,
                    bias=False,
                    quant_method=_method(format),
                )
                sharded = RowParallelLinear(
                    512,
                    256,
                    layer_config=config,
                    bias=False,
                    quant_method=_method(format),
                )
            for module in (reference, sharded):
                _load(module, checkpoint, "row", (256, 512))
                module.finalize_weights()
            expected = reference(x)
            actual = sharded(x.chunk(group.world_size, dim=-1)[group.rank_in_group].contiguous())
            # Positive operands avoid cancellation. Each BF16 partial and its
            # reduction add at most one unit roundoff; gamma bounds their sum.
            operations = group.world_size + 1
            gamma = operations * 2**-8 / (1 - operations * 2**-8)
            torch.testing.assert_close(
                actual, expected, rtol=gamma, atol=0, msg=f"{name}/{format}: logical row projection"
            )
            prepared_output = torch.cat(
                [
                    reference.forward_prepared(
                        reference.prepare_input(part, absmax=part.abs().amax())
                    )
                    for part in x.chunk(4, dim=0)
                ]
            )
            torch.testing.assert_close(
                prepared_output,
                expected,
                rtol=gamma,
                atol=0,
                msg=f"{name}/{format}: reusable activation preparation",
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
                        sequence_group=layout.communicator,
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


@torch.inference_mode()
def _run_sequence_scale(rank, rendezvous):
    from uniserve_worker.nn.parallel import SequenceParallel
    from uniserve_worker.nn.quant.nvfp4 import DynamicW4A4NvFp4LinearMethod

    device = torch.device("cuda", rank)
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    )
    mesh = initialize_model_parallel(
        environment,
        {
            "rows": (
                (3, 1, 2, 0),
                ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (4,))),
            )
        },
    )["rows"]
    group = mesh.get_group("sp")
    graph = None
    try:
        values = ((torch.arange(384 * 512, device=device).view(384, 512) % 11 + 1) / 16).bfloat16()
        values.mul_((torch.arange(384, device=device) // 32 + 1).unsqueeze(-1))
        for method in (DynamicW8A8Fp8LinearMethod(tensorwise=True), DynamicW4A4NvFp4LinearMethod()):
            reference = LinearBase(
                512,
                256,
                layer_config=LayerConfig(Communicator(), None),
                bias=False,
                quant_method=method,
            ).to(device=device, dtype=torch.bfloat16)
            sharded = LinearBase(
                512,
                256,
                layer_config=LayerConfig(Communicator(), None),
                bias=False,
                quant_method=method,
                input_scale_group=group,
            ).to(device=device, dtype=torch.bfloat16)
            weights = (
                (torch.arange(256 * 512, device=device).view(256, 512) % 7 + 1) / 512
            ).bfloat16()
            for linear in (reference, sharded):
                linear.weight.copy_(weights)
                linear.finalize_weights()
            local = values.chunk(4)[group.rank_in_group].clone()

            def execute():
                return group.all_gather(sharded(local), dim=0)

            torch.testing.assert_close(execute(), reference(values), rtol=2**-7, atol=0)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = execute()
            values.mul_(2)
            local.copy_(values.chunk(4)[group.rank_in_group])
            graph.replay()
            torch.testing.assert_close(actual, reference(values), rtol=2**-7, atol=0)
            graph.reset()
            graph = None
            values.div_(2)
    finally:
        if graph is not None:
            graph.reset()
        environment.close()
        dist.destroy_process_group()


def test_tensor_scale_preserves_values_across_sequence_partitions(tmp_path):
    mp.spawn(
        _run_sequence_scale,
        args=((tmp_path / "sequence-scales").as_uri(),),
        nprocs=4,
        join=True,
    )
