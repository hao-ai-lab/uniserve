"""Loaded quantized projections preserve complete source scale domains.

The domains are preserved across TP.
"""

import pytest
import torch
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from uniserve import loading
from uniserve.distributed import DeviceMesh
from uniserve.loading import checkpoint, weights
from uniserve.model import TextSize
from uniserve.nn import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from uniserve.quantization import QuantizationConfig, Quantizer
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    initialize_process_groups,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


class Projections(nn.Module):
    def __init__(self, _):
        super().__init__()
        self.row = RowParallelLinear(512, 256, bias=False)
        self.branches = MergedColumnParallelLinear(
            512,
            dict.fromkeys(("first", "second", "third", "fourth"), 512),
            branch_width=64,
            bias=False,
        )
        self.qkv = QKVParallelLinear(512, 32, 8, 64, bias=False)


def _mapping(model):
    return (
        weights.ModuleMapping(
            model,
            "primary",
            lambda reader: tuple(
                weights.Assignment(parameter, reader.get(name))
                for name, parameter in model.named_parameters()
            ),
            frozenset(name for name, _ in model.named_parameters()),
        ),
    )


@torch.inference_mode()
def _run(rank, rendezvous, directory, source):
    device = torch.device("cuda", rank)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=device,
        init_method=rendezvous,
    ) as owner:
        for ranks in ((3, 1, 2, 0), (3, 1), (0,)):
            mesh = owner.bind(
                DeviceMesh(
                    ranks=ranks, shape=(len(ranks),), axes=("tp",), rank=rank
                ),
                device=device,
            )
            if rank not in ranks:
                continue
            group = mesh.get_group("tp")
            x = (
                (torch.arange(128 * 512, device=device).view(128, 512) % 11 + 1)
                / 16
            ).bfloat16()
            x *= torch.tensor(
                [0.03, 0.3, 3, 30], device=device
            ).repeat_interleave(128)
            for format in ("fp8", "nvfp4", "mxfp8"):
                quantizer = Quantizer(format)
                model = loading.load_model(
                    Projections,
                    None,
                    checkpoint=(
                        checkpoint.Config().resolve(
                            directory, io=loading.Config()
                        ),
                    ),
                    mapping=_mapping,
                    device=device,
                    meshes={"": mesh},
                    weights=weights.Config(
                        quantization={
                            "": QuantizationConfig(quantizer, quantizer)
                        }
                    ),
                ).model
                # Positive values make relative gamma bounds meaningful. Every
                # local BF16 product and TP sum contributes one unit roundoff.
                calls = group.size + 1
                gamma = calls * 2**-8 / (1 - calls * 2**-8)
                stream = CUDAStream.external(torch.cuda.Stream(device=device))
                stream.wait(torch.cuda.current_stream(device))
                with stream, ExecutionContext(model, stream=stream) as context:
                    context.prepare(TextSize(128, 1))
                    local = x.chunk(group.size, dim=-1)[group.rank].contiguous()
                    row_result = model.row(local)
                    branch_result = model.branches(x)
                    qkv_result = model.qkv(x)

                    def invoke():
                        return model.row(local), model.branches(x), model.qkv(x)

                    with CUDAGraph(context=context) as graph:
                        graph.capture(invoke)
                        for multiplier in (1.0, 2.0):
                            if multiplier == 2.0:
                                x.mul_(2)
                                local.copy_(
                                    x.chunk(group.size, dim=-1)[group.rank]
                                )
                            row_result, branch_result, qkv_result = (
                                graph.replay()
                            )
                            decoded = quantizer.quantize(x).dequantize(
                                dtype=torch.float32
                            )
                            matrix = quantizer.quantize(
                                source["row.weight"].to(device)
                            ).dequantize(dtype=torch.float32)
                            torch.testing.assert_close(
                                row_result.float(),
                                F.linear(decoded, matrix),
                                rtol=gamma,
                                atol=0,
                            )
                            for prefix, results in (
                                ("branches", branch_result),
                                ("qkv", qkv_result),
                            ):
                                for name, actual in results.items():
                                    matrix = quantizer.quantize(
                                        source[
                                            f"{prefix}.projections.{name}.weight"
                                        ].to(device)
                                    ).dequantize(dtype=torch.float32)
                                    expected = F.linear(decoded, matrix).chunk(
                                        group.size, dim=-1
                                    )[group.rank]
                                    torch.testing.assert_close(
                                        actual.float(),
                                        expected,
                                        rtol=gamma,
                                        atol=0,
                                    )
                    torch.cuda.synchronize(device)


def test_loaded_quantized_branches_preserve_scale_domains_and_replay(tmp_path):
    model = Projections(None)
    generator = torch.Generator().manual_seed(113)
    source = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        value = (
            torch.rand(parameter.shape, generator=generator) + 0.125
        ).bfloat16()
        if name == "row.weight":
            value *= torch.tensor([0.03, 0.3, 3, 30]).repeat_interleave(128)
        else:
            # Different branch and channel magnitudes expose accidental sharing
            # of source statistics between branches or local output shards.
            value *= (index + 1) * torch.linspace(
                0.03, 30, value.shape[0]
            ).unsqueeze(1)
        source[name] = value
    save_file(source, tmp_path / "model.safetensors")
    mp.spawn(
        _run,
        args=((tmp_path / "projections").as_uri(), tmp_path, source),
        nprocs=4,
        join=True,
    )
