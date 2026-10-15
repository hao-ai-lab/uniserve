"""Every Ulysses head projection returns each member's heads of every token.

A merged projection of whole heads reaches each member's head shard by
gathering all members' input rows or by exchanging the shards each member
projects from its own rows, the latter whole or in overlapped steps. Every
mode, unencoded and with tensor-wide FP8, row-wise FP8 and dynamic NVFP4
activations, returns every token's head shard within one rounding of the
complete encoded projection: tensor-wide statistics span every member's rows
although the members' inputs differ in magnitude. Members cut their inputs
at different boundaries, logical members run opposite to backend order, and
a captured projection replays over new inputs. An exchange whose consumer
stops early leaves no transfer writing its storage.
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
from uniserve.nn import MergedColumnParallelLinear
from uniserve.nn.attention import AttentionParallelConfig, Ulysses
from uniserve.quantization import QuantizationConfig, Quantizer
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    initialize_process_groups,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# Four 256-channel branches of 64-channel heads over a 640-channel input.
HIDDEN, WIDTH, HEAD = 640, 256, 64
BRANCHES = ("q", "k", "v", "gate")
# Each member's 2,432 rows span two whole 1,024-row steps and a partial one.
SHARD, STEP = 2432, 1024
MODES = (
    Ulysses("ulysses"),
    Ulysses("ulysses", projection="exchange"),
    Ulysses("ulysses", projection="exchange", projection_rows=STEP),
)
ENCODINGS = (
    None,
    QuantizationConfig(Quantizer("fp8"), Quantizer("fp8")),
    QuantizationConfig(Quantizer("fp8", axis=0), Quantizer("fp8", axis=0)),
    QuantizationConfig(Quantizer("nvfp4"), Quantizer("nvfp4")),
)


class Projection(nn.Module):
    def __init__(self, _):
        super().__init__()
        self.heads = MergedColumnParallelLinear(
            HIDDEN,
            dict.fromkeys(BRANCHES, WIDTH),
            branch_width=HEAD,
            bias=False,
        )


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
    torch.cuda.set_device(device)
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=device,
        init_method=rendezvous,
    ) as owner:
        mesh = owner.bind(
            DeviceMesh(ranks=(1, 0), shape=(2,), axes=("ulysses",), rank=rank),
            device=device,
        )
        member = mesh.get_group("ulysses").rank
        tokens = 2 * SHARD
        local = slice(member * SHARD, (member + 1) * SHARD)
        shard = slice(member * WIDTH // 2, (member + 1) * WIDTH // 2)
        # Positive inputs keep relative error bounds meaningful; the second
        # member's rows are 30 times larger, so a tensor-wide scale taken
        # from one member's rows alone would saturate the other's.
        x = (
            (
                torch.arange(tokens * HIDDEN, device=device).view(
                    tokens, HIDDEN
                )
                % 13
                + 1
            )
            / 16
        ).bfloat16()
        x[SHARD:] *= 30
        # Each member's producer cuts its rows at different boundaries.
        cut = 700 + 333 * member

        for encoding in ENCODINGS:
            for mode in MODES:
                model = loading.load_model(
                    Projection,
                    None,
                    checkpoint=(
                        checkpoint.Config().resolve(
                            directory, io=loading.Config()
                        ),
                    ),
                    mapping=_mapping,
                    device=device,
                    meshes={"": mesh},
                    attention={"": AttentionParallelConfig(heads=mode)},
                    weights=weights.Config(
                        quantization={} if encoding is None else {"": encoding}
                    ),
                ).model
                projection = model.heads

                def chunks():
                    for start in range(local.start, local.stop, cut):
                        stop = min(start + cut, local.stop)
                        yield slice(start, stop), x[start:stop]

                def project():
                    out = projection.output_buffers(
                        tokens, dtype=x.dtype, device=device
                    )
                    covered = torch.zeros(
                        tokens, dtype=torch.int32, device=device
                    )
                    for rows, _ in projection.forward_chunks(
                        chunks(), token_slice=local, num_tokens=tokens, out=out
                    ):
                        covered[rows] += 1
                    return out, covered

                def check(out, covered):
                    assert bool((covered == 1).all())
                    decoded = (
                        x.float()
                        if encoding is None
                        else encoding.activation.quantize(x).dequantize(
                            dtype=torch.float32
                        )
                    )
                    for name in BRANCHES:
                        matrix = source[f"heads.projections.{name}.weight"].to(
                            device
                        )
                        matrix = (
                            matrix.float()
                            if encoding is None
                            else encoding.weight.quantize(matrix).dequantize(
                                dtype=torch.float32
                            )
                        )
                        # One product rounding and one BF16 output rounding.
                        gamma = 2 * 2**-8 / (1 - 2 * 2**-8)
                        torch.testing.assert_close(
                            out[name].float(),
                            F.linear(decoded, matrix)[:, shard],
                            rtol=gamma,
                            atol=0,
                        )

                stream = CUDAStream.external(torch.cuda.Stream(device=device))
                stream.wait(torch.cuda.current_stream(device))
                with (
                    stream,
                    ExecutionContext(model, stream=stream) as context,
                ):
                    context.prepare(TextSize(tokens, 1))
                    check(*project())

                    if mode.projection == "exchange":
                        # A consumer that stops early leaves no transfer
                        # writing storage the next projection reuses.
                        partial = projection.forward_chunks(
                            chunks(), token_slice=local, num_tokens=tokens
                        )
                        next(partial)
                        partial.close()

                    with CUDAGraph(context=context) as graph:
                        graph.capture(project)
                        x.mul_(2)
                        check(*graph.replay())
                        x.div_(2)
                torch.cuda.synchronize(device)


def test_every_ulysses_projection_returns_each_members_heads(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("the Ulysses projections need two devices")
    model = Projection(None)
    generator = torch.Generator().manual_seed(29)
    source = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        # Different branch and channel magnitudes expose statistics shared
        # across branches or head shards.
        source[name] = (
            (torch.rand(parameter.shape, generator=generator) + 0.125)
            * (index + 1)
            * torch.linspace(0.03, 3, parameter.shape[0]).unsqueeze(1)
        ).bfloat16()
    save_file(source, tmp_path / "model.safetensors")
    mp.spawn(
        _run,
        args=((tmp_path / "rendezvous").as_uri(), tmp_path, source),
        nprocs=2,
        join=True,
    )
