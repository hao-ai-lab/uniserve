"""Remote experts support public prefill, decode and captured execution."""

import argparse
import os
import socket
from contextlib import ExitStack
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file

from tests.python.fixtures.checkpoints import qwen_moe_checkpoint
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.loading import weights
from uniserve.model import TextInput, TextSize
from uniserve.nn.attention import AttentionBatch, PagedInput
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import QuantizationConfig, Quantizer
from uniserve.runtime import (
    CUDAGraph,
    CUDAStream,
    ExecutionContext,
    Microbatches,
    PrefixCache,
)
from uniserve.runtime.expert_exchange import ExpertExchange
from uniserve.runtime.process_groups import initialize_process_groups
from uniserve_models import loading as models

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# BF16 placement parity and eager/graph execution use the established model
# bound. Encoded expert equations are checked separately in test_split_experts;
# different fused kernels can round differently and resolve router ties apart.
RTOL, ATOL = 2e-2, 2e-2
CAPACITY, BLOCK_SIZE = 32, 16


def _encode_checkpoint(root):
    """Store seeded expert matrices in ModelOpt's calibrated NVFP4 layout."""
    path = root / "model.safetensors"
    tensors = load_file(path)
    prefixes = [
        name.removesuffix(".gate_proj.weight")
        for name in tensors
        if ".experts." in name and name.endswith(".gate_proj.weight")
    ]
    for prefix in prefixes:
        up = tensors.pop(prefix + ".up_proj.weight")
        gate = tensors.pop(prefix + ".gate_proj.weight")
        down = tensors.pop(prefix + ".down_proj.weight")
        # Up and gate share the statistical domain of the fused projection;
        # splitting the serialized rows preserves that common tensor scale.
        projections = {
            "up_gate": Quantizer("nvfp4").quantize(
                torch.cat((up, gate)).cuda()
            ),
            "down": Quantizer("nvfp4").quantize(down.cuda()),
        }
        for name, encoded in projections.items():
            fields = {
                key: value.cpu() for key, value in encoded.buffers().items()
            }
            intervals = (
                (("up", slice(0, len(up))), ("gate", slice(len(up), None)))
                if name == "up_gate"
                else (("down", slice(None)),)
            )
            for projection, rows in intervals:
                key = f"{prefix}.{projection}_proj"
                tensors[key + ".weight"] = fields["values"][rows].clone()
                tensors[key + ".weight_scale"] = (
                    fields["block_scale"][rows]
                    .view(torch.float8_e4m3fn)
                    .clone()
                )
                tensors[key + ".weight_scale_2"] = fields[
                    "tensor_scale"
                ].clone()
                tensors[key + ".input_scale"] = torch.tensor(1.0)
    save_file(tensors, path)


def _batch(tokens, start, device):
    count = len(tokens)
    attention = AttentionBatch.single(
        PagedInput.from_blocks(
            query_lengths=(count,),
            prefix_lengths=(start,),
            blocks=((0,),),
            block_size=BLOCK_SIZE,
            causal=True,
            device=device,
        )
    )
    return TextInput(
        torch.tensor(tokens, dtype=torch.int64, device=device),
        torch.arange(start, start + count, device=device),
        attention,
    )


def _context(scope, model, device, exchange=None, *, moe="auto"):
    stream = scope.enter_context(
        CUDAStream.external(torch.cuda.Stream(device=device))
    )
    stream.wait(torch.cuda.current_stream(device))
    cache = (
        scope.enter_context(
            PrefixCache(
                model.cache_config,
                num_units=2,
                block_size=BLOCK_SIZE,
                device=device,
            )
        )
        if hasattr(model, "cache_config")
        else None
    )
    context = scope.enter_context(
        ExecutionContext(
            model,
            stream=stream,
            cache=cache,
            experts=exchange,
            moe=moe,
        )
    )
    context.prepare(TextSize(CAPACITY, 1))
    return context


def _forward(model, inputs):
    hidden = model(inputs)
    return model.compute_logits(
        hidden,
        token_indices=torch.arange(len(inputs.input_ids), device=hidden.device),
    ).gather()


@torch.inference_mode()
def _exercise(
    rank,
    local_rank,
    world_size,
    attention_ranks,
    root,
    init_method,
    attention_tp,
    format="bf16",
    transport="deepep",
):
    device = torch.device("cuda", local_rank)
    source = rank < attention_ranks
    if attention_ranks % attention_tp:
        raise ValueError("attention ranks must form complete tensor groups")
    with initialize_process_groups(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
    ) as groups:
        attention_mesh = None
        for start in range(0, attention_ranks, attention_tp):
            mesh = groups.bind(
                DeviceMesh(
                    ranks=tuple(range(start, start + attention_tp)),
                    shape=(attention_tp,),
                    axes=("tp",),
                    rank=rank,
                ),
                device=device,
            )
            if rank in mesh.ranks:
                attention_mesh = mesh
        metadata = models.read_config(root)
        with torch.device("meta"):
            description = metadata.model_class(metadata.model)
        expert_paths = frozenset(
            path
            for path, module in description.named_modules()
            if isinstance(module, FusedMoE)
        )
        layers = [
            description.get_submodule(path) for path in sorted(expert_paths)
        ]
        first = layers[0]
        intermediate, activation = first.intermediate_size, first.activation
        shapes = {(m.num_experts, m.hidden_size, m.top_k) for m in layers}
        assert len(shapes) == 1
        num_experts, hidden_size, top_k = next(iter(shapes))
        del first, layers, description
        representation = None
        if format == "mxfp8":
            representation = metadata.precisions["mxfp8-experts"]
        elif format == "nvfp4":
            # The reference and remote model load identical expert encodings.
            # Fixed synthetic NVFP4 scales exercise both projection boundaries;
            # This compares placement, not quantization quality against BF16.
            quantization = QuantizationConfig(
                Quantizer(format),
                Quantizer(format, calibrated_scale=1.0),
            )
            representation = weights.Config(
                quantization=dict.fromkeys(expert_paths, quantization)
            )

        # Two independent sequences, each with prefill followed by two decode
        # steps. All ranks use the same declared step order and capacity.
        replica = rank // attention_tp
        prompts = ([1, 3, 5, 7 + replica], [2, 4, 6, 8 + replica])
        expected, inputs = [], [[], []]
        tokens, positions = list(prompts), [0, 0]
        if source and format == "bf16":
            reference = models.load_model(
                metadata,
                device=device,
                weights=representation,
            ).model
            with ExitStack() as scope:
                context = _context(scope, reference, device)
                for lane, prompt in enumerate(prompts):
                    sequence_inputs, sequence_logits = [], []
                    reference_tokens, position = prompt, 0
                    for _ in range(3):
                        batch = _batch(reference_tokens, position, device)
                        sequence_inputs.append(batch)
                        context.bind_attention(batch.attention)
                        with context.activate():
                            logits = _forward(reference, batch)
                        context.stream.synchronize()
                        sequence_logits.append(logits.cpu())
                        position += len(reference_tokens)
                        reference_tokens = [int(logits[-1].argmax())]
                    inputs[lane] = sequence_inputs
                    expected.append(sequence_logits)
            del context, reference, logits
            torch.cuda.empty_cache()

        # No source starts a transport while a peer still loads or compiles
        # the colocated numerical reference.
        dist.all_reduce(
            torch.zeros((), dtype=torch.int32),
            group=groups.process_group._require(),
        )
        model = models.load_model(
            metadata,
            device=device,
            weights=representation,
            modules=None if source else expert_paths,
            exclude_modules=expert_paths if source else frozenset(),
            meshes={"": attention_mesh} if source else None,
            experts=None
            if source
            else Communicator(
                tuple(range(attention_ranks, world_size)),
                rank - attention_ranks,
                "experts",
                device,
            ),
        ).model
        if not source:
            # The same loaded expert modules, in numerical traversal order;
            # the execution owner does not need the nonresident attention tree.
            model = torch.nn.ModuleList(
                module
                for module in model.modules()
                if isinstance(module, FusedMoE)
            )
        exchanges = [
            ExpertExchange(
                groups.process_group,
                max_tokens=CAPACITY,
                top_k=top_k,
                num_experts=num_experts,
                hidden_size=hidden_size,
                device=device,
                transport=transport,
                attention_ranks=attention_ranks,
                intermediate_size=intermediate,
                activation=activation,
            )
            for _ in prompts
        ]
        with ExitStack() as scope:
            contexts = [
                _context(scope, model, device, exchange)
                for exchange in exchanges
            ]
            run = Microbatches(contexts)
            scope.callback(run.close)
            for step in range(3):
                if source:
                    for lane, context in enumerate(contexts):
                        if format != "bf16":
                            inputs[lane].append(
                                _batch(tokens[lane], positions[lane], device)
                            )
                        context.bind_attention(inputs[lane][step].attention)

                def forward(lane):
                    context, exchange = contexts[lane], exchanges[lane]
                    exchange.begin(CAPACITY)
                    try:
                        output = (
                            _forward(model, inputs[lane][step])
                            if source
                            else None
                        )
                        context.join_expert_layers()
                        return output
                    finally:
                        exchange.end()

                calls = [partial(forward, lane) for lane in range(len(prompts))]
                outputs = run(calls)
                torch.cuda.synchronize(device)

                def compare(values, reference):
                    if source:
                        for lane, actual in enumerate(values):
                            wanted = reference[lane]
                            torch.testing.assert_close(
                                actual.cpu(), wanted, rtol=RTOL, atol=ATOL
                            )
                            assert torch.equal(
                                actual[-1].argmax().cpu(), wanted[-1].argmax()
                            )

                # Preserve the eager result before graph replay reuses device
                # buffers. Quantized decode follows this kernel's own tokens;
                # a portable kernel's routing decisions are not its oracle.
                eager = [value.cpu() for value in outputs] if source else []
                if source:
                    for lane, value in enumerate(eager):
                        assert value.shape[0] == len(
                            inputs[lane][step].input_ids
                        )
                        assert torch.isfinite(value).all()
                    if format == "bf16":
                        compare(outputs, [lane[step] for lane in expected])
                with CUDAGraph(context=contexts[0]) as graph:
                    graph.capture(lambda: run(calls))
                    for _ in range(3):
                        outputs = graph.replay()
                        contexts[0].stream.synchronize()
                        compare(outputs, eager)
                if source and format != "bf16":
                    for lane, value in enumerate(eager):
                        positions[lane] += len(tokens[lane])
                        tokens[lane] = [int(value[-1].argmax())]
        for exchange in exchanges:
            exchange.close()


def _local(rank, port, root, attention_tp, format, transport):
    _exercise(
        rank,
        rank,
        2 * attention_tp,
        attention_tp,
        root,
        f"tcp://127.0.0.1:{port}",
        attention_tp,
        format,
        transport,
    )


@pytest.mark.parametrize("attention_tp", [1, 2])
@pytest.mark.parametrize(
    "format,transport",
    [
        ("bf16", "deepep"),
        ("nvfp4", "deepep"),
        ("mxfp8", "megamoe"),
        ("nvfp4", "megamoe"),
    ],
)
def test_public_prefill_and_decode_with_remote_experts(
    tmp_path, attention_tp, format, transport
):
    if torch.cuda.device_count() < 2 * attention_tp:
        pytest.skip(f"requires {2 * attention_tp} CUDA devices")
    qwen_moe_checkpoint(
        tmp_path,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_experts=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    )
    if format == "nvfp4":
        _encode_checkpoint(tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _local,
        args=(port, str(tmp_path), attention_tp, format, transport),
        nprocs=2 * attention_tp,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attention-ranks", type=int, required=True)
    parser.add_argument("--attention-tp", type=int, default=1)
    parser.add_argument(
        "--format", choices=("bf16", "mxfp8", "nvfp4"), default="bf16"
    )
    parser.add_argument(
        "--transport", choices=("deepep", "megamoe"), default="deepep"
    )
    arguments = parser.parse_args()
    _exercise(
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
        arguments.attention_ranks,
        arguments.model,
        "env://",
        arguments.attention_tp,
        arguments.format,
        arguments.transport,
    )
