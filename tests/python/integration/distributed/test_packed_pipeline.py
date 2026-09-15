"""Multimodal capabilities preserve sequence partitions and shared prefix values."""

from contextlib import ExitStack

import pytest
import torch
import torch.multiprocessing as mp

from tests.python.integration.model_loading.test_bagel import _checkpoint as bagel_checkpoint
from tests.python.integration.model_loading.test_sensenova_u1 import _checkpoint as u1_checkpoint
from uniserve import loading
from uniserve.distributed import DeviceMesh
from uniserve.loading import checkpoint, weights
from uniserve.media import image
from uniserve.model import TextInput, TextSize
from uniserve.nn.attention import (
    AttentionParallelConfig,
    PagedInput,
    SequenceLengths,
    Ulysses,
    VarlenInput,
)
from uniserve.runtime import ExecutionContext, PrefixCache, initialize_process_groups
from uniserve_models import bagel, sensenova_u1
from uniserve_worker.bootstrap.inputs import image_builder

pytestmark = pytest.mark.integration


def _load(root, architecture, config, *, mesh=None, attention=None):
    package = bagel if architecture == "bagel" else sensenova_u1
    paths = ("text", "denoiser")
    return loading.load_model(
        package.Model,
        config,
        checkpoint=(checkpoint.Config("primary").resolve(root, io=loading.Config()),),
        mapping=package.checkpoint_mappings,
        device="cpu",
        weights=weights.Config(),
        modules=frozenset(paths),
        meshes=None if mesh is None else {path: mesh for path in paths},
        attention=None if attention is None else {path: attention for path in paths},
    ).model


@torch.inference_mode()
def _run(rank, rendezvous, root, architecture, config, shape, axes):
    with initialize_process_groups(
        rank=rank, local_rank=rank, world_size=4, device="cpu", init_method=rendezvous
    ) as groups:
        mesh = groups.bind(
            DeviceMesh(ranks=(3, 1, 0, 2), shape=shape, axes=axes, rank=rank), device="cpu"
        )
        parallel = (
            AttentionParallelConfig(heads=Ulysses("tokens"))
            if "tokens" in axes
            else AttentionParallelConfig()
        )
        reference = _load(root, architecture, config)
        model = _load(root, architecture, config, mesh=mesh, attention=parallel)
        pipeline = mesh.get_group("pp" if "pp" in axes else ())
        last = pipeline.rank + 1 == pipeline.size
        assert model.text.backbone is model.denoiser.backbone
        with ExitStack() as scope:
            caches = [
                scope.enter_context(
                    PrefixCache(item.text.cache_config, num_blocks=2, block_size=4, device="cpu")
                )
                for item in (reference, model)
            ]
            contexts = [
                scope.enter_context(ExecutionContext(item.text, cache=cache, attention="torch"))
                for item, cache in zip((reference, model), caches, strict=True)
            ]
            for context in contexts:
                context.prepare(TextSize(5, 2))
            prefixes = (0, 0)
            for tokens, positions, queries in (
                ((1, 3, 5, 7, 9), (0, 1, 2, 0, 1), (3, 2)),
                ((11, 13), (3, 2), (1, 1)),
            ):
                batch = PagedInput.from_blocks(
                    blocks=((0,), (1,)),
                    query_lengths=queries,
                    prefix_lengths=prefixes,
                    block_size=4,
                    causal=True,
                    device="cpu",
                )
                inputs = TextInput(
                    torch.tensor(tokens), torch.tensor(positions).expand(3, -1), batch
                )
                outputs = []
                for item, context in zip((reference, model), contexts, strict=True):
                    with context.activate():
                        context.bind_attention(batch)
                        hidden = item.text(inputs)
                        result = item.text.compute_logits(
                            hidden, token_indices=torch.arange(len(tokens))
                        )
                        outputs.append(None if result is None else result.gather())
                if last:
                    torch.testing.assert_close(outputs[1], outputs[0], rtol=2e-2, atol=2e-2)
                else:
                    assert outputs[1] is None
                prefixes = tuple(
                    prefix + query for prefix, query in zip(prefixes, queries, strict=True)
                )
            for name, layout in model.text.cache_config.layers.items():
                for block, length in enumerate(prefixes):
                    actual = caches[1].state(name).read((block,), start=0, length=length)
                    full = caches[0].state(name).read((block,), start=0, length=length)
                    heads = torch.tensor(layout.head_indices)
                    for value, expected in zip(actual, full, strict=True):
                        torch.testing.assert_close(
                            value, expected.index_select(1, heads), rtol=2e-2, atol=2e-2
                        )

        # Diffusion remains a separate homogeneous numerical call, including
        # BAGEL's text-expert markers within its mathematical image layout.
        size = image.Config(8, 8)
        factory = image_builder(reference)
        sample = torch.empty(reference.denoiser.latent_shape("image", size), dtype=torch.bfloat16)
        factory.initialize(size, seed=71, out=sample)
        positions = factory.positions(size, 9, device="cpu")
        count = positions.shape[-1]
        lengths = SequenceLengths.from_lengths((count,), device="cpu")
        attention = VarlenInput(lengths, lengths, (False,))
        inputs = factory.bind(
            samples=(sample,),
            sizes=(size,),
            timesteps=(torch.tensor(0.5),),
            positions=(positions,),
            attention=attention,
            step_index=0,
        )
        outputs = []
        before = sample.clone()
        for item in (reference, model):
            with ExecutionContext(item.denoiser, attention="torch") as context:
                context.prepare(TextSize(count, 1))
                context.bind_attention(attention)
                outputs.append(
                    item.denoiser(inputs, state={}, constants={}, workspace={})["image"][0]
                )
        if last:
            torch.testing.assert_close(outputs[1].tensor, outputs[0].tensor, rtol=2e-2, atol=2e-2)
        else:
            assert outputs[1] is None
        torch.testing.assert_close(sample, before, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ("bagel", "sensenova_u1"))
@pytest.mark.parametrize(
    "shape,axes", (((4,), ("tp",)), ((2, 2), ("pp", "tokens")), ((2, 2), ("tp", "tokens")))
)
def test_partitioned_multimodal_calls_preserve_logits_prefix_and_image_values(
    tmp_path, architecture, shape, axes
):
    if architecture == "bagel":
        _, _, config = bagel_checkpoint(tmp_path)
    else:
        reference, _ = u1_checkpoint(tmp_path, torch.bfloat16)
        config = reference.config
    mp.spawn(
        _run,
        args=((tmp_path / "multimodal").as_uri(), tmp_path, architecture, config, shape, axes),
        nprocs=4,
        join=True,
    )
