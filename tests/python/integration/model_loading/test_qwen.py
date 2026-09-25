"""Public Qwen loading agrees with the checkpoint equations.

Cached decoding agrees with the checkpoint equations as well.
"""

import pytest
import torch

from tests.python.fixtures.checkpoints import qwen_checkpoint
from uniserve import loading
from uniserve.loading import weights
from uniserve.model import EmbeddingReplacement, TextInput, TextSize
from uniserve.nn.attention import PagedInput
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve_models import loading as models
from uniserve_models import qwen3

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("tied,theta", [(False, 10_000.0), (True, 1_000_000.0)])
def test_checkpoint_prefill_decode_and_selected_logits(tmp_path, tied, theta):
    reference = qwen_checkpoint(tmp_path, tied, theta)
    io = loading.Config()
    config = models.read_config(tmp_path, io=io)
    result = models.load_model(
        config, device="cpu", weights=weights.Config(dtype=torch.float32)
    )
    model = result.model
    if tied:
        assert model.backbone.embedding.weight is model.lm_head.weight
    tokens = torch.tensor([[1, 3, 9, 2]])
    with torch.no_grad():
        expected = reference(tokens).logits
    with PrefixCache(
        model.cache_config, num_blocks=2, block_size=4, device="cpu"
    ) as cache:
        with ExecutionContext(model, cache=cache, attention="torch") as context:
            context.prepare(TextSize(4, 1))
            for start, stop in ((0, 3), (3, 4)):
                batch = PagedInput.from_blocks(
                    query_lengths=(stop - start,),
                    prefix_lengths=(start,),
                    blocks=((0,),),
                    block_size=4,
                    causal=True,
                    device="cpu",
                )
                context.bind_attention(batch)
                hidden = model(
                    TextInput(
                        tokens[0, start:stop], torch.arange(start, stop), batch
                    )
                )
                logits = model.compute_logits(
                    hidden, token_indices=torch.arange(stop - start)
                )
                torch.testing.assert_close(
                    logits.gather(),
                    expected[0, start:stop],
                    rtol=1e-5,
                    atol=1e-6,
                )
                assert model.compute_logits(
                    hidden, token_indices=torch.empty(0, dtype=torch.int64)
                ).gather().shape == (0, 37)


def test_embedding_replacement_matches_numerical_embedding_input(tmp_path):
    reference = qwen_checkpoint(tmp_path)
    io = loading.Config()
    model = loading.load_model(
        qwen3.Model,
        qwen3.read_config(tmp_path, io, sources={}),
        checkpoint=tuple(
            source.resolve(tmp_path, io=io)
            for source in qwen3.checkpoint_sources
        ),
        mapping=qwen3.checkpoint_mappings,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    ).model
    tokens = torch.tensor([1, 3, 9])
    embeddings = model.embed_input_ids(tokens).clone()
    embeddings[1].fill_(0.25)
    with torch.no_grad():
        expected = reference(inputs_embeds=embeddings[None]).logits[0]
    batch = PagedInput.from_blocks(
        query_lengths=(3,),
        prefix_lengths=(0,),
        blocks=((0,),),
        block_size=4,
        causal=True,
        device="cpu",
    )
    with PrefixCache(
        model.cache_config, num_blocks=1, block_size=4, device="cpu"
    ) as cache:
        with ExecutionContext(model, cache=cache, attention="torch") as context:
            context.prepare(TextSize(3, 1))
            context.bind_attention(batch)
            inputs = TextInput(
                tokens,
                torch.arange(3),
                batch,
                EmbeddingReplacement(
                    embeddings, torch.tensor([False, True, False])
                ),
            )
            hidden = model(inputs)
            torch.testing.assert_close(
                model.compute_logits(
                    hidden, token_indices=torch.arange(3)
                ).gather(),
                expected,
                rtol=1e-5,
                atol=1e-6,
            )


def _partitioned(rank, rendezvous, root, shape, axes):
    from uniserve.distributed import DeviceMesh
    from uniserve.nn.attention import AttentionParallelConfig, Ulysses
    from uniserve.runtime import initialize_process_groups

    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device="cpu",
        init_method=rendezvous,
    ) as owner:
        mesh = owner.bind(
            DeviceMesh(ranks=(3, 1, 0, 2), shape=shape, axes=axes, rank=rank),
            device="cpu",
        )
        io = loading.Config()
        parallel = (
            AttentionParallelConfig(heads=Ulysses("sp"))
            if "sp" in axes
            else AttentionParallelConfig()
        )
        model = loading.load_model(
            qwen3.Model,
            qwen3.read_config(root, io, sources={}),
            checkpoint=tuple(
                source.resolve(root, io=io)
                for source in qwen3.checkpoint_sources
            ),
            mapping=qwen3.checkpoint_mappings,
            device="cpu",
            weights=weights.Config(dtype=torch.float32),
            meshes={"": mesh},
            attention={"": parallel},
        ).model
        from uniserve.model import EntryPoint
        from uniserve.nn.attention import SequenceLengths, VarlenInput
        from uniserve_worker.model_executor.component_binding import Call
        from uniserve_worker.model_executor.graph_storage import GraphStorage
        from uniserve_worker.model_executor.text_runner import TextRunner
        from uniserve_worker.sampling.metadata import TokenSelection

        expected = torch.load(root / "expected.pt", weights_only=True)
        with PrefixCache(
            model.cache_config, num_blocks=1, block_size=4, device="cpu"
        ) as cache:
            with ExecutionContext(
                model, cache=cache, attention="torch"
            ) as context:
                context.prepare(TextSize(4, 4))
                call = TextRunner(
                    "text",
                    Call("", model, EntryPoint("forward")),
                    torch.device("cpu"),
                    (),
                    None,
                    context,
                    storage=GraphStorage(),
                    devices=(),
                )
                for start, stop in ((0, 3), (3, 4)):
                    batch = PagedInput.from_blocks(
                        blocks=((0,),),
                        query_lengths=(stop - start,),
                        prefix_lengths=(start,),
                        block_size=4,
                        causal=True,
                        device="cpu",
                    )
                    context.bind_attention(batch)
                    hidden = model(
                        TextInput(
                            torch.tensor([1, 3, 9, 2])[start:stop],
                            torch.arange(start, stop),
                            batch,
                        )
                    )
                    logits = model.compute_logits(
                        hidden, token_indices=torch.arange(stop - start)
                    )
                    if (
                        "pp" not in axes
                        or mesh.get_group("pp").rank
                        == mesh.get_group("pp").size - 1
                    ):
                        torch.testing.assert_close(
                            logits.gather(),
                            expected[0, start:stop],
                            rtol=1e-5,
                            atol=1e-6,
                        )
                    else:
                        assert logits is None

                lengths = SequenceLengths.from_lengths(
                    (2, 0, 1, 1), device="cpu"
                )
                attention = VarlenInput(lengths, lengths, (True,) * 4)
                context.bind_attention(attention)
                selected = (
                    call(
                        TextInput(
                            torch.tensor([1, 3, 9, 2]),
                            torch.tensor([0, 1, 0, 0]),
                            attention,
                        ),
                        (
                            TokenSelection.HIDDEN,
                            TokenSelection.ALL_LOGITS,
                            TokenSelection.LAST_LOGITS,
                            TokenSelection.ALL_LOGITS,
                        ),
                    )
                    .materialize()
                    .values
                )
                reference = torch.load(root / "selected.pt", weights_only=True)
                for actual, wanted in zip(selected, reference, strict=True):
                    torch.testing.assert_close(
                        actual, wanted, rtol=1e-5, atol=1e-6
                    )
                call.close()


@pytest.mark.parametrize(
    "shape,axes", [((4,), ("tp",)), ((2, 2), ("pp", "sp"))]
)
def test_partitioned_checkpoint_decoder_matches_complete_model(
    tmp_path, shape, axes
):
    import torch.multiprocessing as mp

    reference = qwen_checkpoint(tmp_path, tied=True)
    with torch.no_grad():
        expected = reference(torch.tensor([[1, 3, 9, 2]])).logits
    torch.save(expected, tmp_path / "expected.pt")
    with torch.no_grad():
        hidden = reference(
            torch.tensor([[1, 3]]), output_hidden_states=True
        ).hidden_states[-1][0]
        logits = tuple(
            reference(torch.tensor([[token]])).logits[0] for token in (9, 2)
        )
    torch.save(
        (hidden, torch.empty((0, 37)), *logits), tmp_path / "selected.pt"
    )

    mp.spawn(
        _partitioned,
        args=((tmp_path / "rendezvous").as_uri(), tmp_path, shape, axes),
        nprocs=4,
        join=True,
    )


def test_decoder_without_prefix_storage(tmp_path):
    from uniserve.nn.attention import SequenceLengths, VarlenInput

    reference = qwen_checkpoint(tmp_path)
    io = loading.Config()
    model = loading.load_model(
        qwen3.Model,
        qwen3.read_config(tmp_path, io, sources={}),
        checkpoint=tuple(
            source.resolve(tmp_path, io=io)
            for source in qwen3.checkpoint_sources
        ),
        mapping=qwen3.checkpoint_mappings,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    ).model
    tokens = torch.tensor([1, 3, 9])
    lengths = SequenceLengths.from_lengths((3,), device="cpu")
    batch = VarlenInput(lengths, lengths, (True,))
    inputs = TextInput(tokens, torch.arange(3), batch)
    with torch.no_grad():
        expected = reference(tokens[None]).logits[0]
    direct = model.compute_logits(
        model(inputs), token_indices=torch.arange(3)
    ).gather()
    torch.testing.assert_close(direct, expected, rtol=1e-5, atol=1e-6)
    with ExecutionContext(model, attention="torch") as context:
        context.prepare(TextSize(3, 1))
        context.bind_attention(batch)
        bound = model.compute_logits(
            model(inputs), token_indices=torch.arange(3)
        ).gather()
        torch.testing.assert_close(bound, expected, rtol=1e-5, atol=1e-6)


def test_text_encoder_retains_checkpoint_layers_and_sequence_boundaries(
    tmp_path,
):
    from uniserve.model import TextEncoder

    reference = qwen_checkpoint(tmp_path)
    io = loading.Config()
    model = loading.load_model(
        qwen3.Model,
        qwen3.read_config(tmp_path, io, sources={}),
        checkpoint=tuple(
            source.resolve(tmp_path, io=io)
            for source in qwen3.checkpoint_sources
        ),
        mapping=qwen3.checkpoint_mappings,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    ).model
    # Conditioning uses the first checkpoint layer's residual stream, before
    # subsequent decoder layers or final normalization.
    model.backbone.norm = torch.nn.Identity()
    encoder = TextEncoder(model.backbone, retained_layers=(0,))
    tokens = (
        torch.tensor([1, 2, 3]),
        torch.tensor([9]),
        torch.empty(0, dtype=torch.long),
    )
    with torch.no_grad():
        expected = tuple(
            reference(row[None], output_hidden_states=True).hidden_states[1][0]
            for row in tokens[:2]
        )
        outputs = encoder.encode(tokens)
    for actual, target in zip(outputs[:2], expected, strict=True):
        torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-6)
    assert outputs[2].shape == (0, 32)
    assert encoder.encode(()) == ()


def _partitioned_encoder(rank, rendezvous, root):
    from uniserve.distributed import DeviceMesh, parallelize_
    from uniserve.model import TextEncoder
    from uniserve.nn.attention import AttentionParallelConfig, Ulysses
    from uniserve.runtime import initialize_process_groups

    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device="cpu",
        init_method=rendezvous,
    ) as owner:
        mesh = owner.bind(
            DeviceMesh(
                ranks=(3, 1, 0, 2),
                shape=(2, 2),
                axes=("pp", "tokens"),
                rank=rank,
            ),
            device="cpu",
        )
        model = models.load_model(
            models.read_config(root),
            device="cpu",
            weights=weights.Config(dtype=torch.float32),
        ).model
        encoder = TextEncoder(model.backbone, retained_layers=(0, 1))
        parallelize_(
            encoder,
            mesh,
            attention=AttentionParallelConfig(heads=Ulysses("tokens")),
        )
        expected = torch.load(root / "encoder-features.pt", weights_only=True)
        sequences = (
            (
                torch.tensor([1, 2, 3]),
                torch.empty(0, dtype=torch.long),
                torch.tensor([9]),
            ),
            (torch.tensor([3]),),
        )
        last = mesh.get_group("pp").rank == 1
        with ExecutionContext(encoder, attention="torch") as context:
            context.prepare(TextSize(4, 3))
            for tokens, wanted in zip(sequences, expected, strict=True):
                actual = encoder.encode(tokens)
                if last:
                    for value, target in zip(actual, wanted, strict=True):
                        torch.testing.assert_close(
                            value, target, rtol=1e-5, atol=1e-6
                        )
                else:
                    assert actual is None
            assert encoder.encode(()) == (() if last else None)


def test_text_encoder_pipeline_restores_sequences_after_empty_token_shards(
    tmp_path,
):
    import torch.multiprocessing as mp

    reference = qwen_checkpoint(tmp_path)
    sequences = (
        (
            torch.tensor([1, 2, 3]),
            torch.empty(0, dtype=torch.long),
            torch.tensor([9]),
        ),
        (torch.tensor([3]),),
    )
    with torch.no_grad():
        expected = tuple(
            tuple(
                reference.model(tokens[None]).last_hidden_state[0]
                if tokens.numel()
                else torch.empty(0, 32)
                for tokens in batch
            )
            for batch in sequences
        )
    torch.save(expected, tmp_path / "encoder-features.pt")
    mp.spawn(
        _partitioned_encoder,
        args=((tmp_path / "rendezvous").as_uri(), tmp_path),
        nprocs=4,
        join=True,
    )


def test_public_partial_loading_exposes_selected_decoder_values(tmp_path):
    from uniserve.nn.attention import SequenceLengths, VarlenInput

    reference = qwen_checkpoint(tmp_path)
    config = models.read_config(tmp_path, modules=frozenset({"backbone"}))
    loaded = models.load_model(
        config, device="cpu", weights=weights.Config(dtype=torch.float32)
    )
    backbone = loaded.model.backbone
    tokens = torch.tensor([1, 3, 9])
    lengths = SequenceLengths.from_lengths((3,), device="cpu")
    with torch.no_grad():
        actual = backbone(
            backbone.embed_input_ids(tokens),
            torch.arange(3),
            VarlenInput(lengths, lengths, (True,)),
        )
        expected = reference.model(tokens[None]).last_hidden_state[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert loaded.model.lm_head.weight.is_meta
    with pytest.raises(ValueError, match="exceeds"):
        models.load_model(config, device="cpu", modules=frozenset({""}))
    with pytest.raises(ValueError, match="mutually exclusive"):
        models.load_model(
            config, device="cpu", precision="bf16", weights=weights.Config()
        )
