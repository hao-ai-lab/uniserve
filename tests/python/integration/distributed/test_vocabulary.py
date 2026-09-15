"""Vocabulary loading and masked embeddings share padded TP ownership.

Gathering shares the same padded TP ownership.
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
from uniserve.model import Logits
from uniserve.nn.linear import VocabParallelEmbedding, VocabParallelHead
from uniserve.runtime import initialize_process_groups

pytestmark = pytest.mark.integration


class Vocabulary(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.embedding = VocabParallelEmbedding(size, 4)
        self.head = VocabParallelHead(4, size)
        self.head.weight = self.embedding.weight


def _mapping(model):
    return (
        weights.ModuleMapping(
            model,
            "primary",
            lambda reader: (
                weights.Assignment(
                    model.embedding.weight, reader.get("weight")
                ),
            ),
            frozenset({"embedding.weight", "head.weight"}),
        ),
    )


def _run(rank, rendezvous, root):
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device="cpu",
        init_method=rendezvous,
    ) as owner:
        mesh = owner.bind(
            DeviceMesh(ranks=(3, 1, 0, 2), shape=(4,), axes=("tp",), rank=rank),
            device="cpu",
        )
        result = loading.load_model(
            Vocabulary,
            37,
            checkpoint=(
                checkpoint.Config().resolve(root, io=loading.Config()),
            ),
            mapping=_mapping,
            device="cpu",
            weights=weights.Config(dtype=torch.float32),
            meshes={"": mesh},
        )
        model = result.model
        assert model.embedding.weight is model.head.weight
        full = torch.arange(37 * 4).reshape(37, 4).float() / 16
        ids = torch.tensor([0, 36, 37, -1])
        expected = torch.cat((full[[0, 36]], torch.zeros(2, 4)), dim=0)
        torch.testing.assert_close(
            model.embedding(ids), expected, rtol=0, atol=0
        )
        for tokens in (0, 3):
            hidden = torch.arange(tokens * 4).reshape(tokens, 4).float() / 8
            logits = Logits(model.head(hidden), model.head.vocab)
            out = torch.empty(37, tokens).T
            assert logits.gather(out=out) is out
            torch.testing.assert_close(
                out, F.linear(hidden, full), rtol=0, atol=0
            )


def test_loaded_vocabulary_shards_preserve_tied_values(tmp_path):
    save_file(
        {"weight": torch.arange(37 * 4).reshape(37, 4).float() / 16},
        tmp_path / "model.safetensors",
    )
    mp.spawn(
        _run,
        args=((tmp_path / "vocab").as_uri(), tmp_path),
        nprocs=4,
        join=True,
    )
