"""Qwen3 language model composition and numerical entry points."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from uniserve.model import (
    DEFAULT_COMPONENT,
    CausalLM,
    ComponentEntry,
    EntryPoint,
)
from uniserve.nn.linear import VocabParallelHead

from .config import Config
from .transformer import Transformer


class Model(CausalLM):
    """The same loaded Qwen computation for Python and serving callers.

    ``CausalLM`` supplies ``forward``, ``embed_input_ids`` and
    ``compute_logits`` over the ``Transformer`` backbone and a
    vocabulary-parallel head. Pipeline binding later drops the embedding
    outside the first stage and the head outside the last one.
    """

    def __init__(self, config: Config):
        super().__init__(
            Transformer(config),
            VocabParallelHead(config.hidden_size, config.vocab_size),
        )
        self.config = config

        # Tied checkpoints store one embedding matrix; the head shares that
        # same Parameter instead of holding a second copy. Construction
        # precedes pipeline binding, so both modules are still present.
        if config.tie_word_embeddings:
            embedding, head = self.backbone.embedding, self.lm_head
            if embedding is None or head is None:
                raise ValueError(
                    "tied embeddings require the embedding and head"
                )
            head.weight = embedding.weight


def entry_points(config: Config) -> Mapping[str, ComponentEntry]:
    """Declare the numerical methods serving ranks may invoke on this model.

    Qwen3 exports the model root as its single component, whatever the
    config. ``forward`` runs on every pipeline stage; token embedding runs
    only on the first stage and logits only on the last.
    """
    return MappingProxyType(
        {
            DEFAULT_COMPONENT: ComponentEntry(
                "",
                (
                    EntryPoint("forward", groups=("tp", "sp", "pp")),
                    EntryPoint(
                        "embed_input_ids", stage="first", groups=("tp",)
                    ),
                    EntryPoint("compute_logits", stage="last", groups=("tp",)),
                ),
            ),
        }
    )
