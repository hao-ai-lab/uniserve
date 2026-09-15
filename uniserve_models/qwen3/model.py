"""Qwen3 language model composition and numerical entry points."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from uniserve.model import CausalLM, EntryPoint
from uniserve.nn.linear import VocabParallelHead

from .config import Config
from .transformer import Transformer


class Model(CausalLM):
    """The same loaded Qwen computation for Python and serving callers."""

    def __init__(self, config: Config):
        super().__init__(
            Transformer(config), VocabParallelHead(config.hidden_size, config.vocab_size)
        )
        self.config = config

        # Tied checkpoints store one embedding matrix; the head shares that
        # same Parameter instead of holding a second copy.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight


entry_paths = MappingProxyType({"model": "forward"})


def entry_points(config: Config) -> Mapping[str, tuple[EntryPoint, ...]]:
    """Declare the numerical methods serving ranks may invoke on this model."""
    return MappingProxyType(
        {
            "": (
                EntryPoint("forward", groups=("tp", "sp", "pp")),
                EntryPoint("embed_input_ids", stage="first", groups=("tp",)),
                EntryPoint("compute_logits", stage="last", groups=("tp",)),
            )
        }
    )
