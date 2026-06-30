"""System-owned interleaved text-decode stepper.

The :class:`InterleavedTextCacheDriver` owns text forward orchestration for
interleaved text+image models: it resolves the per-request paged text KV view
over the **system** KV pool, builds position indexes and attention masks from
system primitives (:func:`build_text_position_indexes`,
:func:`create_block_causal_mask`/:func:`create_causal_mask`), invokes the
model's thin neural forward through the :class:`InterleavedModelOwner` contract,
and hands logits back to the system sampler. The model contributes only compute;
the system owns attention-metadata building and cache residency.

Model-neutral: touches the concrete model only through the duck-typed
``InterleavedModelOwner`` surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

import torch

from ..foundation.errors import model_execution_error
from ..runtime.masks import create_block_causal_mask, create_causal_mask
from ..runtime.paged_text_cache import PagedTextCache
from ..runtime.request_state import append_new_block_ids
from .forward_stream import build_text_position_indexes

if TYPE_CHECKING:
    from ..runtime.kv_pool import PagedKVPool

__all__ = [
    "TextCache",
    "InterleavedModelOwner",
    "InterleavedTextCacheDriver",
    "InterleavedTextStepper",
]


@dataclass
class TextCache:
    """Paged text KV cache and decode state for one interleaved branch."""

    past: Any = None
    block_ids: list[int] = field(default_factory=list)
    t_index: int = -1
    last_logits: torch.Tensor | None = None
    last_token_id: int | None = None


class InterleavedModelOwner(Protocol):
    """Collaborator surface a concrete model must provide to the interleaved
    text/image cache and commit drivers.

    ``InterleavedTextCacheDriver`` and ``GeneratedImageCommitDriver`` own the
    cache-append and commit flow but delegate model- and pool-specific work back
    to the concrete owner through the members declared here. Every member is part
    of the drivers' contract; the concrete owner must define all of them.
    """

    # Collaborator attributes.
    model: Any
    tokenizer: Any
    device: Any
    reqs: dict[int, Any]
    kv_pool: "PagedKVPool | None"
    scratch_pool: "PagedKVPool | None"
    num_layers: int
    eos_id: int
    img_start_id: int
    img_end_id: int

    # Collaborator methods. Model-specific state types (the request state / image
    # state) are kept ``Any`` here: this system component is duck-typed against the
    # concrete model and must not name model-layer types.
    def allocate_scratch_blocks(self, count: int) -> list[int]: ...
    def _state(self, op: dict[str, Any]) -> Any: ...
    def _extend_cache_blocks(self, cache: "TextCache", op: dict[str, Any]) -> None: ...
    def _ensure_host_cache(self, cache: "TextCache") -> None: ...
    def _release_scratch_cache(self, cache: Any) -> None: ...
    def _release_image_state_caches(self, image_state: Any) -> None: ...
    def _prepare_generated_image_for_commit(self, image_state: Any) -> torch.Tensor: ...


class InterleavedTextCacheDriver:
    """Own text prefill/decode cache appends for native decoder models."""

    def __init__(
        self,
        owner: InterleavedModelOwner,
        *,
        request_state_factory: Callable[[], Any],
        image_start_token: str,
    ) -> None:
        self.owner = owner
        self.request_state_factory = request_state_factory
        self.image_start_token = image_start_token

    def state(self, op: dict[str, Any]) -> Any:
        req_id = int(op["req_id"])
        hook = getattr(self.owner, "interleaved_image_state", None)
        if callable(hook):
            return hook(req_id)
        return self.owner.reqs.setdefault(req_id, self.request_state_factory())

    def run_text_logits(self, op: dict[str, Any]) -> torch.Tensor:
        st = self.state(op)
        self.extend_cache_blocks(st.cond, op)
        tokens = op.get("token_ids") or []
        if not tokens:
            eos_id = int(self.owner.eos_id or 0)
            logits = torch.full((1, eos_id + 1), float("-inf"), device=self.owner.device)
            logits[0, eos_id] = 0.0
            return logits

        if st.cond.past is None:
            self.ensure_host_cache(st.cond)
            self.prefix_forward_ids(st.cond, tokens, int(op["pos_range"][0]))
        elif len(tokens) == 1:
            self.append_one(st.cond, int(tokens[0]))
        else:
            self.append_ids(st.cond, tokens)
        return st.cond.last_logits[:, -1, :]

    def extend_cache_blocks(self, cache: TextCache, op: dict[str, Any]) -> None:
        # Host-issued KV block ids belong only to host-KV caches. Scratch caches
        # use worker-local block ids and must not ingest host ids.
        if cache.past is None or getattr(cache.past, "pool", None) is self.owner.kv_pool:
            append_new_block_ids(cache.block_ids, op.get("new_block_ids"))
        if cache.past is not None and getattr(cache.past, "pool", None) is self.owner.kv_pool:
            cache.past.set_blocks(cache.block_ids)

    def ensure_host_cache(self, cache: TextCache) -> None:
        if cache.past is not None:
            return
        if self.owner.kv_pool is None:
            raise model_execution_error("paged KV pool is not initialized")
        cache.past = PagedTextCache(
            self.owner.kv_pool,
            cache.block_ids,
            num_layers=self.owner.num_layers,
        )

    def ensure_scratch_cache(self, cache: TextCache) -> None:
        if cache.past is not None:
            return
        if self.owner.scratch_pool is None:
            raise model_execution_error("scratch KV pool is not initialized")
        cache.past = PagedTextCache(
            self.owner.scratch_pool,
            [],
            num_layers=self.owner.num_layers,
            allocate_blocks=self.owner.allocate_scratch_blocks,
        )

    def prefix_forward_ids(self, cache: TextCache, tokens: list[int], start: int = 0) -> None:
        if cache.past is None:
            raise model_execution_error("text prefix requires an initialized paged cache")
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.owner.device)
        indexes = self.text_indexes(start, len(tokens))
        outputs = self.owner.model.language_model.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask={"full_attention": create_block_causal_mask(indexes[0])},
            past_key_values=cache.past,
            use_cache=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index = int(indexes[0].max().item())
        cache.last_logits = self.owner.model.language_model.lm_head(outputs.last_hidden_state)
        cache.last_token_id = int(tokens[-1])

    def prefix_from_query(self, query: str) -> TextCache:
        cache = TextCache()
        self.ensure_scratch_cache(cache)
        ids, indexes, attn = self.owner.model._build_t2i_text_inputs(self.owner.tokenizer, query)
        outputs = self.owner.model.language_model.model(
            input_ids=ids,
            indexes=indexes,
            attention_mask=attn,
            past_key_values=cache.past,
            use_cache=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index = int(indexes[0].max().item())
        cache.last_logits = self.owner.model.language_model.lm_head(outputs.last_hidden_state)
        cache.last_token_id = int(ids[0, -1].item())
        return cache

    def append_ids(self, cache: TextCache, tokens: list[int]) -> None:
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.owner.device)
        seq_len = input_ids.shape[1]
        embeds = self.owner.model.language_model.get_input_embeddings()(input_ids)
        indexes = self.text_indexes(cache.t_index + 1, seq_len)
        past_len = cache.past.get_seq_length()
        mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=self.owner.device)
        mask[:, :, :, past_len:] = create_causal_mask(seq_len, device=self.owner.device)
        outputs = self.owner.model.language_model(
            inputs_embeds=embeds,
            indexes=indexes,
            attention_mask={"full_attention": mask},
            past_key_values=cache.past,
            use_cache=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index += seq_len
        cache.last_logits = outputs.logits
        cache.last_token_id = int(tokens[-1])

    def append_one(self, cache: TextCache, token_id: int) -> None:
        ids = torch.tensor([token_id], dtype=torch.long, device=self.owner.device)
        indexes = self.text_indexes(cache.t_index + 1, 1)
        outputs = self.owner.model.language_model(
            input_ids=ids.unsqueeze(0),
            indexes=indexes,
            past_key_values=cache.past,
            use_cache=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index += 1
        cache.last_logits = outputs.logits
        cache.last_token_id = int(token_id)

    def ensure_img_start(self, cache: TextCache | None) -> None:
        if cache is None or cache.past is None or cache.last_token_id == self.owner.img_start_id:
            return
        self.append_one(cache, int(self.owner.img_start_id))

    def empty_img_start_prefix(self) -> TextCache:
        query = self.owner.model._build_t2i_query("", append_text=self.image_start_token)
        return self.prefix_from_query(query)

    def text_indexes(self, start: int, seq_len: int) -> torch.Tensor:
        return build_text_position_indexes(int(start), int(seq_len), self.owner.device)


# The design's vocabulary name for this system component.
InterleavedTextStepper = InterleavedTextCacheDriver
