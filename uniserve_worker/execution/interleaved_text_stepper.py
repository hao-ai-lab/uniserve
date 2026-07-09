"""System-owned interleaved text-decode stepper.

The :class:`InterleavedTextCacheDriver` owns text forward orchestration for
interleaved text+image models: it resolves the per-request paged text KV view
over the **system** KV pool, builds position indexes and attention masks from
system primitives (:func:`build_text_position_indexes`,
:func:`create_causal_mask`), invokes the
model's thin neural forward through the :class:`InterleavedModelOwner` contract,
and hands logits back to the system sampler. The model contributes only compute;
the system owns attention-metadata building and cache residency.

Model-neutral: touches the concrete model only through the duck-typed
``InterleavedModelOwner`` surface.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Protocol

import torch

from ..foundation.errors import model_execution_error
from ..nn.logits import forced_eos_logits
from ..runtime.masks import create_causal_mask
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
    "hydrate_cached_prefix_from_op",
    "resolve_op_token_ids",
]


def resolve_op_token_ids(op: Mapping[str, Any]) -> list[int]:
    """Resolve one op's input token ids, synchronizing a pending relay if needed.

    ``last_sampled`` ops from the pipelined decode burst carry the sampled
    token only as a device relay tensor (``op['token_tensor']``); the eager
    path materializes it here (the graph path consumes the tensor directly and
    never synchronizes).
    """

    source = str(op.get("token_source") or "wire")
    tokens = list(op.get("token_ids") or [])
    if source != "last_sampled":
        return tokens
    if len(tokens) != 1:
        raise model_execution_error(
            "decode op requested token_source='last_sampled' but does not have exactly one token"
        )
    relay = op.get("token_tensor")
    if isinstance(relay, torch.Tensor):
        return [int(relay.reshape(-1)[0].item())]
    if int(tokens[0]) < 0:
        raise model_execution_error(
            "decode op requested token_source='last_sampled' but carries neither "
            "a relay tensor nor a resolved token id"
        )
    return [int(tokens[0])]


def hydrate_cached_prefix_from_op(cache: Any, op: Mapping[str, Any]) -> None:
    """Adopt host prefix-cache hits into a fresh local paged text cache.

    Prefix-cache reuse crosses the worker boundary as block IDs plus an op
    ``pos_range``.  A newly-created model cache has the right block table but a
    zero logical length, so its first suffix prefill/decode must advance the
    local length to the cached prefix before writing new K/V.
    """

    past = getattr(cache, "past", None)
    if past is None:
        return
    pos = op.get("pos_range")
    if not isinstance(pos, Sequence) or len(pos) < 1:
        return
    start = int(pos[0])
    if start <= 0 or start <= int(getattr(past, "length", 0)):
        return
    if int(getattr(past, "length", 0)) != 0 or int(getattr(cache, "t_index", -1)) >= 0:
        return
    past.ensure_capacity(start)
    past.length = start
    cache.t_index = start - 1


class TextCache:
    """Paged text KV cache and decode state for one interleaved branch.

    ``last_token_id`` may be committed either as a resolved CPU int or as a
    pending device relay tensor (the pipelined decode burst commits the tensor
    so the graph replay loop never synchronizes on the token value). The
    property materializes a pending tensor on first read; readers only consult
    it at modality transitions, long after the producing kernel has finished.
    """

    def __init__(self) -> None:
        self.past: Any = None
        self.block_ids: list[int] = []
        self.t_index: int = -1
        self.last_logits: torch.Tensor | None = None
        self._last_token_id: int | None = None
        self._last_token_tensor: torch.Tensor | None = None

    @property
    def last_token_id(self) -> int | None:
        if self._last_token_tensor is not None:
            self._last_token_id = int(self._last_token_tensor.reshape(-1)[0].item())
            self._last_token_tensor = None
        return self._last_token_id

    @last_token_id.setter
    def last_token_id(self, value: int | None) -> None:
        self._last_token_tensor = None
        self._last_token_id = None if value is None else int(value)

    def set_last_token_tensor(self, tensor: torch.Tensor) -> None:
        self._last_token_tensor = tensor


class InterleavedModelOwner(Protocol):
    """Collaborator surface a concrete model must provide to the interleaved
    text/image cache and commit drivers.

    ``InterleavedTextCacheDriver`` owns the cache-append flow (and
    ``interleaved_image_commit.GeneratedImageCommitOwner`` extends this surface
    for the commit driver) but delegates model- and pool-specific work back
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
    residency: Any
    num_layers: int
    eos_id: int
    img_start_id: int
    img_end_id: int

    # Collaborator methods. Model-specific state types (the request state / image
    # state) are kept ``Any`` here: this system component is duck-typed against the
    # concrete model and must not name model-layer types.
    def interleaved_text_forward(self, **kwargs: Any) -> Any: ...
    def interleaved_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def interleaved_text_inputs(self, query: str) -> tuple[torch.Tensor, torch.Tensor, Any]: ...
    def interleaved_empty_image_start_query(self, image_start_token: str) -> str: ...
    def interleaved_image_patch_size(self) -> int: ...
    def interleaved_image_downsample_ratio(self) -> float: ...
    def interleaved_image_features(
        self,
        image_input: torch.Tensor,
        *,
        grid_hw: torch.Tensor,
        gen_model: bool = False,
    ) -> torch.Tensor: ...
    def _state(self, op: dict[str, Any]) -> Any: ...
    def _extend_cache_blocks(self, cache: "TextCache", op: dict[str, Any]) -> None: ...
    def _ensure_host_cache(self, cache: "TextCache") -> None: ...
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
        # System-owned decode CUDA graph adapter, constructed lazily on first use
        # so CPU/eager and non-CUDA integrations never import the graph stack.
        self._decode_graph_runner: Any | None = None

    def state(self, op: dict[str, Any]) -> Any:
        req_id = int(op["req_id"])
        hook = getattr(self.owner, "interleaved_image_state", None)
        if callable(hook):
            return hook(req_id)
        return self.owner.reqs.setdefault(req_id, self.request_state_factory())

    def run_text_logits_batch(self, ops: Sequence[Mapping[str, Any]]) -> list[torch.Tensor]:
        """Return one logits row per op, graphing the eligible one-token decode batch.

        The system-owned decode graph adapter owns capture/replay for the eligible
        one-token host-KV decode rows; a ``None`` result (ineligible batch, graph
        miss, or non-fatal fallback) hands the whole batch to the scalar eager
        path so no non-decode / ineligible text semantics ever change.
        """

        op_list = [dict(op) for op in ops]
        if not op_list:
            return []
        graphed = self.try_run_decode_graph_logits_batch(op_list)
        if graphed is not None:
            return graphed
        return [self._run_text_logits_one(op) for op in op_list]

    def try_run_decode_graph_logits_batch(
        self,
        ops: Sequence[Mapping[str, Any]],
    ) -> list[torch.Tensor] | None:
        """Return graph-produced logits for eligible one-token decode rows.

        Unlike :meth:`run_text_logits_batch`, this method does not fall back to
        eager execution. Callers that must account for graph misses explicitly can
        use this seam and keep the existing eager path authoritative.
        """

        op_list = [dict(op) for op in ops]
        if not op_list:
            return []
        return self._decode_graph().maybe_run_batch(self, op_list)

    def run_text_logits(self, op: dict[str, Any]) -> torch.Tensor:
        return self._run_text_logits_one(dict(op))

    def _run_text_logits_one(self, op: dict[str, Any]) -> torch.Tensor:
        st = self.state(op)
        self.extend_cache_blocks(st.cond, op)
        tokens = resolve_op_token_ids(op)
        if not tokens:
            return forced_eos_logits(
                int(self.owner.eos_id or 0), device=self.owner.device, batch_shape=(1,)
            )

        if st.cond.past is None:
            self.ensure_host_cache(st.cond)
            hydrate_cached_prefix_from_op(st.cond, op)
            self.prefix_forward_ids(st.cond, tokens, int(op["pos_range"][0]))
        elif len(tokens) == 1:
            hydrate_cached_prefix_from_op(st.cond, op)
            self.append_one(st.cond, int(tokens[0]))
        else:
            hydrate_cached_prefix_from_op(st.cond, op)
            self.append_ids(st.cond, tokens)
        return st.cond.last_logits[:, -1, :]

    def _decode_graph(self) -> Any:
        runner = self._decode_graph_runner
        if runner is None:
            from .interleaved_text_graph_runner import InterleavedTextDecodeGraphRunner

            runner = InterleavedTextDecodeGraphRunner()
            self._decode_graph_runner = runner
        return runner

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
        allocate_blocks = self.owner.residency.require_allocator_for_pool(
            self.owner.scratch_pool,
            label="scratch KV pool",
        )
        cache.past = PagedTextCache(
            self.owner.scratch_pool,
            [],
            num_layers=self.owner.num_layers,
            allocate_blocks=allocate_blocks,
        )

    def prefix_forward_ids(self, cache: TextCache, tokens: list[int], start: int = 0) -> None:
        if cache.past is None:
            raise model_execution_error("text prefix requires an initialized paged cache")
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.owner.device)
        indexes = self.text_indexes(start, len(tokens))
        seq_len = input_ids.shape[1]
        past_len = cache.past.get_seq_length()
        mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=self.owner.device)
        mask[:, :, :, past_len:] = create_causal_mask(seq_len, device=self.owner.device)
        outputs = self.owner.interleaved_text_forward(
            input_ids=input_ids,
            indexes=indexes,
            text_only_rope=True,
            attention_mask={"full_attention": mask},
            past_key_values=cache.past,
            use_cache=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index = int(indexes[0].max().item())
        cache.last_logits = outputs.logits
        cache.last_token_id = int(tokens[-1])

    def prefix_from_query(self, query: str) -> TextCache:
        cache = TextCache()
        self.ensure_scratch_cache(cache)
        ids, indexes, attn = self.owner.interleaved_text_inputs(query)
        outputs = self.owner.interleaved_text_forward(
            input_ids=ids,
            indexes=indexes,
            attention_mask=attn,
            past_key_values=cache.past,
            use_cache=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index = int(indexes[0].max().item())
        cache.last_logits = outputs.logits
        cache.last_token_id = int(ids[0, -1].item())
        return cache

    def append_ids(self, cache: TextCache, tokens: list[int]) -> None:
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.owner.device)
        seq_len = input_ids.shape[1]
        embeds = self.owner.interleaved_text_embeddings(input_ids)
        indexes = self.text_indexes(cache.t_index + 1, seq_len)
        past_len = cache.past.get_seq_length()
        mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=self.owner.device)
        mask[:, :, :, past_len:] = create_causal_mask(seq_len, device=self.owner.device)
        outputs = self.owner.interleaved_text_forward(
            inputs_embeds=embeds,
            indexes=indexes,
            text_only_rope=True,
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
        outputs = self.owner.interleaved_text_forward(
            input_ids=ids.unsqueeze(0),
            indexes=indexes,
            past_key_values=cache.past,
            use_cache=True,
            text_only_rope=True,
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
        query = self.owner.interleaved_empty_image_start_query(self.image_start_token)
        return self.prefix_from_query(query)

    def text_indexes(self, start: int, seq_len: int) -> torch.Tensor:
        return build_text_position_indexes(int(start), int(seq_len), self.owner.device)


# The design's vocabulary name for this system component.
InterleavedTextStepper = InterleavedTextCacheDriver
