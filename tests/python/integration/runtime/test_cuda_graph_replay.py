"""CUDA-graph capture/replay behavior for the step and span implementations.

Covers the observable contract of ``execution.graph.capture``, ``step``,
``span``, and ``executor``:

* batch-size bucketing rounds a request up to the nearest configured warmup
  bucket (and prefill token bucketing likewise);
* the decode input copy zero-pads a short batch up to the captured bucket and
  extends the ``cache_seqlens_cpu`` / ``kv_seqlens_cpu`` host mirrors;
* ``_share_input_buffer`` pools by ``(name, dtype, device)`` and slices the
  largest captured buffer for smaller buckets;
* eager-vs-graph replay produces identical logits and the returned tensor is
  sliced back to the unpadded request count, with CUDA-graph stats accounting
  the unpadded vs padded token split.

The capture/replay tests need a real CUDA device (available here, so they run).
The bucketing tests are pure-Python and run anywhere.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from uniserve_worker.contracts.attention_plan import (
    GraphBinding,
    PagedDecodePlan,
    PagedVarlenPlan,
)
from uniserve_worker.contracts.forward_batch import ForwardBatch
from uniserve_worker.contracts.forward_context import (
    ForwardContext,
    use_forward_context,
)
from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.contracts.forward_stats import ForwardStats
from uniserve_worker.execution.graph import Executor, span, step
from uniserve_worker.execution.graph.capture import (
    _reset_for_testing,
    _share_input_buffer,
    _share_step_input,
)
from uniserve_worker.execution.graph.executor import _padded_max_tokens
from uniserve_worker.execution.graph.span import Runner as Span
from uniserve_worker.execution.graph.span import copy_inputs as copy_span
from uniserve_worker.execution.graph.span import make_state as make_span
from uniserve_worker.execution.graph.step import Inputs as StepInputs
from uniserve_worker.execution.graph.step import Runner as Step
from uniserve_worker.execution.graph.step import (
    copy_host,
    dense_replacements,
    resolve_backend,
    resolve_prepare,
)
from uniserve_worker.execution.graph.step import copy_inputs as copy_step
from uniserve_worker.execution.graph.step import make_state as make_step
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import BatchedPagedRequestCache

pytestmark = pytest.mark.integration

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="decode CUDA graph capture/replay requires a CUDA device"
)

requires_decode_graph_backend = pytest.mark.skipif(
    resolve_backend(None) is None,
    reason="step replay requires a graph-aware paged decode backend",
)

_VOCAB = 32
_HIDDEN = 16


def _kv_pool(
    device: torch.device,
    *,
    num_blocks: int = 16,
    block_size: int = 4,
    head_dim: int = 8,
) -> PagedKVPool:
    return PagedKVPool(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=head_dim,
        device=device,
        dtype=torch.bfloat16,
    )


def _decode_plan(
    pool: PagedKVPool,
    *,
    block_ids_by_row: list[list[int]],
    cache_seqlens_cpu: tuple[int, ...],
    kv_seqlens_cpu: tuple[int, ...],
    device: torch.device,
) -> PagedDecodePlan:
    batch = len(block_ids_by_row)
    cache = BatchedPagedRequestCache(pool, block_ids_by_row, list(cache_seqlens_cpu))
    block_table = cache.block_table(device=device)
    cache_seqlens = cache.cache_seqlens(device=device)
    return PagedDecodePlan(
        residency_cache=cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=cache_seqlens_cpu,
        kv_seqlens=torch.tensor(kv_seqlens_cpu, dtype=torch.int32, device=device),
        query_lens=torch.ones(batch, dtype=torch.int32, device=device),
        query_lens_cpu=tuple(1 for _ in range(batch)),
        kv_seqlens_cpu=kv_seqlens_cpu,
        decode_page_ids=torch.zeros(batch, dtype=torch.long, device=device),
        decode_page_offsets=torch.zeros(batch, dtype=torch.long, device=device),
    )


@requires_cuda
@pytest.mark.gpu
def test_batched_request_cache_append_plan_can_be_invalidated_for_dynamic_lengths():
    device = torch.device("cuda")
    pool = _kv_pool(device, block_size=8)
    cache = BatchedPagedRequestCache(pool, [[0]], [2])
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([2], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    first_k = torch.full((1, pool.n_kv, pool.head_dim), 10.0, dtype=pool.dtype, device=device)
    first_v = torch.full((1, pool.n_kv, pool.head_dim), -10.0, dtype=pool.dtype, device=device)
    second_k = torch.full((1, pool.n_kv, pool.head_dim), 20.0, dtype=pool.dtype, device=device)
    second_v = torch.full((1, pool.n_kv, pool.head_dim), -20.0, dtype=pool.dtype, device=device)

    cache.append_varlen(
        0,
        first_k,
        first_v,
        [1],
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    cache_seqlens.fill_(3)
    cache.invalidate_append_plan()
    cache.append_varlen(
        0,
        second_k,
        second_v,
        [1],
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    torch.cuda.synchronize()

    k_read, v_read = pool.read(0, [0], start=2, length=2)
    assert k_read is not None
    assert v_read is not None
    torch.testing.assert_close(k_read[0], first_k[0])
    torch.testing.assert_close(v_read[0], first_v[0])
    torch.testing.assert_close(k_read[1], second_k[0])
    torch.testing.assert_close(v_read[1], second_v[0])


def _prefill_plan(
    pool: PagedKVPool,
    *,
    block_ids_by_row: list[list[int]],
    cache_seqlens_cpu: tuple[int, ...],
    query_lens_cpu: tuple[int, ...],
    device: torch.device,
    max_context_len: int = 0,
) -> PagedVarlenPlan:
    cache = BatchedPagedRequestCache(pool, block_ids_by_row, list(cache_seqlens_cpu))
    query_lens = torch.tensor(query_lens_cpu, dtype=torch.int32, device=device)
    kv_lens = torch.tensor(
        [base + query for base, query in zip(cache_seqlens_cpu, query_lens_cpu, strict=True)],
        dtype=torch.int32,
        device=device,
    )
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    return PagedVarlenPlan(
        residency_cache=cache,
        block_table=cache.block_table(device=device),
        cache_seqlens=cache.cache_seqlens(device=device),
        cache_seqlens_cpu=cache_seqlens_cpu,
        query_lens=query_lens,
        query_lens_cpu=query_lens_cpu,
        kv_seqlens=kv_lens,
        kv_seqlens_cpu=tuple(int(x.item()) for x in kv_lens),
        cu_seqlens_q=torch.cat([zero, torch.cumsum(query_lens, dim=0).to(torch.int32)]),
        cu_seqlens_k=torch.cat([zero, torch.cumsum(kv_lens, dim=0).to(torch.int32)]),
        max_seqlen_q=max(query_lens_cpu, default=0),
        max_seqlen_k=max((b + q for b, q in zip(cache_seqlens_cpu, query_lens_cpu, strict=True)), default=0),
        max_context_len=int(max_context_len),
        mode=ForwardMode.EXTEND,
    )


class _LinearLogitsModel:
    """A CUDA-graph-safe one-token model: embed by id, add position, project.

    Static shapes, no host syncs or data-dependent control flow, so it is safe to
    capture. The same closure computes the eager reference so the test can assert
    capture/replay output equality exactly.
    """

    def __init__(self, device: torch.device, *, seed: int) -> None:
        gen = torch.Generator(device=device).manual_seed(seed)
        self.emb = torch.randn(_VOCAB, _HIDDEN, device=device, generator=gen)
        self.proj = torch.randn(_VOCAB, _HIDDEN, device=device, generator=gen)

    def logits(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.emb[input_ids.reshape(-1)] + positions.reshape(-1, 1).float()
        return x @ self.proj.t()

    def forward_text(self, fb: ForwardBatch) -> torch.Tensor:
        return self.logits(fb.input_ids, fb.positions)

    def query_geometry(self) -> tuple[int, float, torch.dtype]:
        # Plausible query geometry matching the test pool; the fake forward
        # never runs attention, but declaring it keeps the model eligible for
        # the decode graph and exercises the real FlashInfer re-plan hook.
        return 1, 1.0, torch.bfloat16


# --------------------------------------------------------------------------- #
# Pure-Python bucketing (no CUDA needed)
# --------------------------------------------------------------------------- #


def test_decode_bucket_rounds_up_to_nearest_configured_bucket():
    runner = Step(name="t", default_warmup_batch_sizes=(1, 2, 4, 8, 16))

    assert runner.bucket_batch_size(3) == 4
    assert runner.bucket_batch_size(5) == 8


def test_decode_bucket_keeps_exact_batch_when_it_equals_a_bucket():
    runner = Step(name="t", default_warmup_batch_sizes=(1, 2, 4, 8, 16))

    assert runner.bucket_batch_size(8) == 8


def test_decode_bucket_falls_back_to_raw_batch_above_largest_bucket():
    runner = Step(name="t", default_warmup_batch_sizes=(1, 2, 4, 8, 16))

    # No configured bucket is >= 20, so the runner uses the request size verbatim.
    assert runner.bucket_batch_size(20) == 20


def test_decode_bucket_skips_disabled_bucket_and_picks_next_larger():
    runner = Step(name="t", default_warmup_batch_sizes=(2, 4, 8))
    runner.disabled.add(4)

    # batch 3 would normally round to 4, but 4 is disabled -> next bucket is 8.
    assert runner.bucket_batch_size(3) == 8


def test_prefill_token_bucket_rounds_up_to_nearest_configured_bucket():
    runner = Span(name="t", default_warmup_token_buckets=(4, 8, 16, 32))

    assert runner.bucket_num_tokens(5) == 8
    assert runner.bucket_num_tokens(16) == 16


def test_prefill_token_bucket_falls_back_to_raw_count_above_largest_bucket():
    runner = Span(name="t", default_warmup_token_buckets=(4, 8, 16, 32))

    assert runner.bucket_num_tokens(33) == 33


def test_prefill_batch_bucket_rounds_up_to_nearest_configured_bucket():
    runner = Span(name="t", default_warmup_batch_sizes=(1, 2, 4, 8))

    assert runner.bucket_batch_size(3) == 4
    assert runner.bucket_batch_size(6) == 8


def test_prefill_batch_bucket_uses_largest_viable_bucket_for_token_bucket():
    runner = Span(name="t", default_warmup_batch_sizes=(1, 2, 4, 8))

    assert runner.bucket_batch_size(1, num_tokens=4) == 8
    assert runner.bucket_batch_size(3, num_tokens=12) == 8


def test_prefill_warmup_uses_declared_batch_capacity():
    runner = Span(
        name="t",
        default_warmup_token_buckets=(4, 8),
        default_warmup_batch_sizes=(1, 2, 4, 8),
    )

    assert runner.warmup_capture_buckets() == ((8, 8), (4, 8))


def test_prefill_kv_bucket_uses_context_capacity_when_available():
    runner = Span(name="t", default_warmup_token_buckets=(4, 8, 16, 32))

    assert runner.bucket_kv_tokens(24, max_context_len=128) == 128
    assert runner.bucket_kv_tokens(160, max_context_len=128) == 160


def test_decode_graph_backend_resolver_accepts_direct_fa4_paged_backend(monkeypatch):
    from uniserve_worker.backends import attention as attention_registry

    backend = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(available=True, paged_kv=True),
        forward_paged=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        attention_registry,
        "normalize_attention_backend_name",
        lambda name: "fa4_cute" if name == "fa4_cute" else str(name or "auto"),
    )
    monkeypatch.setattr(attention_registry, "has_attention_backend", lambda name: name == "fa4_cute")
    monkeypatch.setattr(attention_registry, "get_attention_backend", lambda name: backend)

    assert resolve_backend("fa4_cute") is backend


def test_decode_graph_prepare_accepts_direct_backend_without_plan_hook(monkeypatch):

    backend = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(available=True, paged_kv=True, paged_block_size_multiple=1),
        forward_paged=lambda *args, **kwargs: None,
    )
    before_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(step, "resolve_backend", lambda _name: backend)

    owner = SimpleNamespace(query_geometry=lambda: (4, 0.125, torch.bfloat16))
    kv_pool = SimpleNamespace(
        block_size=64,
        n_kv=2,
        head_dim=128,
        k=torch.empty(1, dtype=torch.bfloat16),
    )
    prepare = resolve_prepare(
        owner=owner,
        kv_pool=kv_pool,
        num_blocks=16,
        attention_preference="fa4_cute",
        before=lambda state, ctx: before_calls.append((state, ctx)),
    )

    assert prepare is not None
    state = SimpleNamespace(batch_size=8, plan=object(), graph_binding=GraphBinding())
    ctx = SimpleNamespace()
    prepare(state, ctx)
    assert before_calls == [(state, ctx)]


def test_decode_graph_prepare_rejects_direct_backend_page_size_mismatch(monkeypatch):

    backend = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(available=True, paged_kv=True, paged_block_size_multiple=256),
        forward_paged=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(step, "resolve_backend", lambda _name: backend)

    owner = SimpleNamespace(query_geometry=lambda: (4, 0.125, torch.bfloat16))
    kv_pool = SimpleNamespace(block_size=64, n_kv=2, head_dim=128, k=torch.empty(1))

    assert resolve_prepare(
        owner=owner,
        kv_pool=kv_pool,
        num_blocks=16,
        attention_preference="fa4_cute",
    ) is None


def test_prefill_graph_warmup_mode_does_not_live_capture_missing_shape():
    runner = Span(name="t", default_enabled=True, default_warmup=True)
    stats = ForwardStats()

    out = runner.maybe_run(
        kv_pool=None,
        num_blocks=0,
        num_tokens=8,
        max_kv_tokens=8,
        batch_size=1,
        input_ids=torch.zeros(8, dtype=torch.long),
        positions=torch.arange(8, dtype=torch.long),
        attention_plan=None,
        last_token_indices=torch.tensor([7], dtype=torch.long),
        raw_num_tokens=8,
        ctx=ForwardContext(stats=stats),
        forward_fn=lambda state: pytest.fail("missing warmup shape must not capture during serving"),
    )

    assert out is None
    assert stats.cuda_graph_misses == 1
    assert stats.cuda_graph_captures == 0


def test_flashinfer_prefill_graph_prepare_replans_live_side_tables(monkeypatch):
    from uniserve_worker.backends.attention import flashinfer as fi

    class RecordingPrefillWrapper:
        def __init__(self) -> None:
            self.calls = []
            self._plan_info = None

        def plan(self, *args, **kwargs) -> None:
            self.calls.append((args, kwargs))
            self._plan_info = object()

    monkeypatch.setattr(fi, "_BatchPrefillWithPagedKVCacheWrapper", object)
    backend = fi.FlashInferAttentionBackend()
    monkeypatch.setattr(
        backend,
        "_workspace",
        lambda device: torch.empty(1, dtype=torch.uint8, device=device),
    )
    wrapper = RecordingPrefillWrapper()
    wrapper_key = fi.WrapperKey("prefill", "cpu", "fa2", scope=1)
    binding = GraphBinding()
    plan = SimpleNamespace(
        block_table=torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 7, 11], dtype=torch.int32),
    )
    backend._prefill_wrappers[wrapper_key] = wrapper
    backend._binding_prefill_graph_wrappers[id(binding)] = (wrapper_key, None)
    stats = ForwardStats()

    def prepare() -> None:
        backend.prepare_paged_prefill_cuda_graph(
            binding,
            plan,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=8,
            page_size=4,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            causal=False,
            scale=0.125,
        )

    with use_forward_context(ForwardContext(stats=stats)):
        prepare()
        plan.cu_seqlens_k.copy_(torch.tensor([0, 8, 14], dtype=torch.int32))
        prepare()

    assert len(wrapper.calls) == 2
    first_args, first_kwargs = wrapper.calls[0]
    second_args, second_kwargs = wrapper.calls[1]
    assert first_args[0].tolist() == [0, 2, 5]
    assert first_args[1].tolist() == [0, 2, 3]
    assert first_args[2].tolist() == [0, 1, 3]
    assert first_args[3].tolist() == [3, 4]
    assert first_kwargs["seq_lens"].tolist() == [7, 4]
    assert first_kwargs["seq_lens_q"].tolist() == [2, 3]
    assert first_kwargs["block_tables"] is plan.block_table
    assert first_kwargs["q_data_type"] == torch.bfloat16
    assert first_kwargs["kv_data_type"] == torch.bfloat16
    assert first_kwargs["o_data_type"] == torch.bfloat16
    assert first_kwargs["causal"] is False
    assert first_kwargs["sm_scale"] == 0.125
    assert second_args[1].tolist() == [0, 2, 4]
    assert second_args[2].tolist() == [0, 1, 3, 4]
    assert second_args[3].tolist() == [4, 2]
    assert second_kwargs["seq_lens"].tolist() == [8, 6]
    assert second_kwargs["seq_lens_q"].tolist() == [2, 3]
    assert backend.paged_prefill_graph_wrapper_planned(binding)
    assert stats.flashinfer_prefill_plan_calls == 2
    assert stats.flashinfer_prefill_plan_rows == 4
    assert stats.flashinfer_prefill_plan_indices == 7


def test_flashinfer_prefill_graph_binding_owns_stable_graph_buffers(monkeypatch):
    from uniserve_worker.backends.attention import flashinfer as fi

    class RecordingPrefillWrapper:
        def __init__(self, workspace, layout, **kwargs) -> None:
            self.workspace = workspace
            self.layout = layout
            self.kwargs = kwargs

    monkeypatch.setattr(fi, "_BatchPrefillWithPagedKVCacheWrapper", RecordingPrefillWrapper)
    backend = fi.FlashInferAttentionBackend()
    monkeypatch.setattr(
        backend,
        "_workspace",
        lambda device: torch.empty(1, dtype=torch.uint8, device=device),
    )
    binding = GraphBinding()
    plan = SimpleNamespace(
        block_table=torch.zeros((3, 7), dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 2, 5, 9], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 7, 11, 20], dtype=torch.int32),
    )

    backend.bind_paged_prefill_graph_wrapper(binding, plan, device="cpu")

    wrapper_key, wrapper = backend._prefill_graph_wrapper_for_binding(binding)
    workspace = backend._prefill_plan_workspaces[wrapper_key]
    assert wrapper_key.batch_size == 3
    assert wrapper_key.max_indices == 21
    assert wrapper.kwargs["use_cuda_graph"] is True
    assert wrapper.kwargs["qo_indptr_buf"] is workspace.qo_indptr
    assert wrapper.kwargs["paged_kv_indptr_buf"] is workspace.kv_indptr
    assert wrapper.kwargs["paged_kv_indices_buf"] is workspace.indices
    assert wrapper.kwargs["paged_kv_last_page_len_buf"] is workspace.last_page_len


def test_prefill_graph_prepare_resolver_uses_owner_geometry(monkeypatch):

    class Backend:
        def __init__(self) -> None:
            self.calls = []
            self.binds = []

        def bind_paged_prefill_graph_wrapper(self, binding, plan, *, device) -> None:
            self.binds.append((binding, plan, torch.device(device)))

        def release_paged_prefill_graph_wrapper(self, binding) -> None:
            del binding

        def prepare_paged_prefill_cuda_graph(self, binding, plan, **kwargs) -> None:
            self.calls.append((binding, plan, kwargs))

    backend = Backend()
    binding = GraphBinding()
    plan = object()
    owner = SimpleNamespace(query_geometry=lambda: (4, 0.125, torch.bfloat16))
    kv_pool = SimpleNamespace(
        n_kv=2,
        head_dim=8,
        block_size=4,
        k=torch.empty(1, dtype=torch.float16),
    )
    monkeypatch.setattr(span, "_resolve_backend", lambda ctx: backend)

    prepare = span.resolve_prepare(
        owner=owner,
        kv_pool=kv_pool,
        attention_preference="flashinfer",
    )

    assert prepare is not None
    state = SimpleNamespace(
        graph_binding=binding,
        plan=plan,
        input_ids=torch.empty(1),
        release_backend=None,
    )
    prepare(state, SimpleNamespace())
    assert backend.binds == [(binding, plan, torch.device("cpu"))]
    assert callable(state.release_backend)
    assert backend.calls == [
        (
            binding,
            plan,
            {
                "num_q_heads": 4,
                "num_kv_heads": 2,
                "head_dim": 8,
                "page_size": 4,
                "q_dtype": torch.bfloat16,
                "kv_dtype": torch.float16,
                "causal": True,
                "scale": 0.125,
            },
        )
    ]


def test_prefill_graph_prepare_resolver_accepts_direct_graph_safe_backend(monkeypatch):

    class Backend:
        def capabilities(self):
            return SimpleNamespace(paged_varlen_cuda_graph=True)

    calls = []
    monkeypatch.setattr(span, "_resolve_backend", lambda ctx: Backend())

    prepare = span.resolve_prepare(
        owner=object(),
        kv_pool=SimpleNamespace(),
        attention_preference="trtllm_mha",
        before=lambda state, ctx: calls.append((state, ctx)),
    )

    assert prepare is not None
    state = SimpleNamespace()
    ctx = SimpleNamespace()
    prepare(state, ctx)
    assert calls == [(state, ctx)]


def test_graph_warmup_context_uses_configured_attention_backend(monkeypatch):
    seen: list[tuple[str, str | None]] = []

    def capture_decode(self, **kwargs):
        seen.append(("decode", kwargs["ctx"].attention_preference))

    def capture_prefill(self, **kwargs):
        seen.append(("prefill", kwargs["ctx"].attention_preference))

    monkeypatch.setattr(Step, "_warmup_capture_buckets", capture_decode)
    monkeypatch.setattr(Span, "_warmup_capture_buckets", capture_prefill)

    Step(name="t", default_enabled=True, default_warmup=True).warmup(
        kv_pool=SimpleNamespace(),
        num_blocks=1,
        device=torch.device("cpu"),
        attention_preference="flashinfer",
        forward_fn=lambda state: state,
    )
    Span(name="t", default_enabled=True, default_warmup=True).warmup(
        kv_pool=SimpleNamespace(),
        num_blocks=1,
        block_size=1,
        device=torch.device("cpu"),
        attention_preference="flashinfer",
        forward_fn=lambda state: state,
    )

    assert seen == [("decode", "flashinfer"), ("prefill", "flashinfer")]


def test_text_graph_runner_routes_cached_prefix_prefill_to_prefill_runner():
    class PrefillStub:
        def enabled(self) -> bool:
            return True

        def bucket_kv_tokens(self, max_kv_tokens: int, *, max_context_len: int = 0) -> int:
            return int(max_context_len or max_kv_tokens)

        def can_use(self, *args, **kwargs) -> bool:
            return True

        def maybe_run(self, **kwargs):
            self.kwargs = kwargs
            return torch.ones(1, _VOCAB)

    prefill = PrefillStub()

    runner = Executor.__new__(Executor)
    runner._span = prefill
    runner.kv_pool = None
    runner.num_blocks = 0
    runner.max_context_len = 128
    runner.attention_preference = None
    runner._span_prepare = lambda *args, **kwargs: lambda state, ctx: None
    plan = PagedVarlenPlan(
        residency_cache=None,
        block_table=torch.empty((1, 0), dtype=torch.int32),
        cache_seqlens=None,
        cache_seqlens_cpu=(64,),
        query_lens=None,
        query_lens_cpu=(8,),
        kv_seqlens=None,
        kv_seqlens_cpu=(72,),
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 72], dtype=torch.int32),
        max_seqlen_q=8,
        max_seqlen_k=72,
        mode=ForwardMode.EXTEND,
    )
    fb = SimpleNamespace(
        batch_size=1,
        last_token_indices=torch.tensor([7], dtype=torch.long),
        spec_token_ids=[],
        num_token_non_padded=8,
    )

    out = runner._maybe_span(
        None,
        torch.zeros(8, dtype=torch.long),
        torch.arange(8, dtype=torch.long),
        fb,
        plan,
        ForwardContext(stats=ForwardStats()),
    )

    assert out is not None
    assert prefill.kwargs["max_kv_tokens"] == 128
    assert prefill.kwargs["batch_size"] == 1


@requires_cuda
def test_text_graph_runner_routes_mixed_extend_decode_to_prefill_runner():
    """A mixed extend+decode group replays through the prefill graph buckets."""

    runner = Executor.__new__(Executor)
    calls = {}
    runner._maybe_step = lambda *a, **k: calls.setdefault("decode", True)
    runner._maybe_span = lambda *a, **k: calls.setdefault("prefill", True) or torch.ones(2, _VOCAB)
    plan = SimpleNamespace(residency_cache=object.__new__(BatchedPagedRequestCache))
    fb = SimpleNamespace(forward_mode=ForwardMode.MIXED, attn_plan=plan)
    out = runner.maybe_run(
        None,
        torch.zeros(9, dtype=torch.long, device="cuda"),
        torch.arange(9, dtype=torch.long, device="cuda"),
        fb,
        ForwardContext(stats=ForwardStats()),
    )
    assert out is not None
    assert "prefill" in calls and "decode" not in calls


def test_prefill_graph_padding_preserves_row_order():
    """Capacity padding leaves real request ordering unchanged."""

    from uniserve_worker.contracts.batches import TextBatch

    class PrefillStub:
        def enabled(self) -> bool:
            return True

        def bucket_num_tokens(self, num_tokens: int) -> int:
            return 80

    runner = Executor.__new__(Executor)
    runner._span = PrefillStub()
    runner.block_size = 64
    ops = (
        {"req_id": 1, "kind": "decode_und", "token_ids": [7], "pos_range": [10, 11]},
        {"req_id": 2, "kind": "prefill_und", "token_ids": list(range(64)), "pos_range": [0, 64]},
    )
    text = TextBatch.from_ops(
        ForwardMode.MIXED,
        ops,
        op_modes=(ForwardMode.DECODE, ForwardMode.EXTEND),
        allow_mixed_text=True,
    )
    assert runner.reorder_mixed_for_padding(text) is text


def test_padded_prefill_tokens_accepts_mixed_mode():
    """Token-bucket padding applies to mixed groups so they hit captured buckets."""

    class PrefillStub:
        def enabled(self) -> bool:
            return True

        def bucket_num_tokens(self, num_tokens: int) -> int:
            return 8

        def padding_batch_size(self, batch_size: int, *, num_tokens: int) -> int:
            del batch_size, num_tokens
            return 4

    runner = Executor.__new__(Executor)
    runner._span = PrefillStub()
    runner.block_size = 64
    text = SimpleNamespace(
        mode=ForwardMode.MIXED,
        spec_token_ids=[(), ()],
        token_ids=[(5,), (1, 2, 3)],
        pos_ranges=[(64, 65), (0, 3)],
    )
    assert runner.padded_num_tokens(text, attention_preference=None) == 8


def test_prefill_padding_does_not_depend_on_request_kv_tail_capacity():
    """Graph padding uses sink pages instead of extending request residency."""

    class PrefillStub:
        def enabled(self) -> bool:
            return True

        def bucket_num_tokens(self, num_tokens: int) -> int:
            return 12

        def padding_batch_size(self, batch_size: int, *, num_tokens: int) -> int:
            del batch_size, num_tokens
            return 4

    runner = Executor.__new__(Executor)
    runner._span = PrefillStub()
    runner.block_size = 4
    text = SimpleNamespace(
        mode=ForwardMode.EXTEND,
        spec_token_ids=[(), ()],
        token_ids=[(1, 2, 3, 4), (5, 6, 7)],
        pos_ranges=[(0, 4), (0, 3)],
    )

    assert runner.padded_num_tokens(text, attention_preference=None) == 12


@requires_cuda
def test_prefill_graph_inputs_preserve_cached_prefix_lengths():
    device = torch.device("cuda")
    pool = _kv_pool(device, num_blocks=16, block_size=4)
    cache = BatchedPagedRequestCache(pool, [[0, 1, 2, 3], [4, 5, 6, 7]], [5, 7])
    query_lens = torch.tensor([2, 3], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([7, 10], dtype=torch.int32, device=device)
    plan = PagedVarlenPlan(
        residency_cache=cache,
        block_table=cache.block_table(device=device),
        cache_seqlens=cache.cache_seqlens(device=device),
        cache_seqlens_cpu=(5, 7),
        query_lens=query_lens,
        query_lens_cpu=(2, 3),
        kv_seqlens=kv_lens,
        kv_seqlens_cpu=(7, 10),
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, 7, 17], dtype=torch.int32, device=device),
        max_seqlen_q=3,
        max_seqlen_k=10,
        mode=ForwardMode.EXTEND,
    )
    state = make_span(
        kv_pool=pool,
        num_blocks=16,
        num_tokens=8,
        max_kv_tokens=16,
        batch_size=3,
        device=device,
    )

    copy_span(
        state,
        input_ids=torch.arange(5, dtype=torch.long, device=device),
        positions=torch.arange(5, dtype=torch.long, device=device),
        attention_plan=plan,
        raw_num_tokens=5,
        last_token_indices=torch.tensor([1, 4], dtype=torch.long, device=device),
    )

    assert state.cache.base_lens == [5, 7, 0]
    assert state.cache.block_ids_by_row[2] == []
    assert state.plan.cache_seqlens_cpu == (5, 7, 0)
    assert state.plan.query_lens_cpu == (2, 6, 0)
    assert state.plan.kv_seqlens_cpu == (7, 13, 0)
    assert state.plan.max_seqlen_k == 16
    assert state.cache_seqlens.cpu().tolist() == [5, 7, 0]
    assert state.query_lens.cpu().tolist() == [2, 6, 0]
    assert state.kv_seqlens.cpu().tolist() == [7, 13, 0]
    assert state.cu_seqlens_q.cpu().tolist() == [0, 2, 8, 8]
    assert state.cu_seqlens_k.cpu().tolist() == [0, 7, 20, 20]


@requires_cuda
def test_prefill_graph_inputs_pad_short_batch_to_graph_bucket():
    device = torch.device("cuda")
    pool = _kv_pool(device, num_blocks=32, block_size=4)
    plan = _prefill_plan(
        pool,
        block_ids_by_row=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        cache_seqlens_cpu=(5, 6, 7),
        query_lens_cpu=(2, 3, 1),
        device=device,
        max_context_len=32,
    )
    state = make_span(
        kv_pool=pool,
        num_blocks=32,
        num_tokens=8,
        max_kv_tokens=32,
        batch_size=4,
        device=device,
        max_context_len=32,
    )

    copy_span(
        state,
        input_ids=torch.arange(6, dtype=torch.long, device=device),
        positions=torch.arange(6, dtype=torch.long, device=device),
        attention_plan=plan,
        raw_num_tokens=6,
        last_token_indices=torch.tensor([1, 4, 5], dtype=torch.long, device=device),
    )
    torch.cuda.synchronize()

    assert state.plan.cache_seqlens_cpu == (5, 6, 7, 0)
    assert state.cache.block_ids_by_row[3] == []
    assert state.plan.query_lens_cpu == (2, 3, 3, 0)
    assert state.plan.kv_seqlens_cpu == (7, 9, 10, 0)
    assert state.cache_seqlens.cpu().tolist() == [5, 6, 7, 0]
    assert state.query_lens.cpu().tolist() == [2, 3, 3, 0]
    assert state.cu_seqlens_q.cpu().tolist() == [0, 2, 5, 8, 8]
    assert state.last_token_indices.cpu().tolist() == [1, 4, 5, 0]


@requires_cuda
def test_prefill_graph_token_tail_uses_sink_pages():
    device = torch.device("cuda")
    pool = _kv_pool(device, num_blocks=32, block_size=4)
    plan = _prefill_plan(
        pool,
        block_ids_by_row=[[5, 6]],
        cache_seqlens_cpu=(5,),
        query_lens_cpu=(2,),
        device=device,
        max_context_len=16,
    )
    state = make_span(
        kv_pool=pool,
        num_blocks=32,
        num_tokens=8,
        max_kv_tokens=16,
        batch_size=1,
        device=device,
        max_context_len=16,
    )

    copy_span(
        state,
        input_ids=torch.arange(2, dtype=torch.long, device=device),
        positions=torch.arange(5, 7, dtype=torch.long, device=device),
        attention_plan=plan,
        raw_num_tokens=2,
        last_token_indices=torch.tensor([1], dtype=torch.long, device=device),
    )
    torch.cuda.synchronize()

    assert state.cache.base_lens == [5]
    assert state.cache.block_ids_by_row == [[5, 6, 0, 0]]
    assert state.plan.query_lens_cpu == (8,)
    assert state.plan.kv_seqlens_cpu == (13,)
    assert state.query_lens.cpu().tolist() == [8]
    assert state.last_token_indices.cpu().tolist() == [1]


def test_padded_max_tokens_isolates_padding_from_cached_prefix():
    plan = PagedVarlenPlan(
        residency_cache=None,
        block_table=torch.empty((1, 0), dtype=torch.int32),
        cache_seqlens=None,
        cache_seqlens_cpu=(1597,),
        query_lens=None,
        query_lens_cpu=(29,),
        kv_seqlens=None,
        kv_seqlens_cpu=(1626,),
        cu_seqlens_q=torch.tensor([0, 29], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 1626], dtype=torch.int32),
        max_seqlen_q=29,
        max_seqlen_k=1626,
        mode=ForwardMode.EXTEND,
    )

    assert (
        _padded_max_tokens(
            plan,
            padded_tokens=64,
            raw_tokens=29,
            batch_size=1,
        )
        == 1626
    )


@requires_cuda
@pytest.mark.gpu
def test_prefill_graph_pads_short_batch_to_bucket_and_returns_unpadded_rows():
    device = torch.device("cuda")
    torch.manual_seed(7)
    model = _LinearLogitsModel(device, seed=7)
    pool = _kv_pool(device, num_blocks=32, block_size=4)
    runner = Span(
        name="t",
        default_enabled=True,
        default_warmup=False,
        default_warmup_batch_sizes=(4,),
    )
    input_ids = torch.tensor([7, 9, 11, 5, 6, 13], dtype=torch.long, device=device)
    positions = torch.arange(6, dtype=torch.long, device=device)
    plan = _prefill_plan(
        pool,
        block_ids_by_row=[[0, 1], [2, 3], [4, 5]],
        cache_seqlens_cpu=(0, 0, 0),
        query_lens_cpu=(2, 2, 2),
        device=device,
        max_context_len=32,
    )
    last_token_indices = torch.tensor([1, 3, 5], dtype=torch.long, device=device)
    reference = model.logits(input_ids, positions).index_select(0, last_token_indices)

    stats = ForwardStats()
    out = runner.maybe_run(
        kv_pool=pool,
        num_blocks=32,
        num_tokens=8,
        max_kv_tokens=32,
        batch_size=3,
        input_ids=input_ids,
        positions=positions,
        attention_plan=plan,
        last_token_indices=last_token_indices,
        raw_num_tokens=6,
        ctx=ForwardContext(stats=stats),
        forward_fn=lambda state: model.logits(state.input_ids, state.positions).index_select(0, state.last_token_indices),
    )
    torch.cuda.synchronize()

    assert out is not None
    assert tuple(out.shape) == (3, _VOCAB)
    torch.testing.assert_close(out, reference)
    assert (8, 4, 32) in runner.states
    assert stats.cuda_graph_captures == 1
    assert stats.cuda_graph_replays == 1
    assert stats.cuda_graph_unpadded_tokens == 6
    assert stats.cuda_graph_padded_tokens == 2


# --------------------------------------------------------------------------- #
# Shared graph input buffer pool (CUDA tensors -> needs a device)
# --------------------------------------------------------------------------- #


@requires_cuda
def test_share_input_buffer_slices_largest_captured_buffer_for_smaller_bucket():
    pool: dict = {}
    device = torch.device("cuda")
    largest = torch.empty((8, 4), dtype=torch.int32, device=device)

    shared_largest = _share_input_buffer(pool, "block_table", largest, strict=True)
    smaller = torch.empty((3, 4), dtype=torch.int32, device=device)
    shared_smaller = _share_input_buffer(pool, "block_table", smaller, strict=True)

    # The first (largest) allocation is kept as-is; the smaller bucket is a view
    # that aliases the largest buffer's storage rather than a fresh allocation.
    assert shared_largest is largest
    assert tuple(shared_smaller.shape) == (3, 4)
    assert shared_smaller.data_ptr() == largest.data_ptr()


@requires_cuda
def test_share_input_buffer_strict_grows_for_larger_late_bucket():
    pool: dict = {}
    device = torch.device("cuda")
    small = torch.empty(4, dtype=torch.int32, device=device)
    shared_small = _share_input_buffer(pool, "cache_seqlens", small, strict=True)

    large = torch.empty(8, dtype=torch.int32, device=device)
    shared_large = _share_input_buffer(pool, "cache_seqlens", large, strict=True)

    # Late larger lazy buckets get a new resident buffer. Existing graph states
    # keep references to their original smaller tensors.
    assert shared_small is small
    assert shared_large is large
    assert shared_large.data_ptr() != shared_small.data_ptr()

    smaller_again = torch.empty(2, dtype=torch.int32, device=device)
    shared_smaller_again = _share_input_buffer(
        pool, "cache_seqlens", smaller_again, strict=True
    )
    assert tuple(shared_smaller_again.shape) == (2,)
    assert shared_smaller_again.data_ptr() == shared_large.data_ptr()


@requires_cuda
def test_share_input_buffer_pools_separately_by_dtype():
    pool: dict = {}
    device = torch.device("cuda")
    int_buf = torch.empty((4, 2), dtype=torch.int32, device=device)
    long_buf = torch.empty((2, 2), dtype=torch.long, device=device)

    shared_int = _share_input_buffer(pool, "shared", int_buf, strict=True)
    shared_long = _share_input_buffer(pool, "shared", long_buf, strict=True)

    # Same name but different dtype -> distinct pool entries, each its own buffer.
    assert shared_int is int_buf
    assert shared_long is long_buf
    assert len(pool) == 2


@requires_cuda
def test_module_level_decode_buffer_pool_slices_then_resets():
    _reset_for_testing()
    device = torch.device("cuda")
    try:
        largest = torch.empty((8, 2), dtype=torch.int32, device=device)
        first = _share_step_input("graph.step.block_table", largest)
        smaller = torch.empty((3, 2), dtype=torch.int32, device=device)
        sliced = _share_step_input("graph.step.block_table", smaller)

        assert first is largest
        assert sliced.data_ptr() == largest.data_ptr()

        _reset_for_testing()

        # After reset the pool is empty, so a fresh smaller tensor becomes the
        # new resident buffer (returned as itself, not a stale slice).
        fresh = torch.empty((2, 2), dtype=torch.int32, device=device)
        assert _share_step_input("graph.step.block_table", fresh) is fresh
    finally:
        _reset_for_testing()


# --------------------------------------------------------------------------- #
# Decode input copy padding (CUDA tensors)
# --------------------------------------------------------------------------- #


@requires_cuda
def test_decode_input_copy_pads_short_batch_to_bucket_with_zeros():
    device = torch.device("cuda")
    pool = _kv_pool(device)
    state = make_step(kv_pool=pool, num_blocks=16, batch_size=4, device=device)
    plan = _decode_plan(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(3, 5),
        kv_seqlens_cpu=(4, 6),
        device=device,
    )
    input_ids = torch.tensor([[11], [22]], dtype=torch.long, device=device)
    positions = torch.tensor([[3], [5]], dtype=torch.long, device=device)

    copy_step(
        state, input_ids=input_ids, positions=positions, attention_plan=plan
    )
    torch.cuda.synchronize()

    # The two real rows are copied; rows [2, 4) of the bucket are zero-padded.
    assert state.input_ids.flatten().tolist() == [11, 22, 0, 0]
    assert state.positions.flatten().tolist() == [3, 5, 0, 0]
    assert state.cache_seqlens.tolist() == [3, 5, 0, 0]
    assert state.plan.kv_seqlens.tolist() == [4, 6, 1, 1]


@requires_cuda
def test_decode_input_copy_extends_cpu_seqlen_mirrors():
    device = torch.device("cuda")
    pool = _kv_pool(device)
    state = make_step(kv_pool=pool, num_blocks=16, batch_size=4, device=device)
    plan = _decode_plan(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(3, 5),
        kv_seqlens_cpu=(4, 6),
        device=device,
    )

    copy_step(
        state,
        input_ids=torch.tensor([[11], [22]], dtype=torch.long, device=device),
        positions=torch.tensor([[3], [5]], dtype=torch.long, device=device),
        attention_plan=plan,
    )

    # cache_seqlens_cpu pads with 0 (no resident KV); kv_seqlens_cpu pads with 1
    # (the padded rows still attend over a single synthetic token).
    assert state.plan.cache_seqlens_cpu == (3, 5, 0, 0)
    assert state.plan.kv_seqlens_cpu == (4, 6, 1, 1)


@requires_cuda
def test_decode_host_input_copy_populates_static_bucket_inputs():
    device = torch.device("cuda")
    pool = _kv_pool(device)
    state = make_step(kv_pool=pool, num_blocks=16, batch_size=4, device=device)
    replacement = torch.tensor([99], dtype=torch.long, device=device)

    copy_host(
        state,
        StepInputs(
            input_ids=(11, 0),
            positions=(3, 5),
            block_ids_by_row=((0,), (1, 2)),
            cache_seqlens_cpu=(3, 5),
            kv_seqlens_cpu=(4, 6),
            token_replacements=((1, replacement),),
        ),
    )
    torch.cuda.synchronize()

    assert state.input_ids.flatten().tolist() == [11, 99, 0, 0]
    assert state.positions.flatten().tolist() == [3, 5, 0, 0]
    assert state.block_table[:2, :2].tolist() == [[0, 0], [1, 2]]
    assert state.block_table[:2, 2:].sum().item() == 0
    assert state.block_table[2:].sum().item() == 0
    assert state.cache_seqlens.tolist() == [3, 5, 0, 0]
    assert state.plan.kv_seqlens.tolist() == [4, 6, 1, 1]
    assert state.plan.decode_page_ids.tolist() == [0, 2, 0, 0]
    assert state.plan.decode_page_offsets.tolist() == [3, 1, 0, 0]
    assert state.plan.cache_seqlens_cpu == (3, 5, 0, 0)
    assert state.plan.kv_seqlens_cpu == (4, 6, 1, 1)


@requires_cuda
def test_decode_host_input_copy_dense_replacements_clear_inactive_bucket_rows():
    device = torch.device("cuda")
    pool = _kv_pool(device)
    state = make_step(kv_pool=pool, num_blocks=16, batch_size=4, device=device)
    state.input_ids.copy_(torch.tensor([[7], [8], [777], [888]], dtype=torch.long, device=device))
    replacements = (
        (0, torch.tensor([11], dtype=torch.long, device=device)),
        (1, torch.tensor([22], dtype=torch.long, device=device)),
    )

    copy_host(
        state,
        StepInputs(
            input_ids=(0, 0),
            positions=(3, 5),
            block_ids_by_row=((0,), (1, 2)),
            cache_seqlens_cpu=(3, 5),
            kv_seqlens_cpu=(4, 6),
            token_replacements=replacements,
        ),
    )
    torch.cuda.synchronize()

    assert state.input_ids.flatten().tolist() == [11, 22, 0, 0]
    assert state.positions.flatten().tolist() == [3, 5, 0, 0]
    assert state.plan.decode_page_ids.tolist() == [0, 2, 0, 0]
    assert state.plan.decode_page_offsets.tolist() == [3, 1, 0, 0]


@requires_cuda
def test_decode_host_input_copy_dense_replacements_preserve_contiguous_relay_span():
    device = torch.device("cuda")
    pool = _kv_pool(device)
    state = make_step(kv_pool=pool, num_blocks=16, batch_size=4, device=device)
    relay_tokens = torch.tensor([31, 41], dtype=torch.long, device=device)
    replacements = (
        (0, relay_tokens[0:1]),
        (1, relay_tokens[1:2]),
    )

    dense = dense_replacements(state, replacements, actual_batch=2)

    assert isinstance(dense, torch.Tensor)
    assert dense.data_ptr() == relay_tokens.data_ptr()
    assert tuple(dense.shape) == (2,)

    copy_host(
        state,
        StepInputs(
            input_ids=(0, 0),
            positions=(3, 5),
            block_ids_by_row=((0,), (1, 2)),
            cache_seqlens_cpu=(3, 5),
            kv_seqlens_cpu=(4, 6),
            token_replacements=replacements,
        ),
    )
    torch.cuda.synchronize()

    assert state.input_ids.flatten().tolist() == [31, 41, 0, 0]
    assert state.positions.flatten().tolist() == [3, 5, 0, 0]
    assert state.plan.decode_page_ids.tolist() == [0, 2, 0, 0]
    assert state.plan.decode_page_offsets.tolist() == [3, 1, 0, 0]


@requires_cuda
def test_decode_host_input_copy_reuses_unchanged_block_table(monkeypatch):
    device = torch.device("cuda")
    pool = _kv_pool(device)
    state = make_step(kv_pool=pool, num_blocks=16, batch_size=4, device=device)
    calls: list[str] = []
    original = step._copy_host_ints_to_device

    def record_copy(values, target, *, dtype, slot, name, view_shape=None):
        calls.append(str(name))
        return original(values, target, dtype=dtype, slot=slot, name=name, view_shape=view_shape)

    monkeypatch.setattr(step, "_copy_host_ints_to_device", record_copy)
    stable_rows = ((0, 4), (1, 2))

    copy_host(
        state,
        StepInputs(
            input_ids=(11, 22),
            positions=(3, 5),
            block_ids_by_row=stable_rows,
            cache_seqlens_cpu=(3, 5),
            kv_seqlens_cpu=(4, 6),
        ),
    )
    calls.clear()

    copy_host(
        state,
        StepInputs(
            input_ids=(33, 44),
            positions=(4, 6),
            block_ids_by_row=stable_rows,
            cache_seqlens_cpu=(4, 6),
            kv_seqlens_cpu=(5, 7),
        ),
    )
    torch.cuda.synchronize()

    assert "graph.step.block_table" not in calls
    assert state.block_table[:2, :2].tolist() == [[0, 4], [1, 2]]
    assert state.cache_seqlens.tolist() == [4, 6, 0, 0]
    calls.clear()

    copy_host(
        state,
        StepInputs(
            input_ids=(55, 66),
            positions=(5, 7),
            block_ids_by_row=((0, 4), (1, 2, 3)),
            cache_seqlens_cpu=(5, 8),
            kv_seqlens_cpu=(6, 9),
        ),
    )
    torch.cuda.synchronize()

    assert "graph.step.block_table" in calls
    assert state.block_table[:2, :3].tolist() == [[0, 4, 0], [1, 2, 3]]
    assert state.cache_seqlens.tolist() == [5, 8, 0, 0]


# --------------------------------------------------------------------------- #
# Eager-vs-graph replay (full capture + replay on CUDA)
# --------------------------------------------------------------------------- #


@requires_cuda
@pytest.mark.gpu
def test_decode_graph_replay_matches_eager_for_exact_bucket_batch():
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = _LinearLogitsModel(device, seed=0)
    pool = _kv_pool(device)
    runner = Step(name="t", default_warmup_batch_sizes=(1, 2, 4))

    input_ids = torch.tensor([[7], [9]], dtype=torch.long, device=device)
    positions = torch.tensor([[3], [5]], dtype=torch.long, device=device)
    plan = _decode_plan(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(3, 5),
        kv_seqlens_cpu=(4, 6),
        device=device,
    )
    reference = model.logits(input_ids, positions)

    stats = ForwardStats()
    out = runner.maybe_run(
        kv_pool=pool,
        num_blocks=16,
        batch_size=2,
        input_ids=input_ids,
        positions=positions,
        attention_plan=plan,
        ctx=ForwardContext(stats=stats),
        forward_fn=lambda state: model.logits(state.input_ids, state.positions),
        prepare_backend=None,
    )
    torch.cuda.synchronize()

    assert out is not None
    # batch == an exact bucket, so the returned tensor has exactly the batch rows.
    assert tuple(out.shape) == (2, _VOCAB)
    torch.testing.assert_close(out, reference)
    # First call is a capture+replay: one capture, one replay, no padding.
    assert stats.cuda_graph_captures == 1
    assert stats.cuda_graph_replays == 1
    assert stats.cuda_graph_unpadded_tokens == 2
    assert stats.cuda_graph_padded_tokens == 0


@requires_cuda
@pytest.mark.gpu
def test_decode_graph_second_call_replays_without_recapture():
    device = torch.device("cuda")
    torch.manual_seed(1)
    model = _LinearLogitsModel(device, seed=1)
    pool = _kv_pool(device)
    runner = Step(name="t", default_warmup_batch_sizes=(1, 2, 4))

    def run(ids, pos, seqlens, stats):
        plan = _decode_plan(
            pool,
            block_ids_by_row=[[0], [1]],
            cache_seqlens_cpu=seqlens,
            kv_seqlens_cpu=tuple(s + 1 for s in seqlens),
            device=device,
        )
        return runner.maybe_run(
            kv_pool=pool,
            num_blocks=16,
            batch_size=2,
            input_ids=ids,
            positions=pos,
            attention_plan=plan,
            ctx=ForwardContext(stats=stats),
            forward_fn=lambda state: model.logits(state.input_ids, state.positions),
            prepare_backend=None,
        )

    first_stats = ForwardStats()
    run(
        torch.tensor([[7], [9]], dtype=torch.long, device=device),
        torch.tensor([[3], [5]], dtype=torch.long, device=device),
        (3, 5),
        first_stats,
    )

    ids2 = torch.tensor([[5], [6]], dtype=torch.long, device=device)
    pos2 = torch.tensor([[1], [2]], dtype=torch.long, device=device)
    reference2 = model.logits(ids2, pos2)
    second_stats = ForwardStats()
    out2 = run(ids2, pos2, (1, 2), second_stats)
    torch.cuda.synchronize()

    # The bucket is already captured: the second call only replays.
    assert second_stats.cuda_graph_captures == 0
    assert second_stats.cuda_graph_replays == 1
    torch.testing.assert_close(out2, reference2)


@requires_cuda
@pytest.mark.gpu
def test_decode_graph_pads_short_batch_to_bucket_and_returns_unpadded_rows():
    device = torch.device("cuda")
    torch.manual_seed(2)
    model = _LinearLogitsModel(device, seed=2)
    pool = _kv_pool(device)
    # Only bucket 4 is configured, so a batch of 3 is padded up to 4.
    runner = Step(name="t", default_warmup_batch_sizes=(4,))

    input_ids = torch.tensor([[7], [9], [11]], dtype=torch.long, device=device)
    positions = torch.tensor([[2], [3], [1]], dtype=torch.long, device=device)
    plan = _decode_plan(
        pool,
        block_ids_by_row=[[0], [1], [2]],
        cache_seqlens_cpu=(2, 3, 1),
        kv_seqlens_cpu=(3, 4, 2),
        device=device,
    )
    reference = model.logits(input_ids, positions)

    stats = ForwardStats()
    out = runner.maybe_run(
        kv_pool=pool,
        num_blocks=16,
        batch_size=3,
        input_ids=input_ids,
        positions=positions,
        attention_plan=plan,
        ctx=ForwardContext(stats=stats),
        forward_fn=lambda state: model.logits(state.input_ids, state.positions),
        prepare_backend=None,
    )
    torch.cuda.synchronize()

    assert out is not None
    # batch 3 rounds up to the captured bucket 4...
    assert runner.resolve_bucket(3) == 4
    # ...but the returned tensor is sliced back to the unpadded request count.
    assert tuple(out.shape) == (3, _VOCAB)
    torch.testing.assert_close(out, reference)
    # The padded row contributes to the padded-token counter, not the unpadded.
    assert stats.cuda_graph_unpadded_tokens == 3
    assert stats.cuda_graph_padded_tokens == 1


@requires_cuda
@pytest.mark.gpu
def test_decode_graph_state_buffers_are_bucket_sized_and_zero_padded():
    device = torch.device("cuda")
    torch.manual_seed(3)
    model = _LinearLogitsModel(device, seed=3)
    pool = _kv_pool(device)
    runner = Step(name="t", default_warmup_batch_sizes=(4,))

    runner.maybe_run(
        kv_pool=pool,
        num_blocks=16,
        batch_size=3,
        input_ids=torch.tensor([[7], [9], [11]], dtype=torch.long, device=device),
        positions=torch.tensor([[2], [3], [1]], dtype=torch.long, device=device),
        attention_plan=_decode_plan(
            pool,
            block_ids_by_row=[[0], [1], [2]],
            cache_seqlens_cpu=(2, 3, 1),
            kv_seqlens_cpu=(3, 4, 2),
            device=device,
        ),
        ctx=ForwardContext(stats=ForwardStats()),
        forward_fn=lambda state: model.logits(state.input_ids, state.positions),
        prepare_backend=None,
    )
    torch.cuda.synchronize()

    state = runner.states[4]
    # The captured buffers are sized to the bucket (4), with the tail zero-padded
    # and the host seqlen mirrors extended (cache pads 0, kv pads 1).
    assert state.batch_size == 4
    assert state.input_ids.flatten().tolist() == [7, 9, 11, 0]
    assert state.cache_seqlens.tolist() == [2, 3, 1, 0]
    assert state.plan.cache_seqlens_cpu == (2, 3, 1, 0)
    assert state.plan.kv_seqlens_cpu == (3, 4, 2, 1)


# --------------------------------------------------------------------------- #
# Executor end-to-end dispatch (CUDA)
# --------------------------------------------------------------------------- #


@requires_cuda
@requires_decode_graph_backend
@pytest.mark.gpu
def test_text_graph_runner_decode_replay_matches_eager_forward():
    device = torch.device("cuda")
    torch.manual_seed(4)
    model = _LinearLogitsModel(device, seed=4)
    # FlashInfer's decode kernels require a real head_dim; the graph runner now
    # re-plans the backend before every replay, so the pool must be plannable.
    pool = _kv_pool(device, head_dim=64)
    runner = Executor(kv_pool=pool, num_blocks=16, block_size=4, device=device)

    input_ids = torch.tensor([[7], [9]], dtype=torch.long, device=device)
    positions = torch.tensor([[2], [3]], dtype=torch.long, device=device)
    plan = _decode_plan(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(2, 3),
        kv_seqlens_cpu=(3, 4),
        device=device,
    )
    fb = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        req_ids=(0, 1),
        input_ids=input_ids,
        positions=positions,
        attn_plan=plan,
    )
    reference = model.forward_text(fb)

    stats = ForwardStats()
    out = runner.maybe_run(model, input_ids, positions, fb, ForwardContext(stats=stats))
    torch.cuda.synchronize()

    assert out is not None
    assert tuple(out.shape) == (2, _VOCAB)
    torch.testing.assert_close(out, reference)
    assert stats.cuda_graph_captures == 1
    assert stats.cuda_graph_replays == 1


@requires_cuda
@pytest.mark.gpu
def test_text_graph_runner_skips_non_decode_extend_mode_without_prefill_graph():
    device = torch.device("cuda")
    torch.manual_seed(5)
    model = _LinearLogitsModel(device, seed=5)
    pool = _kv_pool(device)
    # Default worker config leaves prefill CUDA graphs disabled, so an ENCODE
    # (neither DECODE nor EXTEND) forward is never graphed -> the runner returns
    # None so the caller falls back to eager.
    runner = Executor(kv_pool=pool, num_blocks=16, block_size=4, device=device)

    input_ids = torch.tensor([[7], [9]], dtype=torch.long, device=device)
    positions = torch.tensor([[2], [3]], dtype=torch.long, device=device)
    plan = _decode_plan(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(2, 3),
        kv_seqlens_cpu=(3, 4),
        device=device,
    )
    fb = ForwardBatch(
        forward_mode=ForwardMode.ENCODE,
        req_ids=(0, 1),
        input_ids=input_ids,
        positions=positions,
        attn_plan=plan,
    )

    assert runner.maybe_run(model, input_ids, positions, fb, ForwardContext(stats=ForwardStats())) is None


@requires_cuda
@pytest.mark.gpu
def test_text_graph_runner_skips_cpu_input_tensors():
    device = torch.device("cuda")
    torch.manual_seed(6)
    model = _LinearLogitsModel(device, seed=6)
    pool = _kv_pool(device)
    runner = Executor(kv_pool=pool, num_blocks=16, block_size=4, device=device)

    plan = _decode_plan(
        pool,
        block_ids_by_row=[[0], [1]],
        cache_seqlens_cpu=(2, 3),
        kv_seqlens_cpu=(3, 4),
        device=device,
    )
    input_ids_cpu = torch.tensor([[7], [9]], dtype=torch.long)
    positions_cpu = torch.tensor([[2], [3]], dtype=torch.long)
    fb = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        req_ids=(0, 1),
        input_ids=input_ids_cpu,
        positions=positions_cpu,
        attn_plan=plan,
    )

    # CUDA graphs require device inputs; a CPU forward is not graphed.
    assert (
        runner.maybe_run(model, input_ids_cpu, positions_cpu, fb, ForwardContext(stats=ForwardStats()))
        is None
    )
