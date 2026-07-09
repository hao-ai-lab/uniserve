"""Denoise-step CUDA graph capture/replay on the real transient paged-varlen path.

Covers the observable contract of ``uniserve_worker.execution.denoise_step_graph``:

* replay is bitwise identical to the eager ``_predict_v_batched``-shaped forward
  on the *real* transient attention path (RadixAttention -> transient paged
  varlen -> the dispatcher-selected graph-capable paged-varlen backend);
* a foreign re-plan of the *shared* FlashInfer prefill wrapper between steps
  cannot corrupt a FlashInfer-backed captured graph (the plan-baking regression
  the exclusive wrapper exists for);
* returned velocity/hidden tensors never alias the graph's static output buffers
  (the next replay must not rewrite results a caller retained);
* release frees the per-image graph states and any backend graph-scoped wrapper
  bindings, and a later step recaptures cleanly;
* the ``UNISERVE_DENOISE_STEP_GRAPH`` env gate defaults off;
* capture failures fall back to eager and hard-disable after two strikes.

The capture/replay tests need a CUDA device plus the FlashInfer paged prefill
wrapper; they skip cleanly elsewhere. Key-shape tests are pure Python.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from uniserve_worker.contracts.forward_context import ForwardContext, use_forward_context
from uniserve_worker.contracts.forward_stats import ForwardStats
from uniserve_worker.execution import paged_denoise as paged_denoise_mod
from uniserve_worker.execution.denoise_step_graph import (
    DENOISE_STEP_GRAPH_ENV,
    DenoiseStepGraphRunner,
    maybe_run_denoise_step_graph,
    release_denoise_step_graphs,
)
from uniserve_worker.execution.interleaved_image_denoise import DenoiseRow
from uniserve_worker.execution.paged_denoise import can_run_paged_denoise_attention
from uniserve_worker.nn.attention import RadixAttention
from uniserve_worker.ops import AttentionRegime
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.paged_text_cache import BatchedPagedTextCache, PagedTextCache

pytestmark = pytest.mark.integration

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="denoise-step CUDA graph requires a CUDA device"
)


def _flashinfer_prefill_available() -> bool:
    try:
        from uniserve_worker.backends.attention import flashinfer as fi
    except Exception:
        return False
    return fi._BatchPrefillWithPagedKVCacheWrapper is not None


requires_flashinfer = pytest.mark.skipif(
    not _flashinfer_prefill_available(),
    reason="denoise-step CUDA graph requires the FlashInfer paged prefill wrapper",
)

_HEAD_DIM = 64
_TOKEN_H, _TOKEN_W = 4, 8
_N_TOKENS = _TOKEN_H * _TOKEN_W  # 32 transient tokens per branch
_BASE_LEN = 16                   # one full block of conditioning KV per branch
_BLOCK_SIZE = 16
_DTYPE = torch.bfloat16


class _TinyDenoiseOwner:
    """Minimal ``TextImageDenoiseOwner`` slice driving the real transient path.

    ``interleaved_image_predict_velocity`` mirrors the production contract: a
    projection into q/k/v, the transient paged-varlen attention through
    ``RadixAttention`` (writing transient KV into the caches' scratch pages and
    attending over the conditioning prefix), then a deterministic velocity head
    consuming ``t`` and ``z``. Static shapes, no host syncs — graph-capturable.
    """

    def __init__(self, device: torch.device, pool: PagedKVPool, *, seed: int) -> None:
        gen = torch.Generator(device=device).manual_seed(seed)
        self.device = device
        self.pool = pool
        self.attention_backend = "auto"
        self.attn = RadixAttention(1, 1, _HEAD_DIM, layer_id=0)
        def _w() -> torch.Tensor:
            return torch.randn(_HEAD_DIM, _HEAD_DIM, device=device, dtype=_DTYPE, generator=gen)
        self.wq, self.wk, self.wv, self.wo = _w(), _w(), _w(), _w()

    def _wait_gen_cache_ready(self, cache) -> None:
        return None

    def interleaved_image_predict_velocity(
        self,
        image_embeds: torch.Tensor,
        indexes: torch.Tensor,
        attention_mask,
        cache,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        image_token_num: int,
        image_size: tuple[int, int],
        return_hidden: bool = False,
    ):
        del attention_mask, image_size
        batch, n_tokens, dim = image_embeds.shape
        assert n_tokens == image_token_num
        x = image_embeds + indexes.permute(1, 2, 0).sum(-1, keepdim=True).to(_DTYPE) * 0.001
        q = (x @ self.wq).view(batch, n_tokens, 1, dim).transpose(1, 2)
        k = (x @ self.wk).view(batch, n_tokens, 1, dim).transpose(1, 2)
        v = (x @ self.wv).view(batch, n_tokens, 1, dim).transpose(1, 2)
        view = cache.request_cache_for_transient(0, n_tokens)
        out = self.attn(q, k, v, kv_cache=view, update_cache=True, causal=True)
        hidden = out.transpose(1, 2).reshape(batch, n_tokens, dim) @ self.wo
        velocity = (hidden.float() - z.float()) * (1.0 - t.float()).clamp_min(1e-4)
        velocity = velocity.to(image_embeds.dtype)
        if return_hidden:
            return velocity, hidden
        return velocity


class _FailingOwner(_TinyDenoiseOwner):
    """Owner whose denoise forward always fails (capture warmup raises)."""

    def interleaved_image_predict_velocity(self, *args, **kwargs):
        raise ValueError("synthetic denoise forward failure")


def _make_pool(device: torch.device) -> PagedKVPool:
    return PagedKVPool(
        num_layers=1,
        num_blocks=64,
        block_size=_BLOCK_SIZE,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        device=device,
        dtype=_DTYPE,
    )


@requires_cuda
def test_paged_denoise_uses_transient_varlen_attention_metadata(monkeypatch):
    device = torch.device("cuda")
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=4,
        block_size=4,
        num_kv_heads=1,
        head_dim=8,
        device=device,
        dtype=torch.float16,
    )
    cache = PagedTextCache(pool, [1], num_layers=1, length=0)
    prototype = torch.empty((1, 2, 16), device=device, dtype=torch.float16)
    calls = []

    def fake_can_run_attention(q, k, v, **kwargs):
        del k, v
        calls.append((q, kwargs))
        return True

    monkeypatch.setattr(paged_denoise_mod.ops, "can_run_attention", fake_can_run_attention)

    assert can_run_paged_denoise_attention(cache, prototype=prototype, query_width=8)

    assert len(calls) == 1
    q_probe, kwargs = calls[0]
    assert tuple(q_probe.shape) == (1, 1, 8)
    assert kwargs["regime"] is AttentionRegime.EXTEND
    assert kwargs["kv_cache"] is cache.request_cache_for_transient(0, 2)
    assert kwargs["block_table"].shape == (1, 1)
    assert kwargs["cu_seqlens_q"].tolist() == [0, 2]
    assert kwargs["cu_seqlens_k"].tolist() == [0, 2]
    assert kwargs["max_seqlen_q"] == 2
    assert kwargs["max_seqlen_k"] == 2


def _make_caches(
    pool: PagedKVPool, device: torch.device, *, seed: int, first_block: int = 0
) -> list[PagedTextCache]:
    """Two CFG-branch caches with random conditioning KV and a block allocator."""

    gen = torch.Generator(device=device).manual_seed(seed)
    free_blocks = list(range(first_block + 2, first_block + 32))

    def allocate(count: int) -> list[int]:
        return [free_blocks.pop(0) for _ in range(count)]

    caches = []
    for row in range(2):
        cache = PagedTextCache(
            pool, [first_block + row], num_layers=1, length=_BASE_LEN, allocate_blocks=allocate
        )
        base_kv = torch.randn(
            2, _BASE_LEN, pool.n_kv, pool.head_dim, device=device, dtype=_DTYPE, generator=gen
        )
        pool.write(0, cache.block_ids, start=0, k=base_kv[0], v=base_kv[1])
        caches.append(cache)
    return caches


def _make_rows(
    caches: list[PagedTextCache],
    device: torch.device,
    *,
    seed: int,
    t_value: float,
) -> list[DenoiseRow]:
    """One denoise step's grouped CFG rows (shared img geometry, per-row inputs)."""

    gen = torch.Generator(device=device).manual_seed(seed)
    img = SimpleNamespace(
        token_h=_TOKEN_H,
        token_w=_TOKEN_W,
        height=_TOKEN_H * 16,
        width=_TOKEN_W * 16,
        cond_cache=caches[0],
        tu_cache=caches[1],
        iu_cache=None,
    )
    t = torch.tensor(t_value, device=device, dtype=torch.float32)
    rows = []
    for row_index, cache in enumerate(caches):
        embeds = torch.randn(1, _N_TOKENS, _HEAD_DIM, device=device, dtype=_DTYPE, generator=gen)
        latent = torch.randn(1, _N_TOKENS, _HEAD_DIM, device=device, dtype=_DTYPE, generator=gen)
        pos = torch.arange(_N_TOKENS, device=device, dtype=torch.long)
        indexes = torch.stack(
            [
                torch.full((_N_TOKENS,), _BASE_LEN, device=device, dtype=torch.long),
                pos // _TOKEN_W,
                pos % _TOKEN_W,
            ],
            dim=0,
        )
        step = SimpleNamespace(extra={"image_embeds": embeds}, latent=latent, t=t)
        rows.append(
            DenoiseRow(
                step_index=0,
                step=step,
                branch="cond" if row_index == 0 else "text_uncond",
                img=img,
                indexes=indexes,
                cache=cache,
            )
        )
    return rows


def _eager_reference(owner: _TinyDenoiseOwner, rows: list[DenoiseRow], *, return_hidden: bool = False):
    """The exact eager ``_predict_v_batched`` tail the graph replaces."""

    first = rows[0]
    image_embeds = torch.cat([row.step.extra["image_embeds"] for row in rows], dim=0)
    indexes = torch.stack([row.indexes for row in rows], dim=1).contiguous()
    cache = BatchedPagedTextCache([row.cache for row in rows])
    z = torch.cat([row.step.latent for row in rows], dim=0)
    return owner.interleaved_image_predict_velocity(
        image_embeds,
        indexes,
        {"full_attention": None},
        cache,
        first.step.t,
        z,
        image_token_num=_N_TOKENS,
        image_size=(first.img.width, first.img.height),
        return_hidden=return_hidden,
    )


def _flashinfer_backend():
    from uniserve_worker.backends.attention import get_attention_backend

    return get_attention_backend("flashinfer")


def _scoped_prefill_keys(backend) -> list:
    return [key for key in backend._prefill_wrappers if key.scope is not None]


def _run(runner: DenoiseStepGraphRunner, owner, rows, *, return_hidden: bool = False):
    stats = ForwardStats()
    with torch.inference_mode(), use_forward_context(ForwardContext(stats=stats)):
        out = runner.maybe_run_rows(owner, rows, return_hidden=return_hidden)
    torch.cuda.synchronize()
    return out, stats


# --------------------------------------------------------------------------- #
# Pure-Python key semantics
# --------------------------------------------------------------------------- #


def test_rows_key_rejects_non_cuda_inputs():
    step = SimpleNamespace(
        extra={"image_embeds": torch.zeros(1, 4, 8)},
        latent=torch.zeros(1, 4, 8),
        t=torch.zeros(()),
    )
    img = SimpleNamespace(token_h=2, token_w=2, height=32, width=32)
    cache = SimpleNamespace(pool=SimpleNamespace(), length=16, block_ids=[3, 4])
    row = DenoiseRow(0, step, "cond", img, torch.zeros(3, 4, dtype=torch.long), cache)

    # CPU inputs are never graph-eligible.
    assert DenoiseStepGraphRunner._rows_key([row], False) is None


def test_rows_key_component_stability_and_change_detection():
    if not torch.cuda.is_available():
        pytest.skip("key builder inspects CUDA tensors")
    device = torch.device("cuda")
    pool = SimpleNamespace()
    caches = [
        SimpleNamespace(pool=pool, length=16, block_ids=[1, 2]),
        SimpleNamespace(pool=pool, length=16, block_ids=[3, 4]),
    ]
    img = SimpleNamespace(token_h=2, token_w=2, height=32, width=32)
    t = torch.zeros((), device=device)

    def rows():
        out = []
        for cache in caches:
            step = SimpleNamespace(
                extra={"image_embeds": torch.zeros(1, 4, 8, device=device, dtype=_DTYPE)},
                latent=torch.zeros(1, 4, 8, device=device, dtype=_DTYPE),
                t=t,
            )
            out.append(
                DenoiseRow(0, step, "cond", img, torch.zeros(3, 4, dtype=torch.long, device=device), cache)
            )
        return out

    key_a = DenoiseStepGraphRunner._rows_key(rows(), False)
    key_b = DenoiseStepGraphRunner._rows_key(rows(), False)
    assert key_a is not None and key_a == key_b

    assert DenoiseStepGraphRunner._rows_key(rows(), True) != key_a  # return_hidden in key

    caches[0].length = 17
    assert DenoiseStepGraphRunner._rows_key(rows(), False) != key_a  # base len in key
    caches[0].length = 16

    caches[1].block_ids = [3, 5]
    assert DenoiseStepGraphRunner._rows_key(rows(), False) != key_a  # block ids in key


# --------------------------------------------------------------------------- #
# Capture/replay on the real transient path (CUDA + FlashInfer)
# --------------------------------------------------------------------------- #


@requires_cuda
@requires_flashinfer
@pytest.mark.gpu
def test_denoise_step_graph_replay_is_bitwise_identical_to_eager():
    device = torch.device("cuda")
    pool = _make_pool(device)
    owner = _TinyDenoiseOwner(device, pool, seed=0)
    caches = _make_caches(pool, device, seed=1)
    runner = DenoiseStepGraphRunner(default_enabled=True)

    rows_a = _make_rows(caches, device, seed=10, t_value=0.9)
    with torch.inference_mode():
        eager_a = _eager_reference(owner, rows_a)
    out_a, stats_a = _run(runner, owner, rows_a)

    assert out_a is not None
    assert torch.equal(out_a, eager_a)
    assert stats_a.cuda_graph_captures == 1
    assert stats_a.cuda_graph_replays == 1

    # A later step: same image geometry, new embeds/latent/timestep -> pure replay.
    rows_b = _make_rows(caches, device, seed=20, t_value=0.5)
    with torch.inference_mode():
        eager_b = _eager_reference(owner, rows_b)
    out_b, stats_b = _run(runner, owner, rows_b)

    assert out_b is not None
    assert torch.equal(out_b, eager_b)
    assert stats_b.cuda_graph_captures == 0
    assert stats_b.cuda_graph_replays == 1
    assert len(runner.states) == 1


@requires_cuda
@requires_flashinfer
@pytest.mark.gpu
def test_denoise_step_graph_survives_foreign_shared_wrapper_replans():
    """A foreign varlen forward must not invalidate the captured (baked) plan.

    This is the regression the graph-scoped exclusive prefill wrapper exists
    for: the shared wrapper is re-planned by the interleaved forward below, and
    the captured graph must keep replaying its own plan bit-exactly.
    """

    device = torch.device("cuda")
    pool = _make_pool(device)
    owner = _TinyDenoiseOwner(device, pool, seed=2)
    caches = _make_caches(pool, device, seed=3)
    runner = DenoiseStepGraphRunner(default_enabled=True)

    rows_a = _make_rows(caches, device, seed=30, t_value=0.8)
    out_a, _ = _run(runner, owner, rows_a)
    assert out_a is not None

    # Foreign work: a differently-shaped transient varlen forward through the
    # *shared* prefill wrapper (fresh caches on disjoint blocks, different base
    # geometry).
    foreign_caches = _make_caches(pool, device, seed=4, first_block=32)
    foreign_caches[0].length = _BASE_LEN - 3
    foreign_caches[1].length = _BASE_LEN - 3
    foreign_rows = _make_rows(foreign_caches, device, seed=40, t_value=0.7)
    for row in foreign_rows:
        row.indexes = row.indexes.clone()
        row.indexes[0].fill_(_BASE_LEN - 3)
    with torch.inference_mode():
        _eager_reference(owner, foreign_rows)

    rows_b = _make_rows(caches, device, seed=50, t_value=0.4)
    with torch.inference_mode():
        eager_b = _eager_reference(owner, rows_b)
    out_b, stats_b = _run(runner, owner, rows_b)

    assert out_b is not None
    assert stats_b.cuda_graph_captures == 0 and stats_b.cuda_graph_replays == 1
    assert torch.equal(out_b, eager_b)


@requires_cuda
@requires_flashinfer
@pytest.mark.gpu
def test_denoise_step_graph_outputs_do_not_alias_static_buffers():
    device = torch.device("cuda")
    pool = _make_pool(device)
    owner = _TinyDenoiseOwner(device, pool, seed=5)
    caches = _make_caches(pool, device, seed=6)
    runner = DenoiseStepGraphRunner(default_enabled=True)

    rows_a = _make_rows(caches, device, seed=60, t_value=0.9)
    with torch.inference_mode():
        eager_v_a, eager_h_a = _eager_reference(owner, rows_a, return_hidden=True)
    out_a, _ = _run(runner, owner, rows_a, return_hidden=True)
    assert out_a is not None
    velocity_a, hidden_a = out_a
    assert torch.equal(velocity_a, eager_v_a)
    assert torch.equal(hidden_a, eager_h_a)

    # Replay with different inputs; the previously returned tensors must be
    # untouched (they would be rewritten if they aliased the graph buffers).
    rows_b = _make_rows(caches, device, seed=70, t_value=0.3)
    out_b, _ = _run(runner, owner, rows_b, return_hidden=True)
    assert out_b is not None
    velocity_b, hidden_b = out_b
    assert not torch.equal(hidden_b, hidden_a)
    torch.cuda.synchronize()
    assert torch.equal(velocity_a, eager_v_a)
    assert torch.equal(hidden_a, eager_h_a)
    state = next(iter(runner.states.values()))
    graph_velocity, graph_hidden = state.logits
    assert hidden_a.data_ptr() != graph_hidden.data_ptr()
    assert velocity_a.data_ptr() != graph_velocity.data_ptr()


@requires_cuda
@requires_flashinfer
@pytest.mark.gpu
def test_denoise_step_graph_release_frees_states_and_backend_bindings():
    device = torch.device("cuda")
    pool = _make_pool(device)
    owner = _TinyDenoiseOwner(device, pool, seed=7)
    caches = _make_caches(pool, device, seed=8)
    runner = DenoiseStepGraphRunner(default_enabled=True)
    owner._denoise_step_graph_runner = runner
    backend = _flashinfer_backend()
    scoped_before = set(_scoped_prefill_keys(backend))

    rows = _make_rows(caches, device, seed=80, t_value=0.9)
    out, _ = _run(runner, owner, rows)
    assert out is not None
    assert len(runner.states) == 1
    state = next(iter(runner.states.values()))
    owns_flashinfer_binding = state.release_backend is not None
    if owns_flashinfer_binding:
        assert len(_scoped_prefill_keys(backend)) == len(scoped_before) + 1
    else:
        assert set(_scoped_prefill_keys(backend)) == scoped_before

    image_state = SimpleNamespace(cond_cache=caches[0], tu_cache=caches[1], iu_cache=None)
    release_denoise_step_graphs(owner, image_state)

    assert runner.states == {}
    assert set(_scoped_prefill_keys(backend)) == scoped_before

    # A later image over the same caches recaptures cleanly.
    rows_again = _make_rows(caches, device, seed=90, t_value=0.6)
    with torch.inference_mode():
        eager_again = _eager_reference(owner, rows_again)
    out_again, stats_again = _run(runner, owner, rows_again)
    assert out_again is not None
    assert stats_again.cuda_graph_captures == 1
    assert torch.equal(out_again, eager_again)

    release_denoise_step_graphs(owner, image_state)
    assert runner.states == {}
    assert set(_scoped_prefill_keys(backend)) == scoped_before


@requires_cuda
@requires_flashinfer
@pytest.mark.gpu
def test_denoise_step_graph_env_gate_defaults_off(monkeypatch):
    device = torch.device("cuda")
    pool = _make_pool(device)
    owner = _TinyDenoiseOwner(device, pool, seed=9)
    caches = _make_caches(pool, device, seed=10)
    rows = _make_rows(caches, device, seed=100, t_value=0.9)

    monkeypatch.delenv(DENOISE_STEP_GRAPH_ENV, raising=False)
    with torch.inference_mode(), use_forward_context(ForwardContext(stats=ForwardStats())):
        assert maybe_run_denoise_step_graph(owner, rows) is None
    assert getattr(owner, "_denoise_step_graph_runner", None) is None

    monkeypatch.setenv(DENOISE_STEP_GRAPH_ENV, "1")
    with torch.inference_mode():
        eager = _eager_reference(owner, rows)
    with torch.inference_mode(), use_forward_context(ForwardContext(stats=ForwardStats())):
        out = maybe_run_denoise_step_graph(owner, rows)
    torch.cuda.synchronize()
    assert out is not None
    assert torch.equal(out, eager)
    runner = owner._denoise_step_graph_runner
    release_denoise_step_graphs(
        owner, SimpleNamespace(cond_cache=caches[0], tu_cache=caches[1], iu_cache=None)
    )
    assert runner.states == {}


@requires_cuda
@requires_flashinfer
@pytest.mark.gpu
def test_denoise_step_graph_capture_failure_falls_back_then_hard_disables():
    device = torch.device("cuda")
    pool = _make_pool(device)
    owner = _FailingOwner(device, pool, seed=11)
    caches = _make_caches(pool, device, seed=12)
    runner = DenoiseStepGraphRunner(default_enabled=True)
    backend = _flashinfer_backend()
    scoped_before = set(_scoped_prefill_keys(backend))

    rows = _make_rows(caches, device, seed=110, t_value=0.9)
    out_first, stats_first = _run(runner, owner, rows)
    assert out_first is None
    assert stats_first.cuda_graph_fallbacks == 1
    assert runner.enabled()  # one strike left
    # The failed capture's wrapper binding is rolled back.
    assert set(_scoped_prefill_keys(backend)) == scoped_before

    other_caches = _make_caches(pool, device, seed=13, first_block=32)
    other_rows = _make_rows(other_caches, device, seed=120, t_value=0.8)
    out_second, stats_second = _run(runner, owner, other_rows)
    assert out_second is None
    assert stats_second.cuda_graph_fallbacks == 1
    assert not runner.enabled()  # two strikes -> hard disable

    out_third, stats_third = _run(runner, owner, rows)
    assert out_third is None
    assert stats_third.cuda_graph_fallbacks == 0  # disabled runner does not engage
    assert set(_scoped_prefill_keys(backend)) == scoped_before
