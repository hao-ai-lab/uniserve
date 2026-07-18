"""Qwen3 family conformance through the complete target stack (GPU).

The Stage 11 conformance shape at unit scale: a real Qwen3-topology adapter
(RMSNorm, RoPE, grouped projections, gated MLP, paged reference attention)
runs prefill, greedy decode, and candidate verification through
ExecutionEngine + StandardTransactionExecutor + transactional residency, and
every sampled token and acceptance count must match an independent dense
full-recompute oracle implemented from scratch in this test.
"""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.backends.attention.reference_target import (
    CacheDeviceBinding,
    ReferenceAttentionBackend,
)
from uniserve_worker.contracts.execution import (
    CandidateVerification,
    EngineRef,
    ExecuteBatch,
    ExecuteRow,
    NewSession,
    OperationTag,
    SamplingSpec,
    SequenceStep,
    SessionRef,
)
from uniserve_worker.contracts.residency_batch import ResidencyBatchCapacity
from uniserve_worker.contracts.segment_table import GraphCapacity
from uniserve_worker.execution.engine import ExecutionEngine, StandardTransactionExecutor
from uniserve_worker.models.cache_registrations import qwen3_cache_registration
from uniserve_worker.models.qwen3_target import Qwen3Target, Qwen3TargetConfig
from uniserve_worker.runtime.transactional_residency import ArenaConfig, Residency

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)
_PAGE_TOKENS = 4
_CONFIG = Qwen3TargetConfig(
    vocab_size=64,
    hidden_size=32,
    layers=2,
    routes=1,
    query_heads=4,
    kv_heads=2,
    head_dim=8,
    mlp_hidden=64,
)


# --------------------------------------------------------------------------- #
# Independent dense oracle: textbook causal transformer over the full history.
# --------------------------------------------------------------------------- #


def _oracle_logits(adapter: Qwen3Target, tokens: list[int]) -> torch.Tensor:
    c = adapter.config
    device = adapter.device
    ids = torch.tensor(tokens, device=device, dtype=torch.long)
    positions = torch.arange(len(tokens), device=device)
    x = adapter.embedding.index_select(0, ids)

    def rms(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        var = value.float().pow(2).mean(-1, keepdim=True)
        return (value.float() * torch.rsqrt(var + c.rms_eps)).to(value.dtype) * weight

    def rope(value: torch.Tensor) -> torch.Tensor:
        half = c.head_dim // 2
        freqs = c.rope_theta ** (
            -torch.arange(0, half, device=device, dtype=torch.float32) / half
        )
        angles = positions.float()[:, None] * freqs[None, :]
        cos, sin = torch.cos(angles)[:, None, :], torch.sin(angles)[:, None, :]
        a, b = value[..., :half].float(), value[..., half:].float()
        return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1).to(value.dtype)

    total = len(tokens)
    causal = torch.tril(torch.ones(total, total, device=device, dtype=torch.bool))
    for layer in adapter.layers:
        normed = rms(x, layer.input_norm)
        q = rope(
            (normed @ layer.q.weights[0].T).view(total, c.query_heads, c.head_dim)
        )
        k = rope(
            (normed @ layer.k.weights[0].T).view(total, c.kv_heads, c.head_dim)
        )
        v = (normed @ layer.v.weights[0].T).view(total, c.kv_heads, c.head_dim)
        group = c.query_heads // c.kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * c.head_dim**-0.5
        scores = scores.masked_fill(~causal[None], float("-inf"))
        attended = torch.einsum(
            "hqk,khd->qhd", torch.softmax(scores, dim=-1), v.float()
        ).to(x.dtype)
        x = x + attended.reshape(total, -1) @ layer.o.weights[0].T
        normed = rms(x, layer.post_norm)
        gate = normed @ layer.gate.weights[0].T
        up = normed @ layer.up.weights[0].T
        x = x + (torch.nn.functional.silu(gate) * up) @ layer.down.weights[0].T
    return rms(x, adapter.final_norm) @ adapter.lm_head.T


def _oracle_greedy(adapter: Qwen3Target, tokens: list[int]) -> int:
    return int(_oracle_logits(adapter, tokens)[-1].argmax())


# --------------------------------------------------------------------------- #
# Target stack plumbing.
# --------------------------------------------------------------------------- #


def _stack():
    registration = qwen3_cache_registration(
        layer_count=_CONFIG.layers,
        query_heads=_CONFIG.query_heads,
        kv_heads=_CONFIG.kv_heads,
        qk_head_dim=_CONFIG.head_dim,
        value_head_dim=_CONFIG.head_dim,
        page_tokens=_PAGE_TOKENS,
    )
    residency = Residency(
        _ENGINE, (ArenaConfig(domain_id=1, page_count=65, page_tokens=_PAGE_TOKENS),)
    )
    binding = CacheDeviceBinding(
        layers=_CONFIG.layers,
        pages=65,
        page_tokens=_PAGE_TOKENS,
        kv_heads=_CONFIG.kv_heads,
        head_dim=_CONFIG.head_dim,
        device="cuda",
    )
    adapter = Qwen3Target(
        _CONFIG, ReferenceAttentionBackend(binding), device="cuda", seed=11
    )
    capacity = GraphCapacity(
        rows=2,
        segments=8,
        tokens=64,
        branches=3,
        candidate_tokens=8,
        position_axes=1,
        visibility_payload_entries=0,
        residency=ResidencyBatchCapacity(bindings=6, page_references=32, tokens=64),
    )
    executor = StandardTransactionExecutor(
        residency=residency,
        registration=registration,
        capacities=(capacity,),
        adapter=adapter,
        page_tokens=_PAGE_TOKENS,
    )
    engine = ExecutionEngine(
        engine=_ENGINE,
        executor=executor,
        advertised_operations=frozenset(OperationTag),
        replay_window=16,
    )
    return engine, adapter


def _row(operation, version, admission=None):
    return ExecuteRow(
        row_id=0,
        session=SessionRef(_ENGINE, 41, 1, version),
        operation=operation,
        admission=admission,
        cache_leases=(),
        product_leases=(),
        scheduler_op_id=0,
    )


def _batch(step, row):
    return ExecuteBatch(
        engine_epoch=_ENGINE.engine_epoch,
        step_id=step,
        acknowledged_through=step - 1,
        rows=(row,),
    )


def test_prefill_decode_and_verification_match_the_dense_oracle():
    engine, adapter = _stack()
    history = [3, 1, 4, 1, 5]

    # Prefill through admission.
    admission = NewSession(41, 1, SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0), 42, 64)
    prefill = SequenceStep(tuple(history), 0, 0, 1)
    result = engine.execute(_batch(0, _row(prefill, 0, admission))).result()
    sampled = result.row_results[0].sampled_tokens[0]
    assert sampled == _oracle_greedy(adapter, history)
    history.append(sampled)

    # Three greedy decode steps: paged incremental state must reproduce the
    # dense recompute exactly.
    for step in range(1, 4):
        decode = SequenceStep((history[-1],), len(history) - 1, len(history) - 1, 1)
        result = engine.execute(_batch(step, _row(decode, step))).result()
        sampled = result.row_results[0].sampled_tokens[0]
        assert sampled == _oracle_greedy(adapter, history)
        history.append(sampled)

    # Verification with oracle-perfect candidates: all accepted.
    context = list(history)
    draft = []
    rollout = list(context)
    for _ in range(3):
        rollout.append(_oracle_greedy(adapter, rollout))
        draft.append(rollout[-1])
    verify = SequenceStep(
        (context[-1],),
        len(context) - 1,
        len(context) - 1,
        1,
        CandidateVerification(
            tuple(draft), tuple(range(len(context), len(context) + 3))
        ),
    )
    result = engine.execute(_batch(4, _row(verify, 4))).result()
    assert result.row_results[0].accepted_candidates == 3
    delta = result.session_deltas[0]
    # 1 input token commits fully plus all 3 accepted candidates.
    assert delta.history_length_after == len(context) + 3

    # Verification with a corrupted second candidate: exactly one accepted.
    committed = context + draft
    next_input = result.row_results[0].sampled_tokens[0]
    assert next_input == _oracle_greedy(adapter, committed)
    first = _oracle_greedy(adapter, committed + [next_input])
    wrong = (first, (first + 7) % 64, (first + 9) % 64)
    verify = SequenceStep(
        (next_input,),
        len(committed),
        len(committed),
        1,
        CandidateVerification(
            wrong, tuple(range(len(committed) + 1, len(committed) + 4))
        ),
    )
    result = engine.execute(_batch(5, _row(verify, 5))).result()
    assert result.row_results[0].accepted_candidates == 1
    assert result.session_deltas[0].history_length_after == len(committed) + 2
