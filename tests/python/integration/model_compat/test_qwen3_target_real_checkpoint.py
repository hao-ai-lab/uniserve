"""Real Qwen3-32B through the complete target stack (Stage 11 evidence).

Loads the actual production checkpoint into the target family root (exact
architecture: per-head QK-norm, checkpoint rope theta, bf16) and serves
greedy decode through ExecutionEngine + transactional residency + the
FlashInfer provider. Conformance:

* every step's logits match an independent dense full-recompute of the same
  real weights (textbook math, no shared stack code paths);
* the greedy continuation of a real tokenized prompt detokenizes to
  coherent text — a whole-architecture check that would fail loudly on any
  rope/norm/mapping error.

Requires the pinned checkpoint; skipped where it is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

_CHECKPOINT = os.environ.get(
    "UNISERVE_QWEN3_MODEL",
    str(
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots"
        / "9216db5781bf21249d130ec9da846c4624c16137"
    ),
)


@pytest.mark.skipif(
    not Path(_CHECKPOINT).exists(), reason="Qwen3 checkpoint not present"
)
def test_real_checkpoint_greedy_decode_matches_dense_recompute():
    from uniserve_worker.backends.attention.flashinfer_target import (
        FlashInferTargetBackend,
    )
    from uniserve_worker.backends.attention.reference_target import (
        CacheDeviceBinding,
    )
    from uniserve_worker.contracts.execution import (
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
    from uniserve_worker.contracts.segment_table import (
        AttentionLayerSpec,
        GraphCapacity,
    )
    from uniserve_worker.execution.engine import ExecutionEngine, StandardTransactionExecutor
    from uniserve_worker.models.cache_registrations import qwen3_cache_registration
    from uniserve_worker.models.qwen3_target import (
        Qwen3Target,
        Qwen3TargetConfig,
        load_qwen3_checkpoint,
    )
    from uniserve_worker.runtime.transactional_residency import (
        ArenaConfig,
        Residency,
    )

    raw = json.loads((Path(_CHECKPOINT) / "config.json").read_text())
    config = Qwen3TargetConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        layers=raw["num_hidden_layers"],
        routes=1,
        query_heads=raw["num_attention_heads"],
        kv_heads=raw["num_key_value_heads"],
        head_dim=raw.get(
            "head_dim", raw["hidden_size"] // raw["num_attention_heads"]
        ),
        mlp_hidden=raw["intermediate_size"],
        rope_theta=float(raw.get("rope_theta", 10_000.0)),
        rms_eps=float(raw.get("rms_norm_eps", 1e-6)),
        qk_norm=True,
    )
    page_tokens = 16
    engine_ref = EngineRef(deployment_id=1, engine_epoch=7)
    binding = CacheDeviceBinding(
        layers=config.layers,
        pages=17,
        page_tokens=page_tokens,
        kv_heads=config.kv_heads,
        head_dim=config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    site = AttentionLayerSpec(
        layer_id=0,
        site_id=1,
        domain_id=1,
        query_heads=config.query_heads,
        kv_heads=config.kv_heads,
        qk_head_dim=config.head_dim,
        value_head_dim=config.head_dim,
        scale=config.head_dim**-0.5,
    )
    adapter = Qwen3Target(
        config,
        FlashInferTargetBackend(binding, site),
        device="cuda",
        zero_init=True,
        dtype=torch.bfloat16,
    )
    load_qwen3_checkpoint(adapter, _CHECKPOINT)

    registration = qwen3_cache_registration(
        layer_count=config.layers,
        query_heads=config.query_heads,
        kv_heads=config.kv_heads,
        qk_head_dim=config.head_dim,
        value_head_dim=config.head_dim,
        page_tokens=page_tokens,
    )
    residency = Residency(
        engine_ref,
        (ArenaConfig(domain_id=1, page_count=17, page_tokens=page_tokens),),
    )
    capacity = GraphCapacity(
        rows=1,
        segments=4,
        tokens=64,
        branches=1,
        candidate_tokens=0,
        position_axes=1,
        visibility_payload_entries=0,
        residency=ResidencyBatchCapacity(bindings=2, page_references=16, tokens=64),
    )
    engine = ExecutionEngine(
        engine=engine_ref,
        executor=StandardTransactionExecutor(
            residency=residency,
            registration=registration,
            capacities=(capacity,),
            adapter=adapter,
            page_tokens=page_tokens,
        ),
        advertised_operations=frozenset(OperationTag),
        replay_window=32,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_CHECKPOINT)
    prompt = "The capital of France is"
    history = tokenizer.encode(prompt)
    assert len(history) < 16

    def dense_oracle_greedy(tokens: list[int]) -> int:
        """Independent textbook recompute over the real weights."""

        root = adapter.root
        c = config
        device = root.device
        ids = torch.tensor(tokens, device=device, dtype=torch.long)
        positions = torch.arange(len(tokens), device=device)
        x = root.embedding.index_select(0, ids)

        def rms(value, weight):
            var = value.float().pow(2).mean(-1, keepdim=True)
            return (value.float() * torch.rsqrt(var + c.rms_eps)).to(
                value.dtype
            ) * weight

        def rope(value):
            half = c.head_dim // 2
            freqs = c.rope_theta ** (
                -torch.arange(0, half, device=device, dtype=torch.float32) / half
            )
            angles = positions.float()[:, None] * freqs[None, :]
            cos = torch.cos(angles)[:, None, :]
            sin = torch.sin(angles)[:, None, :]
            a, b = value[..., :half].float(), value[..., half:].float()
            return torch.cat(
                [a * cos - b * sin, b * cos + a * sin], dim=-1
            ).to(value.dtype)

        total = len(tokens)
        causal = torch.tril(
            torch.ones(total, total, device=device, dtype=torch.bool)
        )
        group = c.query_heads // c.kv_heads
        for layer in root.layers:
            normed = rms(x, layer.input_norm)
            q = (normed @ layer.q.weights[0].T).view(
                total, c.query_heads, c.head_dim
            )
            k = (normed @ layer.k.weights[0].T).view(
                total, c.kv_heads, c.head_dim
            )
            v = (normed @ layer.v.weights[0].T).view(
                total, c.kv_heads, c.head_dim
            )
            q = rope(rms(q, layer.q_norm))
            k = rope(rms(k, layer.k_norm))
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)
            scores = (
                torch.einsum("qhd,khd->hqk", q.float(), k.float())
                * c.head_dim**-0.5
            )
            scores = scores.masked_fill(~causal[None], float("-inf"))
            attended = torch.einsum(
                "hqk,khd->qhd", torch.softmax(scores, dim=-1), v.float()
            ).to(x.dtype)
            x = x + attended.reshape(total, -1) @ layer.o.weights[0].T
            normed = rms(x, layer.post_norm)
            gate = normed @ layer.gate.weights[0].T
            up = normed @ layer.up.weights[0].T
            x = x + (torch.nn.functional.silu(gate) * up) @ layer.down.weights[0].T
        return rms(x, root.final_norm) @ root.lm_head.T

    admission = NewSession(
        41, 1, SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0), 42, 256
    )

    def run(step: int, operation, version: int, admit=None):
        return engine.execute(
            ExecuteBatch(
                engine_epoch=engine_ref.engine_epoch,
                step_id=step,
                acknowledged_through=step - 1,
                rows=(
                    ExecuteRow(
                        0,
                        SessionRef(engine_ref, 41, 1, version),
                        operation,
                        admit,
                        (),
                        (),
                        0,
                    ),
                ),
            )
        ).result()

    def check(sampled: int, tokens: list[int], step: int) -> None:
        """Exact argmax agreement, or a certified bf16 near-tie."""

        final = dense_oracle_greedy(tokens)[-1].float()
        oracle_top = int(final.argmax())
        if sampled == oracle_top:
            return
        gap = float(final[oracle_top] - final[sampled])
        assert gap < 0.15, (
            f"step {step}: stack sampled {sampled}, oracle argmax {oracle_top}, "
            f"logit gap {gap:.4f} is not a near-tie"
        )

    result = run(0, SequenceStep(tuple(history), 0, 0, 1), 0, admission)
    sampled = result.row_results[0].sampled_tokens[0]
    check(sampled, history, 0)
    history.append(sampled)
    for step in range(1, 8):
        result = run(
            step,
            SequenceStep((history[-1],), len(history) - 1, len(history) - 1, 1),
            step,
        )
        sampled = result.row_results[0].sampled_tokens[0]
        check(sampled, history, step)
        history.append(sampled)

    continuation = tokenizer.decode(history[len(tokenizer.encode(prompt)) :])
    # Whole-architecture coherence: the real checkpoint must answer sensibly.
    assert "Paris" in continuation, continuation
