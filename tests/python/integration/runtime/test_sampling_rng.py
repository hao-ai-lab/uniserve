"""Counter-based sampling RNG conformance for sequence operations.

A sampled token is a pure function of its semantic coordinates — the request
seed and the sequence position of the drawn token. Batch composition, draw
history, and step retries after a rollback must not change the drawn token.
"""

from __future__ import annotations

import pytest
import torch

from uniserve_worker.contracts import UniModel
from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.contracts.resource_plan import ResourcePlan
from uniserve_worker.execution import ExecutorConfig, ModelExecutor
from uniserve_worker.runtime.residency import KvCacheSpec, ResidencyManager
from uniserve_worker.runtime.resources import ResourceRuntime

pytestmark = pytest.mark.integration

_VOCAB = 64
_SEED = 11


class SpreadLogitsTextModel(UniModel):
    """Thin system-managed text model emitting one fixed, dispersed logits row.

    Every row receives the same mild logits spread, so a temperature-1
    multinomial has real dispersion and the drawn token depends only on the
    system sampler's RNG. ``fail_after_sampling`` raises from the
    post-sampling conditioning hook, failing the step after the draw already
    advanced the request's device generator.
    """

    resource_classes = ("kv_block",)
    resource_plan = ResourcePlan(kv_block="per_block")
    supported_ops = ("prefill_und", "decode_und")
    adapter_mode = "none"
    device = "cpu"
    num_layers = 1
    num_blocks = 8
    block_size = 16

    def __init__(self) -> None:
        self.fail_after_sampling: set[int] = set()

    def kv_cache_spec(self) -> KvCacheSpec:
        return KvCacheSpec(num_layers=1, num_kv_heads=1, head_dim=4, dtype=torch.float32)

    def forward_text(self, forward_batch) -> torch.Tensor:
        return torch.linspace(0.0, 4.0, _VOCAB).repeat(forward_batch.batch_size, 1)

    def maybe_publish_conditioning(self, req_id: int, sampled_token_id: int) -> str | None:
        if int(req_id) in self.fail_after_sampling:
            raise RuntimeError("injected failure after the draw")
        return None


def _fresh() -> tuple[SpreadLogitsTextModel, ModelExecutor]:
    ledger = ResourceRuntime(("kv_block",), totals={"kv_block": 8})
    model = SpreadLogitsTextModel()
    residency = ResidencyManager.build(
        model.kv_cache_spec(),
        num_blocks=8,
        block_size=16,
        device="cpu",
        ledger=ledger,
    )
    executor = ModelExecutor(
        model,
        config=ExecutorConfig(simulation=True),
        resource_runtime=ledger,
        residency=residency,
    )
    return model, executor


def _prefill_batch(step: int, *, req_id: int, block: int, tokens: list[int], seed: int):
    return seal_batch(
        step,
        [
            {
                "req_id": req_id,
                "kind": "prefill_und",
                "token_ids": tokens,
                "pos_range": [0, len(tokens)],
            }
        ],
        new_reqs=[
            {
                "req_id": req_id,
                "block_ids": [block],
                "sampling": {"temperature": 1.0, "seed": seed},
            }
        ],
    )


def _decode_batch(step: int, *, req_id: int, token: int, pos: int):
    return seal_batch(
        step,
        [
            {
                "req_id": req_id,
                "kind": "decode_und",
                "token_ids": [token],
                "pos_range": [pos, pos + 1],
            }
        ],
        base_version=1,
    )


def _sampled(result: dict) -> list[tuple[int, int]]:
    return [(int(row["req_id"]), int(row["sampled_token_id"])) for row in result["per_seq"]]


def test_retry_after_a_post_draw_rollback_redraws_the_same_token():
    model, executor = _fresh()
    _, control = _fresh()
    prefill = _prefill_batch(1, req_id=1, block=0, tokens=[5, 6, 7], seed=_SEED)
    first_token = _sampled(executor.execute(prefill))[0][1]
    control.execute(prefill)

    # The failed attempt samples (the draw happens), then the step fails and
    # rolls back. The retried operation must redraw the identical token even
    # though the failed attempt already consumed randomness.
    model.fail_after_sampling = {1}
    with pytest.raises(RuntimeError, match="injected failure after the draw"):
        executor.execute(_decode_batch(2, req_id=1, token=first_token, pos=3))
    model.fail_after_sampling = set()

    retry = executor.execute(_decode_batch(3, req_id=1, token=first_token, pos=3))
    reference = control.execute(_decode_batch(2, req_id=1, token=first_token, pos=3))
    assert _sampled(retry) == _sampled(reference)


def test_batch_composition_does_not_change_the_drawn_token():
    _, solo_executor = _fresh()
    solo = solo_executor.execute(_prefill_batch(1, req_id=1, block=0, tokens=[5, 6, 7], seed=_SEED))

    _, batched_executor = _fresh()
    batched = batched_executor.execute(
        seal_batch(
            1,
            [
                {"req_id": 2, "kind": "prefill_und", "token_ids": [4, 9, 8], "pos_range": [0, 3]},
                {"req_id": 1, "kind": "prefill_und", "token_ids": [5, 6, 7], "pos_range": [0, 3]},
            ],
            new_reqs=[
                {"req_id": 2, "block_ids": [1], "sampling": {"temperature": 1.0, "seed": 77}},
                {"req_id": 1, "block_ids": [0], "sampling": {"temperature": 1.0, "seed": _SEED}},
            ],
        )
    )

    assert dict(_sampled(solo))[1] == dict(_sampled(batched))[1]
    # Distinct seeds draw independently within one batch.
    assert dict(_sampled(batched))[2] != dict(_sampled(batched))[1]


def test_draw_history_does_not_change_the_token_drawn_at_a_position():
    # Route A: the position-4 token is the request's second draw (prefill then
    # decode). Route B: it is the first draw (one longer prefill). The token
    # depends only on (seed, position), never on how many draws came before.
    _, step_executor = _fresh()
    prefill = step_executor.execute(_prefill_batch(1, req_id=1, block=0, tokens=[5, 6, 7], seed=_SEED))
    token_at_3 = _sampled(prefill)[0][1]
    decode = step_executor.execute(_decode_batch(2, req_id=1, token=token_at_3, pos=3))
    token_at_4 = _sampled(decode)[0][1]

    _, direct_executor = _fresh()
    direct = direct_executor.execute(
        _prefill_batch(1, req_id=1, block=0, tokens=[5, 6, 7, 9], seed=_SEED)
    )

    assert _sampled(direct)[0][1] == token_at_4
    # Neighboring positions draw from decorrelated coordinates.
    assert token_at_3 != token_at_4
