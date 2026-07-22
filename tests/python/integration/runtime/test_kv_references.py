"""Sequence-KV reference conformance for prefix sharing and reclamation.

The runtime ``KvStore`` registers each request's prefix-reference boundary at
admission, guards every declared KV write range against the boundary and the
leased block capacity before any KV mutation, admits multiple read-only
holders of shared prefix blocks while rejecting a lease of a block another
live request may write, hands stale registrations over to new leases, and
reclaims block references when a request is dropped.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from uniserve_worker.contracts import BatchPolicy, UniModel
from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.contracts.forward_context import get_forward_context
from uniserve_worker.contracts.resource_plan import ResourcePlan
from uniserve_worker.execution import ExecutorConfig, ModelExecutor
from uniserve_worker.execution.sequence import SequenceExecutor
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.request_session import SessionStore
from uniserve_worker.runtime.residency import KvCacheSpec, ResidencyManager
from uniserve_worker.runtime.resources import ResourceRuntime

pytestmark = pytest.mark.integration

_VOCAB = 16
_BLOCK_SIZE = 4


def _sessions(monkeypatch: pytest.MonkeyPatch | None = None) -> SessionStore:
    if monkeypatch is not None:
        monkeypatch.setenv("UNISERVE_KV_GUARDS", "1")
    sessions = SessionStore()
    sessions.kv.bind_block_size(_BLOCK_SIZE)
    return sessions


# ---------------------------------------------------------------------------
# Prefix-reference boundary and write-range guarding
# ---------------------------------------------------------------------------


def test_write_below_the_prefix_reference_fails_before_any_state_change():
    sessions = _sessions()
    sessions.create_or_update(7, {"req_id": 7, "block_ids": [0, 1], "prefix_len": 4})
    state = sessions.get(7)
    blocks_before = list(state.block_ids)
    lengths_before = dict(state.kv_lengths)

    below_boundary = {
        "req_id": 7,
        "kind": "prefill_und",
        "token_ids": [5, 6],
        "pos_range": [2, 4],
        "new_block_ids": [2],
    }
    with pytest.raises(WorkerError, match="prefix reference boundary"):
        sessions.resolve_text_row(7, below_boundary)

    # The rejected write ingested nothing: even its new-block lease is untouched.
    assert state.block_ids == blocks_before
    assert state.kv_lengths == lengths_before
    assert state.prefix_len == 4

    # The suffix write at the boundary resolves against the leased chain.
    row = sessions.resolve_text_row(
        7,
        {"req_id": 7, "kind": "prefill_und", "token_ids": [5, 6], "pos_range": [4, 6]},
    )
    assert row.block_ids == (0, 1)
    assert row.base_len == 4


def test_write_beyond_the_leased_block_capacity_is_rejected():
    sessions = _sessions()
    sessions.create_or_update(7, {"req_id": 7, "block_ids": [0]})

    with pytest.raises(WorkerError, match="beyond its"):
        sessions.resolve_text_row(
            7,
            {"req_id": 7, "kind": "decode_und", "token_ids": [1], "pos_range": [4, 5]},
        )

    # The same write is legal once the lease grows to cover it.
    row = sessions.resolve_text_row(
        7,
        {
            "req_id": 7,
            "kind": "decode_und",
            "token_ids": [1],
            "pos_range": [4, 5],
            "new_block_ids": [1],
        },
    )
    assert row.block_ids == (0, 1)


# ---------------------------------------------------------------------------
# Multi-holder prefix sharing and the copy-on-write premise
# ---------------------------------------------------------------------------


def test_shared_prefix_blocks_admit_multiple_read_only_holders(monkeypatch):
    sessions = _sessions(monkeypatch)
    # The prefix writer completed and was dropped; its block stays reusable.
    sessions.create_or_update(1, {"req_id": 1, "block_ids": [0]})
    sessions.drop(1)

    sessions.create_or_update(2, {"req_id": 2, "block_ids": [0, 1], "prefix_len": 4})
    sessions.create_or_update(3, {"req_id": 3, "block_ids": [0, 2], "prefix_len": 4})

    # Both live holders write their own suffix blocks with guards enabled.
    for req_id in (2, 3):
        row = sessions.resolve_text_row(
            req_id,
            {"req_id": req_id, "kind": "prefill_und", "token_ids": [8, 9], "pos_range": [4, 6]},
        )
        assert row.base_len == 4


def test_leasing_a_block_writable_by_a_live_holder_is_rejected():
    sessions = _sessions()
    # A cold request's whole table is its writable region.
    sessions.create_or_update(1, {"req_id": 1, "block_ids": [0, 1]})

    with pytest.raises(WorkerError, match="writable by live request"):
        sessions.create_or_update(2, {"req_id": 2, "block_ids": [0, 3], "prefix_len": 4})


def test_shared_prefix_requires_a_block_aligned_boundary():
    sessions = _sessions()
    # A boundary inside block 1 leaves that block writable by its holder, so
    # the premise that makes copy-on-write unnecessary excludes sharing it.
    sessions.create_or_update(2, {"req_id": 2, "block_ids": [0, 1, 2], "prefix_len": 6})

    with pytest.raises(WorkerError, match="writable by live request"):
        sessions.create_or_update(3, {"req_id": 3, "block_ids": [1, 4], "prefix_len": 4})

    # Blocks wholly below the holder's boundary remain sharable.
    sessions.create_or_update(4, {"req_id": 4, "block_ids": [0, 5], "prefix_len": 4})
    assert sessions.get(4).block_ids == [0, 5]


def test_stale_holder_registrations_are_handed_over_silently(monkeypatch):
    sessions = _sessions(monkeypatch)
    # Absent holder: the request was dropped without a new lease arriving.
    sessions.create_or_update(1, {"req_id": 1, "block_ids": [0]})
    sessions.drop(1)
    sessions.create_or_update(2, {"req_id": 2, "block_ids": [0]})

    # Terminal holder: the request committed but still sits in the table.
    sessions.create_or_update(5, {"req_id": 5, "block_ids": [6]})
    sessions.finish_generation(5, committed=True)
    sessions.create_or_update(6, {"req_id": 6, "block_ids": [6]})

    # The new holders own the blocks exclusively: guarded writes succeed.
    for req_id, block in ((2, 0), (6, 6)):
        row = sessions.resolve_text_row(
            req_id,
            {"req_id": req_id, "kind": "prefill_und", "token_ids": [1, 2], "pos_range": [0, 2]},
        )
        assert row.block_ids == (block,)


# ---------------------------------------------------------------------------
# Executor conformance: reclamation, guarded failure, and concurrency
# ---------------------------------------------------------------------------


class InterleavedSequenceCPUModel(UniModel):
    """Interleaved sequence model driving the shared paged text executor on CPU.

    ``sequence_forward`` appends K/V for every layer through the paged text
    cache and emits logits whose argmax encodes (token, cache length), so the
    sampled token observably depends on the committed KV length the forward
    started from.
    """

    resource_classes = ("kv_block",)
    resource_plan = ResourcePlan(kv_block="per_block")
    supported_ops = ("prefill_und", "decode_und")
    adapter_mode = "none"
    device = "cpu"
    num_layers = 2
    num_blocks = 8
    block_size = _BLOCK_SIZE
    eos_id = 2
    img_start_id = 3

    def __init__(self, residency: ResidencyManager) -> None:
        self.residency = residency
        self.segment_executor = SimpleNamespace(release_staging=lambda cache: None)
        self._driver = SequenceExecutor(self, image_start_token="<img>")

    @property
    def kv_pool(self):
        return self.residency.kv

    @property
    def scratch_pool(self):
        return self.residency.scratch

    def batch_policy(self) -> BatchPolicy:
        return BatchPolicy(max_batch_ops=8, supports_mixed_modes=True)

    def program_state(self, req_id: int):
        kv_view = get_forward_context().kv_view
        assert kv_view is not None
        return kv_view.program(int(req_id))

    def sequence_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(input_ids, num_classes=_VOCAB).float()

    def sequence_forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        indexes: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: Any = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        text_only_rope: bool = False,
        causal_paged_update: bool = False,
        return_all_logits: bool = False,
    ) -> Any:
        assert input_ids is not None and past_key_values is not None
        tokens = input_ids.reshape(1, -1)
        count = int(tokens.shape[1])
        base_len = int(past_key_values.get_seq_length())
        kv = torch.zeros(1, 1, count, 4)
        for layer_idx in range(self.num_layers):
            past_key_values.update(kv, kv, layer_idx)
        logits = torch.zeros(1, count, _VOCAB)
        for offset in range(count):
            token = int(tokens[0, offset])
            logits[0, offset, (token * 7 + base_len + offset) % _VOCAB] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    def forward(self, batch: Any) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for op in batch.ops:
            logits = self._driver.run_text_logits(dict(op))
            outputs.append(
                {
                    "req_id": int(op["req_id"]),
                    "sampled_token_id": int(torch.argmax(logits.reshape(-1)).item()),
                }
            )
        return outputs


def _fresh() -> tuple[InterleavedSequenceCPUModel, ModelExecutor]:
    ledger = ResourceRuntime(("kv_block",), totals={"kv_block": 8})
    residency = ResidencyManager.build(
        KvCacheSpec(num_layers=2, num_kv_heads=1, head_dim=4, dtype=torch.float32),
        num_blocks=8,
        block_size=_BLOCK_SIZE,
        device="cpu",
        ledger=ledger,
    )
    model = InterleavedSequenceCPUModel(residency)
    executor = ModelExecutor(
        model,
        config=ExecutorConfig(simulation=True),
        resource_runtime=ledger,
        residency=residency,
    )
    return model, executor


def _seed_prefix(executor: ModelExecutor, step_id: int) -> None:
    """Write one full prefix block through a request that then completes."""
    executor.execute(
        seal_batch(
            step_id,
            [
                {
                    "req_id": 9,
                    "kind": "prefill_und",
                    "token_ids": [5, 6, 7, 8],
                    "pos_range": [0, 4],
                }
            ],
            new_reqs=[{"req_id": 9, "block_ids": [0], "sampling": {"temperature": 0.0}}],
        )
    )
    executor.drop_request(9)


def _warm_new_req(req_id: int, tail_block: int) -> dict[str, Any]:
    return {
        "req_id": req_id,
        "block_ids": [0, tail_block],
        "prefix_len": 4,
        "sampling": {"temperature": 0.0},
    }


def _warm_ops(req_id: int, *, base_version: int, tail_block: int) -> list[dict[str, Any]]:
    return [
        {
            "req_id": req_id,
            "kind": "prefill_und",
            "token_ids": [1, req_id % _VOCAB],
            "pos_range": [4, 6],
            "base_version": base_version,
        },
        {
            "req_id": req_id,
            "kind": "decode_und",
            "token_ids": [11],
            "pos_range": [6, 7],
            "base_version": base_version + 1,
        },
        {
            "req_id": req_id,
            "kind": "decode_und",
            "token_ids": [12],
            "pos_range": [7, 8],
            "base_version": base_version + 2,
        },
        {
            "req_id": req_id,
            "kind": "decode_und",
            "token_ids": [13],
            "pos_range": [8, 9],
            "new_block_ids": [tail_block + 2],
            "base_version": base_version + 3,
        },
    ]


def _sampled(result: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(row["req_id"]), int(row["sampled_token_id"])) for row in result["per_seq"]]


def test_drop_request_reclaims_block_references(monkeypatch):
    monkeypatch.setenv("UNISERVE_KV_GUARDS", "1")
    _, executor = _fresh()
    _seed_prefix(executor, 1)
    executor.execute(
        seal_batch(
            2,
            [_warm_ops(2, base_version=0, tail_block=1)[0]],
            new_reqs=[_warm_new_req(2, 1)],
        )
    )

    executor.drop_request(2)

    # A cold request re-leasing the dropped request's blocks admits and
    # writes them cleanly with guards enabled.
    result = executor.execute(
        seal_batch(
            3,
            [
                {
                    "req_id": 4,
                    "kind": "prefill_und",
                    "token_ids": [5, 6, 7, 8, 9],
                    "pos_range": [0, 5],
                }
            ],
            new_reqs=[{"req_id": 4, "block_ids": [0, 1], "sampling": {"temperature": 0.0}}],
        )
    )
    assert len(result["per_seq"]) == 1
    assert 2 not in executor.sessions


def test_executor_write_below_prefix_reference_fails_and_rolls_back(monkeypatch):
    monkeypatch.setenv("UNISERVE_KV_GUARDS", "1")
    _, executor = _fresh()
    _, control = _fresh()
    for target in (executor, control):
        _seed_prefix(target, 1)

    # The op re-writes the shared prefix span instead of starting at the
    # declared boundary: a typed failure before any KV write.
    bad = seal_batch(
        2,
        [
            {
                "req_id": 2,
                "kind": "prefill_und",
                "token_ids": [9, 9, 9, 9, 1, 2],
                "pos_range": [0, 6],
            }
        ],
        new_reqs=[_warm_new_req(2, 1)],
    )
    with pytest.raises(WorkerError, match="prefix reference boundary"):
        executor.execute(bad)
    assert 2 not in executor.sessions

    # The suffix admission retries cleanly and matches an executor that never
    # saw the failure.
    steps = _warm_ops(2, base_version=0, tail_block=1)
    outputs = [_sampled(executor.execute(seal_batch(3 + i, [op], new_reqs=[_warm_new_req(2, 1)] if i == 0 else ()))) for i, op in enumerate(steps)]
    control_outputs = [_sampled(control.execute(seal_batch(3 + i, [op], new_reqs=[_warm_new_req(2, 1)] if i == 0 else ()))) for i, op in enumerate(steps)]
    assert outputs == control_outputs


def test_interleaved_sessions_sharing_a_prefix_match_isolated_runs(monkeypatch):
    monkeypatch.setenv("UNISERVE_KV_GUARDS", "1")

    # Interleaved: requests 2 and 3 share prefix block 0 and execute their
    # suffix prefill and decode ops co-batched step by step.
    _, interleaved = _fresh()
    _seed_prefix(interleaved, 1)
    ops_2 = _warm_ops(2, base_version=0, tail_block=1)
    ops_3 = _warm_ops(3, base_version=0, tail_block=2)
    shared: dict[int, list[tuple[int, int]]] = {2: [], 3: []}
    for index, (op_2, op_3) in enumerate(zip(ops_2, ops_3, strict=True)):
        new_reqs = [_warm_new_req(2, 1), _warm_new_req(3, 2)] if index == 0 else ()
        result = interleaved.execute(seal_batch(2 + index, [op_2, op_3], new_reqs=new_reqs))
        for req_id, token in _sampled(result):
            shared[req_id].append((req_id, token))

    # Isolated: each request runs alone against the same seeded prefix.
    isolated: dict[int, list[tuple[int, int]]] = {}
    for req_id, ops, tail_block in ((2, ops_2, 1), (3, ops_3, 2)):
        _, executor = _fresh()
        _seed_prefix(executor, 1)
        outputs: list[tuple[int, int]] = []
        for index, op in enumerate(ops):
            new_reqs = [_warm_new_req(req_id, tail_block)] if index == 0 else ()
            result = executor.execute(seal_batch(2 + index, [op], new_reqs=new_reqs))
            outputs.extend(_sampled(result))
        isolated[req_id] = outputs

    # Guards stayed enabled throughout: zero shared-block writes were
    # recorded, and interleaving did not change either session's tokens.
    assert shared[2] == isolated[2]
    assert shared[3] == isolated[3]
