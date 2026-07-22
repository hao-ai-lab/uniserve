"""Sequence-KV transaction conformance for interleaved execution.

A failed operation must leave every touched request's committed KV length and
block table unchanged, and retrying the same operations must produce the same
tokens and committed state as an uninterrupted run.
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
from uniserve_worker.runtime.residency import KvCacheSpec, ResidencyManager
from uniserve_worker.runtime.resources import ResourceRuntime

pytestmark = pytest.mark.integration

_VOCAB = 16


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
    block_size = 16
    eos_id = 2
    img_start_id = 3

    def __init__(self, residency: ResidencyManager) -> None:
        self.residency = residency
        self.segment_executor = SimpleNamespace(release_staging=lambda cache: None)
        self._driver = SequenceExecutor(self, image_start_token="<img>")
        self.fail_requests: set[int] = set()
        self.fail_mid_layer_requests: set[int] = set()
        self._fail_between_layer_writes = False

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
            if self._fail_between_layer_writes and layer_idx == 1:
                raise RuntimeError("injected KV failure between layer writes")
            past_key_values.update(kv, kv, layer_idx)
        logits = torch.zeros(1, count, _VOCAB)
        for offset in range(count):
            token = int(tokens[0, offset])
            logits[0, offset, (token * 7 + base_len + offset) % _VOCAB] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    def forward(self, batch: Any) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for op in batch.ops:
            req_id = int(op["req_id"])
            self._fail_between_layer_writes = req_id in self.fail_mid_layer_requests
            try:
                logits = self._driver.run_text_logits(dict(op))
            finally:
                self._fail_between_layer_writes = False
            if req_id in self.fail_requests:
                raise RuntimeError("injected KV failure after forward write")
            outputs.append(
                {
                    "req_id": req_id,
                    "sampled_token_id": int(torch.argmax(logits.reshape(-1)).item()),
                }
            )
        return outputs


def _fresh() -> tuple[InterleavedSequenceCPUModel, ModelExecutor]:
    ledger = ResourceRuntime(("kv_block",), totals={"kv_block": 8})
    residency = ResidencyManager.build(
        KvCacheSpec(num_layers=2, num_kv_heads=1, head_dim=4, dtype=torch.float32),
        num_blocks=8,
        block_size=16,
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
    assert executor.flow_store is not None
    return model, executor


def _prefill_batch(step_id: int) -> dict[str, Any]:
    return seal_batch(
        step_id,
        [
            {"req_id": 1, "kind": "prefill_und", "token_ids": [5, 6, 7], "pos_range": [0, 3]},
            {"req_id": 2, "kind": "prefill_und", "token_ids": [4, 9, 8], "pos_range": [0, 3]},
        ],
        new_reqs=[
            {"req_id": 1, "block_ids": [0], "sampling": {"temperature": 0.0}},
            {"req_id": 2, "block_ids": [1], "sampling": {"temperature": 0.0}},
        ],
    )


def _decode_batch(step_id: int) -> dict[str, Any]:
    return seal_batch(
        step_id,
        [
            {
                "req_id": 1,
                "kind": "decode_und",
                "token_ids": [8],
                "pos_range": [3, 4],
                "new_block_ids": [2],
            },
            {
                "req_id": 2,
                "kind": "decode_und",
                "token_ids": [11],
                "pos_range": [3, 4],
                "new_block_ids": [3],
            },
        ],
        base_version=1,
    )


def _committed_text_state(executor: ModelExecutor, req_id: int) -> dict[str, Any]:
    assert executor.flow_store is not None
    program = executor.flow_store.program(req_id)
    session = executor.sessions.get(req_id)
    past = program.cond.past
    return {
        "length": int(past.length) if past is not None else None,
        "cache_blocks": list(past.block_ids) if past is not None else None,
        "t_index": int(program.cond.t_index),
        "session_blocks": list(session.block_ids),
        "ref_len": int(session.prefix_len),
        "kv_lengths": dict(session.kv_lengths),
        "version": int(session.version),
    }


def _sampled(result: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(row["req_id"]), int(row["sampled_token_id"])) for row in result["per_seq"]]


def test_failure_after_forward_kv_write_leaves_committed_length_and_blocks_unchanged():
    model, executor = _fresh()
    _, control = _fresh()

    executor.execute(_prefill_batch(1))
    control.execute(_prefill_batch(1))
    committed = {req_id: _committed_text_state(executor, req_id) for req_id in (1, 2)}
    assert committed[1]["length"] == 3
    assert committed[1]["session_blocks"] == [0]
    assert committed[2]["cache_blocks"] == [1]

    # Request 1's forward succeeds and advances its cache; request 2 fails after
    # its KV span is fully written but before the step commits.
    model.fail_requests = {2}
    with pytest.raises(RuntimeError, match="injected KV failure after forward write"):
        executor.execute(_decode_batch(2))
    model.fail_requests = set()

    for req_id in (1, 2):
        assert _committed_text_state(executor, req_id) == committed[req_id]
    assert executor.resource_runtime.used("kv_block") == 2

    retry = executor.execute(_decode_batch(3))
    reference = control.execute(_decode_batch(2))
    assert _sampled(retry) == _sampled(reference)
    for req_id in (1, 2):
        assert _committed_text_state(executor, req_id) == _committed_text_state(control, req_id)
    after = _committed_text_state(executor, 1)
    assert after["length"] == 4
    assert after["session_blocks"] == [0, 2]
    assert after["cache_blocks"] == [0, 2]
    assert executor.resource_runtime.used("kv_block") == 4


def test_failure_injection_preserves_prefix_reference_blocks_and_lengths():
    model, executor = _fresh()
    _, control = _fresh()

    # A warm admission: block 4 carries a reused prefix, the suffix prefill
    # starts at the declared boundary and writes into the leased tail block.
    warm_ops = [
        {"req_id": 4, "kind": "prefill_und", "token_ids": [5, 6, 7], "pos_range": [16, 19]}
    ]
    warm_new_reqs = [
        {
            "req_id": 4,
            "block_ids": [4, 5],
            "prefix_len": 16,
            "sampling": {"temperature": 0.0},
        }
    ]
    executor.execute(seal_batch(1, warm_ops, new_reqs=warm_new_reqs))
    control.execute(seal_batch(1, warm_ops, new_reqs=warm_new_reqs))
    committed = _committed_text_state(executor, 4)
    assert committed["ref_len"] == 16
    assert committed["session_blocks"] == [4, 5]

    decode = [{"req_id": 4, "kind": "decode_und", "token_ids": [9], "pos_range": [19, 20]}]
    model.fail_requests = {4}
    with pytest.raises(RuntimeError, match="injected KV failure after forward write"):
        executor.execute(seal_batch(2, decode, base_version=1))
    model.fail_requests = set()

    # The failed step left the prefix reference, block table, and committed
    # lane lengths exactly at the prior commit.
    assert _committed_text_state(executor, 4) == committed

    retry = executor.execute(seal_batch(3, decode, base_version=1))
    reference = control.execute(seal_batch(2, decode, base_version=1))
    assert _sampled(retry) == _sampled(reference)
    assert _committed_text_state(executor, 4) == _committed_text_state(control, 4)


def test_failure_between_layer_writes_rolls_back_partial_span_and_retry_matches():
    model, executor = _fresh()
    _, control = _fresh()

    executor.execute(_prefill_batch(1))
    control.execute(_prefill_batch(1))
    committed = {req_id: _committed_text_state(executor, req_id) for req_id in (1, 2)}

    model.fail_mid_layer_requests = {1}
    with pytest.raises(RuntimeError, match="between layer writes"):
        executor.execute(_decode_batch(2))
    model.fail_mid_layer_requests = set()

    for req_id in (1, 2):
        assert _committed_text_state(executor, req_id) == committed[req_id]

    retry = executor.execute(_decode_batch(3))
    reference = control.execute(_decode_batch(2))
    assert _sampled(retry) == _sampled(reference)
    for req_id in (1, 2):
        assert _committed_text_state(executor, req_id) == _committed_text_state(control, req_id)
