"""Pipelined decode-burst semantics for self-managing text models.

The burst launches step k+1 from the device sampled-token relay before step
k's CPU token is read; these tests pin the observable contract: identical
token sequences to the sequential loop, ``last_sampled`` relay ops after the
first step, and exactly one speculative forward (whose input is the stop
token) when a stop token ends the burst early.
"""

from __future__ import annotations

import torch

from uniserve_worker.contracts import UniModel
from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.execution import ExecutorConfig, ModelExecutor
from uniserve_worker.execution.sequence import resolve_op_token_ids


class _ScriptedTextModel(UniModel):
    """Self-managed text model whose argmax follows a scripted token sequence."""

    resource_classes: tuple[str, ...] = ()
    device = "cpu"

    def __init__(self, script: list[int], vocab: int = 16) -> None:
        self.script = list(script)
        self.vocab = int(vocab)
        self.seen_tokens: list[list[int]] = []
        self.seen_sources: list[str] = []

    def run_text_logits_batch(self, ops):
        out = []
        for op in ops:
            self.seen_tokens.append([int(t) for t in resolve_op_token_ids(op)])
            self.seen_sources.append(str(op.get("token_source") or "wire"))
            step = len(self.seen_tokens) - 1
            token = self.script[min(step, len(self.script) - 1)]
            logits = torch.zeros(self.vocab)
            logits[token] = 5.0
            out.append(logits)
        return out


def _burst_op(count: int, stop_ids: list[int]) -> dict:
    return {
        "req_id": 4,
        "kind": "decode_und",
        "token_ids": [11],
        "pos_range": [7, 8],
        "decode_token_count": count,
        "decode_stop_token_ids": stop_ids,
    }


def _run_burst(model: _ScriptedTextModel, op: dict) -> dict:
    runner = ModelExecutor(model, config=ExecutorConfig(simulation=True))
    result = runner.execute(
        seal_batch(
            1,
            [op],
            new_reqs=[{"req_id": 4, "sampling": {"temperature": 0.0}}],
        )
    )
    return result["per_seq"][0]


def test_burst_without_stop_matches_sequential_tokens_and_forward_count():
    model = _ScriptedTextModel([5, 6, 8])
    seq = _run_burst(model, _burst_op(3, stop_ids=[7]))

    assert seq["sampled_token_ids"] == [5, 6, 8]
    assert seq["sampled_token_id"] == 8
    # Exactly count forwards: exhausting the count launches nothing speculative.
    assert model.seen_tokens == [[11], [5], [6]]
    assert model.seen_sources == ["wire", "last_sampled", "last_sampled"]


def test_burst_stop_token_ends_sequence_and_speculative_forward_conditions_on_it():
    model = _ScriptedTextModel([5, 7, 9, 9])
    seq = _run_burst(model, _burst_op(4, stop_ids=[7]))

    # The stop token ends the reported sequence exactly as the sequential loop.
    assert seq["sampled_token_ids"] == [5, 7]
    assert seq["sampled_token_id"] == 7
    # One forward ran speculatively past the stop; its input is the stop token
    # itself (the same conditioning append the non-burst flow performs next),
    # and no token from it is reported.
    assert model.seen_tokens == [[11], [5], [7]]
    assert model.seen_sources == ["wire", "last_sampled", "last_sampled"]


def test_terminal_burst_stop_truncates_result_after_full_speculative_burst():
    model = _ScriptedTextModel([5, 7, 9, 10])
    op = _burst_op(4, stop_ids=[7])
    op["decode_stop_terminal"] = True
    seq = _run_burst(model, op)

    assert seq["sampled_token_ids"] == [5, 7]
    assert seq["sampled_token_id"] == 7
    assert model.seen_tokens == [[11], [5], [7], [9]]
    assert model.seen_sources == ["wire", "last_sampled", "last_sampled", "last_sampled"]


def test_burst_relay_ops_carry_device_relay_tensor():
    captured: list[dict] = []

    class _CapturingModel(_ScriptedTextModel):
        def run_text_logits_batch(self, ops):
            captured.extend(dict(op) for op in ops)
            return super().run_text_logits_batch(ops)

    model = _CapturingModel([3, 4, 5])
    _run_burst(model, _burst_op(3, stop_ids=[]))

    relay_ops = [op for op in captured if op.get("token_source") == "last_sampled"]
    assert len(relay_ops) == 2
    for op in relay_ops:
        tensor = op.get("token_tensor")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert int(tensor.numel()) == 1


def test_relay_started_burst_continues_from_the_previous_burst_tail():
    model = _ScriptedTextModel([5, 6, 8, 9, 10])
    runner = ModelExecutor(model, config=ExecutorConfig(simulation=True))
    first = runner.execute(
        seal_batch(
            1,
            [_burst_op(3, stop_ids=[])],
            new_reqs=[{"req_id": 4, "sampling": {"temperature": 0.0}}],
        )
    )["per_seq"][0]
    second_op = _burst_op(2, stop_ids=[])
    second_op["token_source"] = "last_sampled"
    second_op["token_ids"] = [0]
    second_op["pos_range"] = [10, 11]
    second = runner.execute(seal_batch(2, [second_op], base_version=1))["per_seq"][0]

    assert first["sampled_token_ids"] == [5, 6, 8]
    assert second["sampled_token_ids"] == [9, 10]
    assert model.seen_tokens == [[11], [5], [6], [8], [9]]
    assert model.seen_sources == [
        "wire",
        "last_sampled",
        "last_sampled",
        "last_sampled",
        "last_sampled",
    ]


def test_multi_row_burst_returns_token_lists_for_each_row():
    model = _ScriptedTextModel([5, 6, 7, 8])
    runner = ModelExecutor(model, config=ExecutorConfig(simulation=True))
    result = runner.execute(
        seal_batch(
            1,
            [
                _burst_op(2, stop_ids=[]),
                {
                    "req_id": 5,
                    "kind": "decode_und",
                    "token_ids": [21],
                    "pos_range": [3, 4],
                    "decode_token_count": 2,
                    "decode_stop_token_ids": [],
                },
            ],
            new_reqs=[
                {"req_id": 4, "sampling": {"temperature": 0.0}},
                {"req_id": 5, "sampling": {"temperature": 0.0}},
            ],
        )
    )

    assert result["per_seq"][0]["sampled_token_ids"] == [5, 7]
    assert result["per_seq"][1]["sampled_token_ids"] == [6, 8]
    assert model.seen_tokens == [[11], [21], [5], [6]]
