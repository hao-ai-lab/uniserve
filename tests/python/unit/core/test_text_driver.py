"""Unit tests for the system text driver's sampler + KV-length advance."""
from __future__ import annotations

import torch

import uniserve_worker.execution.text_driver as text_driver_mod
from uniserve_worker.contracts.batches import UniForwardBatch
from uniserve_worker.execution.text_driver import TextDriver
from uniserve_worker.nn.sampler import BatchedSamplingResult, DeferredBatchedSamplingResult
from uniserve_worker.runtime.request_state import RequestStateTable


def _greedy_sampler(captured):
    def fake_sample(batch, sampling_params, recent, allowed, suppress, *, defer_cpu=False):
        del sampling_params, recent, allowed, suppress, defer_cpu
        captured.append(batch)
        tokens = torch.argmax(batch, dim=-1)
        return BatchedSamplingResult(
            samples=[(int(token), None, None) for token in tokens.tolist()],
            device_tokens=tokens,
        )

    return fake_sample


def test_sample_logits_batch_passes_original_logits_to_sampler(monkeypatch):
    logits = torch.tensor([[0.0, 4.0, 1.0], [3.0, 0.5, 2.0]], dtype=torch.float32)
    states = RequestStateTable()
    states.get(1).sampling = {"temperature": 0.0}
    states.get(2).sampling = {"temperature": 0.0}
    captured: list[torch.Tensor] = []
    monkeypatch.setattr(text_driver_mod, "apply_sampling_batched_with_device_tokens", _greedy_sampler(captured))

    out = TextDriver()._sample_logits_batch(
        [{"req_id": 1}, {"req_id": 2}],
        [1, 2],
        logits,
        states,
        None,
        0,
    )

    assert [row.sampled_token_id for row in out] == [1, 0]
    assert captured
    torch.testing.assert_close(captured[0], logits)
    assert captured[0].data_ptr() == logits.data_ptr()


def test_sample_logits_batch_can_defer_cpu_token_materialization(monkeypatch):
    logits = torch.tensor([[0.0, 4.0, 1.0]], dtype=torch.float32)
    states = RequestStateTable()
    states.get(1).sampling = {"temperature": 0.0}

    def fake_sample(batch, sampling_params, recent, allowed, suppress, *, defer_cpu=False):
        del sampling_params, recent, allowed, suppress
        assert defer_cpu is True
        tokens = torch.argmax(batch, dim=-1)
        return DeferredBatchedSamplingResult(
            tokens_cpu=tokens.detach().to("cpu"),
            device_tokens=tokens,
            copy_event=None,
        )

    monkeypatch.setattr(text_driver_mod, "apply_sampling_batched_with_device_tokens", fake_sample)

    out = TextDriver()._sample_logits_batch(
        [{"req_id": 1}], [1], logits, states, None, 0, defer_cpu_results=True
    )

    assert states.get(1).decode_relay.token_id is None
    assert out[0].finalize()["sampled_token_id"] == 1
    assert states.get(1).decode_relay.token_id == 1


def test_deferred_finalize_does_not_overwrite_newer_token_relay(monkeypatch):
    states = RequestStateTable()
    states.get(1).sampling = {"temperature": 0.0}
    calls = 0

    def fake_sample(batch, sampling_params, recent, allowed, suppress, *, defer_cpu=False):
        nonlocal calls
        del sampling_params, recent, allowed, suppress
        calls += 1
        tokens = torch.argmax(batch, dim=-1)
        if calls == 1:
            assert defer_cpu is True
            return DeferredBatchedSamplingResult(
                tokens_cpu=tokens.detach().to("cpu"),
                device_tokens=tokens,
                copy_event=None,
            )
        return BatchedSamplingResult(
            samples=[(int(token), None, None) for token in tokens.tolist()],
            device_tokens=tokens,
        )

    monkeypatch.setattr(text_driver_mod, "apply_sampling_batched_with_device_tokens", fake_sample)

    driver = TextDriver()
    first = driver._sample_logits_batch(
        [{"req_id": 1}], [1], torch.tensor([[0.0, 4.0, 1.0]]), states, None, 0, defer_cpu_results=True
    )
    driver._sample_logits_batch(
        [{"req_id": 1}], [1], torch.tensor([[0.0, 1.0, 4.0]]), states, None, 0
    )

    assert states.get(1).decode_relay.token_id == 2
    assert first[0].finalize()["sampled_token_id"] == 1
    assert states.get(1).decode_relay.token_id == 2


def test_advance_kv_lengths_records_per_request_text_lane():
    states = RequestStateTable()
    text = UniForwardBatch.from_ops(
        [
            {"req_id": 1, "kind": "decode_und", "token_ids": [5], "pos_range": [7, 8]},
            {"req_id": 2, "kind": "prefill_und", "token_ids": [11, 12], "pos_range": [0, 2]},
        ]
    ).as_text(allow_mixed_text=True)

    TextDriver()._advance_kv_lengths(text, states)

    # KV-length advance is system-owned now (lane-keyed bookkeeping), derived
    # from each op's pos_range end.
    assert states.get(1).kv_lengths["text"] == 8
    assert states.get(2).kv_lengths["text"] == 2
