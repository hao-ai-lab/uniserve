from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from uniserve_worker.backends.attention import AttentionCapabilities
from uniserve_worker.backends.attention.text_dispatch import TextBackendGate
from uniserve_worker.ops import (
    AdapterPool,
    AttentionRegime,
    AttentionReq,
    Capabilities,
    CommDispatcher,
    Dispatcher,
    FusedOpPool,
    Handoff,
    tp_all_reduce,
)
from uniserve_worker.ops.providers import _AttentionBackendProvider

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _DispatchReq:
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class _ComputeReq:
    value: int


class _ComputeProvider:
    operator = "toy"

    def __init__(self, name: str) -> None:
        self.name = name

    def capabilities(self):
        return Capabilities(tags=frozenset({self.name, self.operator}))

    def can_run(self, req):
        del req
        return True

    def run(self, req):
        return (self.name, req.value)


class _UnavailableComm:
    name = "deep_ep"
    operator = "moe_a2a"

    def capabilities(self):
        return Capabilities(tags=frozenset({self.name, self.operator}))

    def can_dispatch(self, req, *, mesh=None):
        del req, mesh
        return False

    def dispatch(self, req, *, mesh=None):  # pragma: no cover - can_dispatch blocks this.
        raise AssertionError("unavailable provider must not run")

    def can_combine(self, handoff, *, mesh=None):
        del handoff, mesh
        return False

    def combine(self, handoff, *, mesh=None):  # pragma: no cover - can_combine blocks this.
        raise AssertionError("unavailable provider must not combine")


class _StandardComm:
    name = "standard"
    operator = "moe_a2a"

    def capabilities(self):
        return Capabilities(tags=frozenset({self.name, self.operator}))

    def can_dispatch(self, req, *, mesh=None):
        del req, mesh
        return True

    def dispatch(self, req, *, mesh=None):
        return Handoff(
            format="standard",
            payload=tuple(req.tokens),
            metadata={"mesh": mesh},
        )

    def can_combine(self, handoff, *, mesh=None):
        del mesh
        return handoff.format == "standard"

    def combine(self, handoff, *, mesh=None):
        del mesh
        return {
            "provider": self.name,
            "payload": handoff.payload,
            "metadata": dict(handoff.metadata),
        }


def test_comm_dispatcher_falls_back_and_stamps_handoff_owner():
    dispatcher = CommDispatcher("moe_a2a", [_UnavailableComm(), _StandardComm()])

    handoff = dispatcher.dispatch(_DispatchReq((3, 5)), override="deep_ep", mesh="ep0")
    result = dispatcher.combine(handoff)

    assert handoff.operator == "moe_a2a"
    assert handoff.provider == "standard"
    assert result == {
        "provider": "standard",
        "payload": (3, 5),
        "metadata": {"mesh": "ep0"},
    }


def test_dispatchers_reject_unknown_overrides():
    compute = Dispatcher("toy", [_ComputeProvider("triton"), _ComputeProvider("eager")])
    comm = CommDispatcher("moe_a2a", [_UnavailableComm(), _StandardComm()])

    with pytest.raises(ValueError, match="unknown provider override"):
        compute.run(_ComputeReq(1), override="cutlass")
    with pytest.raises(ValueError, match="unknown comm provider override"):
        comm.dispatch(_DispatchReq((1,)), override="mooncake")


def test_adapter_and_fused_pools_model_composite_provider_axes():
    adapters = AdapterPool()
    adapters.register_pre(
        "deepep_normal",
        "triton",
        lambda handoff: {"compute_tokens": handoff.payload["tokens"]},
    )
    adapters.register_post(
        "triton",
        "deepep_normal",
        lambda value: Handoff(format="deepep_normal", payload={"computed": value}),
    )
    handoff = Handoff(format="deepep_normal", payload={"tokens": (1, 2)})

    compute_req = adapters.adapt_pre(handoff, "triton")
    combine_input = adapters.adapt_post({"out": (2, 4)}, "triton", "deepep_normal")

    assert compute_req == {"compute_tokens": (1, 2)}
    assert combine_input == Handoff(format="deepep_normal", payload={"computed": {"out": (2, 4)}})
    with pytest.raises(KeyError, match="no pre-adapter"):
        adapters.adapt_pre(handoff, "cutlass")

    fused = FusedOpPool()
    fused.register(("deep_ep", "triton"), lambda value: ("fused", value))
    assert fused.get(("deep_ep", "triton"))("tokens") == ("fused", "tokens")
    assert fused.get(("standard", "eager")) is None


class _Axis:
    name = "tp"

    def __init__(self) -> None:
        self.transport = self
        self.calls: list[str] = []

    def all_reduce(self, tensor, op="sum"):
        self.calls.append(op)
        return tensor + 1


def test_tp_all_reduce_facade_runs_through_comm_provider():
    axis = _Axis()
    tensor = torch.tensor([1.0, 2.0])

    out = tp_all_reduce(tensor, "sum", axis=axis)

    torch.testing.assert_close(out, torch.tensor([2.0, 3.0]))
    assert axis.calls == ["sum"]


def test_attention_provider_skips_paged_only_varlen_without_block_table():
    class _PagedOnlyVarlenBackend:
        name = "paged_only_varlen"

        def capabilities(self):
            return AttentionCapabilities(
                varlen_attention=True,
                varlen_paged_kv=True,
                requires_paged_varlen=True,
            )

    provider = _AttentionBackendProvider(_PagedOnlyVarlenBackend())
    q = torch.empty(3, 2, 4)
    k = torch.empty(3, 2, 4)
    v = torch.empty(3, 2, 4)
    cu = torch.tensor([0, 1, 3], dtype=torch.int32)
    req = AttentionReq(
        q=q,
        k=k,
        v=v,
        regime=AttentionRegime.EXTEND,
        causal=True,
        scale=1.0,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=2,
        max_seqlen_k=2,
    )

    assert not provider.can_run(req)
    assert provider.can_run(
        AttentionReq(
            q=q,
            k=torch.empty(4, 2, 2, 4),
            v=torch.empty(4, 2, 2, 4),
            regime=AttentionRegime.EXTEND,
            causal=True,
            scale=1.0,
            block_table=torch.tensor([[0], [1]], dtype=torch.int32),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=2,
            max_seqlen_k=2,
        )
    )


def test_attention_provider_rejects_raw_paged_varlen_page_size_mismatch():
    class _PagedVarlenBackend:
        name = "paged_varlen"

        def capabilities(self):
            return AttentionCapabilities(
                varlen_attention=True,
                varlen_paged_kv=True,
                paged_block_size_multiple=256,
            )

    provider = _AttentionBackendProvider(_PagedVarlenBackend())
    q = torch.empty(3, 2, 4)
    v = torch.empty(4, 64, 2, 4)
    cu = torch.tensor([0, 1, 3], dtype=torch.int32)
    req = AttentionReq(
        q=q,
        k=torch.empty(4, 64, 2, 4),
        v=v,
        regime=AttentionRegime.EXTEND,
        causal=True,
        scale=1.0,
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=2,
        max_seqlen_k=2,
    )

    assert not provider.can_run(req)
    assert provider.can_run(
        AttentionReq(
            q=q,
            k=torch.empty(4, 256, 2, 4),
            v=torch.empty(4, 256, 2, 4),
            regime=AttentionRegime.EXTEND,
            causal=True,
            scale=1.0,
            block_table=torch.tensor([[0], [1]], dtype=torch.int32),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=2,
            max_seqlen_k=2,
        )
    )


def test_text_backend_gate_skips_paged_only_varlen_for_initial_ragged_prefill(monkeypatch):
    class _PagedOnlyVarlenProvider:
        name = "paged_only_varlen"

        def capabilities(self):
            return Capabilities(
                attrs={
                    "varlen_attention": True,
                    "varlen_paged_kv": True,
                    "requires_paged_varlen": True,
                    "min_head_dim": 1,
                    "paged_block_size_multiple": 1,
                }
            )

    class _Dispatcher:
        def ordered(self, override):
            del override
            return (_PagedOnlyVarlenProvider(),)

    import uniserve_worker.ops as ops

    monkeypatch.setattr(ops, "attention_dispatcher", lambda: _Dispatcher())
    gate = TextBackendGate(head_dim=4, block_size=2, device_type="cuda")

    assert not gate._varlen_available(None, needs_paged_kv=False)
    assert gate._varlen_available(None, needs_paged_kv=True)
