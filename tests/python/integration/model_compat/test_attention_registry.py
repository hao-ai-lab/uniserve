"""Attention backend registry tests."""
from __future__ import annotations

import pytest

from uniserve_worker.backends.attention import AttentionCapabilities, registry

pytestmark = pytest.mark.integration


class _FakeBackend:
    def capabilities(self) -> AttentionCapabilities:
        return AttentionCapabilities()


class _OtherFakeBackend(_FakeBackend):
    pass


def test_registry_normalizes_aliases_and_lists_registered_backends(monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(registry, "_BACKENDS", {})

    assert registry.normalize_attention_backend_name("flash-attn-4") == "fa4_cute"
    registry.register_attention_backend("torch_sdpa", backend)

    assert registry.has_attention_backend("torch")
    assert registry.get_attention_backend("sdpa") is backend
    assert registry.list_attention_backends() == ("torch_sdpa",)


def test_registry_registration_is_idempotent_by_backend_type(monkeypatch):
    backend = _FakeBackend()
    monkeypatch.setattr(registry, "_BACKENDS", {})

    first = registry.register_attention_backend("torch_sdpa", backend)
    second = registry.register_attention_backend("torch_sdpa", _FakeBackend())

    assert first is backend
    assert second is backend


def test_registry_rejects_conflicting_registration_and_unknown_lookup(monkeypatch):
    monkeypatch.setattr(registry, "_BACKENDS", {})
    registry.register_attention_backend("torch_sdpa", _FakeBackend())

    with pytest.raises(ValueError, match="different backend"):
        registry.register_attention_backend("torch_sdpa", _OtherFakeBackend())
    with pytest.raises(ValueError, match="unknown attention backend"):
        registry.get_attention_backend("flashinfer")
