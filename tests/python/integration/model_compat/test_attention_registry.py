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


def test_fa4_cute_backend_name_is_registered_when_provider_module_imports():
    from uniserve_kernel import mm_attn_varlen
    from uniserve_worker.backends.attention import get_attention_backend, has_attention_backend

    assert has_attention_backend("fa4_cute")
    backend = get_attention_backend("fa4_cute")
    caps = backend.capabilities()
    assert caps.available is mm_attn_varlen.available()
    assert caps.visible_end is mm_attn_varlen.available()


def test_registry_rejects_conflicting_registration_and_unknown_lookup(monkeypatch):
    monkeypatch.setattr(registry, "_BACKENDS", {})
    registry.register_attention_backend("torch_sdpa", _FakeBackend())

    with pytest.raises(ValueError, match="different backend"):
        registry.register_attention_backend("torch_sdpa", _OtherFakeBackend())
    with pytest.raises(ValueError, match="unknown attention backend"):
        registry.get_attention_backend("flashinfer")
