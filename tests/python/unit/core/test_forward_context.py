"""Unit coverage for the additive TextAttentionMetadataBuilder."""
from __future__ import annotations

import pytest

from uniserve_worker.contracts.forward_context import (
    TextAttentionMetadata,
    TextAttentionMetadataBuilder,
)

pytestmark = pytest.mark.unit


class _FakeCache:
    base_len = 0
    base_lens = ()
    pool = None


def test_builder_build_matches_direct_construction():
    cache = _FakeCache()
    built = (
        TextAttentionMetadataBuilder()
        .cache(cache)
        .block_table(None)
        .cache_seqlens(None)
        .query_lens_cpu((1, 1))
        .max_seqlen_q(7)
        .build()
    )
    direct = TextAttentionMetadata(
        cache=cache,
        block_table=None,
        cache_seqlens=None,
        query_lens_cpu=(1, 1),
        max_seqlen_q=7,
    )
    assert built == direct


def test_builder_returns_self_for_chaining():
    builder = TextAttentionMetadataBuilder()
    assert builder.cache(_FakeCache()) is builder
    assert builder.block_table(None) is builder


def test_builder_defaults_unset_optional_fields():
    built = (
        TextAttentionMetadataBuilder()
        .cache(_FakeCache())
        .block_table(None)
        .cache_seqlens(None)
        .build()
    )
    assert built.cache_seqlens_cpu == ()
    assert built.query_lens is None
    assert built.max_seqlen_k == 0
    assert built.mode is None


def test_builder_build_rejects_missing_required_fields():
    with pytest.raises(ValueError) as exc:
        TextAttentionMetadataBuilder().cache(_FakeCache()).build()
    message = str(exc.value)
    assert "block_table" in message
    assert "cache_seqlens" in message
    assert "cache" not in message.replace("cache_seqlens", "")
