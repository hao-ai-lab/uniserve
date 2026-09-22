"""The checkpoint identity names the files a rank loads.

The head derives it for a local checkpoint directory and the engine derives the
same value in Rust, so one fixture and one golden digest are shared with the
engine's test of the rule.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from uniserve_models.loading import checkpoint_identity
from uniserve_worker.bootstrap.model_loader import verify_checkpoint_identity
from uniserve_worker.errors import WorkerError

pytestmark = pytest.mark.unit

# The identity of ``fixture``, computed once from the identity rule and
# asserted by the engine's test over the same fixture.
GOLDEN = "0977a85cc96b143616b62095d362701b1694cc5259cfec897efae003d06c89bb"


def fixture(root: Path) -> None:
    """Build sidecars, shards, nested, hidden, excluded and linked entries."""
    (root / "config.json").write_bytes(b'{"architectures": ["A"]}\n')
    (root / "tokenizer.model").write_bytes(b"spm")
    (root / "chat_template.jinja").write_bytes(b"{{ messages }}")
    (root / "merges.txt").write_bytes(b"a b\n")
    (root / "model-00001-of-00002.safetensors").write_bytes(bytes([1]) * 300)
    (root / "model-00002-of-00002.safetensors").write_bytes(bytes([2]) * 200)
    (root / "model.safetensors.index.json").write_bytes(b'{"weight_map": {}}')
    (root / "vae").mkdir()
    (root / "vae" / "config.json").write_bytes(b'{"z": 4}')
    (root / "vae" / "weights.pt").write_bytes(bytes([3]) * 50)
    (root / ".gitattributes").write_bytes(b"* filter=lfs")
    (root / "optimizer").mkdir()
    (root / "optimizer" / "state.pt").write_bytes(bytes([4]) * 10)
    (root / "original").mkdir()
    (root / "original" / "params.json").write_bytes(b"{}")

    # A snapshot references its blobs through links: one sidecar and one
    # shard link to files outside the walked tree.
    (root / ".blobs").mkdir()
    (root / ".blobs" / "gen").write_bytes(b'{"eos": 1}')
    (root / ".blobs" / "shard").write_bytes(bytes([5]) * 120)
    os.symlink(root / ".blobs" / "gen", root / "generation_config.json")
    os.symlink(root / ".blobs" / "shard", root / "extra.bin")

    # A linked directory is not entered.
    os.symlink(root / "vae", root / "vae_link")


def test_the_fixture_digests_to_the_shared_golden_value(tmp_path):
    fixture(tmp_path)

    assert checkpoint_identity(tmp_path) == GOLDEN


def test_sidecar_contents_and_shard_sizes_change_the_identity(tmp_path):
    fixture(tmp_path)
    original = checkpoint_identity(tmp_path)

    # A one-byte change in a config sidecar is a different checkpoint.
    (tmp_path / "config.json").write_bytes(b'{"architectures": ["B"]}\n')
    assert checkpoint_identity(tmp_path) != original

    # A shard's contents are not read, but its size is part of the identity,
    # so a truncated or padded shard is detected.
    (tmp_path / "config.json").write_bytes(b'{"architectures": ["A"]}\n')
    assert checkpoint_identity(tmp_path) == original
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(
        bytes([2]) * 201
    )
    assert checkpoint_identity(tmp_path) != original


def test_a_rank_refuses_a_checkpoint_the_head_did_not_derive():
    expected = "a" * 64
    actual = "b" * 64

    # Without an expectation the engine's cross-rank agreement is the check.
    verify_checkpoint_identity(None, actual, rank=1, host="node-b")
    verify_checkpoint_identity(expected, expected, rank=1, host="node-b")

    with pytest.raises(WorkerError) as failure:
        verify_checkpoint_identity(expected, actual, rank=1, host="node-b")

    message = str(failure.value)
    assert "rank 1" in message
    assert "node-b" in message
    assert expected in message
    assert actual in message
