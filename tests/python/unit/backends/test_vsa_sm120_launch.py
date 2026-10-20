"""The SM120 in-place launch runs only under the FlashInfer it repeats.

``_flashinfer_sm120.launch`` reuses internals of FlashInfer's
``bsa_attn_sm120_blk64_fwd``. Under any other installed FlashInfer release
it calls that public function on the caller's views instead. The
internals and the alignment check are stubbed, so no GPU is needed.
"""

import pytest
import torch

from uniserve.runtime.backends.attention.vsa import _flashinfer_sm120

pytestmark = pytest.mark.unit

_TILES, _HEADS = 3, 2
_ROWS = _TILES * _flashinfer_sm120.TILE


@pytest.fixture
def public_calls(monkeypatch):
    """Present the internals, admit every view and record public calls."""
    calls = []
    monkeypatch.setattr(_flashinfer_sm120, "_wrapper", object())
    monkeypatch.setattr(_flashinfer_sm120, "_in_place", lambda value: True)
    monkeypatch.setattr(
        _flashinfer_sm120,
        "bsa_attn_sm120_blk64_fwd",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def _views():
    q, k, v, out = (
        torch.zeros(_ROWS, _HEADS, 128, dtype=torch.bfloat16) for _ in range(4)
    )
    return q, k, v, out


@pytest.mark.usefixtures("public_calls")
def test_the_repeated_release_launches_in_place(monkeypatch):
    monkeypatch.setattr(
        _flashinfer_sm120,
        "_installed_version",
        _flashinfer_sm120.FLASHINFER_VERSION,
    )
    q, k, v, _ = _views()
    assert _flashinfer_sm120._launches_in_place(q, k, v)


@pytest.mark.parametrize("version", ["0.6.17", "0.6.19", "0.7.0"])
def test_another_release_calls_the_public_function(
    monkeypatch, public_calls, version
):
    monkeypatch.setattr(_flashinfer_sm120, "_installed_version", version)
    q, k, v, out = _views()
    indices = torch.zeros(_HEADS, _TILES, _TILES, dtype=torch.int32)
    counts = torch.ones(_HEADS, _TILES, dtype=torch.int32)
    sizes = torch.full((_TILES,), 64, dtype=torch.int32)
    lse = torch.zeros(1, _HEADS, _ROWS)
    scale = 128**-0.5

    assert not _flashinfer_sm120._launches_in_place(q, k, v)
    _flashinfer_sm120.launch(
        q, k, v, indices, counts, sizes, scale=scale, out=out, lse=lse
    )

    # One call, on batch-of-one views of the caller's tensors.
    [(args, kwargs)] = public_calls
    assert len(args) == 5
    for actual, expected in zip(args[:4], (q, k, v, indices)):
        assert actual.shape == (1, *expected.shape)
        assert actual.data_ptr() == expected.data_ptr()
    assert args[4] == _TILES
    assert set(kwargs) == {
        "block_sizes",
        "q2k_block_nums",
        "softmax_scale",
        "out",
        "lse",
    }
    assert kwargs["block_sizes"] is sizes
    assert kwargs["q2k_block_nums"].shape == (1, *counts.shape)
    assert kwargs["q2k_block_nums"].data_ptr() == counts.data_ptr()
    assert kwargs["softmax_scale"] == scale
    assert kwargs["out"].shape == (1, *out.shape)
    assert kwargs["out"].data_ptr() == out.data_ptr()
    assert kwargs["lse"] is lse
