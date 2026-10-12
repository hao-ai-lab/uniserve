"""Resolve token text to token IDs from a checkpoint's tokenizer files.

A checkpoint publishes its vocabulary in several spellings: ``tokenizer.json``
(model vocabulary and ``added_tokens``), ``vocab.json``, ``added_tokens.json``
and the ``added_tokens_decoder`` of ``tokenizer_config.json``. Packages and
the loader read them through ``read_vocabulary`` and ``token_id`` rather than
loading a tokenizer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

__all__ = ["read_vocabulary", "token_id"]


def _object(path: Path) -> dict:
    """Read a JSON object; invalid JSON or a non-object raises ValueError."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"tokenizer metadata {path} must contain an object")
    return value


def _model_vocabulary(path: Path) -> dict[str, object]:
    """Map token text to ID from a ``tokenizer.json`` model and added tokens.

    The model vocabulary is an object from token text to ID or, for a
    Unigram model, a list of ``[text, score]`` rows whose position is the ID;
    ``added_tokens`` is a list of objects carrying ``content`` and ``id``.
    Added tokens override the model vocabulary. Any other layout raises
    ``ValueError``.
    """
    data = _object(path)
    model = data.get("model", {})
    vocabulary = model.get("vocab", {}) if isinstance(model, dict) else None
    added = data.get("added_tokens", [])

    if isinstance(vocabulary, list) and all(
        isinstance(row, list) and row and isinstance(row[0], str)
        for row in vocabulary
    ):
        vocabulary = {row[0]: index for index, row in enumerate(vocabulary)}
    if (
        not isinstance(vocabulary, dict)
        or not isinstance(added, list)
        or any(
            not isinstance(entry, dict)
            or not isinstance(entry.get("content"), str)
            or "id" not in entry
            for entry in added
        )
    ):
        raise ValueError(f"tokenizer metadata {path} has a malformed layout")

    return {
        **vocabulary,
        **{entry["content"]: entry["id"] for entry in added},
    }


def read_vocabulary(root: Path) -> dict[str, object]:
    """Merge every vocabulary spelling of the tokenizer files in ``root``.

    Later spellings override earlier ones for the same token text: the
    ``tokenizer.json`` model and added tokens, ``vocab.json``,
    ``added_tokens.json``, then ``tokenizer_config.json``'s
    ``added_tokens_decoder``. IDs are returned unchecked; ``token_id``
    validates the one it returns. A file with a malformed layout raises
    ``ValueError``.
    """
    vocabulary: dict[str, object] = {}
    if (root / "tokenizer.json").is_file():
        vocabulary.update(_model_vocabulary(root / "tokenizer.json"))
    for filename in ("vocab.json", "added_tokens.json"):
        if (root / filename).is_file():
            vocabulary.update(_object(root / filename))

    path = root / "tokenizer_config.json"
    if path.is_file():
        # ``added_tokens_decoder`` maps each decimal ID to an object whose
        # ``content`` is the token text.
        decoder = _object(path).get("added_tokens_decoder", {})
        if not isinstance(decoder, dict) or any(
            not isinstance(entry, dict)
            or not isinstance(entry.get("content"), str)
            for entry in decoder.values()
        ):
            raise ValueError(
                f"tokenizer metadata {path} has a malformed layout"
            )
        vocabulary.update(
            {entry["content"]: int(index) for index, entry in decoder.items()}
        )
    return vocabulary


def token_id(vocabulary: Mapping[str, object], token: str) -> int:
    """Return the nonnegative integer ID of ``token``.

    Raises:
        ValueError: ``vocabulary`` does not map ``token`` to one.
    """
    index = vocabulary.get(token)
    if type(index) is not int or index < 0:
        raise ValueError(
            f"tokenizer metadata does not define declared token {token!r}"
        )
    return index
