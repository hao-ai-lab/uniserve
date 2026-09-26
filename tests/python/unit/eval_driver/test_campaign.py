"""The frozen campaign preserves its declared populations and partitions."""

import hashlib
import json
from collections import Counter

import pytest

from uniserve_eval.campaign import SHAPES, freeze_corpus

pytestmark = pytest.mark.unit


def test_frozen_partitions_balance_shapes_and_are_reproducible(tmp_path):
    rows = []

    def add(partition, family, shape, **metadata):
        seconds, tokens = shape
        identity = str(len(rows))
        frames = round(seconds * 24)
        rows.append(
            {
                "id": identity,
                "prompt": identity,
                "seconds": seconds,
                "prompt_len": tokens,
                "seed": len(rows),
                "metadata": {
                    "partition": partition,
                    "family": str(family),
                    "variant": identity,
                    "token_ids": [0] * tokens,
                    "tokenizer_repository": "fixture",
                    "tokenizer_revision": "fixture",
                    "provenance": {"kind": "test_fixture"},
                    "prompt_sha256": hashlib.sha256(
                        identity.encode()
                    ).hexdigest(),
                    "aligned_frames": frames + (5 - frames) % 17,
                    **metadata,
                },
            }
        )

    for family in range(6):
        for shape in SHAPES:
            for seed in range(2):
                add("latency", family, shape)
                rows[-1]["seed"] = seed
    for shape in SHAPES:
        for i in range(16):
            add("throughput", i % 6, shape)
    for partition in ("qualification", "warmup"):
        for i, shape in enumerate(SHAPES):
            add(partition, i, shape)
    for concurrency in (1, 2, 4, 8, 16):
        for repetition in (1, 2, 3):
            for i in range(2 * concurrency):
                add(
                    "priming",
                    i % 6,
                    SHAPES[i % 6],
                    concurrency=concurrency,
                    repetition=repetition,
                )

    source = tmp_path / "corpus.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = freeze_corpus(source, tmp_path / "first")
    second = freeze_corpus(source, tmp_path / "second")
    all_throughput = []
    for repetition in (1, 2, 3):
        entry = first["manifests"][f"throughput-r{repetition}"]
        assert entry["count"] == 32
        with open(entry["path"]) as handle:
            records = [json.loads(line) for line in handle]
        counts = Counter((row["seconds"], row["prompt_len"]) for row in records)
        assert sorted(counts.values()) == [5, 5, 5, 5, 6, 6]
        all_throughput.extend(records)
    assert len({row["id"] for row in all_throughput}) == 96
    assert Counter(
        (row["seconds"], row["prompt_len"]) for row in all_throughput
    ) == Counter(dict.fromkeys(SHAPES, 16))
    assert first["manifests"]["latency"]["count"] == 72
    for name, entry in first["manifests"].items():
        assert entry["sha256"] == second["manifests"][name]["sha256"]
    with pytest.raises(FileExistsError):
        freeze_corpus(source, tmp_path / "first")
