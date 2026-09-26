"""Freeze the declared FastH3 workload partitions from authored request rows.

The input corpus contains already-authored prompts and token IDs. This module
does not synthesize text or select prompts using any generated media or timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SHAPES = tuple(
    (seconds, tokens) for seconds in (5, 10, 15) for tokens in (1000, 10000)
)


def freeze_corpus(source: Path, output: Path, *, seed: int = 20260925) -> dict:
    """Validate authored rows and freeze balanced, disjoint JSONL partitions.

    Every row has a ``metadata.partition`` of latency, throughput, warmup,
    qualification or priming. Priming additionally declares concurrency and
    repetition (1 through 3). All prompts and IDs are globally unique.
    Existing output directories are rejected to preserve the frozen corpus.
    """
    rows = [
        json.loads(line)
        for line in source.read_text().splitlines()
        if line.strip()
    ]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate request ID")
    if len({row["prompt"] for row in rows}) != len(rows):
        raise ValueError("every campaign request must have unique prompt bytes")
    for row in rows:
        metadata = row["metadata"]
        shape = (row["seconds"], row["prompt_len"])
        if shape not in SHAPES:
            raise ValueError(f"invalid campaign shape: {shape}")
        if len(metadata["token_ids"]) != row["prompt_len"]:
            raise ValueError("declared token count differs from token IDs")
        digest = hashlib.sha256(row["prompt"].encode()).hexdigest()
        if digest != metadata["prompt_sha256"]:
            raise ValueError("prompt checksum mismatch")
        frames = round(row["seconds"] * 24)
        if metadata["aligned_frames"] != frames + (5 - frames) % 17:
            raise ValueError("incorrect temporal alignment")
        for key in (
            "family",
            "variant",
            "provenance",
            "tokenizer_repository",
            "tokenizer_revision",
        ):
            if not metadata.get(key):
                raise ValueError(f"missing prompt provenance: {key}")

    groups: dict[str, list[dict]] = {}
    rng = random.Random(seed)
    latency = [row for row in rows if row["metadata"]["partition"] == "latency"]
    if len({row["seed"] for row in latency}) != 2:
        raise ValueError("the latency corpus requires two frozen seeds")
    families = {row["metadata"]["family"] for row in latency}
    counts = Counter(
        (row["metadata"]["family"], row["seconds"], row["prompt_len"])
        for row in latency
    )
    if len(families) != 6 or counts != Counter(
        {(family, *shape): 2 for family in families for shape in SHAPES}
    ):
        raise ValueError(
            "latency requires six families x six shapes x two seeds"
        )
    for cell in counts:
        seeds = {
            row["seed"]
            for row in latency
            if (row["metadata"]["family"], row["seconds"], row["prompt_len"])
            == cell
        }
        if len(seeds) != 2:
            raise ValueError("each latency cell requires two distinct seeds")
    rng.shuffle(latency)
    groups["latency"] = latency

    throughput = [
        row for row in rows if row["metadata"]["partition"] == "throughput"
    ]
    cells = {
        shape: [
            row
            for row in throughput
            if (row["seconds"], row["prompt_len"]) == shape
        ]
        for shape in SHAPES
    }
    if any(len(cell) != 16 for cell in cells.values()):
        raise ValueError("throughput requires 16 unique requests per shape")
    for cell in cells.values():
        rng.shuffle(cell)
    for repetition in range(3):
        selected = []
        for index, shape in enumerate(SHAPES):
            count = 6 if index // 2 == repetition else 5
            selected.extend(cells[shape][:count])
            del cells[shape][:count]
        rng.shuffle(selected)
        groups[f"throughput-r{repetition + 1}"] = selected

    for phase in ("warmup", "qualification"):
        selected = [
            row for row in rows if row["metadata"]["partition"] == phase
        ]
        if Counter(
            (row["seconds"], row["prompt_len"]) for row in selected
        ) != Counter(SHAPES):
            raise ValueError(f"{phase} must cover each of the six shapes once")
        groups[phase] = selected
    for concurrency in (1, 2, 4, 8, 16):
        for repetition in (1, 2, 3):
            selected = [
                row
                for row in rows
                if row["metadata"]["partition"] == "priming"
                and row["metadata"].get("concurrency") == concurrency
                and row["metadata"].get("repetition") == repetition
            ]
            if len(selected) != 2 * concurrency:
                raise ValueError(
                    "each repetition requires 2 x concurrency priming rows"
                )
            groups[f"priming-c{concurrency}-r{repetition}"] = selected
    if sum(map(len, groups.values())) != len(rows):
        raise ValueError("unassigned corpus rows")

    output.mkdir(parents=True, exist_ok=False)
    index = {"seed": seed, "manifests": {}}
    for name, selected in groups.items():
        for order, row in enumerate(selected):
            row["metadata"]["execution_order"] = order
        data = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in selected
        ).encode()
        path = output / f"{name}.jsonl"
        path.write_bytes(data)
        index["manifests"][name] = {
            "path": str(path.resolve()),
            "count": len(selected),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze_corpus(args.source, args.output)


if __name__ == "__main__":
    main()
