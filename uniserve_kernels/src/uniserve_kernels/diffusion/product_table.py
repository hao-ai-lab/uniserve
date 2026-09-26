r"""Measure self-conditioning product algorithms into the shipped table.

For every requested row count ``R`` the product of ``[R * canvas, vocab]``
BF16 weights and a ``[vocab, hidden]`` BF16 table, with the workspace of
:func:`~uniserve_kernels.diffusion.canvas.product_scratch_bytes`, times each
algorithm cuBLASLt proposes: one warm-up launch each, then interleaved
rounds with an L2-evicting memset before every launch. The proposal with
the lowest median becomes the table entry for (device name, cuBLASLt
version, positions, hidden, vocab, scratch bytes); entries of other keys in
the file stay. Run it on an otherwise idle device of the model the entries
are for::

    python -m uniserve_kernels.diffusion.product_table --rows 1 2 4 \
        [--canvas 256 --vocab 262144 --hidden 2816 --rounds 9 --output FILE]

Weights are uniform in [0, 2 / vocab), the size of softmax probabilities,
and the table normal with standard deviation 0.05; the algorithms' speed
does not depend on the values beyond power draw.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path

import torch

from . import canvas

FLUSH_BYTES = 1 << 30


def measure(
    rows: int, canvas_length: int, vocab: int, hidden: int, rounds: int
) -> dict:
    """The table entry of one row count, with every proposal's timing."""
    device = torch.device("cuda", torch.cuda.current_device())
    positions = rows * canvas_length
    scratch_bytes = canvas.product_scratch_bytes(positions, hidden)
    generator = torch.Generator(device=device).manual_seed(rows)
    weights = torch.rand(
        positions,
        vocab,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(2.0 / vocab)
    table = torch.randn(
        vocab, hidden, generator=generator, device=device, dtype=torch.bfloat16
    ).mul_(0.05)
    output = torch.empty(positions, hidden, device=device)
    scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=device)

    proposals = canvas._extension().product_proposals(
        weights, table, output, scratch, rounds, FLUSH_BYTES
    )
    timed = [
        {
            **dict(
                zip(
                    canvas.PRODUCT_CONFIG_FIELDS,
                    map(int, row[:-1]),
                    strict=True,
                )
            ),
            "median_us": row[-1] * 1000.0,
        }
        for row in proposals
    ]
    fastest = min(
        (entry for entry in timed if not math.isnan(entry["median_us"])),
        key=lambda entry: entry["median_us"],
    )
    device_name, version, *_ = canvas.product_table_key(
        device, positions, hidden, vocab, scratch_bytes
    )
    return {
        "device": device_name,
        "cublaslt_version": version,
        "m": positions,
        "n": hidden,
        "k": vocab,
        "scratch_bytes": scratch_bytes,
        "algorithm": {
            field: fastest[field] for field in canvas.PRODUCT_CONFIG_FIELDS
        },
        "median_us": fastest["median_us"],
        "proposals": len(timed),
        "first_proposal_median_us": timed[0]["median_us"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=262144)
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", type=Path, default=canvas.PRODUCT_TABLE)
    options = parser.parse_args()

    document = (
        json.loads(options.output.read_text())
        if options.output.exists()
        else {"entries": []}
    )
    fields = ("device", "cublaslt_version", "m", "n", "k", "scratch_bytes")
    entries = {
        tuple(entry[f] for f in fields): entry for entry in document["entries"]
    }
    measured = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    with torch.inference_mode():
        for rows in options.rows:
            entry = measure(
                rows,
                options.canvas,
                options.vocab,
                options.hidden,
                options.rounds,
            )
            entry["measured_utc"] = measured
            entry["rounds"] = options.rounds
            entries[tuple(entry[f] for f in fields)] = entry
            print(
                f"rows {rows}: {entry['algorithm']} "
                f"{entry['median_us']:.1f} us, first proposal "
                f"{entry['first_proposal_median_us']:.1f} us, "
                f"{entry['proposals']} proposals"
            )
            torch.cuda.empty_cache()

    document = {
        "description": (
            "cuBLASLt algorithm of the self-conditioning product per device "
            "model, cuBLASLt version and shape (m positions, n hidden, k "
            "vocabulary, scratch bytes): the proposal with the lowest median "
            "time, written by uniserve_kernels.diffusion.product_table"
        ),
        "entries": [entries[key] for key in sorted(entries)],
    }
    options.output.write_text(json.dumps(document, indent=1) + "\n")


if __name__ == "__main__":
    main()
