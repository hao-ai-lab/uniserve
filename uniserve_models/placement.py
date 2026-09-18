"""Each architecture's computation entries and how they partition over ranks.

A model knows which components it owns and how each one divides work: a
denoiser partitions a sequence, a text encoder partitions tensors, a decoder
divides a timeline into media units, and a muxer assembles an artifact and
divides nothing. Serving infrastructure owns the rest of placement -- which
host and device each rank is, how many ranks there are, and how they are
numbered -- so this module states only what the model itself determines, for a
given number of ranks.

The module is deliberately free of numerical dependencies. Resolving a
checkpoint's entries reads one metadata file and imports nothing from the
model's numerical package, so an engine can ask for a placement without paying
for a framework import.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

__all__ = ["ARCHITECTURE_PACKAGES", "architecture_of", "entries_for"]


#: Architecture name to the package implementing it. Loading reads the same
#: mapping, so an architecture is named in one place.
ARCHITECTURE_PACKAGES: Mapping[str, str] = MappingProxyType(
    {
        "Qwen3ForCausalLM": "uniserve_models.qwen3",
        "Qwen3MoeForCausalLM": "uniserve_models.qwen3",
        "BagelForConditionalGeneration": "uniserve_models.bagel",
        "NEOChatModel": "uniserve_models.sensenova_u1",
        "MiniMaxH3Transformer3DModel": "uniserve_models.minimax_h3",
    }
)


def _tensor_parallel(ranks: int) -> dict:
    """Partition one component's tensors across every rank."""
    return {
        "ranks": list(range(ranks)),
        "parallel_config": {"tensor_parallel_size": ranks},
    }


def _token_entries(ranks: int) -> dict:
    """A decoder-only language model is one component over every rank."""
    return {"model": _tensor_parallel(ranks)}


def _minimax_h3_entries(ranks: int) -> dict:
    """MiniMax H3's five components and the partition each one admits.

    The denoiser divides its sequence by attention head, so its Ulysses degree
    is the rank count. The text encoder divides its tensors. Both decoders
    divide their output timeline into media units, one per rank, and reconstruct
    and encode where they decode. The muxer assembles the encoded units into the
    artifact; it owns no numerical method and divides nothing, so it is placed
    alone.
    """
    members = list(range(ranks))
    unit_decoder = {
        "ranks": members,
        "parallel_config": {},
        "distribution": "temporal_units",
        "units_per_rank": 1,
    }
    return {
        "denoiser": {
            "ranks": members,
            "parallel_config": {
                "sequence_parallel": {
                    "kind": "ulysses",
                    "ulysses_degree": ranks,
                }
            },
        },
        "text_encoder": _tensor_parallel(ranks),
        "video_decoder": dict(unit_decoder),
        "audio_decoder": dict(unit_decoder),
        "muxer": {"ranks": [0], "parallel_config": {}},
    }


#: Architecture name to the entries it declares for a rank count.
_ENTRIES = MappingProxyType(
    {
        "Qwen3ForCausalLM": _token_entries,
        "Qwen3MoeForCausalLM": _token_entries,
        "BagelForConditionalGeneration": _token_entries,
        "NEOChatModel": _token_entries,
        "MiniMaxH3Transformer3DModel": _minimax_h3_entries,
    }
)


#: Where a checkpoint names the architecture it implements, in the order the
#: sources are consulted. A decoder-only checkpoint lists it in its own config.
#: A modular media checkpoint's root names the pipeline rather than the
#: architecture, so the denoising transformer's config names it instead.
_DECLARATIONS = (
    ("config.json", "architectures"),
    ("transformer/config.json", "_class_name"),
)


def architecture_of(path: str | Path) -> str:
    """Return the single architecture a local checkpoint declares.

    Reads the checkpoint's own metadata and nothing else, so identifying a
    checkpoint costs two file reads and no framework import.
    """
    root = Path(path)
    for name, field in _DECLARATIONS:
        metadata = root / name
        if not metadata.is_file():
            continue
        declared = json.loads(metadata.read_text()).get(field, ())
        names = (declared,) if isinstance(declared, str) else tuple(declared)
        if len(names) != 1 or names[0] not in _ENTRIES:
            raise ValueError(
                f"{metadata} must declare one supported architecture; "
                f"found {names!r}"
            )
        return names[0]
    raise ValueError(f"{root} declares no architecture metadata")


def entries_for(path: str | Path, ranks: int) -> dict:
    """Return the computation entries a checkpoint's architecture declares.

    The result maps each component name to the membership and partition it
    admits over ``ranks`` ranks, in the schema the engine's explicit placement
    already accepts. Rank identity -- host and device -- is the caller's.
    """
    if type(ranks) is not int or ranks < 1:
        raise ValueError("a placement requires a positive rank count")
    return _ENTRIES[architecture_of(path)](ranks)


def main() -> int:
    """Print one checkpoint's entries as JSON, for a non-Python caller."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Report the computation entries a checkpoint declares."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--ranks", type=int, required=True)
    arguments = parser.parse_args()

    print(json.dumps(entries_for(arguments.model, arguments.ranks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
