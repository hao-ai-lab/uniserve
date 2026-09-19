"""How a model's components divide work over the ranks that hold them.

A model owns two statements about placement. Which components exist, and how
each one divides: a denoiser divides a sequence by attention head, a text
encoder divides its tensors, a decoder divides its output timeline into media
units, and a muxer assembles the artifact and divides nothing. Everything else
is the serving infrastructure's -- which host and device each rank is, how many
ranks there are, and how they are numbered -- so a model states the division
and shared code states what that division means for a given rank count.

Each model package declares its components beside its numerical code, in a
``placement.toml`` naming one division per component. The declaration is read
rather than imported, so resolving a placement costs two file reads and no
framework import, and an engine can ask for one without loading a model.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "Partition",
    "architecture_of",
    "divisions_of",
    "entries_for",
    "package_of",
]


#: Architecture name to the package under this one implementing it, which is
#: where that architecture's declaration lives. Loading asks for the package
#: through `package_of`, so an architecture names its implementation once.
_PACKAGES: Mapping[str, str] = MappingProxyType(
    {
        "Qwen3ForCausalLM": "qwen3",
        "Qwen3MoeForCausalLM": "qwen3",
        "BagelForConditionalGeneration": "bagel",
        "NEOChatModel": "sensenova_u1",
        "MiniMaxH3Transformer3DModel": "minimax_h3",
    }
)

#: A modular media checkpoint's root names the pipeline assembled around the
#: denoising transformer rather than the transformer that implements the
#: architecture; this reads one as the other.
_PIPELINE_ARCHITECTURES: Mapping[str, str] = MappingProxyType(
    {"MiniMaxH3ModularPipeline": "MiniMaxH3Transformer3DModel"}
)

#: The file in which a package declares one division per component.
_DECLARATION = "placement.toml"


class Partition(StrEnum):
    """How one component divides its work over the ranks that hold it.

    The division is the model's mathematical property and does not depend on
    how wide the instance is; the rank count turns it into a placement.
    """

    #: Divides the component's tensors, every rank holding a shard of each.
    TENSOR = "tensor"
    #: Divides the sequence by attention head, which Ulysses exchanges.
    SEQUENCE = "sequence"
    #: Divides the output timeline, each rank reconstructing its own units.
    MEDIA_UNITS = "media_units"
    #: Divides nothing, so one rank holds the whole component.
    NOTHING = "nothing"


def _placed(partition: Partition, ranks: int) -> dict:
    """State one component's placement over an instance of ``ranks`` ranks.

    The result is a `ComponentConfig` as an explicit ``--workers`` placement
    writes it, so a declared placement and a written one are the same object to
    the engine.
    """
    members = list(range(ranks))
    match partition:
        case Partition.TENSOR:
            return {
                "ranks": members,
                "parallel_config": {"tensor_parallel_size": ranks},
            }
        case Partition.SEQUENCE:
            return {
                "ranks": members,
                "parallel_config": {
                    "sequence_parallel": {
                        "kind": "ulysses",
                        "ulysses_degree": ranks,
                    }
                },
            }
        case Partition.MEDIA_UNITS:
            return {
                "ranks": members,
                "parallel_config": {},
                "distribution": "temporal_units",
                "units_per_rank": 1,
            }
        case Partition.NOTHING:
            return {"ranks": [0], "parallel_config": {}}
    # Reached only if a division is declared that no placement expands, which
    # is a partition added above without the placement it stands for.
    raise ValueError(f"no placement states the {partition} division")


def architecture_of(path: str | Path) -> str:
    """Return the single supported architecture a checkpoint declares.

    A checkpoint names its architecture in its own root metadata, whichever of
    the two roots it carries.
    """
    root = Path(path)
    metadata = root / "config.json"
    if not metadata.is_file():
        metadata = root / "modular_model_index.json"
    declared = json.loads(metadata.read_text()) if metadata.is_file() else {}

    pipeline = _PIPELINE_ARCHITECTURES.get(declared.get("_class_name"))
    architectures = (
        (pipeline,) if pipeline else tuple(declared.get("architectures", ()))
    )
    if len(architectures) != 1 or architectures[0] not in _PACKAGES:
        raise ValueError(
            f"checkpoint must declare one supported architecture; "
            f"found {architectures!r}"
        )
    return architectures[0]


def package_of(path: str | Path) -> str:
    """Return the module implementing a checkpoint's architecture."""
    return f"{__package__}.{_PACKAGES[architecture_of(path)]}"


def divisions_of(architecture: str) -> Mapping[str, Partition]:
    """Return the division each of an architecture's components admits."""
    package = Path(__file__).parent / _PACKAGES[architecture]
    declared = tomllib.loads((package / _DECLARATION).read_text())
    return MappingProxyType(
        {name: Partition(value) for name, value in declared.items()}
    )


def entries_for(path: str | Path, ranks: int) -> dict:
    """Return the components a checkpoint places over ``ranks`` ranks.

    Each component is mapped to the membership and partition it admits, in the
    schema an explicit placement already uses. Rank identity -- which host and
    which device -- is the caller's.
    """
    if type(ranks) is not int or ranks < 1:
        raise ValueError("a placement requires a positive rank count")
    divisions = divisions_of(architecture_of(path))
    return {
        name: _placed(partition, ranks) for name, partition in divisions.items()
    }


def main() -> int:
    """Print one checkpoint's placement as JSON, for a non-Python caller."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Report the components a checkpoint places."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--ranks", type=int, required=True)
    arguments = parser.parse_args()

    print(json.dumps(entries_for(arguments.model, arguments.ranks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
