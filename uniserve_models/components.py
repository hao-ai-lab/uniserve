"""The components a model owns and how each one divides work over ranks.

A model states two things about itself before anything loads it: which
components it has, and how each one divides. A denoiser divides a sequence by
attention head, a text encoder divides its tensors, a decoder divides its
output timeline into media units, and a muxer assembles the artifact and
divides nothing. How wide the instance is, which host and device each rank is,
and how ranks are numbered are the serving infrastructure's; a model states the
division, and shared code states what that division means for a rank count.

Each package declares its components beside its numerical code, in a
``components.toml`` naming one division per component. The declaration is read
rather than imported, so answering costs two file reads and no framework
import, and an engine can ask a checkpoint what it holds without loading it.
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
_DECLARATION = "components.toml"


class Partition(StrEnum):
    """How one component divides its work over the ranks that hold it.

    The division is the model's own mathematical property and does not depend
    on how wide the instance is; a rank count turns it into a rank membership
    and a parallel configuration.
    """

    #: Divides the component's tensors, every rank holding a shard of each.
    TENSOR = "tensor"
    #: Divides the sequence by attention head, which Ulysses exchanges.
    SEQUENCE = "sequence"
    #: Divides the output timeline, each rank reconstructing its own units.
    MEDIA_UNITS = "media_units"
    #: Divides nothing, so one rank holds the whole component.
    NOTHING = "nothing"


def _divided(partition: Partition, ranks: int) -> dict:
    """State how one component divides over an instance of ``ranks`` ranks.

    The result is a `ComponentConfig` as an explicit ``--workers`` argument
    writes it, so a declared component and a written one are the same object to
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
    # Reached only for a division nothing expands, which is a partition added
    # above without the rank membership it stands for.
    raise ValueError(f"nothing states what the {partition} division divides")


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
    """Return a checkpoint's components divided over ``ranks`` ranks.

    Each component is mapped to the rank membership and parallel configuration
    its division gives it at that width, in the schema the engine's `--workers`
    entries already use. Rank identity -- which host and which device -- is the
    caller's.
    """
    if type(ranks) is not int or ranks < 1:
        raise ValueError("components divide over a positive rank count")
    divisions = divisions_of(architecture_of(path))
    return {
        name: _divided(partition, ranks)
        for name, partition in divisions.items()
    }


def main() -> int:
    """Print a checkpoint's divided components as JSON, for another language."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Report the components a checkpoint holds."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--ranks", type=int, required=True)
    arguments = parser.parse_args()

    print(json.dumps(entries_for(arguments.model, arguments.ranks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
