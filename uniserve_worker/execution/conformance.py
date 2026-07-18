"""Mechanical conformance-case generation and the evidence manifest.

Stage 0 ("Freeze Baseline And Proof Protocol") and the Stage 11 adapter
conformance gate of ``specs/unified_forward_execution.md``, with the
companion's completeness rule: the case set is *generated mechanically* from
registration data — a manually authored case list cannot establish
completeness — and the readiness validator regenerates it and compares
hashes, rejecting stale manifests.

For every nonempty subset of a family's advertised operations and every
row-order permutation of that subset, one case describes a batch with one
row per operation in that order (order-equivalent rows are represented by
multiplicity, which the initial generator keeps at one). The runner executes
every generated case against a fresh dormant stack and asserts the
transaction laws: the batch commits, exactly one adapter/root invocation
serves it, and results stay in scheduler order.

The manifest is typed evidence: family identity and geometry, the generated
case identifiers, the canonical case-set hash, configured capacities, and
the frozen benchmark reference points from ``docs/benchmark-protocol.md``
that later fixed-protocol comparisons must name.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass

from ..contracts.execution import OperationTag

__all__ = [
    "ConformanceCase",
    "ConformanceManifest",
    "ManifestError",
    "generate_cases",
    "case_set_hash",
    "build_manifest",
    "validate_manifest",
]

# Frozen serial-benchmark reference points (docs/benchmark-protocol.md,
# snapshot 20260715T0855Z). Later comparisons name these exact values.
_BENCHMARK_REFERENCES: dict[str, dict[str, float]] = {
    "qwen3_sharegpt_r16": {
        "output_tokens_per_s": 1857.14,
        "mean_ttft_ms": 104.19,
        "mean_tpot_ms": 23.99,
    },
    "sensenova_mjhq_t2i_c1": {"mean_image_latency_ms": 3759.051},
    "sensenova_mjhq_t2i_c32": {"images_per_s": 0.240},
    "bagel_mjhq_t2i_c1": {"mean_image_latency_ms": 6913.954},
}


class ManifestError(ValueError):
    """A manifest does not match its regenerated case set."""


@dataclass(frozen=True, slots=True)
class ConformanceCase:
    """One generated operation-composition and row-order case."""

    case_id: str
    family: str
    operations: tuple[OperationTag, ...]  # row order

    def encode(self) -> str:
        tags = ",".join(str(int(tag)) for tag in self.operations)
        return f"{self.family}:{tags}"


@dataclass(frozen=True, slots=True)
class ConformanceManifest:
    family: str
    advertised_operations: tuple[int, ...]
    case_ids: tuple[str, ...]
    case_set_hash: str
    benchmark_references: dict[str, dict[str, float]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)


def generate_cases(
    family: str,
    advertised: frozenset[OperationTag],
) -> tuple[ConformanceCase, ...]:
    """Every nonempty advertised subset in every row order, mechanically."""

    operations = sorted(advertised, key=int)
    cases: list[ConformanceCase] = []
    for size in range(1, len(operations) + 1):
        for subset in itertools.combinations(operations, size):
            for order in itertools.permutations(subset):
                tags = "-".join(tag.name.lower() for tag in order)
                cases.append(
                    ConformanceCase(
                        case_id=f"{family}/{tags}",
                        family=family,
                        operations=tuple(order),
                    )
                )
    return tuple(cases)


def case_set_hash(cases: tuple[ConformanceCase, ...]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.encode().encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def build_manifest(
    family: str,
    advertised: frozenset[OperationTag],
) -> ConformanceManifest:
    cases = generate_cases(family, advertised)
    return ConformanceManifest(
        family=family,
        advertised_operations=tuple(sorted(int(tag) for tag in advertised)),
        case_ids=tuple(case.case_id for case in cases),
        case_set_hash=case_set_hash(cases),
        benchmark_references=_BENCHMARK_REFERENCES,
    )


def validate_manifest(manifest: ConformanceManifest) -> None:
    """Regenerate the case set and reject any drift (readiness rule)."""

    regenerated = generate_cases(
        manifest.family,
        frozenset(OperationTag(tag) for tag in manifest.advertised_operations),
    )
    if tuple(case.case_id for case in regenerated) != manifest.case_ids:
        raise ManifestError(
            f"manifest {manifest.family} lists a stale case set"
        )
    if case_set_hash(regenerated) != manifest.case_set_hash:
        raise ManifestError(
            f"manifest {manifest.family} hash does not match the regenerated set"
        )
