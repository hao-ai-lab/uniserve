"""Declarative weight-loading contracts.

A model family declares one immutable :class:`WeightSpec` as a class attribute.
The system loaders consume it to read checkpoints, apply the declared
rename/stack rules through the parameter weight hooks, validate coverage, and
inject the weights before the model becomes ready. Models declare data only;
checkpoint traversal and transformation are loader behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .weight_utils import StackedParamMapping

__all__ = [
    "GraphSource",
    "NativeSource",
    "Rename",
    "Sidecar",
    "StackedParamMapping",
    "TowerSplit",
    "WeightSpec",
    "weight_spec_of",
]


@dataclass(frozen=True)
class Rename:
    """One ordered checkpoint-name to parameter-name rule.

    ``exact`` rewrites one full tensor name. Otherwise ``source`` is a name
    prefix replaced by ``target``, after which each ``then`` pair is applied as
    a substring substitution over the whole name.
    """

    source: str
    target: str
    exact: bool = False
    then: tuple[tuple[str, str], ...] = ()

    def apply(self, name: str) -> str | None:
        """Return the renamed parameter name, or ``None`` when this rule does not match."""
        if self.exact:
            return self.target if name == self.source else None
        if not name.startswith(self.source):
            return None
        mapped = self.target + name[len(self.source) :]
        for find, replacement in self.then:
            mapped = mapped.replace(find, replacement)
        return mapped


@dataclass(frozen=True)
class TowerSplit:
    """Parameter-name partition between the generation tower and its complement.

    A parameter belongs to the generation tower when its name starts with any
    ``gen_prefixes`` entry or contains any ``gen_infixes`` entry; every other
    parameter belongs to the understanding tower.
    """

    gen_prefixes: tuple[str, ...] = ()
    gen_infixes: tuple[str, ...] = ()

    def is_generation(self, name: str) -> bool:
        return name.startswith(self.gen_prefixes) or any(
            infix in name for infix in self.gen_infixes
        )

    def role_filter(self, tower_role: str | None) -> Callable[[str], bool] | None:
        """Materialization predicate for ``tower_role``; ``None`` loads the whole model."""
        if tower_role is None:
            return None
        if tower_role == "gen":
            return self.is_generation
        if tower_role == "und":
            return lambda name: not self.is_generation(name)
        raise ValueError(f"unknown tower_role {tower_role!r}")


@dataclass(frozen=True)
class Sidecar:
    """A submodule checkpointed in its own file next to the root checkpoint.

    ``module`` is the dotted attribute path of the destination submodule.
    Parameter names containing any ``optional_substrings`` entry may be absent
    from the sidecar file.
    """

    file: str
    module: str
    optional_substrings: tuple[str, ...] = ()


@dataclass(frozen=True)
class NativeSource:
    """Hugging Face native checkpoint bindings for a wrapped serving model.

    The native loader resolves ``config_cls``/``tokenizer_cls`` from the model
    directory, meta-initializes ``module_cls``, and streams the checkpoint into
    it one tensor at a time. ``min_version_key`` names a config entry holding
    the minimum model-code version the checkpoint requires; the loader rejects
    checkpoints newer than ``code_version``.
    """

    config_cls: Any
    module_cls: Any
    tokenizer_cls: Any
    use_fast: bool = False
    extra_special_tokens: dict[str, Any] | None = None
    min_version_key: str | None = None
    code_version: str | None = None


@dataclass(frozen=True)
class GraphSource:
    """Eagerly constructed neural-graph bindings for a composite checkpoint.

    The composite loader builds ``module_cls`` from
    ``config_cls.from_pretrained(model_dir)``, streams the declared root file
    into it, and casts the loaded graph to ``serving_dtype``.
    """

    config_cls: Any
    module_cls: Any
    serving_dtype: str = "bfloat16"


@dataclass(frozen=True)
class WeightSpec:
    """One model family's declarative checkpoint-to-weights contract.

    ``renames`` and ``stacked`` form the closed transform vocabulary applied by
    the loaders; sharding, tying, padded-vocab writes, and quantized-tensor
    handling ride the destination parameters' weight hooks. ``unmatched``
    decides what a checkpoint name with no rename rule targets: ``"keep"``
    maps it to itself, ``"skip"`` declares it outside this load.
    """

    renames: tuple[Rename, ...] = ()
    stacked: tuple[StackedParamMapping, ...] = ()
    unmatched: str = "keep"
    tower: TowerSplit | None = None
    checkpoint_files: tuple[str, ...] = ()
    sidecars: tuple[Sidecar, ...] = ()
    native: NativeSource | None = None
    graph: GraphSource | None = None

    def __post_init__(self) -> None:
        if self.unmatched not in {"keep", "skip"}:
            raise ValueError(f"unknown unmatched policy {self.unmatched!r}")
        if self.native is not None and self.graph is not None:
            raise ValueError("a WeightSpec declares at most one of native and graph sources")

    @property
    def loader(self) -> str:
        """Registry name of the loader that serves this spec's checkpoint layout."""
        if self.native is not None:
            return "native"
        if self.graph is not None:
            return "composite"
        return "default"

    def map_name(self, name: str) -> str | None:
        """Map one checkpoint tensor name through the ordered rename rules."""
        for rule in self.renames:
            mapped = rule.apply(name)
            if mapped is not None:
                return mapped
        return name if self.unmatched == "keep" else None


def weight_spec_of(model_cls: Any) -> WeightSpec:
    """The model class's declared ``weight_spec``, or the identity default."""
    spec = getattr(model_cls, "weight_spec", None)
    return spec if isinstance(spec, WeightSpec) else WeightSpec()
