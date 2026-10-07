"""Checkpoint rotary metadata normalized into typed rotary recipes.

Transformers checkpoints describe rotary positions under ``rope_scaling``
(the older key), ``rope_parameters`` (Transformers 5), both, or neither, and
may repeat ``rope_theta`` and ``partial_rotary_factor`` at the top level.
``read_rotary`` resolves those aliases once at the model-package loading
boundary, so model constructors receive only a theta, a typed
``uniserve.nn.rope`` recipe and a partial width.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from uniserve.nn.rope import (
    DynamicScaling,
    LinearScaling,
    LlamaScaling,
    LongRoPEScaling,
    ProportionalScaling,
    RoPEScaling,
    YaRNScaling,
)

__all__ = ["Rotary", "alias", "read_rotary"]


@dataclass(frozen=True, slots=True)
class Rotary:
    """Resolved rotary metadata of one checkpoint text model.

    ``kind`` is the checkpoint's recipe name (``"default"`` when absent);
    ``scaling`` is None exactly for the default recipe.
    """

    kind: str
    theta: float
    scaling: RoPEScaling | None
    partial_rotary_factor: float


def alias(
    primary: Mapping[str, Any],
    name: str,
    aliases: tuple[Mapping[str, Any], ...],
    default: Any,
    *,
    owner: str,
) -> Any:
    """Return one value for a field that checkpoints may duplicate across maps.

    A ``None`` value counts as absent. Present values must all be equal, or
    this raises ``ValueError`` naming ``owner``; with none present,
    ``default`` is returned.
    """
    values = [
        source[name]
        for source in (primary, *aliases)
        if source.get(name) is not None
    ]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(
            f"{owner} checkpoint has conflicting aliases for {name}"
        )
    return values[0] if values else default


def read_rotary(
    text: Mapping[str, Any],
    *,
    owner: str,
    default_theta: float,
    default_original: int,
) -> Rotary:
    """Resolve a text model's rotary recipe from its checkpoint metadata.

    Args:
        text: The text model's configuration map.
        owner: Model name used in error messages.
        default_theta: The rotary base when no map names ``rope_theta``.
        default_original: The pretraining context length used by recipes
            that need one when the checkpoint omits
            ``original_max_position_embeddings``.

    Raises:
        ValueError: When the rotary maps are not objects, aliases disagree,
            or the recipe is unknown.
    """
    # The recipe name comes from rope_type, falling back to the type key,
    # and every type key present must name the same recipe.
    scaling = text.get("rope_scaling") or {}
    parameters = text.get("rope_parameters") or {}
    if not isinstance(scaling, Mapping) or not isinstance(parameters, Mapping):
        raise ValueError(f"{owner} rotary metadata must be an object")
    kind = alias(
        parameters,
        "rope_type",
        (scaling,),
        parameters.get("type", scaling.get("type", "default")),
        owner=owner,
    )
    if any(
        source.get("type", kind) != kind for source in (parameters, scaling)
    ):
        raise ValueError(
            f"{owner} checkpoint has conflicting rotary type aliases"
        )

    def option(name, default=None):
        return alias(parameters, name, (scaling,), default, owner=owner)

    recipe: RoPEScaling | None = None
    if kind != "default":
        factor = option("factor")
        original = option("original_max_position_embeddings", default_original)
        match kind:
            case "linear":
                recipe = LinearScaling(factor)
            case "dynamic":
                recipe = DynamicScaling(factor)
            case "proportional":
                recipe = ProportionalScaling(factor)
            case "yarn":
                recipe = YaRNScaling(
                    factor,
                    original,
                    option("attention_factor"),
                    option("beta_fast", 32.0),
                    option("beta_slow", 1.0),
                    option("mscale"),
                    option("mscale_all_dim"),
                    option("truncate", True),
                )
            case "longrope":
                short = option("short_factor", ())
                long = option("long_factor", ())
                if not all(
                    isinstance(factors, (list, tuple))
                    for factors in (short, long)
                ):
                    raise ValueError(
                        f"{owner} LongRoPE short and long factors must be lists"
                    )
                recipe = LongRoPEScaling(
                    factor,
                    original,
                    option("attention_factor"),
                    tuple(short),
                    tuple(long),
                )
            case "llama3":
                recipe = LlamaScaling(
                    factor,
                    original,
                    option("low_freq_factor", 1.0),
                    option("high_freq_factor", 4.0),
                )
            case _:
                raise ValueError(f"unsupported {owner} rotary recipe {kind!r}")

    return Rotary(
        kind=kind,
        theta=alias(
            text,
            "rope_theta",
            (parameters, scaling),
            default_theta,
            owner=owner,
        ),
        scaling=recipe,
        partial_rotary_factor=alias(
            text,
            "partial_rotary_factor",
            (parameters, scaling),
            1.0,
            owner=owner,
        ),
    )
