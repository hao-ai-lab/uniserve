"""Mathematical axes for head exchange and context visibility."""

from dataclasses import dataclass


def _axis(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("attention axes must be nonempty names")


@dataclass(frozen=True, slots=True)
class Ulysses:
    """Exchange token and head partitions along one topology axis."""

    axis: str = "ulysses"

    def __post_init__(self) -> None:
        _axis(self.axis)


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextParallelConfig:
    """Gather every owner's K/V along one topology axis.

    Each rank holds the queries of its own rows and attends against the whole
    context, so the owners' keys and values are gathered into its storage
    before the attention call reads them.
    """

    gather_axis: str

    def __post_init__(self) -> None:
        _axis(self.gather_axis)


@dataclass(frozen=True, slots=True, kw_only=True)
class AttentionParallelConfig:
    """Choose independent head and context mechanisms without copying
    degrees.
    """  # noqa: D205

    heads: Ulysses | None = None
    context: ContextParallelConfig | None = None

    def __post_init__(self) -> None:
        if self.heads is not None and not isinstance(self.heads, Ulysses):
            raise TypeError("heads must be Ulysses or None")
        if self.context is not None and not isinstance(
            self.context, ContextParallelConfig
        ):
            raise TypeError("context must be ContextParallelConfig or None")
        if (
            self.heads is not None
            and self.context is not None
            and self.heads.axis == self.context.gather_axis
        ):
            raise ValueError("head and context axes must be independent")
