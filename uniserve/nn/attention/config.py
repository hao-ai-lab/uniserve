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
    """Gather K/V, expose peer-owned K/V, or gather before peer publication."""

    gather_axis: str | None = None
    peer_axis: str | None = None

    def __post_init__(self) -> None:
        axes = tuple(
            axis
            for axis in (self.gather_axis, self.peer_axis)
            if axis is not None
        )
        if not axes or len(set(axes)) != len(axes):
            raise ValueError(
                "context attention requires distinct nonempty axes"
            )
        for axis in axes:
            _axis(axis)


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
            and self.heads.axis
            in (self.context.gather_axis, self.context.peer_axis)
        ):
            raise ValueError("head and context axes must be independent")
