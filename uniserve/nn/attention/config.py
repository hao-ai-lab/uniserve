"""Mathematical axes for head exchange and context visibility."""

from dataclasses import dataclass
from typing import Literal


def _axis(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("attention axes must be nonempty names")


@dataclass(frozen=True, slots=True)
class Ulysses:
    """Exchange token and head partitions along one topology axis.

    ``projection`` selects how a merged head projection whose branches are
    whole heads (``branch_width``) reaches each member's heads; both produce
    the same values:

    - ``"gather"``: every member gathers all members' input rows and projects
      its own head shard, with weights partitioned by head. A member receives
      ``(members - 1) / members`` of the input rows at the input width.
    - ``"exchange"``: every member projects its own rows for all heads, with
      whole weights, and sends each member that member's head shard. A member
      receives the same rows at its shard's projected width.

    ``projection_rows`` applies to ``"exchange"`` only: each member projects
    and sends its rows in steps of that many rows, so a step's transfer
    overlaps the next step's projection and the attention preparation of
    landed rows. ``None`` projects and exchanges the whole shard once and
    prepares no row before the exchange completes. A step must hold whole
    attention tiles.
    """

    axis: str = "ulysses"
    projection: Literal["gather", "exchange"] = "gather"
    projection_rows: int | None = None

    def __post_init__(self) -> None:
        _axis(self.axis)
        if self.projection not in ("gather", "exchange"):
            raise ValueError("Ulysses projection must be gather or exchange")
        if self.projection_rows is not None and (
            type(self.projection_rows) is not int
            or self.projection_rows < 1
            or self.projection != "exchange"
        ):
            raise ValueError(
                "projection_rows is a positive row count of an exchange "
                "projection"
            )


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
