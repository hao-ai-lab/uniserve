"""Analytical diffusion coordinates and their FP32 numerical endpoints."""

import math
from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class Schedule:
    """Trajectory endpoints: FP32 network times, sigmas, and host coordinates.

    All three hold ``num_steps + 1`` aligned entries. ``coordinates`` retains
    the unrounded analytical values that guidance interval comparisons use.
    ``steps`` holds the entry indices ``0..num_steps`` as int64 on the
    endpoints' device, so a step is named by device data (``step``) rather
    than by a host integer baked into a computation.
    """

    timesteps: torch.Tensor
    sigmas: torch.Tensor
    coordinates: tuple[float, ...]
    steps: torch.Tensor = field(init=False)

    def __post_init__(self):
        if (
            self.timesteps.ndim != 1
            or self.sigmas.shape != self.timesteps.shape
            or self.timesteps.numel() < 2
            or len(self.coordinates) != self.timesteps.numel()
            or self.timesteps.device != self.sigmas.device
            or self.timesteps.dtype != torch.float32
            or self.sigmas.dtype != torch.float32
            or not isinstance(self.coordinates, tuple)
            or any(not math.isfinite(value) for value in self.coordinates)
        ):
            raise ValueError(
                "schedules require aligned FP32 endpoints and analytical "
                "coordinates"
            )
        object.__setattr__(
            self,
            "steps",
            torch.arange(
                self.timesteps.numel(),
                dtype=torch.int64,
                device=self.timesteps.device,
            ),
        )

    @property
    def num_steps(self) -> int:
        return len(self.coordinates) - 1

    def step(self, index: int) -> torch.Tensor:
        """Name evaluation ``index`` as a [1] int64 device view of ``steps``.

        The view's value, not its address, identifies the step, so a
        computation captured with one step's view evaluates any step once the
        value is copied into its input.

        Raises:
            ValueError: ``index`` is not an evaluation of this schedule.
        """
        if type(index) is not int or not 0 <= index < self.num_steps:
            raise ValueError("schedule step must be one of its evaluations")
        return self.steps[index : index + 1]


def make_schedule(
    steps: int,
    *,
    shift: float,
    direction: Literal["ascending", "descending"],
    shift_domain: Literal["time", "sigma"],
    device: torch.device | str,
) -> Schedule:
    """Shift a complete linear trajectory, retaining unrounded host coordinates.

    The network time follows direction. Sigma decreases from one to zero for
    either direction, and is computed from the materialized FP32 network time.
    """
    if (
        type(steps) is not int
        or steps < 1
        or not math.isfinite(shift)
        or shift <= 0
    ):
        raise ValueError("schedule steps and shift must be positive")
    if direction not in {"ascending", "descending"} or shift_domain not in {
        "time",
        "sigma",
    }:
        raise ValueError("unknown schedule direction or shift domain")
    coordinates = []
    for index in range(steps + 1):
        value = (
            index / steps if direction == "ascending" else 1.0 - index / steps
        )
        if shift != 1:
            # The shift formula operates on the increasing coordinate; shifting
            # in the sigma domain mirrors the time coordinate into it.
            coordinate = value if shift_domain == "time" else 1.0 - value
            shifted = shift * coordinate / (1.0 + (shift - 1.0) * coordinate)
            value = shifted if shift_domain == "time" else 1.0 - shifted
        coordinates.append(value)

    times = torch.tensor(coordinates, dtype=torch.float32, device=device)
    return Schedule(
        times,
        times if direction == "descending" else 1.0 - times,
        tuple(coordinates),
    )


def _shifted(value, shift):
    """Rectified-flow time shift ``s u / (1 + (s - 1) u)`` of ``value``."""
    return shift * value / (1 + (shift - 1) * value)


def uniform_grid(
    points: int, *, shift: float, device: torch.device | str
) -> Schedule:
    """Shift a descending ``linspace(1, 0, points)`` sigma grid in FP32.

    The grid, its shift and the deduplication of FP32 values the shift
    collapses are evaluated in FP32 on the host, so the schedule does not
    depend on the device; network times are ``1 - sigma`` in FP32. This is
    the released MiniMax-H3 schedule: ``points`` counts sigma points,
    including the clean endpoint, so it evaluates the network
    ``points - 1`` times or fewer after deduplication.

    Raises:
        ValueError: Fewer than two points or a nonpositive shift.
    """
    if (
        type(points) is not int
        or points < 2
        or not math.isfinite(shift)
        or shift <= 0
    ):
        raise ValueError("a sigma grid needs two points and a positive shift")
    base = torch.linspace(1.0, 0.0, points, dtype=torch.float32, device="cpu")
    sigmas = torch.unique_consecutive(_shifted(base, float(shift)))
    timesteps = 1.0 - sigmas
    return Schedule(
        timesteps.to(device),
        sigmas.to(device),
        tuple(float(value) for value in timesteps),
    )


def ladder(
    rungs: tuple[int, ...],
    *,
    shift: float,
    clock: float,
    device: torch.device | str,
) -> Schedule:
    """Shift trained rungs of a ``clock``-step training clock once.

    Each rung divided by ``clock`` gives ``u`` in (0, 1]; sigma is the
    shifted ``u`` evaluated in double precision, followed by the clean
    endpoint, then materialized in FP32. Network times are ``1 - sigma``
    formed in FP32; the analytical coordinates keep the unrounded
    ``1 - sigma``.
    """
    sigmas = tuple(_shifted(rung / clock, shift) for rung in (*rungs, 0))
    sigma = torch.tensor(sigmas, dtype=torch.float32, device=device)
    return Schedule(1.0 - sigma, sigma, tuple(1.0 - value for value in sigmas))


def fixed_point_shift(base, shift: float, max_t: float):
    """Shift ``f_s(u) = s u M / (u (s - 1) + M)`` with fixed point ``M``.

    ``M = max_t`` maps to itself for every shift, so the first node of a
    grid on ``[0, M]`` has the same noise level in every modality; the
    identity shift returns ``base`` unchanged.
    """
    if shift == 1:
        return base
    return shift * base * max_t / (base * (shift - 1.0) + max_t)


def fixed_point_increments(grid: torch.Tensor, shift: float, max_t: float):
    """Return ``f_s(grid[j + 1]) - f_s(grid[j])`` without cancellation.

    The rational difference equals ``s M^2 (b - a) / (D(a) D(b))`` with
    ``D(u) = M + (s - 1) u``; the identity shift keeps the direct
    difference. The increments of a descending grid are negative.
    """
    start, end = grid[:-1], grid[1:]
    if shift == 1:
        return end - start
    return (
        shift
        * max_t**2
        * (end - start)
        / ((max_t + (shift - 1.0) * start) * (max_t + (shift - 1.0) * end))
    )


@dataclass(frozen=True, slots=True)
class BlockGrid:
    """A fine base-clock grid partitioned into evaluation blocks.

    A parallel-decoding (PDD) student predicts one head per fine interval.
    Each evaluation advances one block of intervals ``[nodes[k],
    nodes[k + 1])``: the block's heads, fused with their normalized
    integration weights, form one prediction, and an ordinary Euler step
    between the block's node sigmas applies it, since the block's total
    weight is exactly its node-sigma increment.

    Attributes:
        schedule: Node sigmas (FP32, verbatim) and network times
            ``1 - sigma`` in FP32.
        weights: [intervals] FP64 integration weights ``f(u_{j+1}) -
            f(u_j)`` of every fine interval.
        nodes: Fine-grid indices of the block boundaries.
    """

    schedule: Schedule
    weights: torch.Tensor
    nodes: tuple[int, ...]

    def block_weights(self, block: int) -> torch.Tensor:
        """Return block ``block``'s normalized FP64 head weights."""
        if type(block) is not int or not 0 <= block < len(self.nodes) - 1:
            raise ValueError("block must name one evaluation of the grid")
        weights = self.weights[self.nodes[block] : self.nodes[block + 1]]
        total = weights.sum()
        if not bool(torch.isfinite(total)) or float(total) == 0:
            raise ValueError("a block's integration weight must be nonzero")
        return weights / total


def block_grid(
    intervals: int,
    nodes: tuple[int, ...],
    *,
    shift: float,
    max_t: float,
    device: torch.device | str,
) -> BlockGrid:
    """Build one modality's PDD block schedule and interval weights.

    The fine grid ``linspace(max_t, 0, intervals + 1)`` is FP64 and clamped
    to ``max_t``; node sigmas are the shifted nodes, rounded once to FP32.

    Raises:
        ValueError: Fewer than two intervals, nodes that do not increase
            strictly from 0 to ``intervals``, or a degenerate shift.
    """
    if (
        type(intervals) is not int
        or intervals < 2
        or len(nodes) < 2
        or nodes[0] != 0
        or nodes[-1] != intervals
        or any(left >= right for left, right in zip(nodes, nodes[1:]))
        or not math.isfinite(shift)
        or shift <= 0
        or not 0 < max_t <= 1
    ):
        raise ValueError(
            "a block grid needs nodes from 0 to its interval count and a "
            "positive shift"
        )
    grid = torch.linspace(
        max_t, 0.0, intervals + 1, dtype=torch.float64, device="cpu"
    )
    grid = grid.clamp(max=max_t)
    sigmas = fixed_point_shift(
        grid[torch.tensor(nodes, dtype=torch.long, device="cpu")],
        float(shift),
        max_t,
    ).to(torch.float32)
    timesteps = 1.0 - sigmas
    weights = fixed_point_increments(grid, float(shift), max_t)
    if not bool(torch.all(torch.isfinite(weights) & (weights != 0))):
        raise ValueError("block grid integration weights must be nonzero")
    return BlockGrid(
        Schedule(
            timesteps.to(device),
            sigmas.to(device),
            tuple(float(value) for value in timesteps),
        ),
        weights,
        tuple(nodes),
    )


def fuse_heads(
    parameter: torch.Tensor, weights: torch.Tensor, *, heads: int, start: int
) -> torch.Tensor:
    """Fuse heads ``[start, start + len(weights))`` of a head-major parameter.

    ``parameter`` stacks ``heads`` equal heads along dim 0 (``[heads *
    width, ...]``, a PDD output projection's weight or bias). The result is
    the ``weights``-weighted sum of the selected heads, the weights rounded
    to FP32 and the sum accumulated in FP32, then rounded back to the
    parameter's dtype.

    Raises:
        ValueError: The heads do not divide dim 0 or the block leaves them.
    """
    count = weights.numel()
    if (
        type(heads) is not int
        or heads < 1
        or parameter.shape[0] % heads
        or weights.ndim != 1
        or not 0 <= start < start + count <= heads
    ):
        raise ValueError("fused heads must be a block of whole parameter heads")
    selected = parameter.unflatten(0, (heads, -1))[start : start + count]
    fused = torch.einsum(
        "n,n...->...",
        weights.to(device=parameter.device, dtype=torch.float32),
        selected.to(torch.float32),
    )
    return fused.to(parameter.dtype)
