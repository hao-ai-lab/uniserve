"""Diffusion evaluation grids and their FP32 numerical endpoints.

A ``Grid`` declares where one modality's trajectory evaluates the network:
the points of a checkpoint's trained or released grid, or the step count a
request chooses, and the shift that maps them to noise levels.
``Grid.schedule`` materializes the declaration, with the request parameters
it admits, as the ``Schedule`` endpoints a trajectory steps through. Each
grid is evaluated on the host by fixed precision rules, so a schedule does
not depend on the device that holds it.
"""

import math
from abc import ABC, abstractmethod
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


class Grid(ABC):
    """Where one modality's trajectory evaluates the network.

    A denoiser declares one grid per modality. A request may choose the
    step count, the shift, both or neither, as the grid admits;
    ``schedule`` validates those choices and builds the endpoints.
    """

    @property
    @abstractmethod
    def num_steps(self) -> int | None:
        """Network evaluations of the grid, None when requests choose them."""

    @abstractmethod
    def schedule(
        self,
        *,
        steps: int | None = None,
        shift: float | None = None,
        device: torch.device | str,
    ) -> Schedule:
        """Build the schedule of a request's ``steps`` and ``shift``.

        None leaves a parameter to the grid.

        Raises:
            ValueError: The grid does not admit the requested parameters.
        """


@dataclass(frozen=True, slots=True)
class LinearGrid(Grid):
    """A linear trajectory whose step count each request chooses.

    ``steps + 1`` coordinates divide the network time evenly in
    ``direction``; ``shift`` warps them as ``s u / (1 + (s - 1) u)`` of the
    increasing coordinate, in network time or, with ``shift_domain="sigma"``,
    in the noise level. Sigma decreases from one to zero for either
    direction and is the materialized FP32 network time (``descending``) or
    its complement. A request may replace ``shift``; the shift of one leaves
    the coordinates unwarped.
    """

    shift: float
    direction: Literal["ascending", "descending"]
    shift_domain: Literal["time", "sigma"]

    def __post_init__(self):
        _require_shift(self.shift)
        if self.direction not in {
            "ascending",
            "descending",
        } or self.shift_domain not in {"time", "sigma"}:
            raise ValueError("unknown grid direction or shift domain")

    @property
    def num_steps(self) -> None:
        return None

    def schedule(self, *, steps=None, shift=None, device) -> Schedule:
        if type(steps) is not int or steps < 1:
            raise ValueError("a linear grid needs a positive step count")
        shift = self.shift if shift is None else shift
        _require_shift(shift)

        coordinates = []
        for index in range(steps + 1):
            value = (
                index / steps
                if self.direction == "ascending"
                else 1.0 - index / steps
            )
            if shift != 1:
                # The shift formula operates on the increasing coordinate;
                # shifting in the sigma domain mirrors the time coordinate
                # into it.
                increasing = (
                    value if self.shift_domain == "time" else 1.0 - value
                )
                shifted = _shifted(increasing, shift)
                value = (
                    shifted if self.shift_domain == "time" else 1.0 - shifted
                )
            coordinates.append(value)

        times = torch.tensor(coordinates, dtype=torch.float32, device=device)
        return Schedule(
            times,
            times if self.direction == "descending" else 1.0 - times,
            tuple(coordinates),
        )


class FixedGrid(Grid):
    """A checkpoint's own grid, which requests cannot change.

    A request may restate the step count but not choose another, and may
    not choose a shift.
    """

    @property
    @abstractmethod
    def num_steps(self) -> int: ...

    def schedule(self, *, steps=None, shift=None, device) -> Schedule:
        if shift is not None or steps not in (None, self.num_steps):
            raise ValueError(
                f"the checkpoint's grid evaluates the network {self.num_steps} "
                "times at its trained shift"
            )
        return self.endpoints(device)

    @abstractmethod
    def endpoints(self, device: torch.device | str) -> Schedule:
        """Build the grid's schedule on ``device``."""


@dataclass(frozen=True, slots=True)
class UniformGrid(FixedGrid):
    """A shifted descending ``linspace(1, 0, points)`` sigma grid in FP32.

    The grid, its shift ``s u / (1 + (s - 1) u)`` and the deduplication of
    FP32 values the shift collapses are evaluated in FP32, as the released
    MiniMax-H3 (diffusers) scheduler does; network times are ``1 - sigma``
    in FP32. ``points`` counts sigma points, including the clean endpoint,
    so the network evaluates ``points - 1`` times or fewer after
    deduplication.
    """

    points: int
    shift: float

    def __post_init__(self):
        if type(self.points) is not int or self.points < 2:
            raise ValueError("a sigma grid needs at least two points")
        _require_shift(self.shift)

    @property
    def num_steps(self) -> int:
        return self._sigmas().numel() - 1

    def endpoints(self, device) -> Schedule:
        sigmas = self._sigmas()
        timesteps = 1.0 - sigmas
        return Schedule(
            timesteps.to(device),
            sigmas.to(device),
            tuple(float(value) for value in timesteps),
        )

    def _sigmas(self) -> torch.Tensor:
        base = torch.linspace(
            1.0, 0.0, self.points, dtype=torch.float32, device="cpu"
        )
        return torch.unique_consecutive(_shifted(base, float(self.shift)))


@dataclass(frozen=True, slots=True)
class RungGrid(FixedGrid):
    """A distilled student's trained rungs (a DMD ladder).

    Each rung is an unshifted noise level on a ``clock``-step training
    clock, strictly decreasing; divided by ``clock`` it gives ``u`` in
    (0, 1], which ``shift`` warps once in double precision. The clean
    endpoint follows the last rung, and sigma is then materialized in FP32;
    network times are ``1 - sigma`` formed in FP32, while the analytical
    coordinates keep the unrounded ``1 - sigma``.
    """

    rungs: tuple[int, ...]
    shift: float
    clock: float

    def __post_init__(self):
        if not _is_positive(self.clock):
            raise ValueError("a training clock must be finite and positive")
        if (
            not isinstance(self.rungs, tuple)
            or not self.rungs
            or any(
                type(rung) is not int or not 0 < rung <= self.clock
                for rung in self.rungs
            )
            or any(
                left <= right for left, right in zip(self.rungs, self.rungs[1:])
            )
        ):
            raise ValueError(
                "rungs must be strictly decreasing integers within the "
                "training clock"
            )
        _require_shift(self.shift)

    @property
    def num_steps(self) -> int:
        return len(self.rungs)

    def endpoints(self, device) -> Schedule:
        sigmas = tuple(
            _shifted(rung / self.clock, self.shift) for rung in (*self.rungs, 0)
        )
        sigma = torch.tensor(sigmas, dtype=torch.float32, device=device)
        return Schedule(
            1.0 - sigma, sigma, tuple(1.0 - value for value in sigmas)
        )


@dataclass(frozen=True, slots=True)
class BlockGrid(FixedGrid):
    """A fine base-clock grid partitioned into evaluation blocks.

    A parallel-decoding (PDD) student predicts one head per fine interval of
    ``linspace(max_t, 0, intervals + 1)``, an FP64 grid clamped to
    ``max_t``. ``nodes`` are the fine-grid indices of the block boundaries,
    strictly increasing from 0 to ``intervals``. Each evaluation advances
    one block ``[nodes[k], nodes[k + 1])``: the block's heads, fused with
    their normalized integration weights, form one prediction, and an
    ordinary Euler step between the block's node sigmas applies it, since
    the block's total weight is exactly its node-sigma increment. The
    modality reaches its noise levels through ``f_s(u) = s u M / (u (s - 1)
    + M)``, whose fixed point ``M = max_t`` gives the first node the same
    noise level for every shift; node sigmas are rounded once to FP32 and
    network times are ``1 - sigma`` in FP32.
    """

    intervals: int
    nodes: tuple[int, ...]
    shift: float
    max_t: float

    def __post_init__(self):
        if (
            type(self.intervals) is not int
            or self.intervals < 2
            or not isinstance(self.nodes, tuple)
            or len(self.nodes) < 2
            or any(type(node) is not int for node in self.nodes)
            or self.nodes[0] != 0
            or self.nodes[-1] != self.intervals
            or any(
                left >= right for left, right in zip(self.nodes, self.nodes[1:])
            )
        ):
            raise ValueError(
                "a block grid needs nodes increasing strictly from 0 to its "
                "interval count"
            )
        _require_shift(self.shift)
        if not _is_positive(self.max_t) or self.max_t > 1:
            raise ValueError("a block grid's max_t must lie in (0, 1]")
        weights = self.weights
        if not bool(torch.all(torch.isfinite(weights) & (weights != 0))):
            raise ValueError("block grid integration weights must be nonzero")

    @property
    def num_steps(self) -> int:
        return len(self.nodes) - 1

    @property
    def weights(self) -> torch.Tensor:
        """[intervals] FP64 integration weights ``f(u_{j+1}) - f(u_j)``.

        The rational difference ``s M^2 (b - a) / (D(a) D(b))`` with
        ``D(u) = M + (s - 1) u`` avoids cancellation; the identity shift
        keeps the direct difference. A descending grid's weights are
        negative.
        """
        grid = self._fine()
        start, end = grid[:-1], grid[1:]
        if self.shift == 1:
            return end - start
        shift, max_t = float(self.shift), self.max_t
        return (
            shift
            * max_t**2
            * (end - start)
            / ((max_t + (shift - 1.0) * start) * (max_t + (shift - 1.0) * end))
        )

    def block_weights(self, block: int) -> torch.Tensor:
        """Return block ``block``'s normalized FP64 head weights."""
        if type(block) is not int or not 0 <= block < self.num_steps:
            raise ValueError("block must name one evaluation of the grid")
        weights = self.weights[self.nodes[block] : self.nodes[block + 1]]
        return weights / weights.sum()

    def endpoints(self, device) -> Schedule:
        nodes = self._fine()[
            torch.tensor(self.nodes, dtype=torch.long, device="cpu")
        ]
        if self.shift != 1:
            shift = float(self.shift)
            nodes = (
                shift
                * nodes
                * self.max_t
                / (nodes * (shift - 1.0) + self.max_t)
            )
        sigmas = nodes.to(torch.float32)
        timesteps = 1.0 - sigmas
        return Schedule(
            timesteps.to(device),
            sigmas.to(device),
            tuple(float(value) for value in timesteps),
        )

    def _fine(self) -> torch.Tensor:
        grid = torch.linspace(
            self.max_t,
            0.0,
            self.intervals + 1,
            dtype=torch.float64,
            device="cpu",
        )
        return grid.clamp(max=self.max_t)


def _is_positive(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _require_shift(shift) -> None:
    if not _is_positive(shift):
        raise ValueError("a grid shift must be finite and positive")


def _shifted(value, shift):
    """Rectified-flow time shift ``s u / (1 + (s - 1) u)`` of ``value``."""
    return shift * value / (1 + (shift - 1) * value)


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
