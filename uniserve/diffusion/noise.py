"""Native normal draws and resolution-dependent numerical noise scaling."""

import math
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class NoiseScale:
    """Resolution-dependent scale for initial normal draws.

    ``constant`` keeps ``value``. ``resolution`` multiplies by
    ``sqrt(tokens / base_tokens)``; ``dynamic_sqrt`` additionally takes the
    square root of the result. The scale is capped at ``maximum``.
    """

    value: float
    mode: Literal["constant", "resolution", "dynamic_sqrt"]
    base_tokens: float
    maximum: float

    def __post_init__(self):
        if self.mode not in {"constant", "resolution", "dynamic_sqrt"}:
            raise ValueError("unknown numerical noise scale mode")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.value, self.base_tokens, self.maximum)
        ):
            raise ValueError("noise scale, token baseline and maximum must be finite and positive")

    def scale(self, tokens: int) -> float:
        if type(tokens) is not int or tokens < 1:
            raise ValueError("noise scaling requires a positive token count")
        value = self.value
        if self.mode != "constant":
            value *= math.sqrt(tokens / self.base_tokens)
        if self.mode == "dynamic_sqrt":
            value = math.sqrt(value)
        return min(value, self.maximum)


def normal_noise(seeds: tuple[int, ...], *, out: tuple[torch.Tensor, ...]) -> None:
    """Draw each seed's modalities in tuple order without changing the global RNG.

    Outputs have a common native draw device and retain their complete sample
    dimensions. Callers choose CPU or CUDA storage and perform any subsequent
    distribution after the draw; splitting a draw changes its random sequence.
    """
    if not isinstance(seeds, tuple) or any(type(seed) is not int for seed in seeds):
        raise TypeError("noise seeds must be a tuple of integers")
    if not isinstance(out, tuple) or not out:
        raise ValueError("normal noise requires at least one output modality")
    device = out[0].device
    if any(
        tensor.ndim < 1
        or tensor.shape[0] != len(seeds)
        or tensor.device != device
        or tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        for tensor in out
    ):
        raise ValueError("noise outputs must align with seeds on one native draw device")
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        for tensor in out:
            # A complete native draw precedes copying into possibly strided
            # borrowed storage. Its shape and dtype fix RNG consumption.
            values = torch.randn(
                tensor.shape[1:], dtype=tensor.dtype, device=device, generator=generator
            )
            tensor[index].copy_(values)
