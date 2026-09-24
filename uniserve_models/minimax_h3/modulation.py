"""H3's shared timestep projection and final affine normalization."""

from __future__ import annotations

import torch
from torch import nn

from uniserve.nn import Linear, RMSNorm
from uniserve.nn.timestep import timestep_embedding

from .config import TransformerConfig


class TimestepEmbedding(nn.Module):
    """Apply one checkpoint projection to the video and audio time coordinates.

    H3 trains one shared pair of linear maps. Both named modality projections
    retain that same module and parameter identity; the coordinates differ.
    These FP32 products feed SiLU before the BF16 modulation projections.

    No resident module holds this embedding: checkpoint loading builds it
    transiently to precompute the transformer's per-step modulation products
    for the fixed ladder, then releases it.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.frequency_dim = config.frequency_dim
        self.video_projection = nn.Sequential(
            Linear(
                config.frequency_dim,
                config.time_hidden_dim,
                dtype=torch.float32,
            ),
            nn.SiLU(),
            Linear(
                config.time_hidden_dim, config.time_dim, dtype=torch.float32
            ),
        )
        self.audio_projection = self.video_projection

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps carries [..., 2] coordinates in fixed (video, audio) order.
        if timesteps.ndim < 1 or timesteps.shape[-1] != 2:
            raise ValueError(
                "H3 timesteps must end in the ordered video/audio coordinates"
            )
        features = timestep_embedding(timesteps, self.frequency_dim)
        # One GEMM over both modalities preserves their common FP32 rounding.
        return self.video_projection(features).reshape(*timesteps.shape, -1)


class OutputNorm(nn.Module):
    """Apply per-token shift and scale after the final RMS normalization."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self, hidden: torch.Tensor, modulation: torch.Tensor
    ) -> torch.Tensor:
        # modulation is [..., 2 * hidden]: one shift and one scale per token.
        if (
            modulation.shape[-1] != 2 * hidden.shape[-1]
            or torch.broadcast_shapes(modulation.shape[:-1], hidden.shape[:-1])
            != hidden.shape[:-1]
        ):
            raise ValueError(
                "output modulation must supply shift and scale for every token"
            )
        shift, scale = modulation.chunk(2, dim=-1)
        return self.norm(hidden) * (1.0 + scale) + shift
