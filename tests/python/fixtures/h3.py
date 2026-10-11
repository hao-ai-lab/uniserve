"""MiniMax-H3 configurations of the released checkpoint families.

The released architecture values are stated once, by the sidecars of the
checkpoint fixture ``fixtures/models/fasth3``; the factories below read them
with the production reader and apply per-test overrides.
"""

from dataclasses import replace
from functools import cache
from pathlib import Path

from uniserve.diffusion import BlockGrid, RungGrid, UniformGrid
from uniserve.loading import Config as IOConfig
from uniserve.media import image
from uniserve_models.minimax_h3 import (
    Config,
    DenoiserConfig,
    DenseAttention,
    SparseAttention,
    TransformerConfig,
    audio_vae,
    video_vae,
)
from uniserve_models.minimax_h3.config import DMD_CANVASES, read_config
from uniserve_models.minimax_h3.encoder import TextEncoderConfig

CHECKPOINT = Path(__file__).parent / "models/fasth3"

# The 768p 16:9 canvas, one FastH3 DMD exports generate.
WIDE = image.Config(768, 1344)
# Released audio scheduler shift.
AUDIO_SHIFT = 3.0


@cache
def _released() -> Config:
    return read_config(CHECKPOINT, IOConfig(), sources={})


def transformer_config(**overrides) -> TransformerConfig:
    """The released DiT dimensions, with ``overrides`` applied."""
    transformer = _released().denoisers["transformer"].transformer
    return replace(transformer, **overrides)


def text_encoder_config(**overrides) -> TextEncoderConfig:
    """The released text encoder dimensions, with ``overrides`` applied."""
    return replace(_released().text_encoder, **overrides)


def video_vae_config(**overrides) -> video_vae.Config:
    """The released video VAE configuration, with ``overrides`` applied."""
    return replace(_released().video_vae, **overrides)


def audio_vae_config(**overrides) -> audio_vae.Config:
    """The released audio VAE configuration, with ``overrides`` applied."""
    return replace(_released().audio_vae, **overrides)


def dmd_denoiser(
    transformer: TransformerConfig | None = None,
    *,
    rungs: tuple[int, ...] = (999, 749, 500, 250),
    video_shift: float = 12.0,
    sparsity: float = 0.9,
) -> DenoiserConfig:
    """A FastH3 DMD student: sparse tile-64 attention, text-only.

    It generates the export's training buckets (``DMD_CANVASES``).
    """
    return DenoiserConfig(
        transformer=transformer or transformer_config(),
        grids={
            "video": RungGrid(
                rungs, shift=video_shift, num_train_timesteps=1000.0
            ),
            "audio": RungGrid(
                rungs, shift=AUDIO_SHIFT, num_train_timesteps=1000.0
            ),
        },
        attention=SparseAttention(tile=64, sparsity=sparsity),
        tasks=("t2va",),
        canvases=DMD_CANVASES,
    )


def base_denoiser(
    transformer: TransformerConfig | None = None,
    *,
    tasks: tuple[str, ...] = ("t2va", "fl2va"),
    points: int = 50,
) -> DenoiserConfig:
    """A released diffusers DiT: dense attention over the uniform grid."""
    return DenoiserConfig(
        transformer=transformer or transformer_config(),
        grids={
            "video": UniformGrid(points, shift=12.0),
            "audio": UniformGrid(points, shift=AUDIO_SHIFT),
        },
        attention=DenseAttention(),
        tasks=tasks,
        canvases=None,
    )


def omniref_denoiser(
    transformer: TransformerConfig | None = None,
) -> DenoiserConfig:
    """A FastH3 OmniRef PDD student: 32 heads, tile-128 VSA over segments."""
    return DenoiserConfig(
        transformer=replace(
            transformer or transformer_config(), output_heads=32
        ),
        grids={
            name: BlockGrid(
                32, (0, 4, 8, 12, 16, 20, 24, 28, 32), shift=shift, max_t=0.999
            )
            for name, shift in (("video", 12.0), ("audio", AUDIO_SHIFT))
        },
        attention=SparseAttention(tile=128, sparsity=0.9, reference_keep=0.1),
        tasks=("ref2va",),
        canvases=None,
    )


def fasth3_config() -> Config:
    """A FastH3 t2va export: one DMD denoiser."""
    return Config(
        text_encoder=text_encoder_config(),
        denoisers={"transformer": dmd_denoiser()},
        video_vae=video_vae_config(),
        audio_vae=audio_vae_config(),
    )


def base_config() -> Config:
    """The base release: both DiT partitions."""
    return Config(
        text_encoder=text_encoder_config(),
        denoisers={
            "transformer": base_denoiser(),
            "transformer_ref": base_denoiser(tasks=("ref2va",)),
        },
        video_vae=video_vae_config(),
        audio_vae=audio_vae_config(),
    )
