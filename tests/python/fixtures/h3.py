"""MiniMax-H3 denoiser configurations of the released checkpoint families."""

from dataclasses import replace

from uniserve.diffusion import BlockGrid, RungGrid, UniformGrid
from uniserve.media import image
from uniserve.nn.functional import Rounding
from uniserve_models.minimax_h3 import (
    Config,
    DenoiserConfig,
    DenseAttention,
    SparseAttention,
    TransformerConfig,
    audio_vae,
    video_vae,
)
from uniserve_models.minimax_h3.config import DMD_CANVASES
from uniserve_models.minimax_h3.encoder import TextEncoderConfig

# The 768p 16:9 canvas, one FastH3 DMD exports generate.
WIDE = image.Config(768, 1344)
# Released audio scheduler shift.
AUDIO_SHIFT = 3.0


def dmd_denoiser(
    transformer: TransformerConfig = TransformerConfig(),
    *,
    rungs: tuple[int, ...] = (999, 749, 500, 250),
    video_shift: float = 12.0,
    sparsity: float = 0.9,
) -> DenoiserConfig:
    """A FastH3 DMD student: sparse tile-64 attention, text-only.

    It generates the export's training buckets (``DMD_CANVASES``).

    Its transformer rounds once, as normalization assigns FastH3 exports.
    """
    return DenoiserConfig(
        transformer=replace(transformer, rounding=Rounding.ONCE),
        grids={
            "video": RungGrid(rungs, shift=video_shift, clock=1000.0),
            "audio": RungGrid(rungs, shift=AUDIO_SHIFT, clock=1000.0),
        },
        attention=SparseAttention(tile=64, sparsity=sparsity),
        tasks=("t2va",),
        canvases=DMD_CANVASES,
        max_sequence_rows=None,
    )


def base_denoiser(
    transformer: TransformerConfig = TransformerConfig(),
    *,
    tasks: tuple[str, ...] = ("t2va", "fl2va"),
    points: int = 50,
) -> DenoiserConfig:
    """A released diffusers DiT: dense attention over the uniform grid."""
    return DenoiserConfig(
        transformer=transformer,
        grids={
            "video": UniformGrid(points, shift=12.0),
            "audio": UniformGrid(points, shift=AUDIO_SHIFT),
        },
        attention=DenseAttention(),
        tasks=tasks,
        canvases=None,
        max_sequence_rows=None,
    )


def omniref_denoiser(
    transformer: TransformerConfig = TransformerConfig(),
) -> DenoiserConfig:
    """A FastH3 OmniRef PDD student: 32 heads, multi-region tile-128 VSA."""
    return DenoiserConfig(
        transformer=replace(transformer, output_heads=32),
        grids={
            name: BlockGrid(
                32, (0, 4, 8, 12, 16, 20, 24, 28, 32), shift=shift, max_t=0.999
            )
            for name, shift in (("video", 12.0), ("audio", AUDIO_SHIFT))
        },
        attention=SparseAttention(tile=128, sparsity=0.9, reference_keep=0.1),
        tasks=("ref2va",),
        canvases=None,
        max_sequence_rows=131_072,
    )


def fasth3_config() -> Config:
    """A FastH3 export: one DMD denoiser."""
    return Config(
        text_encoder=TextEncoderConfig(),
        denoisers={"denoiser": dmd_denoiser()},
        video_vae=video_vae.Config(),
        audio_vae=audio_vae.Config(),
    )


def base_config() -> Config:
    """A diffusers root: both released DiTs."""
    return Config(
        text_encoder=TextEncoderConfig(),
        denoisers={
            "denoiser": base_denoiser(),
            "reference_denoiser": base_denoiser(tasks=("ref2va",)),
        },
        video_vae=video_vae.Config(),
        audio_vae=audio_vae.Config(),
    )
