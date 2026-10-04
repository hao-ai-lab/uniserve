"""MiniMax-H3 denoiser configurations of the released checkpoint families."""

from dataclasses import replace

from uniserve.media import image
from uniserve.nn.functional import Rounding
from uniserve_models.minimax_h3 import (
    Config,
    DenoiserConfig,
    DenseAttention,
    DmdLadder,
    PddGrid,
    SparseAttention,
    TransformerConfig,
    UniformGrid,
    audio_vae,
    video_vae,
)
from uniserve_models.minimax_h3.encoder import TextEncoderConfig

# The 16:9 canvas FastH3 DMD exports generate.
WIDE = image.Config(768, 1344)


def dmd_denoiser(
    transformer: TransformerConfig = TransformerConfig(),
    *,
    rungs: tuple[int, ...] = (999, 749, 500, 250),
    video_shift: float = 12.0,
    sparsity: float = 0.9,
) -> DenoiserConfig:
    """A FastH3 DMD student: sparse tile-64 attention, text-only, 16:9.

    Its transformer rounds once, as normalization assigns FastH3 exports.
    """
    return DenoiserConfig(
        transformer=replace(transformer, rounding=Rounding.ONCE),
        schedule=DmdLadder(
            rungs=rungs, video_shift=video_shift, audio_shift=3.0
        ),
        attention=SparseAttention(tile=64, sparsity=sparsity),
        tasks=("t2va",),
        canvases=(WIDE,),
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
        schedule=UniformGrid(points=points, video_shift=12.0, audio_shift=3.0),
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
        schedule=PddGrid(
            intervals=32,
            nodes=(0, 4, 8, 12, 16, 20, 24, 28, 32),
            video_shift=12.0,
            audio_shift=3.0,
            max_t=0.999,
        ),
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
