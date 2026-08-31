"""Linear-precision policy for MiniMax H3 component boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

LinearPrecision: TypeAlias = Literal["fp8", "nvfp4"]
TextEncoderLinearPrecision: TypeAlias = Literal["bf16", "nvfp4"]
VideoVAELinearPrecision: TypeAlias = Literal["fp16", "bf16", "nvfp4"]


@dataclass(frozen=True, slots=True)
class H3LinearPrecisionPolicy:
    transformer_attention: LinearPrecision
    transformer_mlp: LinearPrecision
    text_encoder: TextEncoderLinearPrecision
    video_vae: VideoVAELinearPrecision

    @classmethod
    def resolve(
        cls,
        shorthand: LinearPrecision,
        *,
        transformer_attention: LinearPrecision | None = None,
        transformer_mlp: LinearPrecision | None = None,
        text_encoder: TextEncoderLinearPrecision | None = None,
        video_vae: VideoVAELinearPrecision | None = None,
    ) -> "H3LinearPrecisionPolicy":
        if shorthand == "fp8":
            base = cls(
                transformer_attention="fp8",
                transformer_mlp="fp8",
                text_encoder="bf16",
                video_vae="fp16",
            )
        elif shorthand == "nvfp4":
            base = cls(
                transformer_attention="nvfp4",
                transformer_mlp="nvfp4",
                text_encoder="nvfp4",
                video_vae="nvfp4",
            )
        else:
            raise ValueError(f"unsupported H3 linear precision shorthand {shorthand!r}")
        return cls(
            transformer_attention=transformer_attention or base.transformer_attention,
            transformer_mlp=transformer_mlp or base.transformer_mlp,
            text_encoder=text_encoder or base.text_encoder,
            video_vae=video_vae or base.video_vae,
        )
