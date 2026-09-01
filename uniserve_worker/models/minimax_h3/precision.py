"""Linear-precision policy for MiniMax H3 component boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

LinearPrecision: TypeAlias = Literal["bf16", "fp8", "mxfp8", "nvfp4"]
AttentionLinearPrecision: TypeAlias = Literal["bf16", "fp8", "nvfp4"]
TextEncoderLinearPrecision: TypeAlias = Literal["bf16", "nvfp4"]
VideoVAELinearPrecision: TypeAlias = Literal["fp16", "bf16", "nvfp4"]
H3QuantizationMode: TypeAlias = Literal[
    "quality",
    "balanced",
    "performance",
    "maximum",
]


@dataclass(frozen=True, slots=True)
class H3LinearPrecisionPolicy:
    transformer_attention: AttentionLinearPrecision
    transformer_mlp: LinearPrecision
    text_encoder: TextEncoderLinearPrecision
    video_vae: VideoVAELinearPrecision

    @classmethod
    def from_mode(cls, mode: H3QuantizationMode) -> "H3LinearPrecisionPolicy":
        if mode == "quality":
            return cls(
                transformer_attention="bf16",
                transformer_mlp="bf16",
                text_encoder="bf16",
                video_vae="fp16",
            )
        if mode == "balanced":
            return cls(
                transformer_attention="bf16",
                transformer_mlp="bf16",
                text_encoder="bf16",
                video_vae="nvfp4",
            )
        if mode == "performance":
            return cls(
                transformer_attention="bf16",
                transformer_mlp="fp8",
                text_encoder="bf16",
                video_vae="nvfp4",
            )
        if mode == "maximum":
            return cls(
                transformer_attention="nvfp4",
                transformer_mlp="nvfp4",
                text_encoder="bf16",
                video_vae="nvfp4",
            )
        raise ValueError(f"unsupported H3 quantization mode {mode!r}")

    @classmethod
    def resolve(
        cls,
        shorthand: LinearPrecision,
        *,
        transformer_attention: AttentionLinearPrecision | None = None,
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
        elif shorthand == "mxfp8":
            base = cls(
                transformer_attention="bf16",
                transformer_mlp="mxfp8",
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
        elif shorthand == "bf16":
            base = cls(
                transformer_attention="bf16",
                transformer_mlp="bf16",
                text_encoder="bf16",
                video_vae="fp16",
            )
        else:
            raise ValueError(f"unsupported H3 linear precision shorthand {shorthand!r}")
        return cls(
            transformer_attention=transformer_attention or base.transformer_attention,
            transformer_mlp=transformer_mlp or base.transformer_mlp,
            text_encoder=text_encoder or base.text_encoder,
            video_vae=video_vae or base.video_vae,
        )

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> "H3LinearPrecisionPolicy":
        unknown = set(value) - {"mode", "quant_method", "components"}
        if unknown:
            raise ValueError(f"quantization_config has unknown fields {sorted(unknown)!r}")
        if "mode" in value and "quant_method" in value:
            raise ValueError("quantization_config cannot specify both mode and quant_method")

        mode = value.get("mode")
        method = value.get("quant_method")
        if mode is not None:
            if not isinstance(mode, str) or mode not in {
                "quality",
                "balanced",
                "performance",
                "maximum",
            }:
                raise ValueError(
                    "quantization_config.mode must be 'quality', 'balanced', "
                    "'performance', or 'maximum'"
                )
            base = cls.from_mode(cast(H3QuantizationMode, mode))
        elif method is not None:
            if not isinstance(method, str) or method not in {
                "bf16",
                "fp8",
                "mxfp8",
                "nvfp4",
            }:
                raise ValueError(
                    "quantization_config.quant_method must be 'bf16', 'fp8', 'mxfp8', or 'nvfp4'"
                )
            base = cls.resolve(cast(LinearPrecision, method))
        else:
            base = cls.from_mode("balanced")

        raw_components = value.get("components", {})
        if not isinstance(raw_components, Mapping):
            raise TypeError("quantization_config.components must be an object")
        component_names = {
            "transformer.attention",
            "transformer.mlp",
            "text_encoder",
            "video_vae",
        }
        unknown_components = set(raw_components) - component_names
        if unknown_components:
            raise ValueError(
                f"quantization_config.components has unknown entries {sorted(unknown_components)!r}"
            )

        def component(name: str, choices: set[str]) -> str | None:
            selected = raw_components.get(name)
            if selected is None:
                return None
            if selected not in choices:
                expected = ", ".join(sorted(choices))
                raise ValueError(f"quantization_config.components.{name} must be one of {expected}")
            return str(selected)

        transformer_attention = cast(
            AttentionLinearPrecision | None,
            component("transformer.attention", {"bf16", "fp8", "nvfp4"}),
        )
        transformer_mlp = cast(
            LinearPrecision | None,
            component("transformer.mlp", {"bf16", "fp8", "mxfp8", "nvfp4"}),
        )
        text_encoder = cast(
            TextEncoderLinearPrecision | None,
            component("text_encoder", {"bf16", "nvfp4"}),
        )
        video_vae = cast(
            VideoVAELinearPrecision | None,
            component("video_vae", {"fp16", "bf16", "nvfp4"}),
        )
        return cls(
            transformer_attention=transformer_attention or base.transformer_attention,
            transformer_mlp=transformer_mlp or base.transformer_mlp,
            text_encoder=text_encoder or base.text_encoder,
            video_vae=video_vae or base.video_vae,
        )
