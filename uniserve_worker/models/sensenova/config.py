"""Loader-owned config objects for checkpoints with custom HF config classes."""
from __future__ import annotations

from typing import Any

from transformers import Qwen3Config, Qwen3MoeConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

__all__ = [
    'NeoVisionConfig',
    'NeoLlmConfig',
    'NeoMoeLlmConfig',
    'build_neo_llm_config',
    'NeoChatConfig',
]

logger = logging.get_logger(__name__)


def _first_scalar(value: Any) -> Any:
    """Collapse a per-stage list/tuple config value to its first scalar entry."""
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


# Architecture names used to default an absent sub-config section (the standard
# HF ``is_composition`` contract). Named once so the magic strings are not
# repeated inline in the ``NeoChatConfig`` default-resolution branches.
_DEFAULT_VISION_ARCHITECTURE = "NEOVisionModel"
_DEFAULT_LLM_ARCHITECTURE = "Qwen3ForCausalLM"


def _ensure_layer_types(config: PretrainedConfig) -> None:
    """Populate ``config.layer_types`` if absent or stale.

    Shared by NeoLlmConfig and NeoMoeLlmConfig so the sliding-window layer
    derivation lives in exactly one place.
    """
    existing = getattr(config, "layer_types", None)
    if existing and len(existing) == config.num_hidden_layers:
        return
    use_swa = bool(getattr(config, "use_sliding_window", False)) and getattr(
        config, "sliding_window", None
    ) is not None
    max_window_layers = int(getattr(config, "max_window_layers", 0) or 0)
    config.layer_types = [
        "sliding_attention" if (use_swa and i >= max_window_layers) else "full_attention"
        for i in range(config.num_hidden_layers)
    ]


class NeoVisionConfig(PretrainedConfig):
    model_type = "neo_vision"

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 16,
        hidden_size: int = 1024,
        llm_hidden_size: int | list[int] | tuple[int, ...] = 2048,
        downsample_ratio: float | list[float] | tuple[float, ...] = 0.5,
        rope_theta_vision: float = 10000.0,
        max_position_embeddings_vision: int = 10000,
        min_pixels: int = 65536,
        max_pixels: int = 4194304,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        # Checkpoints may express ``llm_hidden_size``/``downsample_ratio`` as a
        # per-stage list/tuple; the vision tower consumes a single scalar, so
        # normalize to the first entry here (the one normalization owner) rather
        # than re-deriving it in the encoder.
        self.llm_hidden_size = _first_scalar(llm_hidden_size)
        self.downsample_ratio = _first_scalar(downsample_ratio)
        self.rope_theta_vision = rope_theta_vision
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]
        # If we did not descend into a nested ``vision_config`` we are about to
        # treat the top-level config dict as the vision config. Validate the
        # resolved model_type so a mismatched (non-vision) checkpoint surfaces a
        # warning instead of silently producing a misconfigured object.
        resolved_type = config_dict.get("model_type")
        if resolved_type is not None and resolved_type != cls.model_type:
            logger.warning(
                "Instantiating %s from a config of type %r (expected %r); "
                "this checkpoint may not expose a 'vision_config' section.",
                cls.__name__,
                resolved_type,
                cls.model_type,
            )
        return cls.from_dict(config_dict, **kwargs)


class NeoLlmConfig(Qwen3Config):
    def __init__(
        self,
        rope_theta_hw: float = 10000.0,
        max_position_embeddings_hw: int = 10000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw
        self._ensure_layer_types()

    def _ensure_layer_types(self) -> None:
        _ensure_layer_types(self)


class NeoMoeLlmConfig(Qwen3MoeConfig):
    def __init__(
        self,
        rope_theta_hw: float = 10000.0,
        max_position_embeddings_hw: int = 10000,
        gen_num_experts: int | None = None,
        gen_num_experts_per_tok: int | None = None,
        gen_moe_intermediate_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw
        self.gen_num_experts = (
            int(gen_num_experts) if gen_num_experts is not None else int(self.num_experts)
        )
        self.gen_num_experts_per_tok = (
            int(gen_num_experts_per_tok)
            if gen_num_experts_per_tok is not None
            else int(self.num_experts_per_tok)
        )
        self.gen_moe_intermediate_size = (
            int(gen_moe_intermediate_size)
            if gen_moe_intermediate_size is not None
            else int(self.moe_intermediate_size)
        )
        self._ensure_layer_types()

    def _ensure_layer_types(self) -> None:
        _ensure_layer_types(self)


def _is_moe_llm_config(llm_config: Any) -> bool:
    """Heuristically decide whether ``llm_config`` describes a MoE LLM.

    Checkpoints reach the loader with a raw HF config dict, before any concrete
    config class has been instantiated, so there is no reliable type to switch
    on. Detection therefore layers three independent signals, in order of
    decreasing confidence:

    1. ``model_type`` contains ``"moe"`` (the canonical Qwen3 MoE type is
       ``"qwen3_moe"``; dense is ``"qwen3"``).
    2. any entry in ``architectures`` contains ``"moe"`` (covers custom/derived
       MoE classes the loader cannot enumerate ahead of time).
    3. an explicit ``num_experts > 1`` field (the structural signal a router
       config carries; dense Qwen3 configs do not define ``num_experts``).

    The substring matches are deliberately permissive so non-canonical MoE
    checkpoints are still routed to NeoMoeLlmConfig rather than being silently
    loaded as dense.
    """
    if isinstance(llm_config, dict):
        model_type = llm_config.get("model_type", "")
        archs = llm_config.get("architectures") or []
        has_num_experts = "num_experts" in llm_config
        num_experts = llm_config.get("num_experts", 0)
    else:
        model_type = getattr(llm_config, "model_type", "")
        archs = getattr(llm_config, "architectures", None) or []
        has_num_experts = hasattr(llm_config, "num_experts")
        num_experts = getattr(llm_config, "num_experts", 0)

    if isinstance(model_type, str) and "moe" in model_type.lower():
        return True
    if any("moe" in str(arch).lower() for arch in archs):
        return True
    return bool(has_num_experts) and int(num_experts or 0) > 1


def build_neo_llm_config(llm_config: Any) -> Any:
    if isinstance(llm_config, dict):
        if _is_moe_llm_config(llm_config):
            return NeoMoeLlmConfig(**llm_config)
        return NeoLlmConfig(**llm_config)
    return llm_config


class NeoChatConfig(PretrainedConfig):
    model_type = "neo_chat"
    is_composition = True

    def __init__(
        self,
        vision_config: Any | None = None,
        llm_config: Any | None = None,
        use_backbone_lora: int = 0,
        use_llm_lora: int = 0,
        downsample_ratio: float = 0.5,
        template: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = {"architectures": [_DEFAULT_VISION_ARCHITECTURE]}
        if llm_config is None:
            llm_config = {"architectures": [_DEFAULT_LLM_ARCHITECTURE]}
        self.vision_config = (
            NeoVisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config
        )
        self.llm_config = build_neo_llm_config(llm_config)
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        return output
