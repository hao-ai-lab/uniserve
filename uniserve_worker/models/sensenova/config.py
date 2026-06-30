"""Loader-owned config objects for checkpoints with custom HF config classes."""
from __future__ import annotations

from typing import Any

from transformers import Qwen3Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

__all__ = [
    'NeoVisionConfig',
    'NeoLlmConfig',
    'build_neo_llm_config',
    'NeoChatConfig',
]

logger = logging.get_logger(__name__)


def _vision_stage_scalar(value: Any, field_name: str) -> Any:
    """Normalize a vision config field that may be serialized per stage."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{field_name} must not be empty")
        return value[0]
    return value


# Architecture names used to default an absent sub-config section (the standard
# HF ``is_composition`` contract). Named once so the magic strings are not
# repeated inline in the ``NeoChatConfig`` default-resolution branches.
_DEFAULT_VISION_ARCHITECTURE = "NEOVisionModel"
_DEFAULT_LLM_ARCHITECTURE = "Qwen3ForCausalLM"


def _ensure_layer_types(config: PretrainedConfig) -> None:
    """Populate ``config.layer_types`` if absent or stale.

    Shared by SenseNova LLM configs so the sliding-window layer derivation lives
    in exactly one place.
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
        self.llm_hidden_size = _vision_stage_scalar(llm_hidden_size, "llm_hidden_size")
        self.downsample_ratio = _vision_stage_scalar(downsample_ratio, "downsample_ratio")
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
        if not hasattr(self, "rope_theta"):
            rope = getattr(self, "rope_parameters", None) or getattr(self, "rope_scaling", None) or {}
            self.rope_theta = rope.get("rope_theta", 10000.0) if isinstance(rope, dict) else 10000.0
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw
        self._ensure_layer_types()

    def _ensure_layer_types(self) -> None:
        _ensure_layer_types(self)


def build_neo_llm_config(llm_config: Any) -> Any:
    if isinstance(llm_config, dict):
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
