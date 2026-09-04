"""Loader-owned config objects for checkpoints with custom HF config classes."""

from __future__ import annotations

from typing import Any

from transformers import Qwen3Config
from transformers.configuration_utils import PretrainedConfig

__all__ = [
    'NeoVisionConfig',
    'NeoLlmConfig',
    'build_neo_llm_config',
    'NeoChatConfig',
]

def _vision_stage_scalar(value: Any, field_name: str) -> Any:
    """Normalize a vision config field that may be serialized per stage."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{field_name} must not be empty")
        return value[0]
    return value


# Hugging Face composition resolves omitted tower configs from these architecture names.
_DEFAULT_VISION_ARCHITECTURE = "NEOVisionModel"
_DEFAULT_LLM_ARCHITECTURE = "Qwen3ForCausalLM"
_TOKEN_ID_FIELDS = ("bos_token_id", "eos_token_id", "pad_token_id")


def _ensure_layer_types(config: PretrainedConfig) -> None:
    """Derive each decoder layer's attention type from the sliding-window boundary."""
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
    """Normalizes vision-tower patch, width, head, layer, and position settings from checkpoint metadata."""

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
        """Normalize serialized vision geometry into scalar tower configuration."""

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

class NeoLlmConfig(Qwen3Config):
    """Normalizes decoder width, head, layer, expert, rotary, and vocabulary settings for SenseNova."""

    def __init__(
        self,
        rope_theta_hw: float = 10000.0,
        max_position_embeddings_hw: int = 10000,
        **kwargs: Any,
    ) -> None:
        """Normalize rotary metadata and derive per-layer attention modes."""

        super().__init__(**kwargs)
        if not hasattr(self, "rope_theta"):
            rope = getattr(self, "rope_parameters", None) or getattr(self, "rope_scaling", None) or {}
            self.rope_theta = rope.get("rope_theta", 10000.0) if isinstance(rope, dict) else 10000.0
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw
        self._ensure_layer_types()

    def _ensure_layer_types(self) -> None:
        """Populate decoder attention modes from sliding-window configuration."""

        _ensure_layer_types(self)


def build_neo_llm_config(llm_config: Any) -> Any:
    """Materialize mapping-based decoder metadata while preserving config instances."""

    if isinstance(llm_config, dict):
        return NeoLlmConfig(**llm_config)
    return llm_config


class NeoChatConfig(PretrainedConfig):
    """Combines SenseNova language and vision configuration with multimodal token identities."""

    model_type = "neo_chat"
    is_composition = True

    def __init__(
        self,
        vision_config: Any | None = None,
        llm_config: Any | None = None,
        downsample_ratio: float = 0.5,
        template: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Materialize nested language and vision configs with shared token identities."""

        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = {"architectures": [_DEFAULT_VISION_ARCHITECTURE]}
        if llm_config is None:
            llm_config = {"architectures": [_DEFAULT_LLM_ARCHITECTURE]}
        if isinstance(llm_config, dict):
            llm_config = dict(llm_config)
            for field_name in _TOKEN_ID_FIELDS:
                if llm_config.get(field_name) is None:
                    parent_value = getattr(self, field_name, None)
                    if parent_value is not None:
                        llm_config[field_name] = parent_value
        self.vision_config = (
            NeoVisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config
        )
        self.llm_config = build_neo_llm_config(llm_config)
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

    def to_dict(self) -> dict[str, Any]:
        """Serialize nested vision and language configuration with multimodal fields intact."""

        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        return output
