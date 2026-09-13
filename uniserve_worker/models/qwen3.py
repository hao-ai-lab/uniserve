"""Pure Qwen3 forward computation over worker-owned attention and request state.

The model consumes prepared tensor views and delegates KV allocation, cache lifetime,
sampling, batching, and request transitions to the worker runtime.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from uniserve_worker.modeling.geometry import CacheGeometry
from uniserve_worker.modeling.tensors import AttentionMode

from ..loader.handles import WeightHandle
from ..loader.mapping import LoadReport, WeightNameMap, stacked_weight_name
from ..loader.weight_loaders import load_parameter_weight
from ..modeling.components import Call, CallSpec, ComponentSpec
from ..modeling.context import BuildContext
from ..modeling.model import Model
from ..modeling.text import TextMixin
from ..nn import (
    ParallelLMHead,
    local_attention_head_count,
    local_kv_head_count,
    local_kv_head_offset,
)
from ..nn.decoder import qwen
from ..nn.vocab_parallel_embedding import vocabulary_partition

if TYPE_CHECKING:
    from ..loader.component import CheckpointComponent


__all__ = ["Qwen3ForCausalLM"]


_QWEN_STACKED_WEIGHTS: WeightNameMap = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


def _qwen_declared_skip(source_name: str, target_name: str) -> bool:
    """Return whether a checkpoint tensor is an allowed non-parameter artifact."""

    return (
        source_name.endswith(
            (
                "rotary_emb.inv_freq",
                "rotary_emb.cos_cached",
                "rotary_emb.sin_cached",
            )
        )
        or "projector" in source_name.split(".")
        or (source_name.endswith(".bias") and target_name.endswith(".bias"))
    )


def _required_int(config: Mapping[str, object], name: str) -> int:
    """Read a required non-boolean integer from model configuration."""

    raw = config.get(name)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"Qwen3 config requires integer field {name!r}")
    if raw <= 0:
        raise ValueError(f"Qwen3 config field {name!r} must be positive")
    return raw


def _optional_int(
    config: Mapping[str, object],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    """Read and lower-bound an optional integer model setting."""

    raw = config.get(name, default)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < minimum:
        raise ValueError(f"Qwen3 config field {name!r} must be an integer >= {minimum}")
    return raw


def _number(config: Mapping[str, object], name: str, default: float) -> float:
    """Read a numeric model setting while rejecting boolean values."""

    raw = config.get(name, default)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(f"Qwen3 config field {name!r} must be numeric")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Qwen3 config field {name!r} must be finite and positive")
    return value


def _boolean(config: Mapping[str, object], name: str, default: bool) -> bool:
    """Read a boolean model setting with a default."""

    raw = config.get(name, default)
    if not isinstance(raw, bool):
        raise ValueError(f"Qwen3 config field {name!r} must be boolean")
    return raw


def _string(config: Mapping[str, object], name: str, default: str) -> str:
    """Read a textual model setting with a default."""

    raw = config.get(name, default)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Qwen3 config field {name!r} must be a non-empty string")
    return raw


def _parse_qwen_config(config: Mapping[str, object]) -> qwen.Qwen3Config:
    """Validate checkpoint fields and derive local Qwen attention and expert geometry."""

    hidden_size = _required_int(config, "hidden_size")
    num_attention_heads = _required_int(config, "num_attention_heads")
    if hidden_size % num_attention_heads:
        raise ValueError("Qwen3 hidden_size must be divisible by num_attention_heads")
    head_dim = _optional_int(
        config,
        "head_dim",
        hidden_size // num_attention_heads,
        minimum=1,
    )
    num_experts = _optional_int(config, "num_experts", 0, minimum=0)
    num_experts_per_tok = _optional_int(config, "num_experts_per_tok", 1, minimum=1)
    intermediate_size = _required_int(config, "intermediate_size")
    cfg = qwen.Qwen3Config(
        vocab_size=_required_int(config, "vocab_size"),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=_required_int(config, "num_hidden_layers"),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=_required_int(config, "num_key_value_heads"),
        head_dim=head_dim,
        hidden_act=_string(config, "hidden_act", "silu"),
        rms_norm_eps=_number(config, "rms_norm_eps", 1e-6),
        rope_theta=_number(config, "rope_theta", 1_000_000.0),
        max_position_embeddings=_optional_int(config, "max_position_embeddings", 4096, minimum=1),
        attention_bias=_boolean(config, "attention_bias", False),
        tie_word_embeddings=_boolean(config, "tie_word_embeddings", False),
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=_optional_int(
            config,
            "moe_intermediate_size",
            intermediate_size,
            minimum=1,
        ),
    )
    if cfg.head_dim <= 0 or cfg.max_position_embeddings <= 0:
        raise ValueError("Qwen3 head_dim and max_position_embeddings must be positive")
    if cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("Qwen3 attention heads must be divisible by KV heads")
    if cfg.num_experts < 0 or cfg.num_experts_per_tok <= 0:
        raise ValueError("Qwen3 expert counts must be non-negative and top-k must be positive")
    if cfg.num_experts and cfg.num_experts_per_tok > cfg.num_experts:
        raise ValueError("Qwen3 num_experts_per_tok must not exceed num_experts")
    return cfg


class Qwen3ForCausalLM(TextMixin, Model):
    """Qwen3 serving model with a thin tensor-level text core."""

    @classmethod
    def components(cls, config: object) -> tuple[ComponentSpec, ...]:
        """Declare the numerical calls sharing this model graph."""

        return (ComponentSpec("model", (CallSpec(Call.TEXT, groups=("tp", "sp", "pp")),)),)

    def checkpoint_components(self) -> tuple[CheckpointComponent, ...]:
        """Declare the complete language graph populated by checkpoint loading."""

        from ..loader.component import CheckpointComponent

        return (
            CheckpointComponent(
                self, map_weights=self.load_weights, nonresident=self._nonresident_names
            ),
        )

    def load_weights(
        self,
        weights: Iterable[WeightHandle],
    ) -> LoadReport:
        """Load Hugging Face Qwen tensors into the model's packed projections."""

        parameters = dict(self.named_parameters())
        parameter_names = set(parameters)
        report = LoadReport()
        for handle in weights:
            source_name = handle.name
            repaired = f"model.{source_name}" if source_name.startswith("layers.") else source_name
            if (
                repaired == "model.embed_tokens.weight"
                and self._tied_embeddings
                and self.model.pipeline.last
                and not self.model.pipeline.first
            ):
                repaired = "lm_head.weight"
            target_name, shard_id = stacked_weight_name(repaired, _QWEN_STACKED_WEIGHTS)
            if target_name not in parameter_names:
                if repaired in parameter_names:
                    target_name, shard_id = repaired, None
                elif repaired == "lm_head.weight" and self._tied_embeddings:
                    report.skipped.append(source_name)
                    continue
                elif target_name in self._nonresident_names or _qwen_declared_skip(
                    source_name, target_name
                ):
                    report.skipped.append(source_name)
                    continue
                else:
                    report.unexpected.append(source_name)
                    continue
            parameter = parameters[target_name]
            load_parameter_weight(parameter, handle, shard_id)
            report.loaded.add(target_name)
        return report

    def __init__(self, config: Mapping[str, object], context: BuildContext) -> None:
        """Construct the tensor-parallel decoder and publish its serving geometry."""

        if not isinstance(config, Mapping):
            raise TypeError("Qwen3 config must be a mapping")
        cfg = _parse_qwen_config(config)
        layer_config = context.layers["model"]
        super().__init__(cfg)
        self._parallel = layer_config.communicator
        self.model = qwen.Qwen3Model(cfg, layer_config=layer_config.child("model"))
        self.lm_head = (
            ParallelLMHead(
                cfg.hidden_size,
                cfg.vocab_size,
                layer_config=layer_config,
                prefix="lm_head",
                bias=False,
            )
            if self.model.pipeline.last
            else None
        )
        self._tied_embeddings = cfg.tie_word_embeddings
        if cfg.tie_word_embeddings and self.model.pipeline.first and self.model.pipeline.last:
            assert self.lm_head is not None and self.model.embed_tokens is not None
            self.lm_head.weight = self.model.embed_tokens.weight
        layer_parameters = tuple(dict(next(iter(self.model.layers.values())).named_parameters()))
        nonresident = self.model.pipeline.nonresident_layer_names("model.layers", layer_parameters)
        if not self.model.pipeline.first:
            nonresident |= {"model.embed_tokens.weight"}
        if not self.model.pipeline.last:
            nonresident |= {"model.norm.weight", "lm_head.weight"}
        self._nonresident_names = nonresident
        self.num_layers = cfg.num_hidden_layers
        self.head_dim = cfg.head_dim
        self.architecture = "Qwen3ForCausalLM"

        self.cache_geometry = CacheGeometry(
            num_layers=len(self.model.pipeline.layers),
            total_layers=int(self.num_layers),
            layer_offset=self.model.pipeline.layers.start,
            num_attention_heads=local_attention_head_count(
                cfg.num_attention_heads,
                parallel=self._parallel,
                sequence=layer_config.sequence,
            ),
            num_kv_heads=local_kv_head_count(
                cfg.num_key_value_heads,
                parallel=self._parallel,
                sequence=layer_config.sequence,
            ),
            total_kv_heads=int(cfg.num_key_value_heads),
            kv_head_offset=local_kv_head_offset(
                int(cfg.num_key_value_heads),
                parallel=self._parallel,
                sequence=layer_config.sequence,
            ),
            head_dim=cfg.head_dim,
            dtype="bfloat16",
        )
        self.vocab_size = int(cfg.vocab_size)
        self.hidden_size = int(cfg.hidden_size)
        self.text_max_tokens = int(cfg.max_position_embeddings)
        self.text_topology = ("tp",)
        self.text_attention_mode = AttentionMode.PAGED_VARLEN

    @property
    def text_backbone(self):
        return self.model

    @property
    def vocabulary(self):
        return vocabulary_partition(self.vocab_size, self._parallel)
