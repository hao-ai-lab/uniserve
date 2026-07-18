"""Target Qwen3 family adapter over the unified stack.

Stage 6 family port of ``specs/unified_forward_execution.md``: one adapter,
one resident root, one packed traversal. The shared route-aware decoder root
lives in :mod:`uniserve_worker.nn.target_decoder`; this family file owns the
Qwen3 configuration (one text route) and the family output projection —
greedy sampled tokens for requested output positions and greedy candidate
acceptance for verification spans. Family code owns no cache state,
provider, plan, or graph; attention arrives through the injected shared
seam. Weights arrive through the root constructor (checkpoint reuse is
explicitly permitted); tests prove conformance against an independent dense
recompute, the Stage 11 shape at unit scale.
"""
from __future__ import annotations

from uniserve_worker.execution.engine import AdapterPayload, AdapterRowOutcome

from ..contracts.cache_schema import CacheEffect
from ..contracts.residency_batch import ResidencyBatchArrays
from ..contracts.segment_table import GraphCapacity, SegmentTableArrays
from ..nn.grouped_routing import WeightOverlayBank
from ..nn.target_decoder import SharedAttention, TargetDecoderConfig, TargetDecoderRoot

__all__ = ["Qwen3Target", "Qwen3TargetConfig", "project_greedy_outcomes"]

Qwen3TargetConfig = TargetDecoderConfig


def project_greedy_outcomes(
    segments: SegmentTableArrays,
    capacity: GraphCapacity,
    payload: AdapterPayload,
    logits,
) -> tuple[AdapterRowOutcome, ...]:
    """Family output projection: greedy sampling and candidate acceptance.

    Rows with persistent sequence segments sample greedily at their final
    position; verification spans accept candidates while each matches the
    prediction from the previous position; rows with only transient or
    read-only segments (denoise, encode, materialize) project no tokens.
    """

    greedy = logits.argmax(dim=-1)
    sampled: dict[int, int | None] = {}
    accepted: dict[int, int] = {}
    for index in range(capacity.segments):
        if not segments.segment_active[index]:
            continue
        row = segments.row_id[index]
        sampled.setdefault(row, None)
        accepted.setdefault(row, 0)
        begin = segments.token_begin[index]
        count = segments.token_count[index]
        if segments.candidate_count[index]:
            matched = 0
            for offset in range(count):
                predicted = int(greedy[begin + offset - 1])
                if predicted == payload.token_ids[begin + offset]:
                    matched += 1
                else:
                    break
            accepted[row] = matched
            sampled[row] = int(greedy[begin + count - 1])
        elif segments.cache_effect[index] == int(CacheEffect.PERSISTENT_APPEND):
            sampled[row] = int(greedy[begin + count - 1])
    outcomes: list[AdapterRowOutcome] = []
    for row in sorted(sampled):
        token = sampled[row]
        outcomes.append(
            AdapterRowOutcome(
                sampled_tokens=(token,) if token is not None else (),
                accepted_candidates=accepted[row],
            )
        )
    return tuple(outcomes)


class Qwen3Target:
    """One resident root, one packed traversal, compact projected outcomes."""

    def __init__(
        self,
        config: TargetDecoderConfig,
        attention: SharedAttention,
        *,
        device: str = "cuda",
        seed: int = 0,
        overlay_bank: WeightOverlayBank | None = None,
        dtype=None,
        zero_init: bool = False,
    ) -> None:
        import torch

        dtype = dtype if dtype is not None else torch.float32
        if config.routes != 1:
            raise ValueError("Qwen3 registers one text route")
        self.root = TargetDecoderRoot(
            config,
            attention,
            device=device,
            seed=seed,
            overlay_bank=overlay_bank,
            dtype=dtype,
            zero_init=zero_init,
        )

    @property
    def config(self) -> TargetDecoderConfig:
        return self.root.config

    @property
    def device(self):
        return self.root.device

    @property
    def embedding(self):
        return self.root.embedding

    @property
    def layers(self):
        return self.root.layers

    @property
    def final_norm(self):
        return self.root.final_norm

    @property
    def lm_head(self):
        return self.root.lm_head

    def forward(
        self,
        segments: SegmentTableArrays,
        residency: ResidencyBatchArrays,
        capacity: GraphCapacity,
        payload: AdapterPayload,
    ) -> tuple[AdapterRowOutcome, ...]:
        logits = self.root.logits(
            segments, residency, capacity, payload.token_ids, payload.positions
        )
        return project_greedy_outcomes(segments, capacity, payload, logits)


def load_qwen3_checkpoint(target: Qwen3Target, checkpoint_dir: str) -> None:
    """Bind a real Qwen3 checkpoint's weights into the target root.

    Checkpoint reuse is the spec's sanctioned family-port path: the resident
    neural weights are the family's; the runtime architecture around them is
    the target's. Mapping covers the dense Qwen3 layout (embed, per-layer
    norms, q/k/v/o with per-head q/k norms, gated MLP, final norm, untied or
    tied LM head).
    """

    import json
    from pathlib import Path

    from safetensors import safe_open

    directory = Path(checkpoint_dir)
    root = target.root
    layers = root.layers

    def assign(name: str, tensor) -> bool:
        value = tensor.to(root.device, root.dtype)
        if name == "model.embed_tokens.weight":
            root.embedding.copy_(value)
            if tied_lm_head:
                root.lm_head.copy_(value)
            return True
        if name == "lm_head.weight":
            root.lm_head.copy_(value)
            return True
        if name == "model.norm.weight":
            root.final_norm.copy_(value)
            return True
        parts = name.split(".")
        if len(parts) < 4 or parts[1] != "layers":
            return False
        layer = layers[int(parts[2])]
        leaf = ".".join(parts[3:])
        mapping = {
            "input_layernorm.weight": layer.input_norm,
            "post_attention_layernorm.weight": layer.post_norm,
            "self_attn.q_proj.weight": layer.q.weights[0],
            "self_attn.k_proj.weight": layer.k.weights[0],
            "self_attn.v_proj.weight": layer.v.weights[0],
            "self_attn.o_proj.weight": layer.o.weights[0],
            "self_attn.q_norm.weight": layer.q_norm,
            "self_attn.k_norm.weight": layer.k_norm,
            "mlp.gate_proj.weight": layer.gate.weights[0],
            "mlp.up_proj.weight": layer.up.weights[0],
            "mlp.down_proj.weight": layer.down.weights[0],
        }
        destination = mapping.get(leaf)
        if destination is None:
            return False
        destination.copy_(value)
        return True

    config = json.loads((directory / "config.json").read_text())
    tied_lm_head = bool(config.get("tie_word_embeddings", False))
    unmapped: list[str] = []
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                if not assign(name, handle.get_tensor(name)):
                    unmapped.append(name)
    if unmapped:
        raise ValueError(f"unmapped checkpoint tensors: {unmapped[:8]}")
