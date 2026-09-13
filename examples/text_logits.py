"""Compute raw-text next-token logits through the public Python library."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from uniserve_worker.backends.attention.selection import AttentionSelection
from uniserve_worker.backends.attention.torch_sdpa import TorchSDPAAttentionBackend
from uniserve_worker.bootstrap.distributed import initialize_entries, initialize_process_groups
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.attention import physical_columns
from uniserve_worker.loader import LoadRequest, load_model
from uniserve_worker.modeling.batch import TextBatch
from uniserve_worker.modeling.components import Call
from uniserve_worker.modeling.geometry import TextShape
from uniserve_worker.modeling.tensors import AttentionMode, TokenSelection
from uniserve_worker.modeling.text import TextMixin
from uniserve_worker.nn.attention import bind_attention_modules
from uniserve_worker.nn.parallel import ComponentConfig
from uniserve_worker.runtime.branches import bind_branches
from uniserve_worker.runtime.kv_cache import KVCache
from uniserve_worker.runtime.process_groups import ProcessGroups


@torch.inference_mode()
def text_logits(
    checkpoint: str, text: str, *, device: str = "cuda:0"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU input IDs [tokens] and logits [tokens, vocabulary] for raw text.

    The checkpoint supplies tokenization and numerical parameters. This example
    evaluates every prompt position with causal attention, without a chat
    template or token sampling. It supports the Qwen3, BAGEL, and SenseNova text
    capabilities on one device using the public PyTorch SDPA backend.
    """

    # Text preprocessing belongs to the caller. A numerical loading result
    # includes a tokenizer only when its model composition itself requires one.
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, use_fast=False, trust_remote_code=False, local_files_only=True
    )
    input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    with initialize_process_groups(rank=0, local_rank=0, world_size=1, device=device) as groups:
        return _compute(groups, checkpoint, input_ids)


def _compute(
    groups: ProcessGroups, checkpoint: str, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    device = groups.local_device
    bindings = initialize_entries(groups, {"model": ComponentConfig((0,))})
    loaded = load_model(
        LoadRequest(
            model_path=checkpoint,
            execution=WorkerConfig(
                device=str(device), model_dtype="bfloat16", attention_backend="torch_sdpa"
            ),
            bindings=bindings,
        )
    )
    model = loaded.model
    if not isinstance(model, TextMixin):
        raise TypeError("text logits require the text computational capability")
    tokens = input_ids.numel()
    if not 1 <= tokens <= model.text_max_tokens:
        raise ValueError("text length lies outside the model's numerical token limits")

    page_size = 64
    pages = (tokens + page_size - 1) // page_size
    geometry = model.cache_geometry
    cache = KVCache(
        num_layers=geometry.num_layers,
        num_pages=pages,
        page_size=page_size,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        dtype=getattr(torch, geometry.dtype),
        store_dtype=geometry.store_dtype,
        device=device,
    )
    try:
        bind_attention_modules(
            model, cache, AttentionSelection("torch_sdpa", (TorchSDPAAttentionBackend(),))
        )
        bind_branches(model, device=device)
        positions = torch.arange(tokens, dtype=torch.long)
        indexes = torch.stack((positions, torch.zeros_like(positions), torch.zeros_like(positions)))
        metadata = physical_columns(
            pages=(tuple(range(pages)),),
            prefix_lens=(0,),
            query_lens=(tokens,),
            causal_rows=(True,),
            write_rows=(True,),
            positions=(indexes,),
            token_rows=(True,),
            text_local_indices=(tuple(range(tokens)),),
            width=pages,
            block_size=page_size,
            packed=model.text_attention_mode is AttentionMode.PACKED,
        )
        # Input delivery belongs to this caller. Numerical metadata keeps only
        # the already-delivered tensor views and its host-known sequence lengths.
        delivered: dict[str, Any] = {
            field.name: value.to(device)
            for field in fields(metadata)
            if isinstance(value := getattr(metadata, field.name), torch.Tensor)
        }
        metadata = replace(metadata, **delivered)
        batch = TextBatch(
            input_ids.to(device),
            positions.to(device),
            metadata,
            (TokenSelection.ALL_LOGITS,),
        )
        model.eval()
        hidden = model.forward(batch, constants={}, scratch={})
        model.tensor_specs(Call.TEXT, TextShape(tokens)).outputs["hidden_states"].validate(
            hidden, state={}, scratch={}
        )
        output = model.compute_logits(hidden, batch)
        output.validate(
            (model.tensor_specs(Call.TEXT, TextShape(tokens, selection=TokenSelection.ALL_LOGITS)),)
        )
        # Materialization joins vocabulary shards and removes vocabulary padding.
        # The CPU copy finishes the result reader before cache and groups retire.
        return input_ids, output.materialize().values[0].cpu()
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cache.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_ids, logits = text_logits(args.checkpoint, args.text, device=args.device)
    torch.save({"input_ids": input_ids, "logits": logits}, args.output)
    print(f"Saved {tuple(logits.shape)} {logits.dtype} logits to {args.output}")


if __name__ == "__main__":
    main()
