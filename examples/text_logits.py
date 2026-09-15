"""Compute raw-text next-token logits through the public Python library."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from uniserve.execution import TextRunner
from uniserve.model import CausalLM, TextInput, TextSize
from uniserve.nn.attention import PagedInput
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve_models.loading import load_model, read_config
from uniserve_models.processing import load_tokenizer


@torch.inference_mode()
def text_logits(
    checkpoint: str, text: str, *, device: str = "cuda:0"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU token IDs and [tokens, vocabulary] logits for untemplated text.

    Capability discovery selects the language module of Qwen, BAGEL or U1.
    Tokenization, prefix allocation and execution resources belong to this
    caller; the numerical model receives only explicit tokens and index views.
    """

    metadata = read_config(checkpoint, modules=frozenset())
    with torch.device("meta"):
        architecture = metadata.model_class(metadata.model)
    paths = tuple(
        path for path, module in architecture.named_modules() if isinstance(module, CausalLM)
    )
    if len(paths) != 1:
        raise TypeError("this example requires one causal-language-model capability")
    path = paths[0]
    del architecture
    config = read_config(checkpoint, modules=frozenset({path}))
    if config.tokenizer is None:
        raise ValueError("the checkpoint must supply a tokenizer")
    tokenizer = load_tokenizer(config.tokenizer)
    input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.int64)
    count = input_ids.numel()
    if count == 0:
        raise ValueError("text must encode at least one token")
    model = load_model(config, device=device).model.get_submodule(path)
    block_size = 64
    blocks = (count + block_size - 1) // block_size
    attention = PagedInput.from_blocks(
        blocks=(tuple(range(blocks)),),
        query_lengths=(count,),
        prefix_lengths=(0,),
        block_size=block_size,
        causal=True,
        device=device,
    )
    positions = torch.arange(count, dtype=torch.int64, device=device)
    inputs = TextInput(input_ids.to(device), positions, attention)
    with (
        PrefixCache(
            model.cache_config, num_blocks=blocks, block_size=block_size, device=device
        ) as cache,
        ExecutionContext(model, cache=cache, attention="torch") as execution,
    ):
        runner = TextRunner(model, context=execution)
        runner.warmup(TextSize(count, 1))
        hidden = runner.forward(inputs)
        logits = runner.compute_logits(hidden, token_indices=positions)
        if logits is None:
            raise RuntimeError("local vocabulary projection returned no logits")
        return input_ids, logits.gather().to("cpu", copy=True)


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
