"""Read candidate probabilities through the public DiffusionGemma library.

Run with a local checkpoint, --state and --question. This example constructs
one yes/no scaffold. It demonstrates numerical resource ownership; use the
System One HTTP endpoint for its complete question and image contract.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace

import torch
from transformers import AutoTokenizer

from uniserve.model import CanvasInput, TextInput, TextSize
from uniserve.nn.attention import (
    AttentionBatch,
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve.runtime.prefix_cache import plan_units
from uniserve_models import loading


@torch.inference_mode()
def read_candidates(model, prompt, canvas, position, candidates):
    """Read one canvas position over a freshly cached prompt.

    Inputs are one-dimensional token tensors on the model's device. The
    candidate IDs name single vocabulary tokens. Return normalized candidate
    probabilities and their full-vocabulary mass as owned CPU tensors. The
    caller owns model loading; this call owns and releases its KV and native
    execution resources, and does not sample or mutate the canvas.
    """
    device = prompt.device
    length, count = prompt.numel(), canvas.numel()
    if not length or not count or not 0 <= position < count:
        raise ValueError("prompt and canvas must be nonempty with a valid slot")
    if not candidates.numel():
        raise ValueError("the slot needs at least one candidate")

    config = model.text.cache_config
    # Full-attention K/V rows are half as wide as sliding-attention rows in
    # this checkpoint. A 32-token widest page gives 64-token full pages,
    # supported by the native 512-dimensional-head kernels.
    block_size = 32
    plan = plan_units(config, block_size=block_size)
    blocks, next_unit = {}, 1
    for index, table in enumerate(plan.tables):
        page = plan.groups[table.group].page_tokens
        pages = math.ceil(length / page)
        blocks[index] = tuple(range(next_unit, next_unit + pages))
        next_unit += pages

    cache = PrefixCache(
        config, num_units=next_unit, block_size=block_size, device=device
    )
    context = ExecutionContext(
        model,
        cache=cache,
        attention="auto" if device.type == "cuda" else "torch",
    )

    def attention(build):
        entries = {}
        for index, table in enumerate(cache.tables):
            page = cache.groups[table.group].page_tokens
            entry = build(blocks[index], page)
            if entries:
                entry = replace(
                    entry,
                    queries=entries[0].queries,
                    prefixes=entries[0].prefixes,
                )
            entries[index] = entry
        return AttentionBatch(entries, entries[0].queries)

    with cache, context:
        context.prepare(TextSize(max(length, count), 1))
        prefill = attention(
            lambda units, page: PagedInput.from_blocks(
                query_lengths=(length,),
                prefix_lengths=(0,),
                blocks=(units,),
                block_size=page,
                causal=True,
                device=device,
            )
        )
        context.bind_attention(prefill)
        model.text.fill_cache(
            TextInput(prompt, torch.arange(length, device=device), prefill)
        )

        # Every canvas query sees the retained prompt and the whole canvas.
        # The canvas pass reads the cache without appending its own K/V.
        readout = attention(
            lambda units, page: SegmentedInput(
                SequenceLengths.from_lengths((count,), device=device),
                SequenceLengths.from_lengths((length,), device=device),
                BlockTable(
                    torch.tensor([units], dtype=torch.int32, device=device),
                    page,
                ),
                None,
                torch.full((1, count), count, dtype=torch.int32, device=device),
                True,
            )
        )
        context.bind_attention(readout)
        hidden = model.denoiser(
            CanvasInput(
                canvas,
                torch.arange(length, length + count, device=device),
                readout,
            )
        )
        logits = (
            model.denoiser.compute_logits(
                hidden, token_indices=torch.tensor([position], device=device)
            )
            .gather()[0]
            .float()
        )
        selected = logits[candidates]
        probabilities = selected.softmax(-1).cpu()
        mass = (selected.logsumexp(-1) - logits.logsumexp(-1)).exp().cpu()
    return probabilities, mass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--state", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = loading.read_config(
        args.checkpoint, modules=frozenset(("text", "denoiser"))
    )
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    text = (
        f"State:\n{args.state}\n\nQuestion:\n{args.question}\n\n"
        "Answer with exactly one line: 1: yes or 1: no."
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    labels = (" no", " yes")
    ids = [
        tokenizer.encode(label, add_special_tokens=False) for label in labels
    ]
    if any(len(value) != 1 for value in ids):
        raise ValueError("this example requires single-token yes/no candidates")
    diffusion = config.model.diffusion
    # The answer token includes its leading space, so the scaffold ends at
    # the colon. Keep the following newline fixed during the canvas pass.
    scaffold = tokenizer.encode("1:", add_special_tokens=False)
    position = len(scaffold)
    scaffold.append(diffusion.mask_token_id)
    scaffold.extend(tokenizer.encode("\n", add_special_tokens=False))
    scaffold.append(diffusion.end_of_turn_id)
    scaffold.extend(
        [diffusion.pad_token_id] * (diffusion.canvas_length - len(scaffold))
    )
    device = torch.device(args.device)
    model = loading.load_model(config, device=device).model
    probabilities, mass = read_candidates(
        model,
        torch.tensor(prompt, device=device),
        torch.tensor(scaffold, device=device),
        position,
        torch.tensor([value[0] for value in ids], device=device),
    )
    print(
        json.dumps(
            {
                "probabilities": dict(
                    zip(("no", "yes"), probabilities.tolist(), strict=True)
                ),
                "candidate_mass": mass.item(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
