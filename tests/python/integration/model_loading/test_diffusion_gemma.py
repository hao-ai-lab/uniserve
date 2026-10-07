"""Public DiffusionGemma loading agrees with the pinned Transformers model.

A prompt pass through ``Model.text`` writes the K/V cache and yields causal
logits; a canvas pass through ``Model.denoiser`` reads that cache without
writing it and yields soft-capped canvas logits. Both run on the CPU through
the torch attention backend and compare against the Transformers encoder,
its cache and its decoder in FP32. On an SM100 GPU, a model with the
attention shapes of the released checkpoints runs every call through the
automatic native attention providers and reproduces that CPU path.
"""

import math
from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import (
    DynamicCache,
)

from tests.python.fixtures.checkpoints import (
    BEGIN_IMAGE,
    END_IMAGE,
    HIDDEN,
    IMAGE,
    SOFTCAP,
    VOCAB,
    diffusion_gemma_checkpoint,
    load_diffusion_gemma,
)
from uniserve.diffusion.tokens import self_conditioning_embedding
from uniserve.distributed import Communicator
from uniserve.loading import weights
from uniserve.model import (
    CanvasInput,
    EmbeddingReplacement,
    TextInput,
    TextSize,
    VisionInput,
)
from uniserve.nn.attention import (
    AttentionBatch,
    BlockTable,
    PagedInput,
    SegmentedInput,
    SequenceLengths,
)
from uniserve.nn.functional import patchify
from uniserve.nn.moe import FusedMoE
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve.runtime.prefix_cache import plan_units
from uniserve_models import loading as models

pytestmark = pytest.mark.integration

# Both implementations evaluate the same FP32 equations with different kernels,
# whose accumulation order depends on the platform's CPU kernels, so their
# results agree to torch's default FP32 tolerances (rtol 1.3e-6, atol 1e-5).

# Every cache group's table holds eight pages of the unit pool; the groups'
# tables name disjoint units.
BLOCK_SIZE, PAGES = 4, 8
LAYERS = (
    "text.backbone.layers.0.attention.attention",
    "text.backbone.layers.1.attention.attention",
)


def _capped(logits):
    """The reference head's soft cap, applied to its uncapped logits."""
    return torch.tanh(logits.float() / SOFTCAP) * SOFTCAP


def _units(cache, table):
    """Return the units of one numerical table: its own run of the pool."""
    return tuple(range(table * PAGES, (table + 1) * PAGES))


def _layer_units(cache, name):
    return _units(cache, cache.table(name))


def _page_tokens(cache, table):
    return cache.groups[cache.tables[table].group].page_tokens


def _batch(cache, build):
    """Build one entry per table; entries share the first one's lengths."""
    entries = {}
    for table in range(len(cache.tables)):
        entry = build(_units(cache, table), _page_tokens(cache, table))
        if entries:
            first = entries[0]
            entry = replace(
                entry, queries=first.queries, prefixes=first.prefixes
            )
        entries[table] = entry
    return AttentionBatch(entries, entries[0].queries)


def _paged(cache, *, queries, prefix, causal):
    return _batch(
        cache,
        lambda units, page_tokens: PagedInput.from_blocks(
            query_lengths=(queries,),
            prefix_lengths=(prefix,),
            blocks=(units,),
            block_size=page_tokens,
            causal=causal,
            device="cpu",
        ),
    )


def _segmented(cache, *, count, prefix):
    return _batch(
        cache,
        lambda units, page_tokens: SegmentedInput(
            SequenceLengths.from_lengths((count,), device="cpu"),
            SequenceLengths.from_lengths((prefix,), device="cpu"),
            BlockTable(torch.tensor([units], dtype=torch.int32), page_tokens),
            None,
            torch.full((1, count), count, dtype=torch.int32),
            True,
        ),
    )


def _prompt(model, context, tokens, segments, features=None):
    """Write the prompt's K/V segment by segment; return every token's logits.

    ``segments`` are ``(start, stop, causal)`` intervals in order. Features,
    keyed by segment start, replace that segment's token embeddings.
    """
    logits = []
    for start, stop, causal in segments:
        batch = _paged(
            context.cache,
            queries=stop - start,
            prefix=start,
            causal=causal,
        )
        context.bind_attention(batch)
        replacement = None
        if features is not None and start in features:
            values = features[start]
            replacement = EmbeddingReplacement(
                values, torch.ones(values.shape[0], dtype=torch.bool)
            )
        hidden = model.text(
            TextInput(
                tokens[start:stop],
                torch.arange(start, stop),
                batch,
                replacement,
            )
        )
        logits.append(
            model.text.compute_logits(
                hidden, token_indices=torch.arange(stop - start)
            ).gather()
        )
    return torch.cat(logits)


def _canvas(model, context, canvas, prefix, soft=None):
    """Denoise one canvas row over a cached prefix; return its logits."""
    count = canvas.numel()
    batch = _segmented(context.cache, count=count, prefix=prefix)
    context.bind_attention(batch)
    hidden = model.denoiser(
        CanvasInput(canvas, torch.arange(prefix, prefix + count), batch, soft)
    )
    return model.denoiser.compute_logits(
        hidden, token_indices=torch.arange(count)
    ).gather()


def _session(model):
    config = model.text.cache_config
    tables = len(plan_units(config, block_size=BLOCK_SIZE).tables)
    cache = PrefixCache(
        config,
        num_units=tables * PAGES,
        block_size=BLOCK_SIZE,
        device="cpu",
    )
    return cache, ExecutionContext(model, cache=cache, attention="torch")


@pytest.mark.parametrize("prompt_length", [6, 7, 8, 9, 20])
def test_prompt_cache_and_canvas_match_transformers(tmp_path, prompt_length):
    """Prompt logits, cached K/V and canvas logits follow the reference.

    Prompts shorter than, equal to and longer than the seven-token history
    pin the window: each canvas token reads the last seven prompt tokens
    in sliding layers, the whole prompt in full layers, and every canvas
    token in both.
    """
    reference = diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(prompt_length)
    prompt = torch.randint(7, 58, (prompt_length,), generator=generator)
    canvas = torch.randint(0, 58, (9,), generator=generator)
    canvas[::3] = 4
    with torch.no_grad():
        expected = reference(
            input_ids=prompt[None], decoder_input_ids=canvas[None]
        )
        expected_prompt = _capped(
            reference.lm_head(expected.encoder_last_hidden_state[0])
        )

    model = load_diffusion_gemma(tmp_path)
    cache, context = _session(model)
    with cache, context:
        context.prepare(TextSize(32, 1))
        middle = prompt_length // 2
        prompt_logits = _prompt(
            model,
            context,
            prompt,
            ((0, middle, True), (middle, prompt_length, True)),
        )
        torch.testing.assert_close(prompt_logits, expected_prompt)

        # Transformers keeps the last six sliding-layer tokens plus the
        # query's own; the full layer keeps the prompt. Full-layer values
        # are the normalized key projection.
        for name, layer in zip(
            LAYERS, expected.past_key_values.layers, strict=True
        ):
            stored = layer.keys.shape[-2]
            key, value = cache.state(name).read(
                _layer_units(cache, name),
                start=prompt_length - stored,
                length=stored,
            )
            for actual, wanted in ((key, layer.keys), (value, layer.values)):
                torch.testing.assert_close(actual, wanted[0].transpose(0, 1))

        torch.testing.assert_close(
            _canvas(model, context, canvas, prompt_length),
            expected.logits[0],
        )


def test_self_conditioning_matches_transformers(tmp_path):
    """Soft embeddings of previous logits condition the canvas as upstream.

    A first pass without soft embeddings equals a pass whose soft
    embeddings are all zero.
    """
    reference = diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(5)
    prompt = torch.randint(7, 58, (12,), generator=generator)
    canvas = torch.randint(0, 58, (9,), generator=generator)
    previous = torch.randn(9, VOCAB, generator=generator) * 3
    with torch.no_grad():
        expected = reference(
            input_ids=prompt[None],
            decoder_input_ids=canvas[None],
            self_conditioning_logits=previous[None],
        ).logits[0]

    model = load_diffusion_gemma(tmp_path)
    soft = self_conditioning_embedding(
        previous,
        model.text.backbone.embedding.weight[:VOCAB],
        torch.tensor(math.sqrt(HIDDEN)),
    )
    cache, context = _session(model)
    with cache, context:
        context.prepare(TextSize(32, 1))
        _prompt(model, context, prompt, ((0, 12, True),))
        torch.testing.assert_close(
            _canvas(model, context, canvas, 12, soft),
            expected,
        )
        assert torch.equal(
            _canvas(model, context, canvas, 12),
            _canvas(model, context, canvas, 12, torch.zeros_like(soft)),
        )


def test_image_features_and_blocks_match_transformers(tmp_path):
    """Pooled image soft tokens and bidirectional image blocks match upstream.

    Two images of 6x6 and 3x6 patches pool into 2x2 and 1x2 soft tokens.
    Each image block attends to itself in both directions, its prompt
    history through the sliding window, and later text attends causally.
    """
    reference = diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(17)
    images = (
        torch.rand(3, 24, 24, generator=generator),
        torch.rand(3, 12, 24, generator=generator),
    )
    grids = ((6, 6), (3, 6))
    rows = tuple(patchify(image, patch_size=4) for image in images)

    # Transformers pads each image's patch rows to the longest one and marks
    # padding positions -1; real positions are (column, row).
    pixels = torch.zeros(2, 36, 48)
    positions = torch.full((2, 36, 2), -1, dtype=torch.long)
    for index, ((height, width), value) in enumerate(
        zip(grids, rows, strict=True)
    ):
        pixels[index, : value.shape[0]] = value
        y, x = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        positions[index, : value.shape[0]] = torch.stack(
            (x.flatten(), y.flatten()), dim=-1
        )

    text = torch.randint(7, 58, (7,), generator=generator).tolist()
    prompt = torch.tensor(
        [*text[:3], BEGIN_IMAGE, *[IMAGE] * 4, END_IMAGE, *text[3:5]]
        + [BEGIN_IMAGE, *[IMAGE] * 2, END_IMAGE, *text[5:]]
    )
    canvas = torch.randint(0, 58, (9,), generator=generator)
    with torch.no_grad():
        expected_features = reference.model.encoder.get_image_features(
            pixels, positions
        ).pooler_output
        # Transformers' encoder forward discards the image-block masks it
        # builds; its generation path builds them first and passes them in.
        cache = DynamicCache(config=reference.config.get_text_config())
        masks = reference.model.encoder.create_masks_for_generate(
            config=reference.config,
            inputs_embeds=torch.empty(1, prompt.numel(), 0),
            attention_mask=torch.ones(1, prompt.numel(), dtype=torch.long),
            past_key_values=cache,
            position_ids=torch.arange(prompt.numel())[None],
            mm_token_type_ids=(prompt == IMAGE).long()[None],
        )
        expected = reference(
            input_ids=prompt[None],
            attention_mask=masks,
            past_key_values=cache,
            pixel_values=pixels,
            image_position_ids=positions,
            decoder_input_ids=canvas[None],
        )
        expected_prompt = _capped(
            reference.lm_head(expected.encoder_last_hidden_state[0])
        )

    model = load_diffusion_gemma(tmp_path)
    encoder = model.vision_encoder
    grid_tensors = tuple(torch.tensor([grid]) for grid in grids)
    with torch.no_grad():
        features = encoder.connector(
            encoder.network(torch.cat(rows), torch.cat(grid_tensors), grids)
        )
        encoded = encoder.encode(VisionInput(rows, grid_tensors, grids))
    torch.testing.assert_close(features, expected_features)
    # The encoder publishes BF16 features, one tensor per image.
    for actual, wanted in zip(encoded, features.split((4, 2)), strict=True):
        assert torch.equal(actual, wanted.to(torch.bfloat16))

    cache, context = _session(model)
    with cache, context:
        context.prepare(TextSize(32, 1))
        prompt_logits = _prompt(
            model,
            context,
            prompt,
            (
                (0, 4, True),
                (4, 8, False),
                (8, 12, True),
                (12, 14, False),
                (14, 17, True),
            ),
            features={4: features[:4], 12: features[4:]},
        )
        torch.testing.assert_close(prompt_logits, expected_prompt)
        torch.testing.assert_close(
            _canvas(model, context, canvas, 17),
            expected.logits[0],
        )


def test_stacked_and_per_expert_checkpoints_load_identically(tmp_path):
    """Per-expert matrices load into the same rows as stacked tensors."""
    stacked, split = tmp_path / "stacked", tmp_path / "split"
    stacked.mkdir()
    diffusion_gemma_checkpoint(stacked)
    split.mkdir()
    for path in stacked.iterdir():
        if path.suffix == ".json":
            (split / path.name).write_bytes(path.read_bytes())

    # Transformers stacks each expert's gate rows before its up rows.
    state = load_file(stacked / "model.safetensors")
    intermediate = 16
    for name in tuple(state):
        prefix = name.removesuffix(".gate_up_proj")
        if prefix == name:
            continue
        gate_up = state.pop(name)
        down = state.pop(f"{prefix}.down_proj")
        for expert in range(gate_up.shape[0]):
            state[f"{prefix}.{expert}.gate_proj.weight"] = gate_up[
                expert, :intermediate
            ].clone()
            state[f"{prefix}.{expert}.up_proj.weight"] = gate_up[
                expert, intermediate:
            ].clone()
            state[f"{prefix}.{expert}.down_proj.weight"] = down[expert].clone()
    save_file(state, split / "model.safetensors")

    generator = torch.Generator().manual_seed(3)
    prompt = torch.randint(7, 58, (10,), generator=generator)
    canvas = torch.randint(0, 58, (9,), generator=generator)
    results = []
    for root in (stacked, split):
        model = load_diffusion_gemma(root)
        cache, context = _session(model)
        with cache, context:
            context.prepare(TextSize(32, 1))
            results.append(
                (
                    _prompt(model, context, prompt, ((0, 10, True),)),
                    _canvas(model, context, canvas, 10),
                )
            )
    for actual, wanted in zip(*results, strict=True):
        assert torch.equal(actual, wanted)


@pytest.mark.parametrize("layout", ["stacked", "per_expert"])
def test_expert_parallel_ranks_load_their_share_of_every_layer(
    tmp_path, layout
):
    """Each rank of a two-rank expert group keeps its half of the experts.

    Rank ``r`` holds global experts ``[r * E / 2, (r + 1) * E / 2)`` of
    every layer, equal to those rows of a single-rank load, from a stacked
    checkpoint and from per-expert matrices alike.
    """
    root = tmp_path / layout
    root.mkdir()
    diffusion_gemma_checkpoint(root)
    if layout == "per_expert":
        state = load_file(root / "model.safetensors")
        intermediate = 16
        for name in tuple(state):
            prefix = name.removesuffix(".gate_up_proj")
            if prefix == name:
                continue
            gate_up = state.pop(name)
            down = state.pop(f"{prefix}.down_proj")
            for expert in range(gate_up.shape[0]):
                state[f"{prefix}.{expert}.gate_proj.weight"] = gate_up[
                    expert, :intermediate
                ].clone()
                state[f"{prefix}.{expert}.up_proj.weight"] = gate_up[
                    expert, intermediate:
                ].clone()
                state[f"{prefix}.{expert}.down_proj.weight"] = down[
                    expert
                ].clone()
        save_file(state, root / "model.safetensors")

    config = models.read_config(root)
    full = models.load_model(
        config, device="cpu", weights=weights.Config(dtype=torch.float32)
    ).model
    for rank in range(2):
        local = models.load_model(
            config,
            device="cpu",
            weights=weights.Config(dtype=torch.float32),
            experts=Communicator((0, 1), rank),
        ).model
        for (path, whole), (_, part) in zip(
            full.named_modules(), local.named_modules(), strict=True
        ):
            if not isinstance(whole, FusedMoE):
                continue
            count = whole.num_experts // 2
            share = slice(rank * count, (rank + 1) * count)
            assert part.expert_slice == share, path
            assert part.num_experts == whole.num_experts
            for name in ("up_gate", "down"):
                assert torch.equal(
                    getattr(part, name).weight,
                    getattr(whole, name).weight[share],
                ), (path, name)


def test_attention_loading_keeps_shared_experts_unmaterialized(tmp_path):
    """Expert exclusions preserve parent weights and shared aliases."""
    diffusion_gemma_checkpoint(tmp_path)
    full = load_diffusion_gemma(tmp_path)
    expert_paths = frozenset(
        path
        for path, child in full.named_modules()
        if isinstance(child, FusedMoE)
    )
    source = models.read_config(tmp_path, exclude_modules=expert_paths)
    partial = models.load_model(
        source, device="cpu", weights=weights.Config(dtype=torch.float32)
    ).model
    remote = {
        id(parameter)
        for path in expert_paths
        for parameter in partial.get_submodule(path).parameters()
    }
    # Read every alias: encoder and denoiser share the same mathematical
    # backbone, so excluding one path must not reload it through the other.
    complete = dict(full.named_parameters(remove_duplicate=False))
    for name, parameter in partial.named_parameters(remove_duplicate=False):
        if id(parameter) in remote:
            assert parameter.is_meta
        else:
            torch.testing.assert_close(
                parameter, complete[name], rtol=0, atol=0
            )


def test_disagreeing_layer_scalars_are_rejected(tmp_path):
    """One backbone cannot represent differing encoder and decoder scalars."""
    diffusion_gemma_checkpoint(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    state["model.encoder.language_model.layers.1.layer_scalar"] += 0.25
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="layer_scalar of layer 1 differ"):
        load_diffusion_gemma(tmp_path)


def _pipeline_stage(rank, rendezvous, root):
    from uniserve.distributed import DeviceMesh
    from uniserve.runtime import initialize_process_groups

    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device="cpu",
        init_method=rendezvous,
    ) as owner:
        mesh = owner.bind(
            DeviceMesh(ranks=(0, 1), shape=(2,), axes=("pp",), rank=rank),
            device="cpu",
        )
        model = models.load_model(
            models.read_config(root),
            device="cpu",
            weights=weights.Config(dtype=torch.float32),
            meshes={"": mesh},
        ).model
        prompt, canvas, expected_prompt, expected_canvas = torch.load(
            root / "expected.pt", weights_only=True
        )
        count = prompt.numel()
        cache, context = _session(model)
        with cache, context:
            context.prepare(TextSize(32, 1))
            batch = _paged(cache, queries=count, prefix=0, causal=True)
            context.bind_attention(batch)
            hidden = model.text(TextInput(prompt, torch.arange(count), batch))
            prompt_logits = model.text.compute_logits(
                hidden, token_indices=torch.arange(count)
            )
            if rank == 0:
                # The first stage forwards its stream and holds no head, for
                # either capability.
                assert prompt_logits is None
                assert model.text.lm_head is None
                assert model.denoiser.lm_head is None
                assert _canvas_stage(model, context, canvas, count) is None
                return
            torch.testing.assert_close(prompt_logits.gather(), expected_prompt)
            torch.testing.assert_close(
                _canvas_stage(model, context, canvas, count).gather(),
                expected_canvas,
            )


def _canvas_stage(model, context, canvas, prefix):
    """Run one canvas pass on this pipeline stage; return its logits."""
    count = canvas.numel()
    batch = _segmented(context.cache, count=count, prefix=prefix)
    context.bind_attention(batch)
    hidden = model.denoiser(
        CanvasInput(canvas, torch.arange(prefix, prefix + count), batch)
    )
    return model.denoiser.compute_logits(
        hidden, token_indices=torch.arange(count)
    )


def test_pipeline_stages_exchange_the_complete_stream(tmp_path):
    """Two pipeline stages pass one complete stream and match upstream."""
    import torch.multiprocessing as mp

    reference = diffusion_gemma_checkpoint(tmp_path)
    generator = torch.Generator().manual_seed(11)
    prompt = torch.randint(7, 58, (10,), generator=generator)
    canvas = torch.randint(0, 58, (9,), generator=generator)
    with torch.no_grad():
        expected = reference(
            input_ids=prompt[None], decoder_input_ids=canvas[None]
        )
        expected_prompt = _capped(
            reference.lm_head(expected.encoder_last_hidden_state[0])
        )
    torch.save(
        (prompt, canvas, expected_prompt, expected.logits[0]),
        tmp_path / "expected.pt",
    )
    mp.spawn(
        _pipeline_stage,
        args=((tmp_path / "rendezvous").as_uri(), tmp_path),
        nprocs=2,
        join=True,
    )


# Attention shapes of the released checkpoints, which the SM100 kernels
# serve: sliding layers of 16 query and 8 KV heads of width 256, full layers
# of 16 query and 2 KV heads of width 512, and vision heads of width 72. The
# 40-token history window is shorter than the prompt, so sliding tables
# start after retired pages; sliding pages hold 16 tokens and full pages 32.
# Every token weights all experts, so no rounding difference can switch a
# token's expert selection; with unit-variance attention scores (see
# _checkpoint) the logits are then continuous at BF16 resolution.
NATIVE_TEXT = {
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_global_key_value_heads": 2,
    "sliding_window": 41,
    "top_k_experts": 6,
}
NATIVE_VISION = {"hidden_size": 144, "head_dim": 72}
NATIVE_BLOCK_SIZE, NATIVE_PAGES = 16, 32


def _sequence_batch(cache, rows, *, device, canvas=0):
    """Build one call over consecutive chunks of one sequence.

    ``rows`` are ``(start, stop, causal)`` token intervals, one call row
    each; a canvas call instead holds one ``canvas``-token row reading the
    prefix ``[0, start)`` of its only interval. The sequence owns logical
    pages ``0..NATIVE_PAGES`` of every table on its own units. A windowed
    table lists each row's pages from the first one its window reads, so
    earlier pages are retired.
    """
    entries = {}
    for number, table in enumerate(cache.tables):
        group = cache.groups[table.group]
        page = group.page_tokens
        units = tuple(
            range(1 + number * NATIVE_PAGES, 1 + (number + 1) * NATIVE_PAGES)
        )
        windowed = group.window is not None
        blocks, starts = [], []
        for start, stop, _ in rows:
            first = max(start - group.window, 0) // page if windowed else 0
            end = start if canvas else stop
            blocks.append(units[first : max(-(-end // page), first + 1)])
            starts.append(first)

        if canvas:
            ((prefix, _, _),) = rows
            entry = SegmentedInput(
                SequenceLengths.from_lengths((canvas,), device=device),
                SequenceLengths.from_lengths((prefix,), device=device),
                BlockTable(
                    torch.tensor(blocks, dtype=torch.int32, device=device),
                    page,
                    torch.tensor(starts, dtype=torch.int32, device=device)
                    if windowed
                    else None,
                    tuple(starts) if windowed else None,
                ),
                None,
                torch.full((1, canvas), canvas, dtype=torch.int32).to(device),
                True,
            )
        else:
            entry = PagedInput.from_blocks(
                blocks=tuple(blocks),
                query_lengths=tuple(stop - start for start, stop, _ in rows),
                prefix_lengths=tuple(start for start, _, _ in rows),
                block_size=page,
                causal=tuple(causal for _, _, causal in rows),
                device=device,
                start_pages=tuple(starts) if windowed else None,
            )
        if entries:
            # Every table shares the first table's query domain.
            entry = replace(
                entry, queries=entries[0].queries, prefixes=entries[0].prefixes
            )
        entries[number] = entry
    return AttentionBatch(entries, entries[0].queries)


def _native_pass(root, device, attention, tokens, calls, images, canvases):
    """Run every prompt call and canvas read in BF16; return their logits.

    ``calls`` are tuples of ``(start, stop, causal)`` rows of one sequence;
    image token rows take the images' pooled features, which the model's own
    vision encoder computes first. Logits return as FP32 CPU tensors.
    """
    model = models.load_model(
        models.read_config(root),
        device=device,
        weights=weights.Config(dtype=torch.bfloat16),
    ).model
    config = model.text.cache_config
    tables = len(plan_units(config, block_size=NATIVE_BLOCK_SIZE).tables)
    cache = PrefixCache(
        config,
        num_units=1 + tables * NATIVE_PAGES,
        block_size=NATIVE_BLOCK_SIZE,
        device=device,
    )
    context = ExecutionContext(model, cache=cache, attention=attention)
    with cache, context, torch.inference_mode():
        context.prepare(TextSize(128, 4))
        rows, grids = images
        features = torch.cat(
            model.vision_encoder.encode(
                VisionInput(
                    tuple(row.to(device) for row in rows),
                    tuple(
                        torch.tensor([grid], device=device) for grid in grids
                    ),
                    grids,
                )
            )
        )
        tokens = tokens.to(device)
        replaced = tokens == IMAGE
        values = torch.zeros(
            (tokens.numel(), HIDDEN), dtype=features.dtype, device=device
        )
        values[replaced] = features

        logits = []
        for call in calls:
            batch = _sequence_batch(cache, call, device=device)
            context.bind_attention(batch)
            select = torch.cat(
                [torch.arange(start, stop) for start, stop, _ in call]
            ).to(device)
            hidden = model.text(
                TextInput(
                    tokens[select],
                    select,
                    batch,
                    EmbeddingReplacement(values[select], replaced[select]),
                )
            )
            indices = torch.arange(select.numel(), device=device)
            logits.append(
                model.text.compute_logits(
                    hidden, token_indices=indices
                ).gather()
            )
        for prefix, canvas in canvases:
            count = canvas.numel()
            batch = _sequence_batch(
                cache, ((prefix, prefix, False),), device=device, canvas=count
            )
            context.bind_attention(batch)
            hidden = model.denoiser(
                CanvasInput(
                    canvas.to(device),
                    torch.arange(prefix, prefix + count, device=device),
                    batch,
                )
            )
            indices = torch.arange(count, device=device)
            logits.append(
                model.denoiser.compute_logits(
                    hidden, token_indices=indices
                ).gather()
            )
    return [value.float().cpu() for value in logits]


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10,
    reason="native DiffusionGemma attention requires an SM100 GPU",
)
def test_native_cuda_attention_reproduces_the_cpu_torch_path(tmp_path):
    """Every call on CUDA through automatic attention matches the CPU path.

    A 166-token prompt holds two images. One call writes its leading text;
    one call holds the first image block, text, the second image block and
    more text as consecutive rows of the sequence; a third call writes the
    remaining text and a fourth commits a 16-token block. A canvas is read
    after the prompt and after the commit. Both paths run the BF16 model,
    on CUDA through automatic attention and on the CPU through the portable
    torch provider, over the same unit-pool tables whose sliding rows start
    after retired pages. Prompt and canvas logits agree within the BF16
    attention tolerance.
    """
    diffusion_gemma_checkpoint(
        tmp_path, text=NATIVE_TEXT, vision=NATIVE_VISION, unit_scores=True
    )
    generator = torch.Generator().manual_seed(23)
    images = (
        torch.rand(3, 24, 24, generator=generator),
        torch.rand(3, 12, 24, generator=generator),
    )
    grids = ((6, 6), (3, 6))
    rows = tuple(patchify(image, patch_size=4) for image in images)
    text = torch.randint(7, 58, (156,), generator=generator).tolist()
    tokens = torch.tensor(
        [*text[:50], BEGIN_IMAGE, *[IMAGE] * 4, END_IMAGE, *text[50:110]]
        + [BEGIN_IMAGE, *[IMAGE] * 2, END_IMAGE, *text[110:]]
    )
    calls = (
        ((0, 51, True),),
        ((51, 55, False), (55, 117, True), (117, 119, False), (119, 130, True)),
        ((130, 150, True),),
        ((150, 166, True),),
    )
    canvases = []
    for prefix in (150, 166):
        canvas = torch.randint(0, 58, (16,), generator=generator)
        canvas[::3] = 4
        canvases.append((prefix, canvas))

    arguments = (tokens, calls, (rows, grids), canvases)
    expected = _native_pass(tmp_path, "cpu", "torch", *arguments)
    actual = _native_pass(tmp_path, "cuda:0", "auto", *arguments)
    for value, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, wanted, rtol=2e-2, atol=2e-2)
