"""Public DiffusionGemma loading agrees with the pinned Transformers model.

A prompt pass through ``Model.text`` writes the K/V cache and yields causal
logits; a canvas pass through ``Model.denoiser`` reads that cache without
writing it and yields soft-capped canvas logits. Both run on the CPU through
the torch attention backend and compare against the Transformers encoder,
its cache and its decoder in FP32.
"""

import json
import math
from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import (
    DiffusionGemmaConfig,
    DiffusionGemmaForBlockDiffusion,
    DynamicCache,
)

from uniserve.diffusion.tokens import self_conditioning_embedding
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
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve.runtime.prefix_cache import plan_units
from uniserve_models import loading as models

pytestmark = pytest.mark.integration

# Transformers' sliding window counts the query: seven history tokens.
WINDOW = 8
SOFTCAP = 0.5
HIDDEN, VOCAB = 32, 64
IMAGE, BEGIN_IMAGE, END_IMAGE = 60, 58, 59
# Every cache group's table holds eight pages of the unit pool; the groups'
# tables name disjoint units.
BLOCK_SIZE, PAGES = 4, 8
LAYERS = (
    "text.backbone.layers.0.attention.attention",
    "text.backbone.layers.1.attention.attention",
)


def _checkpoint(root):
    """Save a two-layer DiffusionGemma checkpoint; return its reference.

    Layer 0 attends through a sliding window and layer 1 fully, with no
    value projection and proportional rotation of a quarter of its 16-wide
    heads. Norm weights, router scales, layer scalars, vision positions and
    standardization are randomized so each factor is observable, and the
    head's logits are large enough for the small softcap to bend them. The
    tokenizer files declare the canvas's special tokens: pad 0, mask 4 and
    end of turn 6.
    """
    config = DiffusionGemmaConfig(
        text_config={
            "vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 16,
            "num_global_key_value_heads": 1,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": WINDOW,
            "num_experts": 6,
            "top_k_experts": 2,
            "moe_intermediate_size": 16,
            "use_bidirectional_attention": "vision",
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
        },
        vision_config={
            "model_type": "gemma4_vision",
            "hidden_size": 24,
            "intermediate_size": 40,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 12,
            "patch_size": 4,
            "pooling_kernel_size": 3,
            "position_embedding_size": 32,
            "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
            "standardize": True,
            "use_clipped_linears": False,
        },
        canvas_length=16,
        image_token_id=IMAGE,
        boi_token_id=BEGIN_IMAGE,
        eoi_token_id=END_IMAGE,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(313)
    model = DiffusionGemmaForBlockDiffusion(config).eval()
    model.final_logit_softcapping = SOFTCAP

    def uniform(value, low, high):
        value.copy_(torch.rand_like(value) * (high - low) + low)

    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith(".weight") and "norm" in name.split(".")[-2]:
                uniform(value, 0.8, 1.2)
        encoder = model.model.encoder
        for index, layer in enumerate(model.model.decoder.layers):
            uniform(layer.router.scale, 0.5, 1.5)
            uniform(layer.router.per_expert_scale, 0.5, 2.0)
            # Both stored copies of a layer scalar agree, as in the released
            # checkpoints.
            uniform(layer.layer_scalar, 0.3, 1.2)
            encoder.language_model.layers[index].layer_scalar.copy_(
                layer.layer_scalar
            )
        tower = encoder.vision_tower
        tower.patch_embedder.position_embedding_table.normal_(std=0.02)
        tower.std_bias.normal_(std=0.1)
        uniform(tower.std_scale, 0.5, 1.5)

    model.save_pretrained(root)
    metadata = json.loads((root / "config.json").read_text())
    metadata["text_config"]["final_logit_softcapping"] = SOFTCAP
    metadata["vision_soft_tokens_per_image"] = 70
    (root / "config.json").write_text(json.dumps(metadata))
    (root / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [1, 6]})
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {"pad_token": "<pad>", "mask_token": "<mask>", "eot_token": "<e>"}
        )
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 0, "content": "<pad>"},
                    {"id": 4, "content": "<mask>"},
                    {"id": 6, "content": "<e>"},
                ]
            }
        )
    )
    return model


def _load(root):
    return models.load_model(
        models.read_config(root),
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    ).model


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
    reference = _checkpoint(tmp_path)
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

    model = _load(tmp_path)
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
        torch.testing.assert_close(
            prompt_logits, expected_prompt, rtol=1e-5, atol=1e-6
        )

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
                torch.testing.assert_close(
                    actual, wanted[0].transpose(0, 1), rtol=1e-5, atol=1e-6
                )

        torch.testing.assert_close(
            _canvas(model, context, canvas, prompt_length),
            expected.logits[0],
            rtol=1e-5,
            atol=1e-6,
        )


def test_self_conditioning_matches_transformers(tmp_path):
    """Soft embeddings of previous logits condition the canvas as upstream.

    A first pass without soft embeddings equals a pass whose soft
    embeddings are all zero.
    """
    reference = _checkpoint(tmp_path)
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

    model = _load(tmp_path)
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
            rtol=1e-5,
            atol=1e-6,
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
    reference = _checkpoint(tmp_path)
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

    model = _load(tmp_path)
    encoder = model.vision_encoder
    grid_tensors = tuple(torch.tensor([grid]) for grid in grids)
    with torch.no_grad():
        features = encoder.connector(
            encoder.network(torch.cat(rows), torch.cat(grid_tensors), grids)
        )
        encoded = encoder.encode(VisionInput(rows, grid_tensors, grids))
    torch.testing.assert_close(
        features, expected_features, rtol=1e-5, atol=1e-6
    )
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
        torch.testing.assert_close(
            prompt_logits, expected_prompt, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            _canvas(model, context, canvas, 17),
            expected.logits[0],
            rtol=1e-5,
            atol=1e-6,
        )


def test_stacked_and_per_expert_checkpoints_load_identically(tmp_path):
    """Per-expert matrices load into the same rows as stacked tensors."""
    stacked, split = tmp_path / "stacked", tmp_path / "split"
    stacked.mkdir()
    _checkpoint(stacked)
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
        model = _load(root)
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


def test_disagreeing_layer_scalars_are_rejected(tmp_path):
    """One backbone cannot represent differing encoder and decoder scalars."""
    _checkpoint(tmp_path)
    state = load_file(tmp_path / "model.safetensors")
    state["model.encoder.language_model.layers.1.layer_scalar"] += 0.25
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="layer_scalar of layer 1 differ"):
        _load(tmp_path)


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
            torch.testing.assert_close(
                prompt_logits.gather(), expected_prompt, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                _canvas_stage(model, context, canvas, count).gather(),
                expected_canvas,
                rtol=1e-5,
                atol=1e-6,
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

    reference = _checkpoint(tmp_path)
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
