"""Pipeline placement preserves packed decoder outputs and logical KV state."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from tests.python.fixtures.model_execution import model_context
from uniserve_worker.backends.attention.fa4_cute import Fa4CuteAttentionBackend
from uniserve_worker.backends.attention.flashinfer import FlashInferAttentionBackend
from uniserve_worker.backends.attention.selection import AttentionSelection
from uniserve_worker.backends.attention.tuning import FlashInferTuningConfig
from uniserve_worker.bootstrap.distributed import (
    initialize_model_parallel,
    initialize_process_groups,
)
from uniserve_worker.loader.handles import TensorWeightHandle
from uniserve_worker.loader.loader import assign_component
from uniserve_worker.loader.weight_loaders import attach_parameter_loaders
from uniserve_worker.modeling.batch import DiffusionBatch, TextBatch
from uniserve_worker.modeling.components import Call
from uniserve_worker.modeling.geometry import MediaShape, TextShape
from uniserve_worker.modeling.tensors import (
    AttentionMetadata,
    AttentionMode,
    ExpertRoute,
    FlowPatches,
    RouteSpan,
    TokenSelection,
)
from uniserve_worker.models.bagel import BagelConfig, BagelForConditionalGeneration, LLMConfig
from uniserve_worker.models.qwen3 import Qwen3ForCausalLM
from uniserve_worker.models.sensenova.config import NeoChatConfig
from uniserve_worker.models.sensenova.model import NEOChatModel
from uniserve_worker.nn.attention import RadixAttention, bind_attention_modules
from uniserve_worker.nn.attention_storage import attention_exchange_scope
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.mesh import Communicator
from uniserve_worker.nn.parallel import ParallelConfig, SequenceParallel
from uniserve_worker.runtime.attention_storage import allocate_attention_exchange_storage
from uniserve_worker.runtime.kv_cache import KVCache

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _pool(model, device):
    geometry = model.cache_geometry
    pool = KVCache(
        num_layers=geometry.num_layers,
        total_layers=geometry.total_layers,
        layer_offset=geometry.layer_offset,
        num_pages=8,
        page_size=64,
        num_kv_heads=geometry.num_kv_heads,
        total_kv_heads=geometry.total_kv_heads,
        kv_head_offset=geometry.kv_head_offset,
        head_dim=geometry.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    pool.k.zero_()
    pool.v.zero_()
    bind_attention_modules(
        model,
        pool,
        AttentionSelection(
            "flashinfer",
            (FlashInferAttentionBackend(tuning=FlashInferTuningConfig(workspace_size=64 << 20)),),
        ),
    )
    return pool


def _prefill(device):
    return TextBatch(
        attention=AttentionMetadata(
            attention_mode=AttentionMode.PAGED_VARLEN,
            prefix_lens=torch.zeros(2, dtype=torch.int32, device=device),
            query_lens=torch.tensor([3, 2], dtype=torch.int32, device=device),
            out_cache_loc=torch.tensor([64, 65, 66, 128, 129], device=device),
            block_table=torch.tensor([[1], [2]], dtype=torch.int32, device=device),
            seq_lens=torch.tensor([3, 2], dtype=torch.int32, device=device),
            cu_seqlens_q=torch.tensor([0, 3, 5], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, 3, 5], dtype=torch.int32, device=device),
            max_seqlen_q=3,
            max_seqlen_k=3,
            prefix_lens_cpu=(0, 0),
            query_lens_cpu=(3, 2),
            causal_rows_cpu=(True, True),
            seq_lens_cpu=(3, 2),
        ),
        input_ids=torch.tensor([1, 3, 5, 7, 9], device=device),
        positions=torch.tensor([0, 1, 2, 0, 1], device=device),
        selections=(TokenSelection.ALL_LOGITS, TokenSelection.HIDDEN),
    )


def _model(architecture, layer_config, tied=False):
    common = dict(
        vocab_size=65,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    if architecture == "qwen":
        return Qwen3ForCausalLM(
            dict(common, head_dim=128, attention_bias=False, tie_word_embeddings=tied),
            context=model_context(layer_config),
        )
    if architecture == "sensenova":
        config = NeoChatConfig(
            vision_config=dict(
                hidden_size=8,
                llm_hidden_size=512,
                downsample_ratio=0.5,
                patch_size=2,
                num_channels=3,
                rope_theta_vision=10000.0,
                max_position_embeddings_vision=128,
            ),
            llm_config=dict(
                common,
                head_dim=128,
                attention_bias=False,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                rope_theta_hw=10000.0,
                max_position_embeddings_hw=128,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            ),
            downsample_ratio=0.5,
            max_image_seq_len=16,
            fm_head_layers=2,
        )
        return NEOChatModel(config, context=model_context(layer_config))
    return BagelForConditionalGeneration(
        BagelConfig(
            llm=LLMConfig(**common),
            start_of_image_id=61,
            end_of_image_id=62,
            max_latent_size=2,
            vit_hidden_size=8,
            vit_intermediate_size=16,
            vit_num_hidden_layers=1,
            vit_num_attention_heads=2,
            vit_patch_size=14,
            vit_image_size=224,
            vit_max_num_patch_per_side=16,
        ),
        context=model_context(layer_config),
    )


def _diffusion_batch(architecture, device):
    image_tokens = 1 if architecture == "sensenova" else 3
    latent_width = 48 if architecture == "sensenova" else 64
    side = 4 if architecture == "sensenova" else 16
    dtype = torch.float32 if architecture == "sensenova" else torch.bfloat16
    latent = torch.arange(latent_width, device=device, dtype=dtype).view(1, -1) / 128
    patches = None
    if architecture == "sensenova":
        patches = FlowPatches(
            pixels=torch.arange(48, device=device, dtype=torch.bfloat16).view(4, 12) / 128,
            grid=torch.tensor([[2, 2]], device=device),
            noise_scale=torch.tensor(0.3, device=device),
        )
    spans = (RouteSpan(ExpertRoute.FLOW, 0, image_tokens),)
    if architecture == "bagel":
        spans = (
            RouteSpan(ExpertRoute.TEXT, 0, 1),
            RouteSpan(ExpertRoute.FLOW, 1, 1),
            RouteSpan(ExpertRoute.TEXT, 2, 1),
        )
    attention = AttentionMetadata(
        attention_mode=AttentionMode.PACKED,
        prefix_lens=torch.zeros(1, dtype=torch.int32, device=device),
        query_lens=torch.tensor([image_tokens], dtype=torch.int32, device=device),
        out_cache_loc=torch.empty(0, dtype=torch.int64, device=device),
        has_cache_writes=False,
        block_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor([0, image_tokens], dtype=torch.int32, device=device),
        attention_indexes=torch.zeros((3, image_tokens), dtype=torch.long, device=device),
        visible_end=torch.full((1, image_tokens), image_tokens, dtype=torch.int32, device=device),
        route_spans=spans,
        max_seqlen_q=image_tokens,
        max_seqlen_k=64,
        query_lens_cpu=(image_tokens,),
        prefix_lens_cpu=(0,),
        seq_lens_cpu=(image_tokens,),
        causal_rows_cpu=(False,),
        causal=False,
        fully_visible=True,
    )
    return DiffusionBatch(
        latents={"image": (latent,)},
        timesteps={"image": (torch.tensor(0.25, device=device),)},
        positions=(torch.zeros(1, dtype=torch.long, device=device),),
        conditioning={"image": (patches,)},
        sequence_lengths=(image_tokens,),
        shapes=(MediaShape(side, side),),
        attention=attention,
    )


def _load(model, architecture, weights):
    attach_parameter_loaders(model, device="cpu", dtype=torch.float32)
    if architecture == "qwen":
        report = model.load_weights(weights)
        assert not report.unexpected
        invalid = model.load_weights(
            (TensorWeightHandle("model.layers.0.unknown.weight", torch.zeros(1)),)
        )
        assert invalid.unexpected == ["model.layers.0.unknown.weight"]
    else:
        for component, source in zip(model.checkpoint_components(), weights, strict=True):
            assign_component(component, source, device="cpu")


def _checkpoint_name(architecture: str, source: str, name: str) -> str:
    """Serialize synthetic global tensors in the checkpoint's external namespace."""

    if architecture != "bagel" or source != "primary":
        return name
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    if name.startswith("lm."):
        return "language_model.model." + name.removeprefix("lm.")
    if name.startswith("vision.encoder.encoder.post_layernorm."):
        return "vit_model.vision_model.post_layernorm." + name.removeprefix(
            "vision.encoder.encoder.post_layernorm."
        )
    if name.startswith("vision.encoder.encoder."):
        return (
            ("vit_model.vision_model.encoder." + name.removeprefix("vision.encoder.encoder."))
            .replace(".mlp.0.", ".mlp.fc1.")
            .replace(".mlp.2.", ".mlp.fc2.")
        )
    if name.startswith("vision.encoder."):
        return "vit_model.vision_model.embeddings." + name.removeprefix("vision.encoder.")
    if name.startswith("vision.projection."):
        return "connector." + name.removeprefix("vision.projection.")
    if name.startswith("vision.position_embed."):
        return "vit_pos_embed." + name.removeprefix("vision.position_embed.")
    return name


def _run_pipeline(
    rank: int,
    rendezvous: str,
    architecture: str,
    tensor_size: int,
    pipeline_size: int,
    sequence_size: int,
    long_rows: bool = False,
) -> None:
    device = torch.device("cuda", rank)
    environment = initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=pipeline_size * tensor_size * sequence_size,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    )
    members = tuple(range(pipeline_size * tensor_size * sequence_size))
    parallel = ParallelConfig(
        tensor_parallel_size=tensor_size,
        pipeline_parallel_size=pipeline_size,
        sequence_parallel=SequenceParallel()
        if sequence_size == 1
        else SequenceParallel("ulysses", (sequence_size,)),
    )
    meshes = initialize_model_parallel(
        environment,
        {
            "ordered": (members, parallel),
            "reversed": (members[::-1], parallel),
        },
    )
    graph = None
    try:
        with torch.inference_mode():
            for tied in (False, True) if architecture == "qwen" and not long_rows else (False,):
                reference = _model(architecture, LayerConfig(Communicator(), None), tied)
                generator = torch.Generator().manual_seed(641)
                for parameter in reference.parameters():
                    if parameter.ndim == 1:
                        parameter.fill_(1)
                    else:
                        parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
                weights = (
                    tuple(
                        TensorWeightHandle(name, value.detach().clone())
                        for name, value in reference.named_parameters()
                    )
                    if architecture == "qwen"
                    else tuple(
                        tuple(
                            TensorWeightHandle(
                                _checkpoint_name(architecture, component.source, name),
                                value.detach().clone(),
                            )
                            for name, value in component.module.named_parameters()
                            if component.included is None or name in component.included
                        )
                        for component in reference.checkpoint_components()
                    )
                )
                del reference
                for mesh in meshes.values():
                    reference = _model(architecture, LayerConfig(mesh.get_group("tp"), None), tied)
                    model = _model(
                        architecture,
                        LayerConfig(
                            mesh.get_group("tp"),
                            None,
                            pipeline=mesh.get_group("pp"),
                            sequence=mesh.get_group("ulysses"),
                        ),
                        tied,
                    )
                    # Parallel and graph execution retain the established
                    # paged-attention BF16 error bound across reduction orders.
                    tolerance = 2e-2
                    for loaded in (reference, model):
                        _load(loaded, architecture, weights)
                        loaded.to(device=device, dtype=torch.bfloat16)
                    reference_pool, pool = _pool(reference, device), _pool(model, device)
                    batch = _prefill(device)
                    if architecture != "qwen":
                        batch = replace(
                            batch,
                            attention=replace(
                                batch.attention,
                                attention_mode=AttentionMode.PACKED,
                                attention_indexes=torch.stack(
                                    (
                                        batch.positions,
                                        torch.zeros_like(batch.positions),
                                        torch.zeros_like(batch.positions),
                                    )
                                ),
                                visible_end=torch.tensor(
                                    [[1, 2, 3], [1, 2, 0]], dtype=torch.int32, device=device
                                ),
                                route_spans=(RouteSpan(ExpertRoute.TEXT, 0, 5),),
                            ),
                        )

                    def execute(selected_model, selected_batch):
                        hidden = selected_model(selected_batch, constants={}, scratch={})
                        selected_model.tensor_specs(
                            Call.TEXT,
                            TextShape(selected_batch.input_ids.numel(), selected_batch.row_count),
                        ).outputs["hidden_states"].validate(hidden, state={}, scratch={})
                        output = selected_model.compute_logits(hidden, selected_batch)
                        output.validate(
                            tuple(
                                selected_model.tensor_specs(
                                    Call.TEXT, TextShape(count, selection=selection)
                                )
                                for count, selection in zip(
                                    selected_batch.attention.query_lens_cpu,
                                    selected_batch.selections,
                                    strict=True,
                                )
                            )
                        )
                        return output

                    expected = execute(reference, batch).materialize()
                    actual = execute(model, batch).materialize()
                    for result, wanted in zip(actual.values, expected.values, strict=True):
                        torch.testing.assert_close(result, wanted, rtol=tolerance, atol=tolerance)

                    batch = replace(
                        batch,
                        input_ids=torch.tensor([11, 13], device=device),
                        positions=torch.tensor([3, 2], device=device),
                        selections=(TokenSelection.LAST_LOGITS,) * 2,
                        attention=replace(
                            batch.attention,
                            attention_mode=AttentionMode.PAGED_DECODE,
                            prefix_lens=torch.tensor([3, 2], dtype=torch.int32, device=device),
                            seq_lens=torch.tensor([4, 3], dtype=torch.int32, device=device),
                            query_lens=torch.ones(2, dtype=torch.int32, device=device),
                            out_cache_loc=torch.tensor([67, 130], device=device),
                            cu_seqlens_q=None,
                            cu_seqlens_k=None,
                            prefix_lens_cpu=(3, 2),
                            seq_lens_cpu=(4, 3),
                            query_lens_cpu=(1, 1),
                        ),
                    )
                    execute(model, batch)
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = execute(model, batch)
                    for tokens in ((17, 19), (23, 29)):
                        batch.input_ids.copy_(torch.tensor(tokens, device=device))
                        graph.replay()
                        expected = execute(reference, batch).materialize()
                        actual = captured.materialize()
                        eager = execute(model, batch).materialize()
                        for result, wanted in zip(actual.values, eager.values, strict=True):
                            torch.testing.assert_close(
                                result, wanted, rtol=tolerance, atol=tolerance
                            )
                        for result, wanted in zip(actual.values, expected.values, strict=True):
                            torch.testing.assert_close(
                                result, wanted, rtol=tolerance, atol=tolerance
                            )
                    for local_layer in range(pool.num_layers):
                        for actual_cache, expected_cache in zip(
                            pool.layer_cache(local_layer, 0),
                            reference_pool.layer_cache(pool.layer_offset + local_layer, 0),
                            strict=True,
                        ):
                            start = pool.kv_head_offset - reference_pool.kv_head_offset
                            expected_cache = expected_cache.narrow(2, start, pool.n_kv)
                            torch.testing.assert_close(
                                actual_cache, expected_cache, rtol=tolerance, atol=tolerance
                            )
                    torch.cuda.synchronize(device)
                    graph.reset()
                    graph = None
                    if architecture != "qwen":
                        diffusion = _diffusion_batch(architecture, device)
                        expected = reference.forward_diffusion(
                            diffusion, state={}, constants={}, scratch={}
                        ).values["image"]
                        prediction = model.forward_diffusion(
                            diffusion, state={}, constants={}, scratch={}
                        )
                        prediction.validate(
                            tuple(
                                model.tensor_specs(Call.DIFFUSION, shape)
                                for shape in diffusion.shapes
                            ),
                            state={},
                            scratch={},
                        )
                        actual = prediction.values["image"]
                        # Non-output pipeline stages participate in communication
                        # without fabricating a local prediction head.
                        if model.diffusion_pipeline.last:
                            for result, wanted in zip(actual, expected, strict=True):
                                torch.testing.assert_close(
                                    result, wanted, rtol=tolerance, atol=tolerance
                                )
                    if long_rows:
                        # PACKED graph execution requires a graph-capable
                        # provider; FlashInfer's segmented forward plans on CPU.
                        for loaded, cache in ((reference, reference_pool), (model, pool)):
                            bind_attention_modules(
                                loaded,
                                cache,
                                AttentionSelection("fa4_cute", (Fa4CuteAttentionBackend(),)),
                            )
                        storage = allocate_attention_exchange_storage(
                            (
                                module
                                for module in model.modules()
                                if isinstance(module, RadixAttention)
                            ),
                            max_tokens=131075,
                            dtype=torch.bfloat16,
                        )
                        with attention_exchange_scope(storage.views):
                            _compare_long_packed_rows(model, reference, architecture, device)
                        del storage
                    del model, reference, pool, reference_pool
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize(device)
        environment.close()


def _compare_long_packed_rows(model, reference, architecture, device):
    # The logical activation exceeds one 32 MiB per-peer exchange interval.
    # Short independent requests keep attention work linear in the row count;
    # the last request and the last sequence owner both have partial capacity.
    rows = 131075
    lengths = (128,) * (rows // 128) + (rows % 128,)
    boundaries = torch.tensor((0, *lengths), device=device, dtype=torch.int32).cumsum(0).int()
    positions = torch.arange(rows, device=device).remainder(128)
    inputs = (
        torch.arange(rows * 512, device=device).remainder(17).reshape(rows, 512).bfloat16() / 32
    )
    spans = (RouteSpan(ExpertRoute.TEXT, 0, rows),)
    if architecture != "qwen":
        spans = (
            RouteSpan(ExpertRoute.TEXT, 0, 65537),
            RouteSpan(ExpertRoute.FLOW, 65537, rows - 65537),
        )
    batch = replace(
        _prefill(device),
        input_ids=torch.ones(rows, device=device, dtype=torch.long),
        positions=positions,
        selections=(TokenSelection.HIDDEN,) * len(lengths),
        attention=replace(
            _prefill(device).attention,
            attention_mode=AttentionMode.PACKED,
            prefix_lens=torch.zeros(len(lengths), dtype=torch.int32, device=device),
            query_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
            out_cache_loc=torch.empty(0, dtype=torch.long, device=device),
            has_cache_writes=False,
            block_table=torch.zeros((len(lengths), 1), dtype=torch.int32, device=device),
            cu_seqlens_q=boundaries,
            cu_seqlens_k=boundaries,
            visible_end=torch.tensor(lengths, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(-1, 128)
            .contiguous(),
            attention_indexes=torch.stack(
                (positions, torch.zeros_like(positions), torch.zeros_like(positions))
            ),
            route_spans=spans,
            max_seqlen_q=128,
            max_seqlen_k=128,
            prefix_lens_cpu=(0,) * len(lengths),
            query_lens_cpu=lengths,
            seq_lens_cpu=lengths,
            causal_rows_cpu=(False,) * len(lengths),
        ),
    )

    def execute(selected):
        if architecture == "qwen":
            return selected.model(inputs.clone(), batch.attention, positions=positions)
        decoder = (
            selected.language_model.model if architecture == "sensenova" else selected.model.lm
        )
        return decoder(inputs.clone(), batch.attention, positions=positions)

    graph = torch.cuda.CUDAGraph()
    try:
        expected = execute(reference)
        actual = execute(model)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            captured = execute(model)
        inputs.add_(0.0625)
        graph.replay()
        expected = execute(reference)
        eager = execute(model)
        torch.testing.assert_close(captured, eager, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)
    finally:
        graph.reset()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
@pytest.mark.parametrize("architecture", ["qwen", "sensenova", "bagel"])
def test_packed_decoder_streamed_rows_preserve_routed_outputs_under_replay(tmp_path, architecture):
    mp.spawn(
        _run_pipeline,
        args=(f"file://{tmp_path / 'row_intervals'}", architecture, 1, 1, 2, True),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
@pytest.mark.parametrize("architecture", ["qwen", "sensenova", "bagel"])
@pytest.mark.parametrize(
    "tensor_size,pipeline_size,sequence_size",
    [
        (1, 2, 1),
        (2, 2, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 1, 2),
        (1, 1, 4),
        (4, 1, 1),
    ],
)
def test_packed_pipeline_preserves_outputs_and_cache_regions(
    tmp_path: Path, architecture: str, tensor_size: int, pipeline_size: int, sequence_size: int
):
    world_size = pipeline_size * tensor_size * sequence_size
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} CUDA devices are required")
    mp.spawn(
        _run_pipeline,
        args=(
            f"file://{tmp_path / 'pipeline'}",
            architecture,
            tensor_size,
            pipeline_size,
            sequence_size,
        ),
        nprocs=world_size,
        join=True,
    )
