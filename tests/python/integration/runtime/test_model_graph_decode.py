"""Model decode replay consumes live metadata through strided page-table views."""

from dataclasses import replace

import pytest
import torch

from uniserve_worker.backends.attention.flashinfer import FlashInferAttentionBackend
from uniserve_worker.backends.attention.tuning import FlashInferTuningConfig
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.forward_batch import (
    AttentionMetadata,
    AttentionMode,
    AttentionSelection,
    ForwardBatch,
    TokenSelection,
)
from uniserve_worker.execution.input_buffers import InputGeometry
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.models.sensenova.config import NeoChatConfig
from uniserve_worker.models.sensenova.model import NEOChatModel
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.mesh import Communicator
from uniserve_worker.protocol.batch import ForwardMode
from uniserve_worker.runtime.kv_cache import KVCache

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture
@torch.inference_mode()
def numerical_model(request):
    torch.manual_seed(619)
    device = torch.device("cuda", 0)
    config = NeoChatConfig(
        vision_config={
            "hidden_size": 8,
            "llm_hidden_size": 512,
            "downsample_ratio": 0.5,
            "patch_size": 2,
            "num_channels": 3,
            "rope_theta_vision": 10000.0,
            "max_position_embeddings_vision": 128,
        },
        llm_config={
            "vocab_size": 64,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "attention_bias": False,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 256,
            "rope_theta_hw": 10000.0,
            "max_position_embeddings_hw": 128,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        downsample_ratio=0.5,
        max_image_seq_len=16,
        fm_head_layers=2,
    )
    if getattr(request, "param", "sensenova") == "qwen":
        from uniserve_worker.models.qwen3 import Qwen3ForCausalLM

        model = Qwen3ForCausalLM(
            dict(
                vocab_size=64,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=128,
                max_position_embeddings=256,
            ),
            layer_config=LayerConfig(Communicator(), None),
        )
    else:
        model = NEOChatModel(config, layer_config=LayerConfig(Communicator(), None))
    model.to(device=device, dtype=torch.bfloat16)
    for parameter in model.parameters():
        if parameter.ndim == 1:
            parameter.fill_(1)
        else:
            parameter.normal_(std=0.05)
    pool = KVCache(
        num_layers=2,
        num_pages=16,
        page_size=64,
        num_kv_heads=2,
        head_dim=128,
        device=device,
        dtype=torch.bfloat16,
    )
    pool.k.normal_(std=0.1)
    pool.v.normal_(std=0.1)
    backend = FlashInferAttentionBackend(tuning=FlashInferTuningConfig(workspace_size=64 << 20))
    selection = AttentionSelection("flashinfer", (backend,))
    model.bind_cache_pool(pool, selection)
    yield model, pool, selection, device
    torch.cuda.synchronize(device)
    pool.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("selection_kind", [TokenSelection.LAST_LOGITS, TokenSelection.HIDDEN])
@torch.inference_mode()
def test_model_decode_replay_consumes_live_strided_page_metadata(selection_kind, numerical_model):
    model, pool, selection, device = numerical_model
    rows = 3
    # A context-bounded view retains the wider staging allocation's row stride.
    # Only the first four physical pages per request contain live KV tokens.
    tables = torch.zeros((rows, 4096), dtype=torch.int32, device=device)
    tables[:, :4].copy_(torch.arange(1, 13, dtype=torch.int32, device=device).view(rows, 4))
    batch = ForwardBatch(
        attention=AttentionMetadata(
            attention_mode=AttentionMode.PAGED_DECODE,
            prefix_lens=torch.tensor([62, 61, 60], dtype=torch.int32, device=device),
            query_lens=torch.ones(rows, dtype=torch.int32, device=device),
            out_cache_loc=torch.tensor([126, 381, 636], dtype=torch.int64, device=device),
            block_table=tables[:, :1319],
            seq_lens=torch.tensor([63, 62, 61], dtype=torch.int32, device=device),
            max_seqlen_k=1319 * 64,
            prefix_lens_cpu=(62, 61, 60),
            query_lens_cpu=(1,) * rows,
            seq_lens_cpu=(63, 62, 61),
        ),
        forward_mode=ForwardMode.DECODE,
        row_count=rows,
        request_pool_indices=torch.arange(1, rows + 1, device=device),
        token_row_indices=tuple(range(rows)),
        input_ids=torch.tensor([1, 3, 5], device=device),
        positions=torch.tensor([62, 61, 60], dtype=torch.int64, device=device),
        token_selections=(selection_kind,) * rows,
        decode_force_finish=torch.zeros(rows, dtype=torch.bool, device=device),
    )
    runner = ModelRunner(
        model,
        WorkerConfig(
            device=str(device),
            block_size=64,
            graph_policy="full",
            prefill_cuda_graph=True,
            decode_graph_batch_sizes=(rows,),
            prefill_graph_token_sizes=(),
        ),
        attention=selection,
    )
    predicates = torch.ones(rows + 1, dtype=torch.bool, device=device)
    runner.configure_inputs(
        geometry=InputGeometry(rows, rows, rows, 1319, 512),
        kv_cache=pool,
        latent_pool=None,
        decode_predicates=predicates,
        max_operations=rows,
        request_slots=rows,
        max_tokens=rows,
        latent_capacity_units=0,
        decode_context_blocks=1319,
        variants=frozenset((ForwardMode.DECODE,)),
        max_inflight=1,
    )
    entry = next(iter(runner.entries.values()))

    def forward(value):
        return model.project(model(value.input_ids, value.positions, value), value)

    try:
        runner.capture_batch(entry, batch, forward)
        runner.complete_startup()
        for index, (lengths, tokens) in enumerate(
            (((198, 211, 200), (9, 11, 13)), ((199, 212, 201), (4, 2, 19)))
        ):
            batch.input_ids.copy_(torch.tensor(tokens, device=device))
            batch.attention.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
            batch.attention.prefix_lens.copy_(batch.attention.seq_lens - 1)
            batch.positions.copy_(batch.attention.prefix_lens)
            page_columns = batch.attention.prefix_lens.long() // 64
            pages = batch.attention.block_table.gather(1, page_columns[:, None])[:, 0]
            batch.attention.out_cache_loc.copy_(
                pages.long() * 64 + batch.attention.prefix_lens % 64
            )
            # Eager invocations use distinct plan identities after metadata
            # changes; Graph replay owns its persistent binding separately.
            batch = replace(
                batch,
                attention=replace(
                    batch.attention,
                    binding=batch.attention.binding + 10000,
                    seq_lens_cpu=lengths,
                    prefix_lens_cpu=tuple(n - 1 for n in lengths),
                ),
            )
            predicates[1:].copy_(torch.tensor([True, False, True], device=device))
            batch.decode_force_finish.copy_(torch.tensor([False, False, True], device=device))
            execution = runner.run_batch(entry, batch, forward, eligible=True)
            actual = execution
            expected = forward(batch)
            for result, reference in zip(actual.values, expected.values, strict=True):
                torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)
            if selection_kind is TokenSelection.HIDDEN:
                # Hidden-state consumers must not receive vocabulary samples.
                assert execution.greedy is None
            else:
                assert execution.greedy is not None
                torch.testing.assert_close(
                    execution.greedy.active, torch.tensor([True, False, True], device=device)
                )
                torch.testing.assert_close(
                    execution.greedy.finish, torch.tensor([False, False, True], device=device)
                )
                torch.testing.assert_close(
                    execution.greedy.continuation, torch.tensor([True, False, False], device=device)
                )
                logits = torch.cat(expected.materialize().values, dim=0)
                torch.testing.assert_close(execution.greedy.tokens, logits.argmax(dim=-1))
    finally:
        torch.cuda.synchronize(device)
        runner.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("numerical_model", ["qwen"], indirect=True)
@torch.inference_mode()
def test_model_prefill_padding_preserves_live_outputs(numerical_model):
    from uniserve_worker.execution.runners.prefill import stage_text

    model, pool, selection, device = numerical_model
    runner = ModelRunner(
        model,
        WorkerConfig(
            device=str(device),
            block_size=64,
            graph_policy="full",
            prefill_cuda_graph=True,
            decode_graph_batch_sizes=(),
            prefill_graph_token_sizes=(8, 16),
        ),
        attention=selection,
    )
    runner.configure_inputs(
        geometry=InputGeometry(8, 16, 16, 4, 512),
        kv_cache=pool,
        latent_pool=None,
        decode_predicates=torch.ones(4, dtype=torch.bool, device=device),
        max_operations=3,
        request_slots=3,
        max_tokens=16,
        latent_capacity_units=0,
        decode_context_blocks=4,
        variants=frozenset((ForwardMode.PREFILL,)),
        max_inflight=1,
    )
    entry = next(iter(runner.entries.values()))
    buffers = entry.input_buffers
    assert buffers is not None

    def forward(batch):
        return model.project(model(batch.input_ids, batch.positions, batch), batch)

    try:
        runner.capture(tokenizer=None, latents=None)
        runner.complete_startup()
        with pool.startup_pages(3) as pages:
            for lengths in ((3,), (3, 2), (4, 5), (1, 5, 4)):
                tokens = tuple(tuple(range(1, length + 1)) for length in lengths)
                batch = stage_text(
                    buffers,
                    pool,
                    tokens,
                    tuple((page,) for page in pages[: len(lengths)]),
                    packed=False,
                )
                expected = forward(batch).clone()
                actual = runner.run_batch(entry, batch, forward, eligible=True)
                for result, reference in zip(actual.values, expected.values, strict=True):
                    torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)
    finally:
        torch.cuda.synchronize(device)
        runner.close()
