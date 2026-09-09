"""Model decode replay consumes live metadata through strided page-table views."""

from dataclasses import replace

import pytest
import torch

from uniserve_worker.backends.attention.flashinfer import FlashInferAttentionBackend
from uniserve_worker.backends.attention.tuning import FlashInferTuningConfig
from uniserve_worker.execution.cuda_graph import CudaGraphRunner
from uniserve_worker.execution.forward_batch import (
    AttentionMode,
    AttentionSelection,
    ForwardBatch,
    ModelPhase,
    TokenSelection,
)
from uniserve_worker.models.sensenova.config import NeoChatConfig
from uniserve_worker.models.sensenova.model import NEOChatModel
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.mesh import Communicator
from uniserve_worker.runtime.cache_pool import CachePool

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_model_decode_replay_consumes_live_strided_page_metadata():
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
    model = NEOChatModel(config, layer_config=LayerConfig(Communicator(), None))
    model.to(device=device, dtype=torch.bfloat16)
    for parameter in model.parameters():
        if parameter.ndim == 1:
            parameter.fill_(1)
        else:
            parameter.normal_(std=0.05)
    cache = model.cache_geometry
    pool = CachePool(
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
    backend = FlashInferAttentionBackend(
        tuning=FlashInferTuningConfig(workspace_size=64 << 20)
    )
    selection = AttentionSelection("flashinfer", (backend,))
    model.bind_cache_pool(pool, selection)
    rows = 3
    # A context-bounded view retains the wider staging allocation's row stride.
    # Only the first four physical pages per request contain live KV tokens.
    tables = torch.zeros((rows, 4096), dtype=torch.int32, device=device)
    tables[:, :4].copy_(torch.arange(1, 13, dtype=torch.int32, device=device).view(rows, 4))
    batch = ForwardBatch(
        phase=ModelPhase.TEXT,
        row_count=rows,
        forward_mode=AttentionMode.PAGED_DECODE,
        req_pool_indices=torch.arange(1, rows + 1, device=device),
        seq_lens=torch.tensor([62, 61, 60], dtype=torch.int32, device=device),
        query_lens=torch.ones(rows, dtype=torch.int32, device=device),
        out_cache_loc=torch.tensor([126, 381, 636], dtype=torch.int64, device=device),
        block_table=tables[:, :1319],
        kv_lens=torch.tensor([63, 62, 61], dtype=torch.int32, device=device),
        max_seqlen_k=1319 * 64,
        seq_lens_cpu=(62, 61, 60),
        query_lens_cpu=(1,) * rows,
        kv_lens_cpu=(63, 62, 61),
        token_row_indices=tuple(range(rows)),
        input_ids=torch.tensor([1, 3, 5], device=device),
        positions=torch.tensor([62, 61, 60], dtype=torch.int64, device=device),
        token_selections=(TokenSelection.LAST_LOGITS,) * rows,
    )
    runner = CudaGraphRunner(
        enabled=True,
        prefill_enabled=True,
        cache=cache,
        cache_pool=pool,
        attention=selection,
        block_size=64,
        weight_version=0,
        memory_budget_bytes=512 << 20,
        decode_batch_sizes=(rows,),
        decode_context_blocks=1319,
    )

    def forward(value):
        return model.project(model(value.input_ids, value.positions, value), value)

    try:
        runner.execute("text", batch, forward, eligible=True)
        runner.execute("text", batch, forward, eligible=True)
        runner.complete_startup()
        for lengths, tokens in (((198, 211, 200), (9, 11, 13)), ((199, 212, 201), (4, 2, 19))):
            batch.input_ids.copy_(torch.tensor(tokens, device=device))
            batch.kv_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
            batch.seq_lens.copy_(batch.kv_lens - 1)
            batch.positions.copy_(batch.seq_lens)
            page_columns = batch.seq_lens.long() // 64
            pages = batch.block_table.gather(1, page_columns[:, None])[:, 0]
            batch.out_cache_loc.copy_(pages.long() * 64 + batch.seq_lens % 64)
            # Eager invocations use distinct plan identities after metadata
            # changes; Graph replay owns its persistent binding separately.
            batch = replace(
                batch,
                binding=batch.binding + 10000,
                kv_lens_cpu=lengths,
                seq_lens_cpu=tuple(n - 1 for n in lengths),
            )
            actual = runner.execute("text", batch, forward, eligible=True).output
            expected = forward(batch)
            for result, reference in zip(actual.values, expected.values, strict=True):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize(device)
        runner.close()
