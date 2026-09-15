"""Captured worker calls preserve live text values and prefix state."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from uniserve import loading
from uniserve.loading import weights
from uniserve.model import TextSize
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve_models import loading as models
from uniserve_worker.bootstrap.cache import cache_info
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.graph_inputs import BatchGraph, pad_text
from uniserve_worker.execution.input_buffers import (
    InputBufferConfig,
    InputBuffers,
)
from uniserve_worker.execution.sampling import TokenSelection
from uniserve_worker.execution.startup import stage_text
from uniserve_worker.execution.text import TextCall
from uniserve_worker.runtime.cache_manager import CacheManager

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _save_checkpoint(root):
    config = Qwen3Config(
        vocab_size=37,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=64,
        rope_theta=10000,
        attention_bias=True,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(93)
    reference = Qwen3ForCausalLM(config).eval().to(dtype=torch.bfloat16)
    reference.save_pretrained(root)
    return reference.to("cuda:0")


def _snapshot(cache):
    return tuple(
        (tensor, tensor.clone())
        for name in cache.config.layers
        for values in cache.state(name).transfer_views(tuple(range(8))).values()
        for tensor in values
    )


@torch.inference_mode()
@pytest.mark.parametrize("provider", ["trtllm", "flash_attn_4", "flashinfer"])
@pytest.mark.parametrize("decode_capacity", [2, 4])
def test_text_graph_replay_uses_live_lengths_tokens_and_cache_blocks(
    tmp_path, provider, decode_capacity
):
    reference = _save_checkpoint(tmp_path)
    model = models.load_model(
        models.read_config(tmp_path, io=loading.Config()),
        device="cuda:0",
        weights=weights.Config(dtype=torch.bfloat16),
    ).model
    call = TextCall(model)
    worker = WorkerConfig(
        device="cuda:0", model_dtype="bfloat16", block_size=16
    )
    cache = PrefixCache(
        model.cache_config, num_blocks=8, block_size=16, device="cuda:0"
    )
    manager = CacheManager(
        cache, info=cache_info(model, worker, num_blocks=8), request_pool_size=4
    )
    # Captured table views use two columns of wider caller-owned backing.
    # Input borrowing must preserve strides instead of relying on a clone
    # having made those views contiguous.
    buffers = InputBuffers(
        config=InputBufferConfig(4, 64, 64, 4, 128), device="cuda:0"
    )
    stream = torch.cuda.Stream(device="cuda:0")
    stream.wait_stream(torch.cuda.current_stream())
    context = ExecutionContext(
        model, cache=cache, attention=provider, stream=stream
    )
    graphs = []
    try:
        with torch.cuda.stream(stream), context.activate():
            context.prepare(TextSize(64, 4))
            expected_rows = []
            for lengths in ((2, 5), (6, 1, 4), (16,)):
                sequences = tuple(
                    tuple(
                        (index + token * 3) % 36 + 1 for token in range(length)
                    )
                    for index, length in enumerate(lengths)
                )
                pages = tuple(
                    (2 * index, 2 * index + 1) for index in range(len(lengths))
                )
                batch = stage_text(buffers, manager, sequences, pages)
                padded = pad_text(batch, 4, 16, 2, False)
                if not graphs:
                    saved = _snapshot(cache)
                    graph = BatchGraph.capture(
                        context,
                        padded,
                        lambda item: call.last_logits(item.inputs),
                        cache=cache,
                    )
                    graphs.append(graph)
                    for actual, expected in saved:
                        torch.testing.assert_close(
                            actual, expected, rtol=0, atol=0
                        )
                output = (
                    graphs[0].replay(padded, rows=len(lengths)).materialize()
                )
                for actual, sequence in zip(
                    output.values, sequences, strict=True
                ):
                    expected = reference(
                        torch.tensor(sequence, device="cuda:0")[None]
                    ).logits[0, -1:]
                    torch.testing.assert_close(
                        actual, expected, rtol=2e-2, atol=2e-2
                    )
                expected_rows.append(output.values[0].clone())
            # Published output copies must survive a subsequent replay.
            assert not torch.equal(expected_rows[0], expected_rows[-1])

            predicate = torch.tensor(
                [False, True, True, True, True], device="cuda:0"
            )
            decode_graph = None
            for length in (2, 19, 31):
                sequences = tuple(
                    tuple(
                        (token * 7 + index) % 36 + 1
                        for token in range(length + 1)
                    )
                    for index in range(2)
                )
                pages = ((4, 5), (0, 1))
                prompt = stage_text(
                    buffers,
                    manager,
                    tuple(sequence[:-1] for sequence in sequences),
                    pages,
                )
                context.bind_attention(prompt.inputs.attention)
                call(prompt.inputs, prompt.token_selections)
                batch = stage_text(
                    buffers,
                    manager,
                    tuple((sequence[-1],) for sequence in sequences),
                    pages,
                    prefixes=(length,) * 2,
                    decode=True,
                )
                batch.decode_force_finish[0] = length == 19
                padded = pad_text(
                    batch, decode_capacity, decode_capacity, 2, True
                )
                if decode_graph is None:
                    saved = _snapshot(cache)
                    decode_graph = BatchGraph.capture(
                        context,
                        padded,
                        lambda item: call.last_logits(item.inputs),
                        cache=cache,
                        predicates=predicate,
                    )
                    graphs.append(decode_graph)
                    for actual, expected in saved:
                        torch.testing.assert_close(
                            actual, expected, rtol=0, atol=0
                        )
                output = decode_graph.replay(padded, rows=2).materialize()
                for actual, sequence in zip(
                    output.values, sequences, strict=True
                ):
                    expected = reference(
                        torch.tensor(sequence, device="cuda:0")[None]
                    ).logits[0, -1:]
                    torch.testing.assert_close(
                        actual, expected, rtol=2e-2, atol=2e-2
                    )
                assert output.greedy.tokens.tolist() == [
                    value.argmax().item() for value in output.values
                ]
                assert output.greedy.finish.tolist() == [length == 19, False]
            # Different graph shapes share the entry's staged columns. An
            # earlier prefill must still execute correctly after decode has
            # changed the same storage's lengths, IDs, positions and writes.
            sequences = ((3, 5, 7, 9), (11, 13))
            batch = stage_text(buffers, manager, sequences, ((2, 3), (6, 7)))
            output = (
                graphs[0]
                .replay(pad_text(batch, 4, 16, 2, False), rows=2)
                .materialize()
            )
            for actual, sequence in zip(output.values, sequences, strict=True):
                expected = reference(
                    torch.tensor(sequence, device="cuda:0")[None]
                ).logits[0, -1:]
                torch.testing.assert_close(
                    actual, expected, rtol=2e-2, atol=2e-2
                )
    finally:
        stream.synchronize()
        for graph in graphs:
            graph.graph.close()
        context.close()
        buffers.close()
        manager.close()


@torch.inference_mode()
@pytest.mark.parametrize("provider", ["trtllm", "flash_attn_4", "flashinfer"])
@pytest.mark.parametrize(
    "selection", [TokenSelection.HIDDEN, TokenSelection.LAST_LOGITS]
)
def test_noncausal_prefill_graph_preserves_live_prefixes_and_sequence_outputs(
    tmp_path, provider, selection
):
    reference = _save_checkpoint(tmp_path)
    model = models.load_model(
        models.read_config(tmp_path, io=loading.Config()),
        device="cuda:0",
        weights=weights.Config(dtype=torch.bfloat16),
    ).model
    call = TextCall(model)
    worker = WorkerConfig(
        device="cuda:0", model_dtype="bfloat16", block_size=16
    )
    cache = PrefixCache(
        model.cache_config, num_blocks=8, block_size=16, device="cuda:0"
    )
    manager = CacheManager(
        cache, info=cache_info(model, worker, num_blocks=8), request_pool_size=4
    )
    buffers = InputBuffers(
        config=InputBufferConfig(4, 64, 64, 4, 128), device="cuda:0"
    )
    stream = torch.cuda.Stream(device="cuda:0")
    stream.wait_stream(torch.cuda.current_stream())
    context = ExecutionContext(
        model, cache=cache, attention=provider, stream=stream
    )
    graph = None
    retained = None
    try:
        with torch.cuda.stream(stream), context.activate():
            context.prepare(TextSize(64, 4))
            for lengths, prefix in (((2, 5), 3), ((6, 1, 4), 17), ((16,), 0)):
                sequences = tuple(
                    tuple(
                        (index + token * 3) % 36 + 1
                        for token in range(prefix + length)
                    )
                    for index, length in enumerate(lengths)
                )
                pages = tuple(
                    (2 * index, 2 * index + 1) for index in range(len(lengths))
                )
                if prefix:
                    prompt = stage_text(
                        buffers,
                        manager,
                        tuple(value[:prefix] for value in sequences),
                        pages,
                    )
                    context.bind_attention(prompt.inputs.attention)
                    call(prompt.inputs, prompt.token_selections)
                batch = stage_text(
                    buffers,
                    manager,
                    tuple(value[prefix:] for value in sequences),
                    pages,
                    prefixes=(prefix,) * len(lengths),
                    selection=selection,
                    causal=False,
                )
                padded = pad_text(batch, 4, 16, 2, False)
                if graph is None:
                    saved = _snapshot(cache)
                    graph = BatchGraph.capture(
                        context,
                        padded,
                        lambda item: (
                            call.last_logits(item.inputs)
                            if selection is TokenSelection.LAST_LOGITS
                            else call(item.inputs, item.token_selections)
                        ),
                        cache=cache,
                    )
                    for actual, expected in saved:
                        torch.testing.assert_close(
                            actual, expected, rtol=0, atol=0
                        )
                output = graph.replay(padded, rows=len(lengths)).materialize()
                for actual, sequence in zip(
                    output.values, sequences, strict=True
                ):
                    # Prefix queries retain causal visibility. Appended image
                    # queries see the complete prefix and current image span.
                    size = len(sequence)
                    mask = torch.zeros(
                        (size, size), device="cuda:0", dtype=torch.bfloat16
                    )
                    if prefix:
                        mask[:prefix].masked_fill_(
                            torch.arange(size, device="cuda:0")[None]
                            > torch.arange(prefix, device="cuda:0")[:, None],
                            torch.finfo(mask.dtype).min,
                        )
                    expected = reference(
                        torch.tensor(sequence, device="cuda:0")[None],
                        attention_mask=mask[None, None],
                        output_hidden_states=True,
                    )
                    expected = (
                        expected.hidden_states[-1][0, prefix:]
                        if selection is TokenSelection.HIDDEN
                        else expected.logits[0, -1:]
                    )
                    torch.testing.assert_close(
                        actual, expected, rtol=2e-2, atol=2e-2
                    )
                if retained is not None:
                    for actual, expected in retained:
                        torch.testing.assert_close(
                            actual, expected, rtol=0, atol=0
                        )
                retained = tuple(
                    (value, value.clone()) for value in output.values
                )
    finally:
        stream.synchronize()
        if graph is not None:
            graph.graph.close()
        context.close()
        buffers.close()
        manager.close()


@torch.inference_mode()
def test_worker_runner_prepares_and_executes_declared_text_calls(tmp_path):
    from uniserve_worker.bootstrap.capacity import input_buffer_config
    from uniserve_worker.execution.model_runner import ModelRunner
    from uniserve_worker.execution.rows import ForwardRow
    from uniserve_worker.execution.sampling import TokenSelection
    from uniserve_worker.protocol.identity import ComputationId, RequestKey
    from uniserve_worker.protocol.operation import (
        Bounds,
        ForwardMode,
        ScheduledRequest,
    )

    reference = _save_checkpoint(tmp_path)
    model = models.load_model(
        models.read_config(tmp_path, io=loading.Config()),
        device="cuda:0",
        weights=weights.Config(dtype=torch.bfloat16),
    ).model
    config = WorkerConfig(
        device="cuda:0",
        model_dtype="bfloat16",
        block_size=16,
        max_request_pool_size=2,
        max_batch_operations=2,
        max_batch_tokens=32,
        max_sequence_tokens=32,
        prefill_cuda_graph=True,
        prefill_graph_token_sizes=(16, 32),
        decode_graph_batch_sizes=(1, 2),
    )
    runner = ModelRunner(model, config)
    cache = PrefixCache(
        model.cache_config, num_blocks=8, block_size=16, device="cuda:0"
    )
    manager = CacheManager(
        cache,
        info=cache_info(model, config, num_blocks=8),
        request_pool_size=2,
        max_blocks_per_request=2,
    )
    predicates = torch.tensor([False, True, True], device="cuda:0")
    try:
        runner.configure_inputs(
            input_config=input_buffer_config(model, config),
            kv_cache=manager,
            latent_pool=None,
            decode_predicates=predicates,
            max_operations=2,
            request_slots=2,
            max_tokens=32,
            latent_capacity_units=0,
            decode_context_blocks=2,
            variants=(),
            max_inflight=1,
        )
        runner.capture(tokenizer=None, latents=None)
        runner.complete_startup()
        manager.block_tables.install(((1, 0, (4, 5), 32), (2, 0, (6, 7), 32)))
        sequences = ((3, 7, 2, 9), (2, 5, 8))
        for mode in (ForwardMode.PREFILL, ForwardMode.DECODE):
            rows, operations = [], []
            for index, sequence in enumerate(sequences):
                prefix = 0 if mode is ForwardMode.PREFILL else len(sequence) - 1
                tokens = (
                    sequence[:-1]
                    if mode is ForwardMode.PREFILL
                    else sequence[-1:]
                )
                rows.append(
                    ForwardRow(
                        forward_mode=mode,
                        token_ids=torch.tensor(tokens),
                        positions=torch.arange(prefix, prefix + len(tokens)),
                        request_pool_idx=index + 1,
                        seq_len=prefix,
                        write_kv=True,
                        selection=TokenSelection.LAST_LOGITS,
                        decode_predicate=predicates[index + 1 : index + 2]
                        if mode is ForwardMode.DECODE
                        else None,
                        decode_predicate_tagged=mode is ForwardMode.DECODE,
                    )
                )
                operations.append(
                    ScheduledRequest(
                        RequestKey(1, index, 0),
                        ComputationId(1, index),
                        None,
                        mode,
                        Bounds(),
                    )
                )
            result = runner.run_forward_group(
                tuple(rows),
                operations=tuple(operations),
                cache=manager,
                tables=manager.block_tables,
                states=None,
            ).materialize()
            for value, sequence in zip(result.values, sequences, strict=True):
                tokens = (
                    sequence[:-1] if mode is ForwardMode.PREFILL else sequence
                )
                expected = reference(
                    torch.tensor(tokens, device="cuda:0")[None]
                ).logits[0, -1:]
                torch.testing.assert_close(
                    value, expected, rtol=2e-2, atol=2e-2
                )
            assert result.request_pool_indices.tolist() == [1, 2]
    finally:
        runner.close()
        manager.close()


@torch.inference_mode()
def test_loaded_worker_warmup_retires_its_request_resources(tmp_path):
    from uniserve.distributed import Communicator
    from uniserve_worker.bootstrap.components import supported_operations
    from uniserve_worker.worker import Worker

    _save_checkpoint(tmp_path)
    model = models.load_model(
        models.read_config(tmp_path), device="cuda:0"
    ).model
    config = WorkerConfig(
        device="cuda:0",
        block_size=16,
        kv_token_capacity=128,
        max_sequence_tokens=32,
        max_batch_operations=2,
        max_batch_tokens=32,
        max_request_pool_size=2,
        prefill_cuda_graph=True,
        prefill_graph_token_sizes=(16, 32),
        decode_graph_batch_sizes=(1, 2),
    )
    with Worker(
        model,
        worker_config=config,
        sampling_group=Communicator(device=torch.device("cuda:0")),
        tokenizer=None,
        allowed_work_variants=supported_operations(model),
        pipeline_depth=2,
        completion_payload_bytes=1 << 16,
    ) as worker:
        worker.warmup()
        assert worker.requests.request_ids() == ()
        worker.warmup()
        assert worker.requests.request_ids() == ()
