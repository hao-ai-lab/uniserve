"""Captured worker calls preserve live text values and prefix state."""

from contextlib import contextmanager

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from uniserve import loading
from uniserve.loading import weights
from uniserve.model import CausalLM
from uniserve.runtime import PrefixCache
from uniserve_models import loading as models
from uniserve_models.stub import image_processor as stub_image_processor
from uniserve_worker.bootstrap.cache import cache_info
from uniserve_worker.bootstrap.inputs import capability
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.model_executor.input_buffers import TokenBufferConfig
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    ForwardMode,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.storage.kv_cache import KVCacheManager

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


@contextmanager
def _text_runner(model, provider, decode_capacity=2, image_processor=None):
    """Capture a text worker's graphs and yield a call executor.

    With an ``image_processor`` declaring feature injection, the worker also
    captures prefill graphs for non-causal rows that replace their
    embeddings, as image feature rows do.
    """
    config = WorkerConfig(
        device="cuda:0",
        model_dtype="bfloat16",
        block_size=16,
        attention_backend=provider,
        max_batch_calls=4,
        max_request_pool_size=4,
        max_batch_tokens=64,
        max_sequence_tokens=64,
        prefill_cuda_graph=True,
        prefill_graph_token_sizes=(16,),
        decode_graph_batch_sizes=(1, decode_capacity),
    )
    cache = PrefixCache(
        model.cache_config, num_units=16, block_size=16, device="cuda:0"
    )
    manager = KVCacheManager(
        cache,
        info=cache_info(model, config, num_units=16),
        request_pool_size=4,
        table_width=4,
    )
    runner = ModelExecutor(model, config, image_processor=image_processor)
    predicates = torch.tensor([False, True, True, True, True], device="cuda:0")
    try:
        runner.configure_inputs(
            input_config=TokenBufferConfig(
                max_rows=4,
                max_tokens=64,
                max_text_tokens=64,
                table_widths=(4,),
                hidden_size=128,
            ),
            kv_cache=manager,
            latent_pool=None,
            decode_predicates=predicates,
            max_calls=4,
            request_slots=4,
            latent_capacity_units=0,
            table_widths=(2,),
            max_inflight=1,
        )
        saved = _snapshot(cache)
        runner.capture(tokenizer=None, latents=None)
        for actual, expected in saved:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        def execute(
            tokens,
            pages,
            *,
            prefixes=None,
            decode=False,
            selection=TokenSelection.LAST_LOGITS,
            causal=True,
            finish=False,
            embeddings=False,
        ):
            mode = ForwardMode.DECODE if decode else ForwardMode.PREFILL
            prefixes = (0,) * len(tokens) if prefixes is None else prefixes
            manager.block_tables.install(
                tuple(
                    (
                        index + 1,
                        0,
                        0,
                        tuple(page + 1 for page in blocks),
                        len(blocks) * 16,
                    )
                    for index, blocks in enumerate(pages)
                )
            )
            rows = tuple(
                TokenRow(
                    forward_mode=mode,
                    token_ids=torch.tensor(sequence, dtype=torch.int64),
                    # Supplied embeddings equal the token embeddings, as an
                    # image row's features stand in for its placeholders.
                    token_embeddings=capability(
                        model, CausalLM
                    ).embed_input_ids(torch.tensor(sequence, device="cuda:0"))
                    if embeddings
                    else None,
                    positions=torch.arange(prefix, prefix + len(sequence)),
                    selection=selection,
                    request_pool_idx=index + 1,
                    seq_len=prefix,
                    write_kv=True,
                    causal=causal,
                    decode_predicate=predicates[index + 1 : index + 2]
                    if decode
                    else None,
                    decode_predicate_tagged=decode,
                    decode_force_finish=finish and index == 0,
                )
                for index, (sequence, prefix) in enumerate(
                    zip(tokens, prefixes, strict=True)
                )
            )
            calls = tuple(
                Call(
                    request_key=RequestKey(1, index, 0),
                    call_id=CallId(1, index),
                    coordinates=CallCoordinates(),
                    kind=mode,
                    bounds=Bounds(),
                )
                for index in range(len(rows))
            )
            return runner.run_forward_group(
                rows,
                calls=calls,
                cache=manager,
                tables=manager.block_tables,
                states=None,
            ).materialize()

        yield execute
    finally:
        runner.close()
        manager.close()


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
    with _text_runner(model, provider, decode_capacity) as execute:
        for lengths in ((2, 5), (6, 1, 4), (16,)):
            sequences = tuple(
                tuple((index + token * 3) % 36 + 1 for token in range(length))
                for index, length in enumerate(lengths)
            )
            pages = tuple(
                (2 * index, 2 * index + 1) for index in range(len(lengths))
            )
            output = execute(sequences, pages)
            for actual, sequence in zip(output.values, sequences, strict=True):
                expected = reference(
                    torch.tensor(sequence, device="cuda:0")[None]
                ).logits[0, -1:]
                torch.testing.assert_close(
                    actual, expected, rtol=2e-2, atol=2e-2
                )

        retained = None
        for length, count in ((2, 2), (19, 1), (31, 2)):
            sequences = tuple(
                tuple(
                    (token * 7 + index) % 36 + 1 for token in range(length + 1)
                )
                for index in range(count)
            )
            pages = ((4, 5), (0, 1))[:count]
            execute(tuple(sequence[:-1] for sequence in sequences), pages)
            output = execute(
                tuple((sequence[-1],) for sequence in sequences),
                pages,
                prefixes=(length,) * count,
                decode=True,
                finish=length == 19,
            )
            for actual, sequence in zip(output.values, sequences, strict=True):
                expected = reference(
                    torch.tensor(sequence, device="cuda:0")[None]
                ).logits[0, -1:]
                torch.testing.assert_close(
                    actual, expected, rtol=2e-2, atol=2e-2
                )
            assert output.greedy.tokens.tolist() == [
                value.argmax().item() for value in output.values
            ]
            assert output.greedy.finish.tolist() == [
                length == 19 and index == 0 for index in range(count)
            ]
            if retained is not None:
                for actual, saved in zip(*retained, strict=True):
                    torch.testing.assert_close(actual, saved, rtol=0, atol=0)
            # Decode output is borrowed. A caller retaining it across another
            # invocation explicitly clones it through the output contract.
            retained = (
                output.clone().values,
                tuple(v.clone() for v in output.values),
            )
        # Different graph shapes share the input buffers. An
        # earlier prefill must still execute correctly after decode has
        # changed the same storage's lengths, IDs, positions and writes.
        sequences = ((3, 5, 7, 9), (11, 13))
        output = execute(sequences, ((2, 3), (6, 7)))
        for actual, sequence in zip(output.values, sequences, strict=True):
            expected = reference(
                torch.tensor(sequence, device="cuda:0")[None]
            ).logits[0, -1:]
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@torch.inference_mode()
def test_graph_statistics_count_live_and_padding_tokens(tmp_path):
    """Graph padding counters count tokens, not rows.

    A replay reports the live query tokens and the token slots its bucket
    adds beyond them.
    """
    _save_checkpoint(tmp_path)
    model = models.load_model(
        models.read_config(tmp_path, io=loading.Config()),
        device="cuda:0",
        weights=weights.Config(dtype=torch.bfloat16),
    ).model
    pages = ((0, 1), (2, 3))
    with _text_runner(model, "flashinfer", decode_capacity=4) as execute:
        # Two prompts of 2 and 5 tokens pad to the only prefill bucket, 8
        # rows of 16 tokens.
        prefill = execute(((1, 2), (3, 4, 5, 6, 7)), pages)
        # Two single-token rows pad to the 4-row decode bucket.
        decode = execute(((8,), (9,)), pages, prefixes=(2, 5), decode=True)

    assert prefill.stats.cuda_graph_replays == 1
    assert prefill.stats.cuda_graph_unpadded_tokens == 7
    assert prefill.stats.cuda_graph_padded_tokens == 16 - 7
    assert decode.stats.cuda_graph_replays == 1
    assert decode.stats.cuda_graph_unpadded_tokens == 2
    assert decode.stats.cuda_graph_padded_tokens == 4 - 2


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
    retained = None
    with _text_runner(
        model, provider, image_processor=stub_image_processor()
    ) as execute:
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
                execute(tuple(value[:prefix] for value in sequences), pages)
            output = execute(
                tuple(value[prefix:] for value in sequences),
                pages,
                prefixes=(prefix,) * len(lengths),
                selection=selection,
                causal=False,
                embeddings=True,
            )
            for actual, sequence in zip(output.values, sequences, strict=True):
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
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            retained = tuple((value, value.clone()) for value in output.values)


@torch.inference_mode()
def test_worker_runner_prepares_and_executes_declared_text_calls(tmp_path):
    from uniserve_worker.bootstrap.capacity import input_buffer_config
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.model_executor.input_batch import TokenRow
    from uniserve_worker.protocol.call import (
        Bounds,
        Call,
        CallCoordinates,
        ForwardMode,
    )
    from uniserve_worker.protocol.identity import CallId, RequestKey
    from uniserve_worker.sampling.metadata import TokenSelection

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
        max_batch_calls=2,
        max_batch_tokens=32,
        max_sequence_tokens=32,
        prefill_cuda_graph=True,
        prefill_graph_token_sizes=(16, 32),
        decode_graph_batch_sizes=(1, 2),
    )
    runner = ModelExecutor(model, config)
    cache = PrefixCache(
        model.cache_config, num_units=8, block_size=16, device="cuda:0"
    )
    manager = KVCacheManager(
        cache,
        info=cache_info(model, config, num_units=8),
        request_pool_size=2,
        table_width=2,
    )
    predicates = torch.tensor([False, True, True], device="cuda:0")
    try:
        runner.configure_inputs(
            input_config=input_buffer_config(model, config),
            kv_cache=manager,
            latent_pool=None,
            decode_predicates=predicates,
            max_calls=2,
            request_slots=2,
            latent_capacity_units=0,
            table_widths=(2,),
            max_inflight=1,
        )
        runner.capture(tokenizer=None, latents=None)
        runner.complete_startup()
        manager.block_tables.install(
            ((1, 0, 0, (4, 5), 32), (2, 0, 0, (6, 7), 32))
        )
        sequences = ((3, 7, 2, 9), (2, 5, 8))
        for mode in (ForwardMode.PREFILL, ForwardMode.DECODE):
            rows, calls = [], []
            for index, sequence in enumerate(sequences):
                prefix = 0 if mode is ForwardMode.PREFILL else len(sequence) - 1
                tokens = (
                    sequence[:-1]
                    if mode is ForwardMode.PREFILL
                    else sequence[-1:]
                )
                rows.append(
                    TokenRow(
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
                calls.append(
                    Call(
                        request_key=RequestKey(1, index, 0),
                        call_id=CallId(1, index),
                        coordinates=CallCoordinates(),
                        kind=mode,
                        bounds=Bounds(),
                    )
                )
            result = runner.run_forward_group(
                tuple(rows),
                calls=tuple(calls),
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
    from uniserve_worker.bootstrap.components import supported_calls
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
        max_batch_calls=2,
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
        allowed_calls=supported_calls(model),
        queue_depth=2,
        completion_payload_bytes=1 << 16,
    ) as worker:
        worker.warmup()
        assert worker.requests.request_ids() == ()
        worker.warmup()
        assert worker.requests.request_ids() == ()
