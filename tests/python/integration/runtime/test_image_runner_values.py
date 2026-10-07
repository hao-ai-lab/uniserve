"""Worker image staging preserves public numerical predictions.

It also preserves cache state.
"""

from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.checkpoints import (
    bagel_checkpoint,
    load_bagel,
    sensenova_checkpoint,
)
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.media import image
from uniserve.model import TextSize
from uniserve.nn.attention import AttentionBatch
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve_worker.bootstrap.cache import cache_info
from uniserve_worker.bootstrap.capacity import input_buffer_config
from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.diffusion import (
    flow_rows,
    image_state,
    prefix_row,
)
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.attention import from_blocks
from uniserve_worker.model_executor.component_binding import ComponentBinding
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    ForwardMode,
    ImageParams,
    MediaCall,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.storage.kv_cache import KVCacheManager
from uniserve_worker.storage.latent_pool import LatentPool

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
@pytest.mark.parametrize(
    "name,generation_device",
    [("bagel", None), ("sensenova_u1", None), ("sensenova_u1", "cuda:1")],
)
def test_guided_image_calls_reuse_graphs_without_writing_conditioning(
    tmp_path, name, generation_device
):
    if name == "bagel":
        _, _, config = bagel_checkpoint(tmp_path)
        model = load_bagel(tmp_path, config)
    else:
        model, _ = sensenova_checkpoint(tmp_path, torch.bfloat16)
    model.text.to("cuda:0")
    model.denoiser.to("cuda:0")
    if name == "sensenova_u1":
        model.vision_encoder.to("cuda:0")
    if generation_device is not None:
        from uniserve import loading
        from uniserve.loading import checkpoint, weights
        from uniserve_models import sensenova_u1
        from uniserve_worker.bootstrap.model_loader import _devices

        model = loading.load_model(
            sensenova_u1.Model,
            model.config,
            checkpoint=(
                checkpoint.Config("primary").resolve(
                    tmp_path, io=loading.Config()
                ),
            ),
            mapping=sensenova_u1.checkpoint_mappings,
            weights=weights.Config(dtype=torch.bfloat16),
            device="cuda:0",
            devices=_devices(model, generation_device),
        ).model
    group = Communicator((0,), 0, device=torch.device("cuda:0"))
    bindings = {
        "model": ComponentBinding(
            "model",
            ComponentConfig((0,)),
            group,
            DeviceMesh(
                ranks=(0,),
                rank=0,
                shape=tuple(
                    size
                    for _, size in ComponentConfig(
                        (0,)
                    ).parallel_config.dimensions
                ),
                axes=tuple(
                    axis
                    for axis, _ in ComponentConfig(
                        (0,)
                    ).parallel_config.dimensions
                ),
            ),
            torch.device("cuda:0"),
        )
    }
    config = WorkerConfig(
        device="cuda:0",
        generation_device=generation_device,
        attention_backend="torch",
        block_size=16,
        max_sequence_tokens=32,
        max_batch_calls=3,
        max_request_pool_size=3,
        max_batch_tokens=64,
        prefill_cuda_graph=True,
        prefill_graph_token_sizes=(16,),
        decode_graph_batch_sizes=(1,),
        flow_graph_shapes=((16, 16),),
        flow_graph_batch_sizes=(1,),
    )
    runner = ModelExecutor(model, config, bindings=bindings)
    size = image.Config(16, 16)
    factory = runner.image_builder
    shape = factory.denoiser.latent_shape("image", size)
    cache = PrefixCache(
        model.text.cache_config, num_blocks=8, block_size=16, device="cuda:0"
    )
    manager = KVCacheManager(
        cache,
        info=cache_info(model.text, config, num_blocks=8),
        request_pool_size=3,
        max_blocks_per_request=2,
    )
    latents = LatentPool(
        request_pool_size=3,
        num_pages=4,
        page_units=16,
        latent_width=shape[1],
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    oracle = ExecutionContext(model.denoiser, cache=cache, attention="torch")
    try:
        runner.configure_inputs(
            input_config=input_buffer_config(model, config),
            kv_cache=manager,
            latent_pool=latents,
            decode_predicates=torch.tensor(
                [False, True, True, True], device="cuda:0"
            ),
            max_calls=3,
            request_slots=3,
            max_tokens=64,
            latent_capacity_units=16,
            decode_context_blocks=2,
            max_inflight=1,
        )
        runner.capture(tokenizer=None, latents=latents)
        runner.complete_startup()
        manager.block_tables.install(
            ((1, 0, (1, 2), 32), (2, 0, (3, 4), 32), (3, 0, (5, 6), 32))
        )
        oracle.prepare(TextSize(64, 3))
        params = ImageParams(
            steps=3,
            height=16,
            width=16,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_renorm_type="none",
            seed=51,
        )
        trajectory = image_state(factory, size, params)
        sample = torch.empty(shape, dtype=torch.bfloat16, device="cuda:0")
        factory.initialize(size, seed=51, out=sample)
        retained = []
        for index, prefix_length in enumerate((2, 5)):
            schedule = trajectory.schedules["image"]
            branches = trajectory.guidance.branches(schedule, index)
            prefixes = tuple(
                tuple(
                    (token + branch * 3) % 30 + 1
                    for token in range(prefix_length)
                )
                for branch in range(len(branches))
            )
            trajectory.kv.entries = {
                branch: (slot + 1, 0, prefix_length, 32)
                for slot, branch in enumerate(branches)
            }

            def calls(mode):
                return tuple(
                    Call(
                        RequestKey(1, slot, 0),
                        CallId(1, index),
                        CallCoordinates(),
                        mode,
                        Bounds(),
                    )
                    for slot in range(len(branches))
                )

            result = runner.run_forward_group(
                tuple(
                    prefix_row(tokens, (slot + 1, 0, 0, 32))
                    for slot, tokens in enumerate(prefixes)
                ),
                calls=calls(ForwardMode.PREFILL),
                cache=manager,
                tables=manager.block_tables,
                states=None,
            )
            result.materialize()
            saved = tuple(
                (tensor, tensor.clone())
                for layer in cache.config.layers
                for fields in cache.state(layer)
                .transfer_views(tuple(range(8)))
                .values()
                for tensor in fields
            )
            time = schedule.timesteps[index].to("cuda:0")
            next_time = schedule.timesteps[index + 1].to("cuda:0")
            positions = tuple(
                factory.positions(size, prefix_length, device="cuda:0")
                for _ in branches
            )
            attention = from_blocks(
                pages=tuple(
                    (slot * 2 + 1, slot * 2 + 2)
                    for slot in range(len(branches))
                ),
                query_lengths=(factory.sequence_length(size),) * len(branches),
                prefix_lengths=(prefix_length,) * len(branches),
                block_size=16,
                causal=(False,) * len(branches),
                write=(False,) * len(branches),
            )
            from uniserve_worker.model_executor.cuda_graph import map_tensors

            attention = AttentionBatch.single(
                map_tensors(attention, lambda value: value.to("cuda:0"))
            )
            inputs = factory.bind(
                samples=(sample,) * len(branches),
                sizes=(size,) * len(branches),
                timesteps=(time,) * len(branches),
                positions=positions,
                attention=attention,
                step=schedule.step(index).to("cuda:0"),
            )
            with oracle.activate():
                oracle.bind_attention(attention)
                expected = tuple(
                    value.tensor.clone()
                    for value in model.denoiser(
                        inputs,
                        state={},
                        constants=oracle.constants,
                        workspace=oracle.workspace,
                    )["image"]
                )
            rows = flow_rows(
                factory,
                trajectory,
                sample,
                branches,
                time,
                conditioning_position=prefix_length,
                device=torch.device("cuda:0"),
            )
            result = runner.run_forward_group(
                rows,
                calls=calls(MediaCall.DENOISING),
                cache=manager,
                tables=manager.block_tables,
                states=None,
            ).materialize()
            for value, reference in zip(result.values, expected, strict=True):
                torch.testing.assert_close(
                    value, reference, rtol=2e-2, atol=2e-3
                )
            retained.extend(zip(result.values, expected, strict=True))
            for value, reference in saved:
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
            velocity = trajectory.guidance.combine(
                dict(zip(branches, expected, strict=True)),
                schedule,
                index,
            )
            expected_sample = (
                sample + velocity * (next_time - time).to(sample.dtype)
            ).to(sample.dtype)
            runner.diffusion_entry(calls(MediaCall.DENOISING)[0]).integrate(
                trajectory, sample, time, result.values, index
            )
            torch.testing.assert_close(
                sample, expected_sample, rtol=2e-2, atol=2e-3
            )
        for value, reference in retained:
            torch.testing.assert_close(value, reference, rtol=2e-2, atol=2e-3)
    finally:
        runner.close()
        oracle.close()
        latents.close()
        manager.close()


@torch.inference_mode()
@pytest.mark.parametrize("name", ["bagel", "sensenova_u1"])
def test_loaded_image_worker_completes_request_warmup(tmp_path, name):
    from uniserve.model import EmbeddingReplacement, TextInput
    from uniserve.nn.attention import (
        AttentionBatch,
        SequenceLengths,
        VarlenInput,
    )
    from uniserve_worker.bootstrap.components import supported_calls
    from uniserve_worker.model_executor.input_batch import TokenRow
    from uniserve_worker.sampling.metadata import TokenSelection
    from uniserve_worker.worker import Worker

    if name == "bagel":
        _, _, architecture = bagel_checkpoint(tmp_path)
        model = load_bagel(tmp_path, architecture)
    else:
        model, _ = sensenova_checkpoint(tmp_path, torch.bfloat16)
    model.text.to("cuda:0")
    model.denoiser.to("cuda:0")
    if name == "sensenova_u1":
        model.vision_encoder.to("cuda:0")
    placement = ComponentConfig((0,))
    group = Communicator((0,), 0, device=torch.device("cuda:0"))
    mesh = DeviceMesh(
        ranks=(0,),
        rank=0,
        shape=tuple(size for _, size in placement.parallel_config.dimensions),
        axes=tuple(axis for axis, _ in placement.parallel_config.dimensions),
    )
    bindings = {
        "model": ComponentBinding(
            "model", placement, group, mesh, torch.device("cuda:0")
        )
    }
    config = WorkerConfig(
        device="cuda:0",
        attention_backend="torch",
        block_size=16,
        kv_token_capacity=128,
        max_sequence_tokens=32,
        max_batch_calls=2,
        max_request_pool_size=2,
        max_batch_tokens=64,
        prefill_cuda_graph=True,
        prefill_graph_token_sizes=(16,),
        decode_graph_batch_sizes=(1,),
        flow_graph_shapes=((16, 16),),
        flow_graph_batch_sizes=(1,),
    )
    from uniserve.processing import ImageProcessor, PatchTransform

    processor = (
        ImageProcessor(
            vit=PatchTransform(2, 0.5, 16, 256), staging_dtype=torch.bfloat16
        )
        if name == "sensenova_u1"
        else None
    )
    with Worker(
        model,
        image_processor=processor,
        worker_config=config,
        bindings=bindings,
        sampling_group=group,
        tokenizer=None,
        allowed_calls=supported_calls(model),
        queue_depth=2,
        completion_payload_bytes=1 << 16,
        components=(("model", placement),),
    ) as worker:
        worker.warmup()
        assert worker.requests.request_ids() == ()
        worker.kv_cache.block_tables.install(((1, 0, (4, 5), 32),))
        tokens = torch.tensor([3, 7, 2], device="cuda:0")
        positions = torch.tensor(
            [[0, 1, 2], [0, 1, 0], [0, 2, 0]], device="cuda:0"
        )
        features = (
            torch.arange(3 * model.text.backbone.hidden_size, device="cuda:0")
            .reshape(3, model.text.backbone.hidden_size)
            .to(torch.bfloat16)
            / 100
        )
        mask = torch.tensor([False, True, False], device="cuda:0")
        lengths = SequenceLengths.from_lengths((3,), device="cuda:0")
        inputs = TextInput(
            tokens,
            positions,
            AttentionBatch.single(VarlenInput(lengths, lengths, (True,))),
            EmbeddingReplacement(features, mask),
        )
        # Feature appends use noncausal visibility with either hidden states
        # or logits; all forms consume live spatial positions and features.
        for selection in (TokenSelection.HIDDEN, TokenSelection.LAST_LOGITS):
            visual = replace(
                inputs,
                attention=AttentionBatch.single(
                    VarlenInput(lengths, lengths, (False,))
                ),
            )
            with ExecutionContext(model.text, attention="torch") as context:
                context.prepare(TextSize(3, 1))
                expected = model.text(visual)
                if selection is TokenSelection.LAST_LOGITS:
                    expected = model.text.compute_logits(
                        expected,
                        token_indices=torch.tensor([2], device="cuda:0"),
                    ).gather()
            row = TokenRow(
                forward_mode=ForwardMode.PREFILL,
                token_ids=tokens,
                positions=positions,
                token_embeddings=features,
                token_embedding_mask=mask,
                request_pool_idx=1,
                seq_len=0,
                write_kv=True,
                causal=False,
                selection=selection,
            )
            call = Call(
                RequestKey(1, 0, 0),
                CallId(1, 0),
                CallCoordinates(),
                ForwardMode.PREFILL,
                Bounds(),
            )
            actual = worker.runner.run_forward_group(
                (row,),
                calls=(call,),
                cache=worker.kv_cache,
                tables=worker.kv_cache.block_tables,
                states=None,
            ).materialize()
            torch.testing.assert_close(
                actual.values[0], expected, rtol=2e-2, atol=2e-2
            )

        # Replace the visual prefix with causal text before testing ordinary
        # continuation against a fully causal reference below.
        with ExecutionContext(model.text, attention="torch") as context:
            context.prepare(TextSize(3, 1))
            expected = model.text.compute_logits(
                model.text(inputs),
                token_indices=torch.tensor([2], device="cuda:0"),
            ).gather()
        row = TokenRow(
            forward_mode=ForwardMode.PREFILL,
            token_ids=tokens,
            positions=positions,
            token_embeddings=features,
            token_embedding_mask=mask,
            request_pool_idx=1,
            seq_len=0,
            write_kv=True,
            selection=TokenSelection.LAST_LOGITS,
        )
        call = Call(
            RequestKey(1, 0, 0),
            CallId(1, 0),
            CallCoordinates(),
            ForwardMode.PREFILL,
            Bounds(),
        )
        actual = worker.runner.run_forward_group(
            (row,),
            calls=(call,),
            cache=worker.kv_cache,
            tables=worker.kv_cache.block_tables,
            states=None,
        ).materialize()
        torch.testing.assert_close(
            actual.values[0], expected, rtol=2e-2, atol=2e-2
        )

        # Decode after spatial feature positions must read the same prefix
        # while ordinary continuation tokens use zero height/width coordinates.
        with ExecutionContext(model.text, attention="torch") as context:
            context.prepare(TextSize(5, 1))
            for token in (5, 4):
                prefix = tokens.numel()
                next_token = tokens.new_tensor([token])
                next_position = positions.new_tensor([[prefix], [0], [0]])
                tokens = torch.cat((tokens, next_token))
                positions = torch.cat((positions, next_position), dim=1)
                features = torch.cat((features, torch.zeros_like(features[:1])))
                mask = torch.cat((mask, torch.zeros_like(mask[:1])))
                lengths = SequenceLengths.from_lengths(
                    (prefix + 1,), device="cuda:0"
                )
                inputs = TextInput(
                    tokens,
                    positions,
                    AttentionBatch.single(
                        VarlenInput(lengths, lengths, (True,))
                    ),
                    EmbeddingReplacement(features, mask),
                )
                expected = model.text.compute_logits(
                    model.text(inputs), token_indices=next_position[0]
                ).gather()
                actual = worker.runner.run_forward_group(
                    (
                        TokenRow(
                            forward_mode=ForwardMode.DECODE,
                            token_ids=next_token,
                            positions=next_position[0],
                            request_pool_idx=1,
                            seq_len=prefix,
                            write_kv=True,
                            selection=TokenSelection.LAST_LOGITS,
                        ),
                    ),
                    calls=(
                        Call(
                            RequestKey(1, 0, 0),
                            CallId(1, 0),
                            CallCoordinates(),
                            ForwardMode.DECODE,
                            Bounds(),
                        ),
                    ),
                    cache=worker.kv_cache,
                    tables=worker.kv_cache.block_tables,
                    states=None,
                ).materialize()
                torch.testing.assert_close(
                    actual.values[0], expected, rtol=2e-2, atol=2e-2
                )

        if processor is not None:
            import base64
            import io

            from PIL import Image

            from tests.python.fixtures.depth_one import (
                ar_params,
                configure_physical_pool,
                encode_call,
                execution_batch,
                finalized_report,
                root_parent,
            )
            from uniserve.model import VisionInput
            from uniserve_worker.model_executor.image_inputs import (
                prepare_image,
            )
            from uniserve_worker.protocol.call import CallStatus
            from uniserve_worker.protocol.tensor import DeviceDim, ShapeBound

            configure_physical_pool(
                cache_pages=worker.info.kv_cache.num_blocks,
                request_pool_size=worker.info.request_slots,
                block_size=16,
                commit_marker_tokens=0,
                max_cfg_branches=3,
                latent_page_units=worker.info.latent_page_units,
                latent_downsample=(
                    worker.runner.image_builder.denoiser.downsample
                ),
            )
            encoded = io.BytesIO()
            Image.new("RGB", (16, 16), (64, 96, 128)).save(
                encoded, format="PNG"
            )
            payload = base64.b64encode(encoded.getvalue()).decode("ascii")
            pixels = prepare_image(
                processor,
                MediaCall.VISION_ENCODING,
                payload,
                device=torch.device("cuda:0"),
                input_images=1,
            )
            expected_features = model.vision_encoder.encode(
                VisionInput(
                    (pixels.pixels,), (pixels.grid,), (pixels.grid_shape,)
                )
            )[0]
            admission = ar_params(0, input_images=1)
            encode = encode_call(
                admission.request_key,
                call_id=CallId(1, 0),
                predecessor=root_parent(admission),
                image_base64=payload,
                encoder_handle=11,
            )
            encode = replace(
                encode,
                encoder_output=replace(
                    encode.encoder_output,
                    shape_bound=ShapeBound(
                        (DeviceDim(expected_features.numel()),)
                    ),
                ),
            )
            result = finalized_report(
                worker,
                worker.submit(
                    execution_batch(
                        batch_id=1,
                        admissions=(admission,),
                        calls=(encode,),
                    )
                ),
            )
            assert result.completions[0].status is CallStatus.OK
            read = worker.tensor_store.consume(
                encode.encoder_output, consumer_call_id=CallId(2, 0)
            )
            try:
                torch.testing.assert_close(
                    read.tensor, expected_features, rtol=2e-2, atol=2e-2
                )
            finally:
                worker.tensor_store.complete_reads((read,))
