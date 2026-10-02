//! Direct FlatBuffers construction from borrowed protocol values.
//!
//! The parent `codec` module reaches this submodule only through `request`
//! and `response`, called by `encode_request` and `encode_response`, which
//! first validate submitted batches, worker info, and batch outputs. The
//! builders perform no checks of their own and cannot fail; they rely on that
//! validation for narrowing casts such as the parallel degrees in `parallel`.
//!
//! FlatBuffers builds bottom-up: a table can reference only objects already
//! written to the builder. Each function first serializes its strings,
//! vectors, and child tables, then creates its own table from their offsets.
//! The schema's structs (`CallId`, `CallKind`, `TokenLogprob`) are stored
//! inline, so they are built as plain values rather than through the builder.
//!
//! Decoding lives in the parent module's `*_from_table` and `*_from_fb`
//! functions, so a schema field written here must also be read there.
//! Collections are written as vectors even when empty, with two exceptions:
//! `allowed_token_ids` is written only when `Some`, because absence (no
//! whitelist) and an empty vector mean different things, and `locator`
//! writes only the vectors of the selected transport. The decoders read most
//! absent vectors as empty, but reject an absent
//! `WorkerInfo.transfer_backends` or `ComponentInfo.outputs`.

use super::*;
use flatbuffers::WIPOffset;

fn request_key<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &RequestKey,
) -> WIPOffset<fbs::RequestKey<'a>> {
    let request_id = v.request_id.0;
    fbs::RequestKey::create(
        b,
        &fbs::RequestKeyArgs {
            engine_id: v.engine_id,
            request_id,
            request_epoch: v.request_epoch,
        },
    )
}

fn buffer_id<'a>(b: &mut FlatBufferBuilder<'a>, v: &BufferId) -> WIPOffset<fbs::BufferId<'a>> {
    let owner = Some(request_key(b, &v.owner));
    let producer_call_id = fbs::CallId::new(
        v.producer_call_id.batch_id,
        v.producer_call_id.request_index,
    );
    fbs::BufferId::create(
        b,
        &fbs::BufferIdArgs {
            owner,
            producer_call_id: Some(&producer_call_id),
            output_index: v.output_index,
            generation: v.generation,
        },
    )
}

/// Writes a tensor reference. The Rust value stores its storage identity as
/// flat fields; the wire nests it in the schema-required `BufferId` table,
/// assembled by `TensorRef::buffer_id`.
fn tensor_ref<'a>(b: &mut FlatBufferBuilder<'a>, v: &TensorRef) -> WIPOffset<fbs::TensorRef<'a>> {
    let (extents, dynamic_axis) = shape_bound_to_parts(&v.shape_bound);
    let id = Some(buffer_id(b, &v.buffer_id()));
    let dtype = dtype_to_fb(v.dtype);
    let extents = Some(b.create_vector(&extents));
    fbs::TensorRef::create(
        b,
        &fbs::TensorRefArgs {
            id,
            dtype,
            extents,
            dynamic_axis,
        },
    )
}

fn rng<'a>(b: &mut FlatBufferBuilder<'a>, v: &Rng) -> WIPOffset<fbs::Rng<'a>> {
    let draw_layout = draw_layout_to_fb(v.draw_layout);
    fbs::Rng::create(
        b,
        &fbs::RngArgs {
            seed: v.seed,
            semantic_index_base: v.semantic_index_base,
            draw_layout,
        },
    )
}

fn coordinates<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &CallCoordinates,
) -> WIPOffset<fbs::CallCoordinates<'a>> {
    fbs::CallCoordinates::create(
        b,
        &fbs::CallCoordinatesArgs {
            logical_position: v.logical_position,
            kv_visible_len: v.kv_visible_len,
            kv_computed_len: v.kv_computed_len,
            flow_step: v.flow_step,
        },
    )
}

fn sampling_state<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &SamplingState,
) -> WIPOffset<fbs::SamplingState<'a>> {
    // `None` stays absent so that `call_from_table` decodes it back to
    // `None`; a present empty whitelist is an all-masked distribution.
    let allowed_token_ids = v
        .allowed_token_ids
        .as_deref()
        .map(|ids| b.create_vector(ids));
    let suppressed_token_ids = Some(b.create_vector(&v.suppressed_token_ids));
    let finish_token_ids = Some(b.create_vector(&v.finish_token_ids));
    let transition_token_ids = Some(b.create_vector(&v.transition_token_ids));
    fbs::SamplingState::create(
        b,
        &fbs::SamplingStateArgs {
            allowed_token_ids,
            suppressed_token_ids,
            finish_token_ids,
            transition_token_ids,
            force_finish: v.force_finish,
        },
    )
}

fn sampling<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &SamplingParams,
) -> WIPOffset<fbs::SamplingParams<'a>> {
    let logit_bias = {
        let values = v
            .logit_bias
            .iter()
            .map(|&(token_id, bias)| {
                fbs::TokenBias::create(b, &fbs::TokenBiasArgs { token_id, bias })
            })
            .collect::<Vec<_>>();
        Some(b.create_vector(&values))
    };
    let bad_words_ids = {
        let values = v
            .bad_words_ids
            .iter()
            .map(|ids| {
                let items = Some(b.create_vector(ids));
                fbs::U32List::create(b, &fbs::U32ListArgs { items })
            })
            .collect::<Vec<_>>();
        Some(b.create_vector(&values))
    };
    // `None` (no restriction) stays absent; `SamplingParams::validate`
    // rejects a present empty whitelist.
    let allowed_token_ids = v
        .allowed_token_ids
        .as_deref()
        .map(|ids| b.create_vector(ids));
    let logprob_token_ids = Some(b.create_vector(&v.logprob_token_ids));
    let forced_token_ids = Some(b.create_vector(&v.forced_token_ids));
    fbs::SamplingParams::create(
        b,
        &fbs::SamplingParamsArgs {
            temperature: v.temperature,
            top_k: v.top_k,
            top_p: v.top_p,
            ignore_eos: v.ignore_eos,
            seed: v.seed,
            min_p: v.min_p,
            repetition_penalty: v.repetition_penalty,
            frequency_penalty: v.frequency_penalty,
            presence_penalty: v.presence_penalty,
            logit_bias,
            // `sampling_from_table` rejects a value that does not fit the
            // receiver's `usize`.
            min_tokens: v.min_tokens as u64,
            n_logprobs: v.n_logprobs,
            bad_words_ids,
            allowed_token_ids,
            return_logprobs: v.return_logprobs,
            return_prompt_logprobs: v.return_prompt_logprobs,
            n_prompt_logprobs: v.n_prompt_logprobs,
            logprob_token_ids,
            typical_p: v.typical_p,
            forced_token_ids,
        },
    )
}

fn image<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &uniserve_core::ImageParams,
) -> WIPOffset<fbs::ImageParams<'a>> {
    let cfg_renorm_type = Some(b.create_string(v.cfg_renorm_type.as_str()));
    let cfg_interval_lo = v.cfg_interval.0;
    let cfg_interval_hi = v.cfg_interval.1;
    let negative_prompt = Some(b.create_string(&v.negative_prompt));
    let image_prompts = {
        let items = v
            .image_prompts
            .iter()
            .map(|s| b.create_string(s))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    fbs::ImageParams::create(
        b,
        &fbs::ImageParamsArgs {
            steps: v.steps,
            cfg_text_scale: v.cfg_text_scale,
            cfg_img_scale: v.cfg_img_scale,
            cfg_renorm_type,
            cfg_renorm_min: v.cfg_renorm_min,
            cfg_interval_lo,
            cfg_interval_hi,
            timestep_shift: v.timestep_shift,
            height: v.height,
            width: v.width,
            seed: v.seed,
            negative_prompt,
            max_images: v.max_images,
            image_prompts,
            retain_images: v.retain_images,
        },
    )
}

fn ar<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &ArRequestParams,
) -> WIPOffset<fbs::ArRequestParams<'a>> {
    let sampling = Some(sampling(b, &v.sampling));
    let negative_token_ids = Some(b.create_vector(&v.negative_token_ids));
    let finish_token_ids = Some(b.create_vector(&v.finish_token_ids));
    fbs::ArRequestParams::create(
        b,
        &fbs::ArRequestParamsArgs {
            sampling,
            negative_token_ids,
            finish_token_ids,
            initial_position: v.initial_position,
        },
    )
}

fn diffusion<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &DiffusionSamplingParams,
) -> WIPOffset<fbs::DiffusionSamplingParams<'a>> {
    fbs::DiffusionSamplingParams::create(
        b,
        &fbs::DiffusionSamplingParamsArgs {
            num_frames: v.num_frames,
            video_units: v.video_units,
            num_inference_steps: v.num_inference_steps,
            seed: v.seed,
            width: v.width,
            height: v.height,
        },
    )
}

fn canvas(value: uniserve_core::Canvas) -> fbs::Canvas {
    fbs::Canvas::new(value.width, value.height)
}

fn audio_clip(value: &uniserve_core::AudioClip) -> fbs::AudioClip {
    fbs::AudioClip::new(
        value.sample_rate,
        value.start_sample,
        value.source_samples,
        value.samples,
    )
}

fn video_condition<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &uniserve_core::VideoCondition,
) -> WIPOffset<fbs::VideoCondition<'a>> {
    use uniserve_core::{ConditionMedia, ConditionRole};

    let source_name = Some(b.create_string(&v.source.name));
    let vision = v.vision.as_ref().map(|vision| {
        let frame_indices = Some(b.create_vector(&vision.frame_indices));
        fbs::ConditionVision::create(
            b,
            &fbs::ConditionVisionArgs {
                grid: Some(&fbs::VisionGrid::new(
                    vision.grid.t,
                    vision.grid.h,
                    vision.grid.w,
                )),
                tokens: vision.tokens,
                frame_indices,
            },
        )
    });
    let latent_units = Some(b.create_vector(&v.latent_units));
    let (image, video, audio) = match &v.media {
        ConditionMedia::Image(fit) => (
            Some(fbs::ImageFit::new(
                &canvas(fit.resized),
                fit.left,
                fit.top,
                &canvas(fit.size),
            )),
            None,
            None,
        ),
        ConditionMedia::Video { clip, soundtrack } => (
            None,
            Some(fbs::VideoClip::new(
                &canvas(clip.canvas),
                clip.start_frame,
                clip.frames,
                clip.vae_frames,
            )),
            soundtrack.as_ref().map(audio_clip),
        ),
        ConditionMedia::Audio(clip) => (None, None, Some(audio_clip(clip))),
    };
    fbs::VideoCondition::create(
        b,
        &fbs::VideoConditionArgs {
            role: match v.role {
                ConditionRole::FirstFrame => fbs::ConditionRole::FirstFrame,
                ConditionRole::LastFrame => fbs::ConditionRole::LastFrame,
                ConditionRole::Reference => fbs::ConditionRole::Reference,
            },
            source_name,
            source_bytes: v.source.bytes,
            image: image.as_ref(),
            video: video.as_ref(),
            audio: audio.as_ref(),
            vision,
            latent_units,
            audio_rows: v.audio_rows,
        },
    )
}

fn video_admission<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &VideoAdmission,
) -> WIPOffset<fbs::VideoAdmission<'a>> {
    let text_tags = Some(b.create_vector(&v.text_tags));
    let conditions = v
        .conditions
        .iter()
        .map(|condition| video_condition(b, condition))
        .collect::<Vec<_>>();
    let conditions = Some(b.create_vector(&conditions));
    fbs::VideoAdmission::create(
        b,
        &fbs::VideoAdmissionArgs {
            task: match v.task {
                uniserve_core::VideoTask::T2va => fbs::VideoTask::T2va,
                uniserve_core::VideoTask::Fl2va => fbs::VideoTask::Fl2va,
                uniserve_core::VideoTask::Ref2va => fbs::VideoTask::Ref2va,
            },
            text_tags,
            conditions,
        },
    )
}

fn admission<'a>(b: &mut FlatBufferBuilder<'a>, v: &NewRequest) -> WIPOffset<fbs::NewRequest<'a>> {
    let request_key = Some(request_key(b, &v.request_key));
    let ar = v.ar.as_ref().map(|value| ar(b, value));
    let image = v.image.as_ref().map(|value| image(b, value));
    let diffusion = v.diffusion.as_ref().map(|value| diffusion(b, value));
    let video = v.video.as_ref().map(|value| video_admission(b, value));
    let prompt_token_ids = Some(b.create_vector(&v.prompt_token_ids));
    fbs::NewRequest::create(
        b,
        &fbs::NewRequestArgs {
            request_key,
            request_pool_idx: v.request_pool_idx,
            ar,
            image,
            diffusion,
            prompt_token_ids,
            input_images: v.input_images,
            video,
        },
    )
}

fn block_table<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &BlockTable,
) -> WIPOffset<fbs::BlockTable<'a>> {
    let page_ids = {
        let values = v.page_ids.iter().map(|page| page.0).collect::<Vec<_>>();
        Some(b.create_vector(&values))
    };
    fbs::BlockTable::create(
        b,
        &fbs::BlockTableArgs {
            request_pool_idx: v.request_pool_idx,
            group_id: v.group_id,
            page_ids,
            allocated_tokens: v.allocated_tokens,
        },
    )
}

fn cache_pages<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &CachePageAllocation,
) -> WIPOffset<fbs::CachePageAllocation<'a>> {
    let page_ids = {
        let values = v.page_ids.iter().map(|page| page.0).collect::<Vec<_>>();
        Some(b.create_vector(&values))
    };
    fbs::CachePageAllocation::create(
        b,
        &fbs::CachePageAllocationArgs {
            request_pool_idx: v.request_pool_idx,
            group_id: v.group_id,
            page_ids,
        },
    )
}

fn latent_params<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &LatentParams,
) -> WIPOffset<fbs::LatentParams<'a>> {
    let request_key = Some(request_key(b, &v.request_key));
    let call_id = fbs::CallId::new(v.call_id.batch_id, v.call_id.request_index);
    let page_table = Some(b.create_vector(&v.page_table));
    fbs::LatentParams::create(
        b,
        &fbs::LatentParamsArgs {
            request_key,
            call_id: Some(&call_id),
            page_table,
            latent_units: v.latent_units,
            height: v.height,
            width: v.width,
            start_step: v.start_step,
            step_count: v.step_count,
        },
    )
}

fn decode_range<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &DecodeRange,
) -> WIPOffset<fbs::DecodeRange<'a>> {
    let request_key = Some(request_key(b, &v.request_key));
    let call_id = fbs::CallId::new(v.call_id.batch_id, v.call_id.request_index);
    fbs::DecodeRange::create(
        b,
        &fbs::DecodeRangeArgs {
            request_key,
            call_id: Some(&call_id),
            cursor: v.cursor,
            max_units: v.max_units,
        },
    )
}

fn buffer_allocation<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &BufferAllocation,
) -> WIPOffset<fbs::BufferAllocation<'a>> {
    let buffer = Some(buffer_id(b, &v.buffer));
    fbs::BufferAllocation::create(
        b,
        &fbs::BufferAllocationArgs {
            buffer,
            offset: v.offset,
            bytes: v.bytes,
        },
    )
}

fn call<'a>(b: &mut FlatBufferBuilder<'a>, v: &Call) -> WIPOffset<fbs::Call<'a>> {
    let request_key = Some(request_key(b, &v.request_key));
    let call_id = fbs::CallId::new(v.call_id.batch_id, v.call_id.request_index);
    let coordinates = Some(coordinates(b, &v.coordinates));
    let component = Some(b.create_string(&v.component));
    let code = computation_to_fb(v.code);

    // The schema has no `Bounds` table: the bounds are scalar fields of the
    // `Call` table, which `call_from_table` reassembles.
    let max_tokens = v.bounds.max_tokens;
    let max_kv_pages = v.bounds.max_kv_pages;
    let max_latent_bytes = v.bounds.max_latent_bytes;
    let max_completion_bytes = v.bounds.max_completion_bytes;
    let max_transfer_bytes = v.bounds.max_transfer_bytes;

    let inputs = {
        let items = v
            .inputs
            .iter()
            .map(|item| tensor_ref(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let outputs = {
        let items = v
            .outputs
            .iter()
            .map(|item| tensor_ref(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };

    let token_input = v.token_input.as_ref().map(|value| tensor_ref(b, value));
    let token_output = v.token_output.as_ref().map(|value| tensor_ref(b, value));
    let vision_input = v.vision_input.as_ref().map(|value| tensor_ref(b, value));
    let latent_feature_input = v
        .latent_feature_input
        .as_ref()
        .map(|value| tensor_ref(b, value));
    let encoder_output = v.encoder_output.as_ref().map(|value| tensor_ref(b, value));
    let latent_input = v.latent_input.as_ref().map(|value| tensor_ref(b, value));
    let latent_output = v.latent_output.as_ref().map(|value| tensor_ref(b, value));
    let image_input = v.image_input.as_ref().map(|value| tensor_ref(b, value));
    let image_output = v.image_output.as_ref().map(|value| tensor_ref(b, value));
    let completion_output = v
        .completion_output
        .as_ref()
        .map(|value| tensor_ref(b, value));
    let transition_output = v
        .transition_output
        .as_ref()
        .map(|value| tensor_ref(b, value));
    let predicate = v.predicate.as_ref().map(|value| tensor_ref(b, value));

    let rng = v.rng.as_ref().map(|value| rng(b, value));
    let sampling_state = v
        .sampling_state
        .as_ref()
        .map(|value| sampling_state(b, value));
    let input_token_ids = Some(b.create_vector(&v.input_token_ids));
    let input_image = v.input_image.as_deref().map(|image| b.create_string(image));
    let kv_input = v.kv_input.as_ref().map(|value| buffer_id(b, value));
    let kv_output = v.kv_output.as_ref().map(|value| buffer_id(b, value));
    let consumer_slots = Some(b.create_vector(&v.consumer_slots));

    fbs::Call::create(
        b,
        &fbs::CallArgs {
            request_key,
            call_id: Some(&call_id),
            coordinates,
            component,
            code: Some(&code),
            max_tokens,
            max_kv_pages,
            max_latent_bytes,
            max_completion_bytes,
            max_transfer_bytes,
            inputs,
            outputs,
            token_input,
            token_output,
            vision_input,
            latent_feature_input,
            encoder_output,
            latent_input,
            latent_output,
            image_input,
            image_output,
            completion_output,
            transition_output,
            predicate,
            rng,
            sampling_state,
            input_token_ids,
            input_image,
            kv_input,
            kv_output,
            consumer_slots,
        },
    )
}

fn endpoint<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &WorkerEndpoint,
) -> WIPOffset<fbs::WorkerEndpoint<'a>> {
    let worker_id = Some(b.create_string(&v.worker_id));
    let node = Some(b.create_string(&v.node));
    let address_space = Some(b.create_string(&v.address_space));
    let incarnation = Some(b.create_string(&v.incarnation));
    fbs::WorkerEndpoint::create(
        b,
        &fbs::WorkerEndpointArgs {
            worker_id,
            rank: v.rank,
            node,
            address_space,
            incarnation,
        },
    )
}

/// Writes one storage locator. `Locator` is a flat table whose `transport`
/// discriminant selects which coordinate fields are meaningful.
fn locator<'a>(b: &mut FlatBufferBuilder<'a>, v: &Locator) -> WIPOffset<fbs::Locator<'a>> {
    // Transport-independent tensor metadata. `..Default::default()` leaves
    // every other field at its schema default; the match below sets the
    // discriminant and only that transport's fields, which are the only
    // coordinate fields `transfer_locator_from_table` reads.
    let mut args = fbs::LocatorArgs {
        source: Some(endpoint(b, &v.source)),
        nbytes: v.nbytes,
        dtype: Some(b.create_string(&v.dtype)),
        shape: Some(b.create_vector(&v.shape)),
        offset: Some(b.create_vector(&v.offset)),
        device: Some(b.create_string(&v.device)),
        ..Default::default()
    };

    match &v.transport {
        TransferTransport::Local { endpoint, key } => {
            args.transport = fbs::TransferTransportKind::Local;
            args.endpoint = Some(b.create_string(endpoint));
            args.key = *key;
        }
        TransferTransport::PosixShm { endpoint, name } => {
            args.transport = fbs::TransferTransportKind::PosixShm;
            args.endpoint = Some(b.create_string(endpoint));
            args.name = Some(b.create_string(name));
        }
        TransferTransport::Channel { endpoint, payload } => {
            args.transport = fbs::TransferTransportKind::Channel;
            args.endpoint = Some(b.create_string(endpoint));
            args.payload = Some(b.create_vector(payload));
        }
        TransferTransport::CudaVmm {
            endpoint,
            publication_id,
            storage_size_bytes,
            storage_offsets_bytes,
            span_lengths,
            span_counts,
            tensor_stride,
            ready_event_handle,
            allocation_handle,
            acknowledgment_offset,
        } => {
            args.transport = fbs::TransferTransportKind::CudaVmm;
            args.endpoint = Some(b.create_string(endpoint));
            args.publication_id = Some(b.create_string(publication_id));
            args.storage_size_bytes = *storage_size_bytes;
            args.storage_offsets_bytes = Some(b.create_vector(storage_offsets_bytes));
            args.span_lengths = Some(b.create_vector(span_lengths));
            args.span_counts = Some(b.create_vector(span_counts));
            args.tensor_stride = Some(b.create_vector(tensor_stride));
            args.ready_event_handle = Some(b.create_vector(ready_event_handle));
            args.allocation_handle = Some(b.create_vector(allocation_handle));
            args.acknowledgment_offset = *acknowledgment_offset;
        }
    }

    fbs::Locator::create(b, &args)
}

fn tensor_transfer<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &TensorTransfer,
) -> WIPOffset<fbs::TensorTransfer<'a>> {
    let shape = Some(b.create_vector(&v.shape));
    let locations = {
        let items = v
            .locations
            .iter()
            .map(|item| locator(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    fbs::TensorTransfer::create(b, &fbs::TensorTransferArgs { shape, locations })
}

fn kv_transfer<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &KvTransfer,
) -> WIPOffset<fbs::KvTransfer<'a>> {
    let tensors = {
        let items = v
            .tensors
            .iter()
            .map(|item| tensor_transfer(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let source = Some(buffer_id(b, &v.source));
    let destination = Some(b.create_string(&v.destination));
    let base = v.base.as_ref().map(|value| buffer_id(b, value));
    let compute_dtype = Some(b.create_string(&v.compute_dtype));
    fbs::KvTransfer::create(
        b,
        &fbs::KvTransferArgs {
            tensors,
            source,
            destination,
            base,
            base_extent: v.base_extent,
            published_extent: v.published_extent,
            group_id: v.group_id,
            compute_dtype,
            page_size: v.page_size,
        },
    )
}

fn transfer_handle<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &TransferHandle,
) -> WIPOffset<fbs::TransferHandle<'a>> {
    // A FlatBuffers union is a type tag plus an untyped table offset; each arm
    // returns the tag that matches the table it builds.
    let (value_type, value) = match v {
        TransferHandle::Encoder {
            height,
            width,
            payload_kind,
            tensor,
        } => {
            let tensor = Some(tensor_transfer(b, tensor));
            let payload_kind = match payload_kind {
                FeatureKind::Vision => fbs::FeatureKind::Vision,
                FeatureKind::Latent => fbs::FeatureKind::Latent,
            };
            (
                fbs::TransferData::EncoderTransfer,
                fbs::EncoderTransfer::create(
                    b,
                    &fbs::EncoderTransferArgs {
                        height: *height,
                        width: *width,
                        payload_kind,
                        tensor,
                    },
                )
                .as_union_value(),
            )
        }
        TransferHandle::DeviceProduct {
            height,
            width,
            value_range,
            tensor,
        } => {
            let tensor = Some(tensor_transfer(b, tensor));
            let value_range = Some(b.create_string(value_range));
            (
                fbs::TransferData::DeviceProductTransfer,
                fbs::DeviceProductTransfer::create(
                    b,
                    &fbs::DeviceProductTransferArgs {
                        height: *height,
                        width: *width,
                        value_range,
                        tensor,
                    },
                )
                .as_union_value(),
            )
        }
        TransferHandle::Latent {
            height,
            width,
            latent_units,
            step,
            tensor,
        } => {
            let tensor = Some(tensor_transfer(b, tensor));
            (
                fbs::TransferData::LatentTransfer,
                fbs::LatentTransfer::create(
                    b,
                    &fbs::LatentTransferArgs {
                        height: *height,
                        width: *width,
                        latent_units: *latent_units,
                        step: *step,
                        tensor,
                    },
                )
                .as_union_value(),
            )
        }
    };
    fbs::TransferHandle::create(
        b,
        &fbs::TransferHandleArgs {
            value_type,
            value: Some(value),
        },
    )
}

fn publication<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &TensorPublication,
) -> WIPOffset<fbs::TensorPublication<'a>> {
    let product = Some(tensor_ref(b, &v.product));
    let value = Some(transfer_handle(b, &v.value));
    fbs::TensorPublication::create(b, &fbs::TensorPublicationArgs { product, value })
}

fn command<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &BatchCommand,
) -> WIPOffset<fbs::BatchCommandEnvelope<'a>> {
    let (command_type, command) = match v {
        BatchCommand::Start { request } => {
            let request = Some(admission(b, request));
            (
                fbs::BatchCommand::StartCommand,
                fbs::StartCommand::create(b, &fbs::StartCommandArgs { request }).as_union_value(),
            )
        }
        BatchCommand::Finish {
            request_key: key,
            retained_buffers,
        } => {
            let request_key = Some(request_key(b, key));
            let buffers = retained_buffers
                .iter()
                .map(|buffer| buffer_id(b, buffer))
                .collect::<Vec<_>>();
            let retained_buffers = Some(b.create_vector(&buffers));
            (
                fbs::BatchCommand::FinishCommand,
                fbs::FinishCommand::create(
                    b,
                    &fbs::FinishCommandArgs {
                        request_key,
                        retained_buffers,
                    },
                )
                .as_union_value(),
            )
        }
        BatchCommand::Free { buffer } => {
            let buffer = Some(buffer_id(b, buffer));
            (
                fbs::BatchCommand::FreeCommand,
                fbs::FreeCommand::create(b, &fbs::FreeCommandArgs { buffer }).as_union_value(),
            )
        }
    };
    fbs::BatchCommandEnvelope::create(
        b,
        &fbs::BatchCommandEnvelopeArgs {
            command_type,
            command: Some(command),
        },
    )
}

/// Writes a submitted batch. The schema has no `ForwardBatch` table: its
/// aligned columns become top-level `Batch` vectors, with
/// `ForwardBatch::call_indices` written as `forward_call_indices`.
fn batch<'a>(b: &mut FlatBufferBuilder<'a>, v: &Batch) -> WIPOffset<fbs::Batch<'a>> {
    let calls = {
        let items = v.calls.iter().map(|item| call(b, item)).collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let block_tables = {
        let items = v
            .block_tables
            .iter()
            .map(|item| block_table(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let new_cache_pages = {
        let items = v
            .new_cache_pages
            .iter()
            .map(|item| cache_pages(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let forward_call_indices = Some(b.create_vector(&v.forward.call_indices));
    let request_pool_indices = Some(b.create_vector(&v.forward.request_pool_indices));
    let seq_lens = Some(b.create_vector(&v.forward.seq_lens));
    let query_lens = Some(b.create_vector(&v.forward.query_lens));
    let write_kv = Some(b.create_vector(&v.forward.write_kv));
    let latent_params = {
        let items = v
            .latent_params
            .iter()
            .map(|item| latent_params(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let decode_ranges = {
        let items = v
            .decode_ranges
            .iter()
            .map(|item| decode_range(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let buffer_allocations = {
        let items = v
            .buffer_allocations
            .iter()
            .map(|item| buffer_allocation(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let commands = {
        let items = v
            .commands
            .iter()
            .map(|item| command(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let input_products = {
        let items = v
            .input_products
            .iter()
            .map(|item| publication(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let kv_inputs = {
        let items = v
            .kv_inputs
            .iter()
            .map(|item| kv_transfer(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    fbs::Batch::create(
        b,
        &fbs::BatchArgs {
            batch_id: v.batch_id,
            collective_seq: v.collective_seq,
            calls,
            block_tables,
            new_cache_pages,
            forward_call_indices,
            request_pool_indices,
            seq_lens,
            query_lens,
            write_kv,
            latent_params,
            decode_ranges,
            buffer_allocations,
            commands,
            input_products,
            kv_inputs,
        },
    )
}

fn timing<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &TimingCounters,
) -> WIPOffset<fbs::TimingCounters<'a>> {
    fbs::TimingCounters::create(
        b,
        &fbs::TimingCountersArgs {
            queued_us: v.queued_us,
            device_us: v.device_us,
            copy_us: v.copy_us,
            host_us: v.host_us,
        },
    )
}

fn finish_flags<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &FinishFlags,
) -> WIPOffset<fbs::FinishFlags<'a>> {
    let eos = v.eos;
    let length = v.length;
    let stop = v.stop;
    fbs::FinishFlags::create(b, &fbs::FinishFlagsArgs { eos, length, stop })
}

fn token_logprob(v: &TokenLogprob) -> fbs::TokenLogprob {
    fbs::TokenLogprob::new(v.token_id, v.logprob, v.rank)
}

fn position_logprobs<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &[TokenLogprob],
) -> WIPOffset<fbs::PositionLogprobs<'a>> {
    let values = v.iter().map(token_logprob).collect::<Vec<_>>();
    let entries = Some(b.create_vector(&values));
    fbs::PositionLogprobs::create(b, &fbs::PositionLogprobsArgs { entries })
}

fn media_output<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &MediaOutput,
) -> WIPOffset<fbs::MediaOutput<'a>> {
    // `ArtifactHandle` has a single variant, so this pattern is irrefutable
    // and `handle_type` is fixed. On the wire, a new variant also needs a
    // member of the schema's `ArtifactHandle` union, a match here, and a
    // decoder arm in `completion_record_from_table`.
    let ArtifactHandle::PosixShm { name } = &v.handle;
    let name = Some(b.create_string(name));
    let handle = Some(
        fbs::PosixShmArtifact::create(b, &fbs::PosixShmArtifactArgs { name }).as_union_value(),
    );
    fbs::MediaOutput::create(
        b,
        &fbs::MediaOutputArgs {
            handle_type: fbs::ArtifactHandle::PosixShmArtifact,
            handle,
            bytes: v.bytes,
        },
    )
}

fn completion<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &RequestOutput,
) -> WIPOffset<fbs::RequestOutput<'a>> {
    let request_key = Some(request_key(b, &v.request_key));
    let call_id = fbs::CallId::new(v.call_id.batch_id, v.call_id.request_index);
    let status = call_status_to_fb(v.status);
    let product_generations = Some(b.create_vector(&v.product_generations));
    let error_code = v.error_code.map(error_code_to_fb);
    let timing_counters = Some(timing(b, &v.timing_counters));
    let code = computation_to_fb(v.code);
    let committed_tokens = Some(b.create_vector(&v.committed_tokens));
    let top_logprobs = {
        let items = v.top_logprobs.iter().map(token_logprob).collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let prompt_logprobs = {
        let items = v
            .prompt_logprobs
            .iter()
            .map(|item| position_logprobs(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let finish_flags = Some(finish_flags(b, &v.finish_flags));
    let media_output = v.media_output.as_ref().map(|value| media_output(b, value));
    let kv_output = v.kv_output.as_ref().map(|value| kv_transfer(b, value));
    fbs::RequestOutput::create(
        b,
        &fbs::RequestOutputArgs {
            request_key,
            call_id: Some(&call_id),
            status,
            product_generations,
            error_code,
            timing_counters,
            code: Some(&code),
            position: v.position,
            kv_visible_len: v.kv_visible_len,
            kv_computed_len: v.kv_computed_len,
            num_completed_steps: v.num_completed_steps,
            committed_tokens,
            sampled_logprob: v.sampled_logprob,
            top_logprobs,
            prompt_logprobs,
            finish_flags,
            media_output,
            kv_output,
        },
    )
}

fn counters<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &BTreeMap<String, u64>,
) -> WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fbs::StringU64Pair<'a>>>> {
    let pairs = v
        .iter()
        .map(|(key, &value)| {
            let key = Some(b.create_string(key));
            fbs::StringU64Pair::create(b, &fbs::StringU64PairArgs { key, value })
        })
        .collect::<Vec<_>>();
    b.create_vector(&pairs)
}

fn forward_stats<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &ForwardStats,
) -> WIPOffset<fbs::ForwardStats<'a>> {
    let mode_counts = Some(counters(b, &v.mode_counts));
    let mode_tokens = Some(counters(b, &v.mode_tokens));
    let mode_us = Some(counters(b, &v.mode_us));
    let attention_backend_counts = Some(counters(b, &v.attention_backend_counts));
    let cuda_graph_runtime_mode_counts = Some(counters(b, &v.cuda_graph_runtime_mode_counts));
    let spec_verify_path_counts = Some(counters(b, &v.spec_verify_path_counts));
    let component_us = Some(counters(b, &v.component_us));
    fbs::ForwardStats::create(
        b,
        &fbs::ForwardStatsArgs {
            mode_counts,
            mode_tokens,
            mode_us,
            attention_launches: v.attention_launches,
            attention_us: v.attention_us,
            attention_backend_counts,
            cuda_graph_captures: v.cuda_graph_captures,
            cuda_graph_replays: v.cuda_graph_replays,
            cuda_graph_misses: v.cuda_graph_misses,
            cuda_graph_fallbacks: v.cuda_graph_fallbacks,
            cuda_graph_unpadded_tokens: v.cuda_graph_unpadded_tokens,
            cuda_graph_padded_tokens: v.cuda_graph_padded_tokens,
            cuda_graph_runtime_mode_counts,
            text_decode_token_relay_hits: v.text_decode_token_relay_hits,
            text_decode_token_relay_misses: v.text_decode_token_relay_misses,
            text_decode_position_relay_hits: v.text_decode_position_relay_hits,
            text_decode_position_relay_misses: v.text_decode_position_relay_misses,
            flashinfer_decode_plan_calls: v.flashinfer_decode_plan_calls,
            flashinfer_decode_plan_reuses: v.flashinfer_decode_plan_reuses,
            flashinfer_decode_plan_rows: v.flashinfer_decode_plan_rows,
            flashinfer_decode_plan_indices: v.flashinfer_decode_plan_indices,
            flashinfer_decode_graph_plan_calls: v.flashinfer_decode_graph_plan_calls,
            flashinfer_decode_graph_plan_reuses: v.flashinfer_decode_graph_plan_reuses,
            spec_verify_rows: v.spec_verify_rows,
            spec_verify_draft_tokens: v.spec_verify_draft_tokens,
            spec_verify_accepted_tokens: v.spec_verify_accepted_tokens,
            spec_verify_rejected_tokens: v.spec_verify_rejected_tokens,
            spec_verify_committed_tokens: v.spec_verify_committed_tokens,
            spec_verify_path_counts,
            component_us,
        },
    )
}

fn batch_output<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &BatchOutput,
) -> WIPOffset<fbs::BatchOutput<'a>> {
    let completions = {
        let items = v
            .completions
            .iter()
            .map(|item| completion(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let products = {
        let items = v
            .products
            .iter()
            .map(|item| publication(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let forward_stats = v
        .forward_stats
        .as_ref()
        .map(|value| forward_stats(b, value));
    fbs::BatchOutput::create(
        b,
        &fbs::BatchOutputArgs {
            batch_id: v.batch_id,
            completions,
            products,
            worker_exec_us: v.worker_exec_us,
            forward_stats,
        },
    )
}

fn kv_group<'a>(b: &mut FlatBufferBuilder<'a>, v: &KvCacheGroup) -> WIPOffset<fbs::KvGroup<'a>> {
    // `window` and `sink` are meaningful only for sliding-window groups; a
    // full-attention group writes zeros, which `kv_group_from_table` ignores.
    let (kind, window, sink) = match v.kind {
        KvGroupKind::Full => (fbs::KvGroupKind::Full, 0, 0),
        KvGroupKind::SlidingWindow { window, sink } => {
            (fbs::KvGroupKind::SlidingWindow, window, sink)
        }
    };
    fbs::KvGroup::create(
        b,
        &fbs::KvGroupArgs {
            num_blocks: v.num_blocks,
            kind,
            window,
            sink,
        },
    )
}

fn kv_cache<'a>(b: &mut FlatBufferBuilder<'a>, v: &KvCacheInfo) -> WIPOffset<fbs::KVCacheInfo<'a>> {
    let groups = {
        let items = v
            .groups
            .iter()
            .map(|item| kv_group(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let dtype = Some(b.create_string(v.dtype.as_str()));
    fbs::KVCacheInfo::create(
        b,
        &fbs::KVCacheInfoArgs {
            block_size: v.block_size,
            num_blocks: v.num_blocks,
            num_layers: v.num_layers,
            total_layers: v.total_layers,
            layer_offset: v.layer_offset,
            num_kv_heads: v.num_kv_heads,
            total_kv_heads: v.total_kv_heads,
            kv_head_offset: v.kv_head_offset,
            head_dim: v.head_dim,
            bytes_per_token: v.bytes_per_token,
            groups,
            dtype,
        },
    )
}

/// Writes a component's parallel degrees.
///
/// The `as u32` casts cannot truncate for a validated `WorkerInfo`:
/// `WorkerInfo::validate` requires positive degrees whose product fits `u32`.
fn parallel<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &uniserve_core::ParallelConfig,
) -> WIPOffset<fbs::ParallelConfig<'a>> {
    use uniserve_core::SequenceParallel;
    let (sequence_parallel_type, sequence_parallel) = match v.sequence_parallel {
        SequenceParallel::Local => (
            fbs::SequenceParallel::LocalSequence,
            fbs::LocalSequence::create(b, &fbs::LocalSequenceArgs {}).as_union_value(),
        ),
        SequenceParallel::Ulysses { ulysses_degree } => (
            fbs::SequenceParallel::UlyssesSequence,
            fbs::UlyssesSequence::create(
                b,
                &fbs::UlyssesSequenceArgs {
                    ulysses_degree: ulysses_degree as u32,
                },
            )
            .as_union_value(),
        ),
        SequenceParallel::Allgather { allgather_degree } => (
            fbs::SequenceParallel::GatherSequence,
            fbs::GatherSequence::create(
                b,
                &fbs::GatherSequenceArgs {
                    allgather_degree: allgather_degree as u32,
                },
            )
            .as_union_value(),
        ),
        SequenceParallel::Hybrid {
            ulysses_degree,
            allgather_degree,
        } => (
            fbs::SequenceParallel::HybridSequence,
            fbs::HybridSequence::create(
                b,
                &fbs::HybridSequenceArgs {
                    ulysses_degree: ulysses_degree as u32,
                    allgather_degree: allgather_degree as u32,
                },
            )
            .as_union_value(),
        ),
    };
    fbs::ParallelConfig::create(
        b,
        &fbs::ParallelConfigArgs {
            tensor_parallel_size: v.tensor_parallel_size as u32,
            pipeline_parallel_size: v.pipeline_parallel_size as u32,
            sequence_parallel_type,
            sequence_parallel: Some(sequence_parallel),
        },
    )
}

fn output_info<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &crate::OutputInfo,
) -> WIPOffset<fbs::OutputInfo<'a>> {
    let (extents, dynamic_axis) = shape_bound_to_parts(&v.shape_bound);
    let name = Some(b.create_string(&v.name));
    let dtype = dtype_to_fb(v.dtype);
    let extents = Some(b.create_vector(&extents));
    fbs::OutputInfo::create(
        b,
        &fbs::OutputInfoArgs {
            name,
            dtype,
            extents,
            dynamic_axis,
        },
    )
}

fn component_info<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &crate::ComponentInfo,
) -> WIPOffset<fbs::ComponentInfo<'a>> {
    let name = Some(b.create_string(&v.name));
    let ranks = {
        let values = v
            .config
            .ranks
            .iter()
            .map(|&rank| rank as u64)
            .collect::<Vec<_>>();
        Some(b.create_vector(&values))
    };
    let parallel_config = Some(parallel(b, &v.config.parallel_config));
    // Wire `Local` encodes an absent distribution; `distribution_from_fb`
    // maps it back to `None`.
    let distribution = match v.config.distribution {
        None => fbs::ComponentDistribution::Local,
        Some(uniserve_core::ComponentDistribution::TemporalUnits) => {
            fbs::ComponentDistribution::TemporalUnits
        }
    };
    // `WorkerInfo::validate` bounds `units_per_rank` to `u32`.
    let units_per_rank = v.config.units_per_rank as u32;
    let outputs = {
        let items = v
            .outputs
            .iter()
            .map(|item| output_info(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    fbs::ComponentInfo::create(
        b,
        &fbs::ComponentInfoArgs {
            name,
            ranks,
            parallel_config,
            distribution,
            units_per_rank,
            outputs,
        },
    )
}

fn info<'a>(b: &mut FlatBufferBuilder<'a>, v: &WorkerInfo) -> WIPOffset<fbs::WorkerInfo<'a>> {
    let model_name = Some(b.create_string(&v.model_name));
    let endpoint = Some(endpoint(b, &v.endpoint));
    let device = Some(b.create_string(&v.device));
    let transfer_backends = {
        let items = v
            .transfer_backends
            .iter()
            .map(|s| b.create_string(s))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let model_dtype = Some(b.create_string(&v.model_dtype));
    let attention_backend = Some(b.create_string(&v.attention_backend));
    let weight_formats = {
        let items = v
            .weight_formats
            .iter()
            .map(|s| b.create_string(s))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let activation_formats = {
        let items = v
            .activation_formats
            .iter()
            .map(|s| b.create_string(s))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let components = {
        let items = v
            .components
            .iter()
            .map(|item| component_info(b, item))
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let supported_calls = {
        let items = v
            .supported_calls
            .iter()
            .copied()
            .map(computation_to_fb)
            .collect::<Vec<_>>();
        Some(b.create_vector(&items))
    };
    let kv_cache = v.kv_cache.as_ref().map(|value| kv_cache(b, value));
    // The `MediaCall` to component-name map is written as a vector of pairs
    // in key order; `info_from_table` rejects a repeated call.
    let media_components = {
        let values = v
            .media_components
            .iter()
            .map(|(call, component)| {
                let component = Some(b.create_string(component));
                fbs::MediaComponent::create(
                    b,
                    &fbs::MediaComponentArgs {
                        call: media_call_to_fb(*call),
                        component,
                    },
                )
            })
            .collect::<Vec<_>>();
        Some(b.create_vector(&values))
    };
    let checkpoint_identity = Some(b.create_string(&v.checkpoint_identity));
    let video_denoiser = v.video_denoiser.as_ref().map(|value| {
        let tasks = value
            .tasks
            .iter()
            .map(|task| b.create_string(task))
            .collect::<Vec<_>>();
        let tasks = Some(b.create_vector(&tasks));
        let canvases = value
            .canvases
            .iter()
            .map(|canvas| fbs::Canvas::new(canvas.width, canvas.height))
            .collect::<Vec<_>>();
        let canvases = Some(b.create_vector(&canvases));
        fbs::VideoDenoiserInfo::create(
            b,
            &fbs::VideoDenoiserInfoArgs {
                tasks,
                schedule_points: value.schedule_points,
                video_shift: value.video_shift,
                audio_shift: value.audio_shift,
                canvases,
                max_sequence_rows: value.max_sequence_rows,
            },
        )
    });
    fbs::WorkerInfo::create(
        b,
        &fbs::WorkerInfoArgs {
            model_name,
            endpoint,
            device,
            transfer_backends,
            fabric_handles: v.fabric_handles,
            world_size: v.world_size,
            model_dtype,
            attention_backend,
            weight_formats,
            activation_formats,
            components,
            supported_calls,
            queue_depth: v.queue_depth,
            max_batch_calls: v.max_batch_calls,
            max_batch_tokens: v.max_batch_tokens,
            request_slots: v.request_slots,
            kv_cache,
            latent_page_units: v.latent_page_units,
            latent_pages: v.latent_pages,
            buffer_pool_bytes: v.buffer_pool_bytes,
            encoder_cache_entries: v.encoder_cache_entries,
            encoder_entry_bytes: v.encoder_entry_bytes,
            max_unresolved_calls: v.max_unresolved_calls,
            media_components,
            num_inference_steps: v.num_inference_steps,
            host_lane_capacity: v.host_lane_capacity,
            checkpoint_identity,
            video_denoiser,
        },
    )
}

fn error_call<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &ErrorCallIdentity,
) -> WIPOffset<fbs::ErrorCallIdentity<'a>> {
    let request_key = Some(request_key(b, &v.request_key));
    let call_id = fbs::CallId::new(v.call_id.batch_id, v.call_id.request_index);
    fbs::ErrorCallIdentity::create(
        b,
        &fbs::ErrorCallIdentityArgs {
            request_key,
            call_id: Some(&call_id),
        },
    )
}

/// Writes the root `WorkerRequest` table. Only `Submit` carries a batch;
/// `request_from_table` rejects a batch on any other kind.
pub(super) fn request<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &WorkerRequest,
) -> WIPOffset<fbs::WorkerRequest<'a>> {
    let batch = v.batch().map(|value| batch(b, value));
    fbs::WorkerRequest::create(
        b,
        &fbs::WorkerRequestArgs {
            kind: request_kind_to_fb(v.kind()),
            message_id: v.message_id(),
            batch,
        },
    )
}

/// Writes the root `WorkerResponse` table.
///
/// Besides `kind` and `message_id`, each kind writes only its own fields:
/// `Info` and `Result` their payload, `Ok` nothing, and `Error` the error
/// metadata. `response_from_table` rejects any other combination.
pub(super) fn response<'a>(
    b: &mut FlatBufferBuilder<'a>,
    v: &WorkerResponse,
) -> WIPOffset<fbs::WorkerResponse<'a>> {
    let mut args = fbs::WorkerResponseArgs {
        kind: response_kind_to_fb(v.kind()),
        message_id: v.message_id(),
        ..Default::default()
    };
    match v {
        WorkerResponse::Info { info: value, .. } => args.info = Some(info(b, value)),
        WorkerResponse::Result { result, .. } => args.result = Some(batch_output(b, result)),
        WorkerResponse::Ok { .. } => {}
        WorkerResponse::Error { error, .. } => {
            args.message = Some(b.create_string(&error.message));
            args.code = error.code.as_deref().map(|s| b.create_string(s));
            // `fatal` is an optional scalar (`bool = null`): `Some` writes it
            // even when false, and `response_from_table` requires it on error
            // responses.
            args.fatal = Some(error.fatal);
            args.phase = error.phase.as_deref().map(|s| b.create_string(s));
            args.route = error.route.as_deref().map(|s| b.create_string(s));
            let calls = error
                .calls
                .iter()
                .map(|call| error_call(b, call))
                .collect::<Vec<_>>();
            args.calls = Some(b.create_vector(&calls));
        }
    }
    fbs::WorkerResponse::create(b, &args)
}
