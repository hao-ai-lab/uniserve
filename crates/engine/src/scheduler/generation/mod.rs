//! Generation lifecycle state and worker-call planning.
//!
//! A request advances through context ingestion, understanding decode, image
//! generation, optional feedback, and terminal publication. Every planned
//! output is named by request epoch, producer call, point, and generation.

use std::collections::HashSet;
use uniserve_worker_ipc::{CallCoordinates, ForwardMode, MediaCall, TransferMode};

use uniserve_core::{GenerationRequest, ImageIngestStep, RequestId, SamplingParams};
use uniserve_worker_ipc::{
    Bounds, BufferId, Call, CallId, CallKind, CallStatus, DType, DimBound, DrawLayout, RequestKey,
    RequestOutput, Rng, SamplingState, ShapeBound, TensorRef,
};

use crate::scheduler::image_artifact::png_artifact_dims_b64;

/// Builds an unstamped product reference for a planned call output. The
/// owning `request_key` and `producer_call_id` are placeholder until
/// [`register_call`] stamps the real identity. The shape
/// bound is empty (it carries identity, not a device geometry).
fn output_tensor(output_index: u16, dtype: DType) -> TensorRef {
    TensorRef {
        request_key: RequestKey::new(0, RequestId(0), 0),
        producer_call_id: CallId::new(0, 0),
        output_index,
        generation: 0,
        dtype,
        shape_bound: ShapeBound::default(),
    }
}

/// Builds a product with an explicit byte bound.
fn bounded_tensor(output_index: u16, dtype: DType, shape_bound: ShapeBound) -> TensorRef {
    let mut product = output_tensor(output_index, dtype);
    product.shape_bound = shape_bound;
    product
}

/// Computes a bounded element count for a dynamic dimension.
fn dynamic_element_bound(bytes: u64, dtype: DType) -> Result<ShapeBound, PlanningError> {
    let elements = bytes.div_ceil(dtype.element_bytes());
    let max = u32::try_from(elements).map_err(|_| PlanningError::ProductBoundTooLarge { bytes })?;
    if max == 0 {
        return Err(PlanningError::MissingProductBound);
    }
    Ok(ShapeBound {
        dims: vec![DimBound::Device { max }],
    })
}

/// Computes the encoded size bound for a PNG artifact.
fn png_base64_bound(width: u32, height: u32) -> Result<u64, PlanningError> {
    let raw = u64::from(height)
        .checked_mul(u64::from(width).saturating_mul(3).saturating_add(1))
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let png = raw
        .checked_mul(2)
        .and_then(|value| value.checked_add(1 << 20))
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    Ok(png.div_ceil(3).saturating_mul(4))
}

const RANKED_LOGPROB_BYTES: u64 = 12;

/// Computes the maximum serialized log-probability payload required by a request.
fn logprob_result_bytes(
    sampling: &SamplingParams,
    prompt_positions: u32,
) -> Result<Option<u64>, PlanningError> {
    let generated = sampling.generated_logprobs_requested();
    let prompt = sampling.prompt_logprobs_requested();
    if !generated && !prompt {
        return Ok(None);
    }
    let requested_ids = sampling
        .logprob_token_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len() as u64;
    let generated_entries = generated.then_some(
        1_u64
            .checked_add(u64::from(sampling.n_logprobs))
            .and_then(|value| value.checked_add(requested_ids))
            .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?,
    );
    let prompt_entries = prompt.then_some(
        1_u64
            .checked_add(u64::from(sampling.n_prompt_logprobs))
            .and_then(|value| value.checked_add(requested_ids))
            .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?,
    );
    let generated_bytes = generated_entries
        .unwrap_or(0)
        .checked_mul(RANKED_LOGPROB_BYTES)
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let per_prompt_bytes = 4_u64
        .checked_add(
            prompt_entries
                .unwrap_or(0)
                .checked_mul(RANKED_LOGPROB_BYTES)
                .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?,
        )
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let prompt_bytes = u64::from(prompt_positions)
        .checked_mul(per_prompt_bytes)
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let bytes = generated_bytes
        .checked_add(if generated { 4 } else { 0 })
        .and_then(|value| value.checked_add(prompt_bytes))
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    Ok(Some(bytes))
}

/// Lifecycle phase for a canonical generation request.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GenerationPhase {
    /// Encode staged input images before continuing text prefill.
    Encode,
    IngestState,
    Prefill,
    DecodeUnd,
    CloseKv,
    PublishKv,
    PrepareGen,
    DenoiseGen,
    CommitGen,
    FeedbackEncode,
    FeedbackState,
}

/// Image extension consumes an encoder feature rather than token inputs.
pub(super) fn consumes_image_features(call: &Call) -> bool {
    call.code == CallKind::Forward(ForwardMode::Prefill)
        && (call.vision_input.is_some() || call.latent_feature_input.is_some())
}

/// Feedback encoders and KV writes produce a predicate for their device successor.
/// Input-image encoding has no such successor until its host result is accepted.
pub(super) fn is_feedback_computation(call: &Call) -> bool {
    (matches!(
        call.code,
        CallKind::Media(MediaCall::VisionEncoding) | CallKind::Media(MediaCall::LatentEncoding)
    ) || consumes_image_features(call))
        && call.completion_output.is_some()
}

/// Prompt extension samples a token and has no image-feature input.
pub(super) fn is_prompt_extend(call: &Call) -> bool {
    call.code == CallKind::Forward(ForwardMode::Prefill)
        && !consumes_image_features(call)
        && call.token_output.is_some()
}

impl super::RequestState {
    /// Applies one validated computation in request order. Pending identities are
    /// consumed once before this update; inactive successors never reach it.
    pub(crate) fn process_generation_result(
        &mut self,
        call: &Call,
        record: &RequestOutput,
    ) -> Result<(), GenerationResultError> {
        if call.call_id.batch_id == 0 {
            return Err(GenerationResultError::MissingCallId);
        }
        if record.status == CallStatus::Predicated {
            return Ok(());
        }
        if matches!(call.code, CallKind::Forward(_)) {
            // A forward reports the extent it initialized, which a verifier
            // leaves above the prefix it accepted.
            self.kv_computed_len = record.kv_computed_len;
        }
        match call.code {
            CallKind::Forward(ForwardMode::Prefill) if is_prompt_extend(call) => {
                let count = call.input_token_ids.len().min(u32::MAX as usize) as u32;
                self.num_computed_prompt_tokens =
                    self.num_computed_prompt_tokens.saturating_add(count);
                self.logical_position = self.logical_position.saturating_add(count);
                self.kv_visible_len = self.kv_visible_len.saturating_add(count);
            }
            CallKind::Forward(ForwardMode::Prefill) if consumes_image_features(call) => {
                self.kv_visible_len = record.kv_visible_len;
                if is_feedback_computation(call) {
                    self.feedback_encoder_index = self.feedback_encoder_index.saturating_add(1);
                    self.feedback_features = None;
                    if self.feedback_encoder_index
                        == self.req.image_generation.feedback_encoders.len()
                    {
                        self.logical_position = self
                            .logical_position
                            .saturating_add(self.req.image_generation.num_feedback_positions);
                        self.feedback_encoder_index = 0;
                        self.feedback_source = None;
                        self.feedback_image_b64 = None;
                        self.phase = GenerationPhase::DecodeUnd;
                    } else {
                        self.phase = GenerationPhase::FeedbackEncode;
                    }
                    self.replayable = false;
                } else {
                    let image = &self.req.multimodal_inputs.images[self.num_ingested_images];
                    self.image_encoder_index = self.image_encoder_index.saturating_add(1);
                    self.input_image_features = None;
                    if self.image_encoder_index == image.encoders.len() {
                        self.logical_position =
                            self.logical_position.saturating_add(image.num_positions);
                        self.num_ingested_images = self.num_ingested_images.saturating_add(1);
                        self.image_encoder_index = 0;
                        self.phase = GenerationPhase::Prefill;
                    } else {
                        self.phase = GenerationPhase::Encode;
                    }
                }
            }
            CallKind::Forward(ForwardMode::Prefill) => {
                self.kv_visible_len = record.kv_visible_len;
                self.phase = GenerationPhase::PublishKv;
                self.replayable = false;
            }
            CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                let count = record.committed_tokens.len().max(1).min(u32::MAX as usize) as u32;
                self.logical_position = self.logical_position.saturating_add(count);
                self.kv_visible_len = self.kv_visible_len.saturating_add(count);
            }
            CallKind::Transfer(TransferMode::KvPublish) => {
                self.image_conditioning = call.kv_output;
                self.phase = GenerationPhase::PrepareGen;
            }
            CallKind::Media(MediaCall::LatentPreparation)
            | CallKind::Media(MediaCall::Denoising) => {
                self.image_latent = call.latent_output.clone();
                if self.image_latent.is_none() {
                    return Err(GenerationResultError::MissingLatentProduct);
                }
                if call.code == CallKind::Media(MediaCall::LatentPreparation) {
                    self.num_completed_denoise_steps = 0;
                    self.phase = GenerationPhase::DenoiseGen;
                } else {
                    self.num_completed_denoise_steps =
                        record.num_completed_steps.min(u32::from(u16::MAX)) as u16;
                    self.replayable = false;
                }
            }
            CallKind::Media(MediaCall::ImageDecoding) => {
                // Decoding the artifact releases the trajectory, so the request
                // leaves the flow and its next call enters at step zero.
                self.num_completed_denoise_steps = 0;
                self.feedback_source = call.image_output.clone();
                self.feedback_encoder_index = 0;
                self.feedback_features = None;
                self.phase = GenerationPhase::FeedbackEncode;
                self.replayable = false;
            }
            CallKind::Media(MediaCall::VisionEncoding)
            | CallKind::Media(MediaCall::LatentEncoding)
                if is_feedback_computation(call) =>
            {
                self.feedback_features = call.encoder_output.clone();
                self.phase = GenerationPhase::FeedbackState;
                self.replayable = false;
            }
            _ => {}
        }
        Ok(())
    }
}

/// Builds the immutable auxiliary feature product produced by an encode call.
fn encoder_output(
    step: ImageIngestStep,
    limits: &uniserve_core::GenerationLimits,
) -> Result<TensorRef, PlanningError> {
    let bytes = match step {
        ImageIngestStep::VaeEncode => limits.max_latent_feature_bytes,
        ImageIngestStep::VitEncode => limits.max_vision_feature_bytes,
    };
    Ok(bounded_tensor(
        0,
        DType::BF16,
        dynamic_element_bound(bytes, DType::BF16)?,
    ))
}

/// Builds one immutable image-latent generation. The output remains addressed by its
/// exact call identity and logical generation until the scheduler releases
/// it after all registered readers have fenced.
fn latent_output(output_index: u16, bytes: u64, dtype: DType) -> Result<TensorRef, PlanningError> {
    Ok(bounded_tensor(
        output_index,
        dtype,
        dynamic_element_bound(bytes, dtype)?,
    ))
}

/// Builds the actual computation before assigning storage and execution identities.
fn computation(request: &GenerationRequest, code: CallKind) -> Call {
    Call {
        consumer_slots: Vec::new(),
        token_input: None,

        request_key: RequestKey::new(0, request.request_id, 0),
        call_id: CallId::new(0, 0),
        coordinates: CallCoordinates::default(),
        component: "model".into(),
        code,
        bounds: Bounds {
            max_tokens: 1,
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        token_output: None,
        vision_input: None,
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,
        predicate: None,
        rng: None,
        input_image: None,
        kv_input: None,
        kv_output: None,
        input_token_ids: Vec::new(),
        sampling_state: None,
    }
}

/// Plans a prompt slice using its physical KV prefix and logical sampling end.
pub(super) fn plan_prompt(
    request: &GenerationRequest,
    start: u32,
    end: u32,
    sampling_state: Option<SamplingState>,
) -> Result<Call, PlanningError> {
    if start >= end || end as usize > request.prompt_token_ids.len() {
        return Err(PlanningError::InvalidPromptRange {
            start,
            end,
            prompt_tokens: request.prompt_token_ids.len(),
        });
    }
    let mut call = computation(request, CallKind::Forward(ForwardMode::Prefill));
    call.bounds.max_tokens = end.saturating_sub(start);
    call.input_token_ids = request.prompt_token_ids[start as usize..end as usize].to_vec();
    call.token_output = Some(output_tensor(0, DType::I64));
    call.transition_output = sampling_state
        .as_ref()
        .is_some_and(|state| !state.transition_token_ids.is_empty())
        .then(|| output_tensor(3, DType::U8));
    call.sampling_state = sampling_state;
    call.rng = Some(Rng {
        seed: request.sampling.seed.unwrap_or(0),
        semantic_index_base: u64::from(end),
        draw_layout: DrawLayout::TargetSampling,
    });
    let prompt_positions = if request.sampling.prompt_logprobs_requested() {
        end.saturating_sub(start)
            .saturating_sub(u32::from(start == 0))
    } else {
        0
    };
    finish_plan(request, call, prompt_positions)
}

/// Plans a sampled continuation; relay inputs leave host token values empty.
pub(super) fn plan_decode(
    request: &GenerationRequest,
    logical_position: u32,
    spec_token_ids: Option<Vec<u32>>,
    input_token: u32,
    relay_input: bool,
    sampling_state: Option<SamplingState>,
) -> Result<Call, PlanningError> {
    let draft_count = spec_token_ids.as_ref().map_or(0, Vec::len);
    let code = if draft_count == 0 {
        CallKind::Forward(ForwardMode::Decode)
    } else {
        CallKind::Forward(ForwardMode::Verify)
    };
    let mut call = computation(request, code);
    call.bounds.max_tokens = (1 + draft_count).min(u32::MAX as usize) as u32;
    call.input_token_ids = match spec_token_ids {
        Some(drafts) if !drafts.is_empty() => drafts,
        _ if relay_input => Vec::new(),
        _ => vec![input_token],
    };
    call.token_output = Some(output_tensor(0, DType::I64));
    call.transition_output = sampling_state
        .as_ref()
        .is_some_and(|state| !state.transition_token_ids.is_empty())
        .then(|| output_tensor(3, DType::U8));
    call.sampling_state = sampling_state;
    call.rng = Some(Rng {
        seed: request.sampling.seed.unwrap_or(0),
        semantic_index_base: u64::from(logical_position.saturating_add(1)),
        draw_layout: DrawLayout::TargetSampling,
    });
    finish_plan(request, call, 0)
}

/// Plans a feature write, retaining an exact token count only when the encoder supplies one.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_image_extend(
    limits: &uniserve_core::GenerationLimits,
    request: &GenerationRequest,
    encoder: uniserve_core::ImageEncoderInput,
    feature: TensorRef,
    feedback: bool,
    sample_continuation: bool,
    sampling_state: Option<SamplingState>,
    sampling_index: Option<u64>,
) -> Result<Call, PlanningError> {
    let capacity = encoder.kv_token_capacity(limits)?;
    let mut call = computation(request, CallKind::Forward(ForwardMode::Prefill));
    call.bounds.max_tokens = capacity;
    match encoder.encoder {
        ImageIngestStep::VitEncode => call.vision_input = Some(feature),
        ImageIngestStep::VaeEncode => call.latent_feature_input = Some(feature),
    }
    call.completion_output = feedback.then(|| output_tensor(0, DType::U8));
    call.token_output = sample_continuation.then(|| output_tensor(1, DType::I64));
    call.transition_output = (sample_continuation
        && sampling_state
            .as_ref()
            .is_some_and(|state| !state.transition_token_ids.is_empty()))
    .then(|| output_tensor(3, DType::U8));
    call.sampling_state = sampling_state;
    if call.token_output.is_some() {
        call.rng = sampling_index.map(|semantic_index_base| Rng {
            seed: request.sampling.seed.unwrap_or(0),
            semantic_index_base,
            draw_layout: DrawLayout::TargetSampling,
        });
    }
    finish_plan(request, call, 0)
}

/// Encodes host image bytes or an existing device image into a feature tensor.
pub(super) fn plan_encode(
    limits: &uniserve_core::GenerationLimits,
    request: &GenerationRequest,
    step: ImageIngestStep,
    image_base64: String,
    source: Option<TensorRef>,
    feedback: bool,
) -> Result<Call, PlanningError> {
    if feedback {
        if !request.feeds_back_images() || request.image_generation.feedback_source.is_none() {
            return Err(PlanningError::FeedbackDisabled);
        }
    } else if source.is_none() && image_base64.is_empty() {
        return Err(PlanningError::MissingImageInput);
    }
    let code = match step {
        ImageIngestStep::VaeEncode => CallKind::Media(MediaCall::LatentEncoding),
        ImageIngestStep::VitEncode => CallKind::Media(MediaCall::VisionEncoding),
    };
    let mut call = computation(request, code);
    if source.is_none() && !image_base64.is_empty() {
        call.input_image = Some(image_base64.into());
    }
    call.image_input = source;
    call.encoder_output = Some(encoder_output(step, limits)?);
    if feedback {
        call.completion_output = Some(output_tensor(1, DType::U8));
    }
    finish_plan(request, call, 0)
}

/// Writes the closing token into KV before publication, without sampling another token.
pub(super) fn plan_close_kv(
    request: &GenerationRequest,
    token: u32,
    relay_input: bool,
) -> Result<Call, PlanningError> {
    let mut call = computation(request, CallKind::Forward(ForwardMode::Prefill));
    if !relay_input {
        call.input_token_ids.push(token);
    }
    call.completion_output = Some(output_tensor(0, DType::U8));
    finish_plan(request, call, 0)
}

/// Declares the physically visible KV range as a transferable input to diffusion.
pub(super) fn plan_kv_publish(
    kv_bytes_per_token: u64,
    request: &GenerationRequest,
    physical_kv_len: u32,
) -> Result<Call, PlanningError> {
    if physical_kv_len == 0 || kv_bytes_per_token == 0 {
        return Err(PlanningError::MissingProductBound);
    }
    let mut call = computation(request, CallKind::Transfer(TransferMode::KvPublish));
    call.bounds.max_tokens = 0;
    call.bounds.max_transfer_bytes = kv_bytes_per_token
        .checked_mul(u64::from(physical_kv_len))
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    call.kv_output = Some(BufferId {
        owner: call.request_key,
        producer_call_id: call.call_id,
        output_index: 0,
        generation: 0,
    });
    call.completion_output = Some(output_tensor(1, DType::U8));
    finish_plan(request, call, 0)
}

/// Creates initial noise from the image's deterministic RNG coordinate and conditioning.
pub(super) fn plan_diffusion_prepare(
    limits: &uniserve_core::GenerationLimits,
    latent_dtype: Option<DType>,
    request: &GenerationRequest,
    image_id: u32,
    conditioning: BufferId,
) -> Result<Call, PlanningError> {
    if !request.generates_images() {
        return Err(PlanningError::GenerationBranchDisabled);
    }
    let mut call = computation(request, CallKind::Media(MediaCall::LatentPreparation));
    call.kv_input = Some(conditioning);
    call.latent_output = Some(latent_output(
        0,
        request.image_latent_bytes(limits)?,
        latent_dtype.ok_or(PlanningError::MissingLatentDType)?,
    )?);
    call.completion_output = Some(output_tensor(1, DType::U8));
    call.rng = Some(Rng {
        seed: request.image.seed.unwrap_or(0),
        semantic_index_base: u64::from(image_id),
        draw_layout: DrawLayout::FlowNoise,
    });
    finish_plan(request, call, 0)
}

/// Plans an exact denoising interval against its input latent and conditioning tensors.
pub(super) fn plan_diffusion_step(
    limits: &uniserve_core::GenerationLimits,
    latent_dtype: Option<DType>,
    request: &GenerationRequest,
    step_count: u16,
    conditioning: BufferId,
    latent: TensorRef,
) -> Result<Call, PlanningError> {
    if !request.generates_images() {
        return Err(PlanningError::GenerationBranchDisabled);
    }
    let mut call = computation(request, CallKind::Media(MediaCall::Denoising));
    call.bounds.max_tokens = u32::from(step_count.max(1));
    call.kv_input = Some(conditioning);
    call.latent_input = Some(latent);
    call.latent_output = Some(latent_output(
        0,
        request.image_latent_bytes(limits)?,
        latent_dtype.ok_or(PlanningError::MissingLatentDType)?,
    )?);
    call.completion_output = Some(output_tensor(1, DType::U8));
    finish_plan(request, call, 0)
}

/// Decodes the final latent and declares public image bytes and optional feedback data.
pub(super) fn plan_diffusion_finalize(
    request: &GenerationRequest,
    latent: TensorRef,
) -> Result<Call, PlanningError> {
    if !request.generates_images() {
        return Err(PlanningError::GenerationBranchDisabled);
    }
    let mut call = computation(request, CallKind::Media(MediaCall::ImageDecoding));
    call.latent_input = Some(latent);
    call.image_output = feedback_image_output(request)?;
    call.completion_output = Some(output_tensor(2, DType::U8));
    finish_plan(request, call, 0)
}

/// Derives storage bounds directly from the computation's actual inputs and outputs.
fn finish_plan(
    request: &GenerationRequest,
    mut call: Call,
    prompt_positions: u32,
) -> Result<Call, PlanningError> {
    let max_latent_bytes = call
        .encoder_output
        .iter()
        .chain(call.latent_output.iter())
        .chain(call.image_output.iter())
        .map(TensorRef::max_bytes)
        .max()
        .unwrap_or(0);
    let image_completion_bytes = if call.code == CallKind::Media(MediaCall::ImageDecoding) {
        png_base64_bound(request.image.width, request.image.height)?
    } else {
        0
    };
    let logprob_bytes = if call.token_output.is_some() {
        logprob_result_bytes(&request.sampling, prompt_positions)?.unwrap_or(0)
    } else {
        0
    };
    let max_transfer_bytes = call.bounds.max_transfer_bytes;
    call.bounds = Bounds {
        max_tokens: call.bounds.max_tokens,
        max_kv_pages: 0,
        max_latent_bytes,
        max_completion_bytes: logprob_bytes.saturating_add(image_completion_bytes),
        max_transfer_bytes,
    };
    Ok(call)
}

/// Declares the device image retained for a subsequent feedback encoder.
/// Encoded PNG bytes are delivered separately through the completion media handle.
fn feedback_image_output(request: &GenerationRequest) -> Result<Option<TensorRef>, PlanningError> {
    if request.feeds_back_images()
        && request.image_generation.feedback_source
            == Some(uniserve_core::FeedbackSource::DeviceProduct)
    {
        return Ok(Some(bounded_tensor(
            1,
            DType::BF16,
            dynamic_element_bound(
                u64::from(request.image.height)
                    .saturating_mul(u64::from(request.image.width))
                    .saturating_mul(3)
                    .saturating_mul(DType::BF16.element_bytes()),
                DType::BF16,
            )?,
        )));
    }
    Ok(None)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Failure while constructing the next worker call.
pub(crate) enum PlanningError {
    #[error(transparent)]
    Capacity(#[from] uniserve_core::GenerationResourceError),
    #[error("invalid prompt range {start}..{end} for {prompt_tokens} input tokens")]
    InvalidPromptRange {
        start: u32,
        end: u32,
        prompt_tokens: usize,
    },
    #[error("image generation branch is disabled")]
    GenerationBranchDisabled,
    #[error("generated-image feedback is disabled")]
    FeedbackDisabled,
    #[error("image encoding requires image bytes or a source product")]
    MissingImageInput,
    #[error("product bound must be nonzero")]
    MissingProductBound,
    #[error("worker did not report a latent dtype")]
    MissingLatentDType,
    #[error("product bound {bytes} bytes exceeds the IPC representation")]
    ProductBoundTooLarge { bytes: u64 },
    #[error("product generation counter exhausted")]
    ProductGenerationExhausted,
}

/// Assigns final identities to the selected computation and its allocated outputs.
/// Identity exhaustion is checked before changing the shared generation counter.
pub(super) fn register_call(
    call: &mut Call,
    request_key: RequestKey,
    next_product_generation: &mut u64,
) -> Result<(), PlanningError> {
    let call_id = call.call_id;
    let required_generations = call
        .tensor_outputs()
        .filter(|product| product.generation == 0)
        .count()
        + usize::from(call.kv_output.is_some_and(|output| output.generation == 0));
    let first_generation = (*next_product_generation).max(1);
    if required_generations > 0 {
        let last_generation = first_generation
            .checked_add(required_generations as u64 - 1)
            .ok_or(PlanningError::ProductGenerationExhausted)?;
        if last_generation > u64::from(u32::MAX) {
            return Err(PlanningError::ProductGenerationExhausted);
        }
    }
    let mut acquire_generation = || {
        let generation =
            u32::try_from((*next_product_generation).max(1)).expect("generation preflight");
        *next_product_generation = u64::from(generation) + 1;
        generation
    };
    call.request_key = request_key;
    if let Some(output) = &mut call.kv_output {
        output.owner = request_key;
        output.producer_call_id = call_id;
        if output.generation == 0 {
            output.generation = acquire_generation();
        }
    }
    for product in call.tensor_outputs_mut() {
        product.request_key = request_key;
        product.producer_call_id = call_id;
        if product.generation == 0 {
            product.generation = acquire_generation();
        }
    }
    Ok(())
}

/// Validates a result against submitted inputs and immutable request limits.
pub(crate) fn validate_generation_result(
    call: &Call,
    image_kv: Option<(u32, Option<u32>)>,
    start_step: Option<u32>,
    state: &super::RequestState,
    record: &RequestOutput,
    media: Option<&uniserve_core::SharedMedia>,
) -> Result<(), GenerationResultError> {
    let request = &state.req;
    if record.request_key != call.request_key {
        return Err(GenerationResultError::Identity {
            detail: "request_mismatch",
        });
    }
    if call.call_id.batch_id == 0 || record.call_id != call.call_id {
        return Err(GenerationResultError::Identity {
            detail: "call_id_mismatch",
        });
    }
    if record.status == CallStatus::Predicated {
        if call.predicate.as_ref().is_none()
            || !record.committed_tokens.as_slice().is_empty()
            || !record.product_generations.is_empty()
        {
            return Err(GenerationResultError::Status {
                detail: "invalid_predicated_result",
            });
        }
        return Ok(());
    }
    let call_variant = call.code;
    if record.status != CallStatus::Ok {
        return Err(GenerationResultError::Status {
            detail: "call_failed",
        });
    }
    if call.code == CallKind::Media(MediaCall::Denoising) {
        let expected_step = start_step
            .ok_or(GenerationResultError::Progress {
                detail: "denoise_input_step_missing",
            })?
            .saturating_add(call.bounds.max_tokens);
        if record.num_completed_steps != expected_step {
            return Err(GenerationResultError::Progress {
                detail: "denoise_step_mismatch",
            });
        }
    }
    if !record
        .product_generations
        .iter()
        .copied()
        .eq(call.tensor_outputs().map(|tensor| tensor.generation))
    {
        return Err(GenerationResultError::Product {
            detail: "tensor_generation_mismatch",
        });
    }
    // Read the PNG header to check the requested dimensions. Full image
    // decoding belongs to the image consumer, outside result processing.
    let image_png = media.and_then(|value| std::str::from_utf8(value.as_bytes()).ok());
    if call_variant == CallKind::Media(MediaCall::ImageDecoding) {
        if media.is_some_and(|value| value.len() as u64 > call.bounds.max_completion_bytes) {
            return Err(GenerationResultError::Product {
                detail: "image_capacity_exceeded",
            });
        }
        let actual =
            image_png
                .and_then(png_artifact_dims_b64)
                .ok_or(GenerationResultError::Product {
                    detail: "missing_image_dimensions",
                })?;
        if actual != (request.image.height, request.image.width) {
            return Err(GenerationResultError::Product {
                detail: "image_dimensions_mismatch",
            });
        }
    }
    if consumes_image_features(call) {
        let (start, num_kv_tokens) = image_kv.ok_or(GenerationResultError::Progress {
            detail: "image_kv_input_missing",
        })?;
        let actual = record.kv_visible_len;
        if num_kv_tokens.is_some_and(|tokens| actual != start.saturating_add(tokens))
            || (num_kv_tokens.is_none()
                && (actual < start || actual > start.saturating_add(call.bounds.max_tokens)))
        {
            return Err(GenerationResultError::Progress {
                detail: "image_kv_mismatch",
            });
        }
    }
    let sampled_tokens = record.committed_tokens.as_slice();
    let produces_token = call.token_output.is_some();
    let allowed_tokens = call
        .sampling_state
        .as_ref()
        .and_then(|sampling| sampling.allowed_token_ids.as_deref())
        .or(request.sampling.allowed_token_ids.as_deref());
    if !produces_token && !sampled_tokens.is_empty() {
        return Err(GenerationResultError::Token {
            detail: "unexpected_sampled_token",
        });
    }
    if produces_token && sampled_tokens.is_empty() {
        return Err(GenerationResultError::Token {
            detail: "missing_sampled_token",
        });
    }
    if call_variant == CallKind::Forward(ForwardMode::Verify) {
        let drafts = call.input_token_ids.as_slice();
        let listed = record.committed_tokens.as_slice();
        let terminal_prefix = !listed.is_empty()
            && listed.len() <= drafts.len()
            && listed == &drafts[..listed.len()]
            && state
                .finish_token_ids
                .contains(listed.last().expect("nonempty verified prefix"));
        if !terminal_prefix {
            let accepted = listed.len().saturating_sub(1);
            if listed.is_empty()
                || accepted > drafts.len()
                || listed[..accepted] != drafts[..accepted]
            {
                return Err(GenerationResultError::Token {
                    detail: "verified_draft_prefix_mismatch",
                });
            }
        }
    }
    if produces_token {
        let max_tokens = if call_variant == CallKind::Forward(ForwardMode::Verify) {
            call.bounds.max_tokens as usize
        } else {
            1
        };
        if sampled_tokens.len() > max_tokens {
            return Err(GenerationResultError::Token {
                detail: "text_token_count_mismatch",
            });
        }
    }
    if let Some(allowed) = allowed_tokens
        && sampled_tokens
            .iter()
            .any(|token_id| !allowed.contains(token_id))
    {
        return Err(GenerationResultError::Token {
            detail: "sampled_token_outside_allowed_set",
        });
    }
    let sampled_token = sampled_tokens.last().copied();
    let generated_candidates = record.top_logprobs.as_slice();
    match (
        produces_token
            && request.sampling.generated_logprobs_requested()
            && matches!(
                call_variant,
                CallKind::Forward(ForwardMode::Prefill)
                    | CallKind::Forward(ForwardMode::Decode)
                    | CallKind::Forward(ForwardMode::Verify)
            ),
        sampled_token,
        generated_candidates.is_empty(),
    ) {
        (false, _, false) | (true, None, false) => {
            return Err(GenerationResultError::Logprob {
                detail: "unexpected_generated_logprobs",
            });
        }
        (true, Some(_), true) => {
            return Err(GenerationResultError::Logprob {
                detail: "missing_generated_logprobs",
            });
        }
        (true, Some(token_id), false) => {
            if generated_candidates[0].token_id != token_id {
                return Err(GenerationResultError::Logprob {
                    detail: "generated_logprob_token_mismatch",
                });
            }
            let mut token_ids = std::collections::HashSet::new();
            if generated_candidates.iter().any(|candidate| {
                candidate.rank == 0
                    || !candidate.logprob.is_finite()
                    || !token_ids.insert(candidate.token_id)
            }) {
                return Err(GenerationResultError::Logprob {
                    detail: "invalid_generated_logprob_candidates",
                });
            }
        }
        (false, _, true) | (true, None, true) => {}
    }
    // Prefill's sampling coordinate is its exclusive prompt-token end, so
    // the first input can be identified even while a cancelled request drains.
    let prompt_tokens = (is_prompt_extend(call) && request.sampling.prompt_logprobs_requested())
        .then(|| {
            &call.input_token_ids[usize::from(
                call.rng.as_ref().is_some_and(|rng| {
                    rng.semantic_index_base == call.input_token_ids.len() as u64
                }) && !call.input_token_ids.is_empty(),
            )..]
        });
    match (
        prompt_tokens,
        (!record.prompt_logprobs.is_empty()).then_some(record.prompt_logprobs.as_slice()),
    ) {
        (None, Some(positions)) if !positions.is_empty() => {
            return Err(GenerationResultError::Logprob {
                detail: "unexpected_prompt_logprobs",
            });
        }
        (Some(expected), actual) => {
            let actual = actual.unwrap_or_default();
            if actual.len() != expected.len() {
                return Err(GenerationResultError::Logprob {
                    detail: "prompt_logprob_count_mismatch",
                });
            }
            for (&expected_token, candidates) in expected.iter().zip(actual) {
                let Some(first) = candidates.first() else {
                    return Err(GenerationResultError::Logprob {
                        detail: "empty_prompt_logprob_position",
                    });
                };
                if first.token_id != expected_token {
                    return Err(GenerationResultError::Logprob {
                        detail: "prompt_logprob_token_mismatch",
                    });
                }
                let mut seen = std::collections::HashSet::new();
                if candidates
                    .iter()
                    .any(|candidate| candidate.rank == 0 || !seen.insert(candidate.token_id))
                {
                    return Err(GenerationResultError::Logprob {
                        detail: "invalid_prompt_logprob_candidates",
                    });
                }
            }
        }
        (None, None | Some(_)) => {}
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// Invalid execution identity, numerical result, or produced value.
pub(crate) enum GenerationResultError {
    #[error("call has no registered identity")]
    MissingCallId,
    #[error("generation call produced no latent product")]
    MissingLatentProduct,
    #[error("worker status invalid: {detail}")]
    Status { detail: &'static str },
    #[error("worker identity invalid: {detail}")]
    Identity { detail: &'static str },
    #[error("worker progress invalid: {detail}")]
    Progress { detail: &'static str },
    #[error("worker product invalid: {detail}")]
    Product { detail: &'static str },
    #[error("worker token result invalid: {detail}")]
    Token { detail: &'static str },
    #[error("worker logprob result invalid: {detail}")]
    Logprob { detail: &'static str },
}

impl GenerationResultError {
    /// Returns structured generation-error details.
    pub(crate) fn detail(&self) -> &'static str {
        match self {
            Self::Status { detail }
            | Self::Identity { detail }
            | Self::Progress { detail }
            | Self::Product { detail }
            | Self::Token { detail }
            | Self::Logprob { detail } => detail,
            Self::MissingCallId => "missing_call_id",
            Self::MissingLatentProduct => "missing_latent_product",
        }
    }
}
