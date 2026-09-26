//! Generation lifecycle state and worker-call planning.
//!
//! A request advances through context ingestion, then either understanding
//! decode, image generation and optional feedback, or, for a readout, canvas
//! denoising, and finally terminal publication ([`GenerationPhase`]).
//! [`RequestState`] holds the accepted progress of one admitted token request.
//!
//! Each `plan_*` builder returns one call whose identities are placeholders
//! and whose `Bounds` come from `finish_plan`; `register_call` then stamps the
//! request key and assigns product generations. Every planned output is named
//! by request key (which includes the request epoch), producer call, output
//! index, and generation.
//!
//! On completion, `validate_generation_result` checks a worker result against
//! the submitted call and the request's limits, and
//! `RequestState::process_generation_result` applies its accepted progress.
//! Token semantics (stop conditions, image-branch triggers, public events)
//! are resolved afterwards in `output`.

use super::{EncoderCachePin, FlowPrefixState, Phase, RequestAllocations, TerminalIntent};
use crate::kv::{BlockTable, KvAllocation};

use std::collections::HashSet;
use uniserve_worker_ipc::{
    CallCoordinates, DEFAULT_COMPONENT, ForwardMode, LatentParams, MediaCall, TransferMode,
};

use uniserve_core::{GenerationRequest, ImageIngestStep, RequestId, SamplingParams};
use uniserve_worker_ipc::{
    Bounds, BufferId, Call, CallId, CallKind, CallStatus, DType, DimBound, DrawLayout, Readout,
    RequestKey, RequestOutput, Rng, SamplingState, ShapeBound, TensorRef,
};

use crate::scheduler::image_artifact::png_artifact_dims_b64;

/// Builds an unstamped product reference for a planned call output.
///
/// The owning `request_key` and `producer_call_id` are placeholders until
/// [`register_call`] stamps the real identity, and generation zero marks the
/// product for a fresh generation there. The shape bound is empty: it carries
/// identity, not a device geometry. `output_index` must be unique among the
/// call's outputs, including `kv_output`; `Call::validate` rejects repeats.
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

/// Converts a byte bound into a one-dimensional, device-sized element bound of
/// `dtype`, rounding up.
///
/// Fails with `ProductBoundTooLarge` when the element count exceeds
/// `u32::MAX` and with `MissingProductBound` when it is zero.
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

/// Computes the base64 length bound for a generated PNG artifact.
///
/// The raw size counts three bytes per pixel plus one byte per row. The bound
/// doubles it, adds 1 MiB, and applies base64's expansion of three bytes to
/// four characters. It reaches the worker as part of `max_completion_bytes`,
/// and `validate_generation_result` rejects a larger artifact.
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

/// Serialized size of one ranked logprob entry (`TokenLogprob` in the IPC
/// schema).
const RANKED_LOGPROB_BYTES: u64 = 12;

/// Computes the maximum serialized log-probability payload required by a request.
///
/// Returns `None` when neither generated nor prompt logprobs are requested.
/// Each reported row costs 4 bytes plus `RANKED_LOGPROB_BYTES` per entry; a
/// row's entries are its own token, the requested top-k candidates
/// (`n_logprobs` or `n_prompt_logprobs`), and the distinct ids of
/// `logprob_token_ids`. A call reports at most one generated row and
/// `prompt_positions` prompt rows. The worker
/// (`uniserve_worker.execution.commit`) costs its actual payload the same way
/// and rejects one above `max_completion_bytes`.
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
///
/// `RequestState::phase` records the accepted phase. Scheduling projects the
/// phase of the next call from it and the calls still in flight
/// (`Scheduler::next_generation_phase`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GenerationPhase {
    /// Encode staged input images before continuing text prefill.
    Encode,
    /// Write the current input image's encoder feature into KV.
    IngestState,
    /// Prefill prompt tokens.
    Prefill,
    /// Decode understanding (text) tokens.
    DecodeUnd,
    /// Write the pending token (`RequestState::next_token`) into KV without
    /// sampling, before an image branch publishes the KV.
    CloseKv,
    /// Publish the visible KV range as diffusion conditioning.
    PublishKv,
    /// Prepare the initial latent of the image.
    PrepareGen,
    /// Run the image's denoising steps.
    DenoiseGen,
    /// Decode the final latent into the image. Never stored in
    /// `RequestState::phase`: `Scheduler::next_generation_phase` projects it
    /// once every denoising step is scheduled, while the accepted phase stays
    /// `DenoiseGen`.
    CommitGen,
    /// Encode the generated image with the next feedback encoder.
    FeedbackEncode,
    /// Write the feedback encoder's feature into KV.
    FeedbackState,
    /// Denoise a readout request's canvas rows over its prompt, a bounded
    /// group of rows per call, until every row has reported its slots.
    Readout,
}

/// Image extension consumes an encoder feature rather than token inputs.
pub(super) fn consumes_image_features(call: &Call) -> bool {
    call.code == CallKind::Forward(ForwardMode::Prefill)
        && (call.vision_input.is_some() || call.latent_feature_input.is_some())
}

/// Feedback encoders and KV writes produce a predicate for their device successor.
/// Input-image encoding has no such successor until its host result is accepted.
///
/// `plan_encode` and `plan_image_extend` declare `completion_output` only for
/// feedback, which is what separates the feedback variants here.
pub(super) fn is_feedback_computation(call: &Call) -> bool {
    (matches!(
        call.code,
        CallKind::Media(MediaCall::VisionEncoding) | CallKind::Media(MediaCall::LatentEncoding)
    ) || consumes_image_features(call))
        && call.completion_output.is_some()
}

/// Prompt extension writes prompt tokens into KV: a prefill with no
/// image-feature input and no completion predicate. It samples the next token
/// unless its request is a readout, whose prompt only conditions its canvases;
/// the `CloseKv` write is the prefill that declares a completion instead.
pub(super) fn is_prompt_extend(call: &Call) -> bool {
    call.code == CallKind::Forward(ForwardMode::Prefill)
        && !consumes_image_features(call)
        && call.completion_output.is_none()
}

impl RequestState {
    /// Applies one validated computation in request order. Pending identities are
    /// consumed once before this update; inactive successors never reach it.
    ///
    /// Updates the request's cursors, KV extents, encoder indices, denoising
    /// progress, and phase for the call kind. The completion loop in
    /// `execution` calls this, after removing the call from the in-flight
    /// queue, only for a `CallStatus::Ok` result that
    /// `validate_generation_result` accepted, on a request of the call's epoch
    /// whose chain is not invalidated and that has no terminal intent or
    /// deferred finish. Phase changes that depend on token values (entering
    /// `CloseKv`, ending a text round) belong to `output`.
    ///
    /// Returns an error, before changing any state, for a call id with batch
    /// zero, a latent call without a latent product, or a denoising result
    /// that has no submitted interval or that `Denoising::accept` rejects. The
    /// caller then finishes the request with an error.
    pub(crate) fn process_generation_result(
        &mut self,
        call: &Call,
        latent: Option<&LatentParams>,
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
                // The worker reports the KV extent the feature write reached.
                // Logical positions advance by the image's position count once,
                // after its last encoder.
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
            // A prefill that neither extends the prompt nor reads a feature is
            // the `CloseKv` write, after which the KV is published.
            CallKind::Forward(ForwardMode::Prefill) => {
                self.kv_visible_len = record.kv_visible_len;
                self.phase = GenerationPhase::PublishKv;
                self.replayable = false;
            }
            CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
                // A result that commits no token still consumes one position;
                // `output` resolves it as the primary EOS.
                let count = record.committed_tokens.len().max(1).min(u32::MAX as usize) as u32;
                self.logical_position = self.logical_position.saturating_add(count);
                self.kv_visible_len = self.kv_visible_len.saturating_add(count);
            }
            CallKind::Forward(ForwardMode::TokenDenoising) => {
                // The call covers the rows after the accepted ones whose
                // canvases it carries; their log-probabilities follow the
                // accepted ones in report order.
                let rows = self
                    .readout_rows_covering(self.readout_rows, call.input_token_ids.len())
                    .ok_or(GenerationResultError::Progress {
                        detail: "readout_rows_mismatch",
                    })?;
                self.readout_rows += rows;
                self.readout_logprobs
                    .extend_from_slice(&record.candidate_logprobs);
            }
            CallKind::Transfer(TransferMode::KvPublish) => {
                self.image_conditioning = call.kv_output;
                self.phase = GenerationPhase::PrepareGen;
            }
            CallKind::Media(MediaCall::LatentPreparation)
            | CallKind::Media(MediaCall::Denoising) => {
                if call.latent_output.is_none() {
                    return Err(GenerationResultError::MissingLatentProduct);
                }
                if call.code == CallKind::Media(MediaCall::LatentPreparation) {
                    self.denoising.open(call.latent_output.clone());
                    self.phase = GenerationPhase::DenoiseGen;
                } else {
                    let accepted = latent.is_some_and(|interval| {
                        self.denoising.accept(
                            interval,
                            record.num_completed_steps,
                            call.latent_output.clone(),
                        )
                    });
                    if !accepted {
                        return Err(GenerationResultError::Progress {
                            detail: "denoise_step_mismatch",
                        });
                    }
                    self.replayable = false;
                }
            }
            CallKind::Media(MediaCall::ImageDecoding) => {
                // Decoding the artifact releases the trajectory, so the request
                // leaves the flow and its next call enters at step zero.
                self.denoising.close();
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

/// Builds one immutable image-latent generation. The output remains addressed
/// by its exact call identity and logical generation until the scheduler frees
/// it; a denoising or image-decoding completion frees the latent it consumed.
fn latent_output(output_index: u16, bytes: u64, dtype: DType) -> Result<TensorRef, PlanningError> {
    Ok(bounded_tensor(
        output_index,
        dtype,
        dynamic_element_bound(bytes, dtype)?,
    ))
}

/// Builds the actual computation before assigning storage and execution identities.
///
/// Every optional input and output is unset, the call id and the request
/// key's engine id and epoch are zero placeholders, and the bound is one
/// token; the `plan_*` builders fill in the rest.
fn computation(request: &GenerationRequest, code: CallKind) -> Call {
    Call {
        consumer_slots: Vec::new(),
        token_input: None,

        request_key: RequestKey::new(0, request.request_id, 0),
        call_id: CallId::new(0, 0),
        coordinates: CallCoordinates::default(),
        component: DEFAULT_COMPONENT.into(),
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
        readout: None,
        sampling_state: None,
    }
}

/// Plans a prefill of prompt tokens `start..end`, with a sampled-token output
/// unless the request is a readout.
///
/// The RNG coordinate is `end`, the exclusive prompt end. A readout's prompt
/// only conditions its canvases, so its prefill samples nothing and declares
/// no RNG coordinate. Fails with `InvalidPromptRange` for an empty range or
/// one past the prompt.
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
    if request.is_readout() {
        return finish_plan(request, call, 0);
    }
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
    // Prompt position zero reports no logprob, so a chunk starting there
    // covers one position fewer; `validate_generation_result` recovers the
    // same offset from the call's RNG coordinate.
    let prompt_positions = if request.sampling.prompt_logprobs_requested() {
        end.saturating_sub(start)
            .saturating_sub(u32::from(start == 0))
    } else {
        0
    };
    finish_plan(request, call, prompt_positions)
}

/// Plans one token-denoising pass over the readout rows `rows` of `request`.
///
/// The call carries the rows' canvases back to back as its input tokens and
/// their slots as a `Readout` whose slot tokens index that concatenation;
/// candidates keep the request's report order, and each returns one FP32
/// log-probability. Fails with `InvalidReadoutRows` for an empty range or
/// one past the request's rows.
pub(super) fn plan_readout(
    request: &GenerationRequest,
    rows: std::ops::Range<usize>,
) -> Result<Call, PlanningError> {
    let selected = request
        .readout
        .get(rows.clone())
        .filter(|selected| !selected.is_empty())
        .ok_or(PlanningError::InvalidReadoutRows {
            start: rows.start,
            end: rows.end,
            rows: request.readout.len(),
        })?;

    let mut call = computation(request, CallKind::Forward(ForwardMode::TokenDenoising));
    let mut readout = Readout {
        slot_tokens: Vec::new(),
        candidate_offsets: vec![0],
        candidate_ids: Vec::new(),
    };
    for row in selected {
        // Slot positions are row-relative; the call addresses them in its
        // concatenated canvas tokens.
        let offset = call.input_token_ids.len() as u32;
        for slot in &row.slots {
            readout.slot_tokens.push(offset + slot.position);
            readout.candidate_ids.extend_from_slice(&slot.candidates);
            readout
                .candidate_offsets
                .push(readout.candidate_ids.len() as u32);
        }
        call.input_token_ids.extend_from_slice(&row.token_ids);
    }
    call.bounds.max_tokens = u32::try_from(call.input_token_ids.len())
        .map_err(|_| PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let completion_bytes = (readout.candidate_ids.len() as u64).saturating_mul(4);
    call.readout = Some(readout);
    let mut call = finish_plan(request, call, 0)?;
    call.bounds.max_completion_bytes = completion_bytes;
    Ok(call)
}

/// Plans a sampled continuation; relay inputs leave host token values empty.
///
/// With `spec_token_ids` drafts the call is a verify that carries the drafts
/// as its input tokens and may commit up to one token more than their count.
/// Otherwise it is a decode whose input is `input_token`, or nothing when
/// `relay_input` continues from the predecessor's device-resident token.
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
///
/// The call is a prefill that reads `feature` and may write up to the
/// encoder's `kv_token_capacity` KV tokens. `feedback` declares the completion
/// predicate that feedback writes report. `sample_continuation` declares a
/// sampled-token output, drawn at `sampling_index` when one is given.
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
///
/// A device `source` takes precedence over `image_base64`. Fails with
/// `FeedbackDisabled` for a feedback encode on a request without image
/// feedback or a feedback source, and with `MissingImageInput` for an
/// input-image encode that has neither bytes nor a source.
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
///
/// `publication_bytes` bounds the publication of the visible extent over
/// every cache group (`KvCacheInfo::publication_bytes`); it also makes the
/// call hold one of the scheduler's transfer reservations. Fails with
/// `MissingProductBound` when it is zero, which includes an empty extent.
pub(super) fn plan_kv_publish(
    publication_bytes: u64,
    request: &GenerationRequest,
) -> Result<Call, PlanningError> {
    if publication_bytes == 0 {
        return Err(PlanningError::MissingProductBound);
    }
    let mut call = computation(request, CallKind::Transfer(TransferMode::KvPublish));
    call.bounds.max_tokens = 0;
    call.bounds.max_transfer_bytes = publication_bytes;
    // Owner and producer are the placeholders from `computation`;
    // `register_call` stamps them.
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
///
/// `max_latent_bytes` is the largest encoder, latent, or image output.
/// `max_completion_bytes` covers the logprob payload of a token-producing call
/// plus, for image decoding, the base64 PNG. `max_kv_units` starts at zero;
/// `Scheduler::plan_computation` sets it for forward calls.
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
        max_kv_units: 0,
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
    #[error("invalid readout rows {start}..{end} of {rows} rows")]
    InvalidReadoutRows {
        start: usize,
        end: usize,
        rows: usize,
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
///
/// The caller must already have set `call.call_id` to the call's batch and
/// row. Every output with generation zero, `kv_output` first and then the
/// tensor outputs in `Call::tensor_outputs` order, receives the next value of
/// the engine-wide counter; generations are nonzero and fit in `u32`. On
/// error nothing is changed, and the caller treats it as engine-fatal.
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
    // The preflight above guarantees every generation this call assigns fits.
    let mut acquire_generation = || {
        let generation = u32::try_from((*next_product_generation).max(1))
            .map_err(|_| PlanningError::ProductGenerationExhausted)?;
        *next_product_generation = u64::from(generation) + 1;
        Ok::<_, PlanningError>(generation)
    };
    call.request_key = request_key;
    if let Some(output) = &mut call.kv_output {
        output.owner = request_key;
        output.producer_call_id = call_id;
        if output.generation == 0 {
            output.generation = acquire_generation()?;
        }
    }
    for product in call.tensor_outputs_mut() {
        product.request_key = request_key;
        product.producer_call_id = call_id;
        if product.generation == 0 {
            product.generation = acquire_generation()?;
        }
    }
    Ok(())
}

/// Validates a result against submitted inputs and immutable request limits.
///
/// `image_kv` and `latent` are the inputs frozen when the call was submitted
/// (`InflightInput::Generation`); `media` is the artifact payload delivered
/// with the result. A predicated result is valid only when the call carried a
/// predicate and the result reports no tokens and no products. Returns the
/// first violation found; the caller finishes the request with an error on
/// any `Err`.
pub(crate) fn validate_generation_result(
    call: &Call,
    image_kv: Option<(u32, Option<u32>)>,
    latent: Option<&LatentParams>,
    state: &RequestState,
    record: &RequestOutput,
    media: Option<&uniserve_core::SharedMedia>,
) -> Result<(), GenerationResultError> {
    let request = &state.req;

    // Identity and status.
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

    // Denoising must end exactly at the end of its submitted interval.
    if call.code == CallKind::Media(MediaCall::Denoising) {
        let interval = latent.ok_or(GenerationResultError::Progress {
            detail: "denoise_input_step_missing",
        })?;
        if !super::Denoising::completes(interval, record.num_completed_steps) {
            return Err(GenerationResultError::Progress {
                detail: "denoise_step_mismatch",
            });
        }
    }

    // The worker reports one generation per tensor output, in
    // `Call::tensor_outputs` order.
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

    // Read the PNG header to check the requested dimensions. The output path
    // (`image_done_event`) decodes the full image later.
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

    // A readout reports one finite log-probability per declared candidate,
    // and no other call reports any.
    if let Some(readout) = &call.readout {
        if record.candidate_logprobs.len() != readout.candidate_count() {
            return Err(GenerationResultError::Product {
                detail: "readout_candidate_count_mismatch",
            });
        }
        if record
            .candidate_logprobs
            .iter()
            .any(|logprob| !logprob.is_finite())
        {
            return Err(GenerationResultError::Product {
                detail: "readout_logprob_not_finite",
            });
        }
    } else if !record.candidate_logprobs.is_empty() {
        return Err(GenerationResultError::Product {
            detail: "unexpected_candidate_logprobs",
        });
    }

    // A feature write must reach exactly `start + tokens` when the encoder
    // declared an exact token count, and otherwise stay within the call's
    // token capacity above `start`.
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

    // Sampled tokens: presence, verified-draft prefix, count, and allowed set.
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
    // A verify commits its accepted drafts plus the target's own token, except
    // that a finish token among the drafts ends the commit there without an
    // extra token.
    if call_variant == CallKind::Forward(ForwardMode::Verify) {
        let drafts = call.input_token_ids.as_slice();
        let listed = record.committed_tokens.as_slice();
        let terminal_prefix = !listed.is_empty()
            && listed.len() <= drafts.len()
            && listed == &drafts[..listed.len()]
            && listed
                .last()
                .is_some_and(|token| state.finish_token_ids.contains(token));
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

    // Generated logprobs are expected exactly when a forward call samples a
    // token and the request asked for them; the first candidate must be the
    // last committed token, and candidates need nonzero ranks, finite values,
    // and distinct ids.
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

    // Prompt logprobs cover every input token of a prompt-extending prefill
    // except prompt position zero. A prefill's RNG coordinate is its exclusive
    // prompt end (`plan_prompt`), so a coordinate equal to the input length
    // marks the chunk starting at position zero; this reads only the call,
    // not request progress, which cancellation may have stopped.
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

/// Engine-owned state for one admitted request.
pub(crate) struct RequestState {
    pub req: GenerationRequest,
    /// Sorted device finish tokens (see `finish_token_ids` in the scheduler
    /// module).
    pub(crate) finish_token_ids: Vec<u32>,
    /// Storage the request holds, installed at admission: its request-pool
    /// slot and per-group KV block tables, plus the latent pages and buffer
    /// spans its calls reserve later. Each table owns its physical page
    /// references and therefore has exactly the request's lifetime.
    pub(super) allocations: Option<RequestAllocations>,
    /// Negative-prompt KV prefix for multi-branch guidance, allocated for the
    /// first denoising call and freed once every denoising step completes or
    /// the request finishes.
    pub(super) flow_prefix: Option<FlowPrefixState>,
    /// Admission generation used to reject results from prior request lifetimes.
    pub(crate) request_epoch: u64,
    /// Most recent accepted state-producing call.
    pub(crate) last_state_call_id: CallId,
    /// Last device token retained until a consumer is registered.
    pub(crate) latest_token: Option<TensorRef>,
    /// A false device predicate invalidated the unresolved successor chain.
    /// Already-submitted descendants must drain before scheduling resumes from
    /// the last host-observed execution result.
    pub(crate) speculative_chain_invalidated: bool,
    pub(super) output: super::output::RequestOutput,
    /// Current accepted computation stage. Completion processing and output
    /// resolution advance it; scheduling changes it only on an encoder-cache
    /// hit, which moves the request to `IngestState` without a call.
    pub(super) phase: Phase,
    /// Prompt token range already accepted by the worker.
    pub(super) num_computed_prompt_tokens: u32,
    /// Context images whose every encoder feature has been written into KV.
    pub(super) num_ingested_images: usize,
    /// Next encoder of the current context image.
    pub(super) image_encoder_index: usize,
    /// Encoder feature of the current context image awaiting its KV write.
    pub(super) input_image_features: Option<TensorRef>,
    /// A single committed round-close token was deferred: it is the next
    /// decode's input, and that decode's completion decides the round.
    pub(super) round_closing: bool,
    /// Model position of the accepted sequence. Tokens advance it by one and
    /// an image by its position count, so it differs from `kv_visible_len`
    /// when image inputs are present.
    pub(super) logical_position: u32,
    /// Accepted KV length visible to attention, in tokens.
    pub(super) kv_visible_len: u32,
    /// Accepted tokens whose KV is initialized. A verifier initializes its
    /// rejected drafts, so this exceeds the visible extent until the next
    /// forward makes exactly what it initializes visible.
    pub(super) kv_computed_len: u32,
    /// Committed token the next decode or `CloseKv` write continues from.
    pub(super) next_token: u32,
    pub(super) num_generated_tokens: usize,
    /// Current image branch; `Scheduler::begin_image` increments it, so the
    /// first image is 1.
    pub(super) image_id: u32,
    pub(super) num_generated_images: usize,
    /// Set when an image branch opens and cleared once
    /// `Scheduler::promote_gen_branch_reservation` confirms its KV capacity.
    pub(super) image_reservation_pending: bool,
    /// Solver progress of the current image's latent trajectory.
    pub(super) denoising: super::Denoising,
    /// Exact generations retained for conditioning, denoising, and feedback.
    pub(super) image_conditioning: Option<uniserve_worker_ipc::BufferId>,
    /// Next feedback encoder of the generated image.
    pub(super) feedback_encoder_index: usize,
    /// Device image product of the last image decode, when feedback reads
    /// the device product.
    pub(super) feedback_source: Option<TensorRef>,
    /// Feedback encoder feature awaiting its KV write.
    pub(super) feedback_features: Option<TensorRef>,
    /// Whether the current epoch is registered with its execution workers.
    pub(super) worker_registered: bool,
    /// Allocation serial of the request's KV units already declared to the
    /// workers: units allocated at or after it are fresh for the next
    /// dispatch (`KvAllocation::allocated_units`).
    pub(super) num_kv_units_sent: u64,
    /// Whether admission reserves the complete multimodal KV requirement.
    pub(super) reserve_worstcase: bool,
    /// Worst-case KV token extent computed at submission; its units are
    /// reserved only when `reserve_worstcase` is set.
    pub(super) max_reserved_kv_tokens: usize,
    /// Per-group chained hashes of the complete prompt pages, retained for
    /// publishing reusable prompt pages as prefill completes.
    pub(super) prefix_page_hashes: Vec<Vec<u64>>,
    /// Per group, the leading prompt pages already published to the prefix
    /// cache, or reused from it.
    pub(super) prefix_published: Vec<usize>,
    /// Accepted text and control tokens used by penalties and trigger matching.
    pub(super) generated_token_ids: Vec<u32>,
    /// Tokens since the last model round boundary, used by suffix-trigger matching.
    pub(super) round_token_ids: Vec<u32>,
    pub(super) text_tokens_since_image: usize,
    /// Host image bytes retained only while artifact-based feedback needs them.
    pub(super) feedback_image_b64: Option<String>,
    /// False after a completed computation creates state unavailable from the input.
    pub(super) replayable: bool,
    /// Encoder-cache entries this request holds; released when the request
    /// finishes.
    pub(super) encoder_cache_pins: Vec<EncoderCachePin>,
    /// Request-local encoder outputs and feedback sources outside the encoder
    /// cache, freed by `Scheduler::free_transient_products` or when the
    /// request finishes.
    pub(super) transient_encoder_products: Vec<TensorRef>,
    /// Leading readout rows whose candidate log-probabilities have been
    /// accepted.
    pub(super) readout_rows: usize,
    /// Accepted candidate log-probabilities of those rows, in report order.
    pub(super) readout_logprobs: Vec<f32>,
    /// Unix time, in seconds, at which the request state was created.
    pub queued_at: f64,
    pub(crate) terminal_intent: TerminalIntent,
}

impl RequestState {
    /// Returns the request allocations, which admission installs.
    pub(super) fn allocations(&self) -> Option<&RequestAllocations> {
        self.allocations.as_ref()
    }

    /// Returns mutable access to the request allocations, which admission installs.
    pub(super) fn allocations_mut(&mut self) -> Option<&mut RequestAllocations> {
        self.allocations.as_mut()
    }

    /// Returns the request-pool row, which admission assigns.
    pub(super) fn request_pool_idx(&self) -> Option<u32> {
        self.allocations().map(RequestAllocations::request_slot)
    }

    /// Returns the request's block tables, one per KV group once admitted.
    pub(super) fn block_tables(&self) -> Option<&[BlockTable]> {
        self.allocations().map(RequestAllocations::block_tables)
    }

    /// Returns mutable access to the request's KV tables once admitted.
    pub(super) fn kv_mut(&mut self) -> Option<&mut KvAllocation> {
        self.allocations_mut()
            .map(|allocations| &mut allocations.kv)
    }

    /// Returns whether the request includes context images.
    pub(super) fn has_context_images(&self) -> bool {
        !self.req.multimodal_inputs.images.is_empty()
    }

    /// Returns whether execution continues after committing generation output.
    pub(super) fn continues_after_gen_commit(&self) -> bool {
        self.req.continues_after_image()
    }

    /// Returns whether the request can open a generation branch.
    pub(super) fn can_open_gen_branch(&self) -> bool {
        self.req.generates_images()
            && self.num_generated_images < self.req.image.max_images as usize
    }

    /// Returns whether generation starts after context ingestion.
    pub(super) fn starts_gen_after_context(&self) -> bool {
        self.req.starts_with_image() && self.can_open_gen_branch()
    }

    /// Returns whether the text request can be replayed.
    pub(crate) fn is_replayable_text(&self) -> bool {
        self.replayable
    }

    /// Returns how many readout rows from row `start` a token-denoising call
    /// of `canvas_tokens` canvas tokens covers, or `None` when no whole
    /// number of rows there has exactly that many tokens.
    pub(super) fn readout_rows_covering(
        &self,
        start: usize,
        canvas_tokens: usize,
    ) -> Option<usize> {
        let mut remaining = canvas_tokens;
        let mut rows = 0;
        for row in self.req.readout.get(start..)? {
            if remaining == 0 {
                break;
            }
            remaining = remaining.checked_sub(row.token_ids.len())?;
            rows += 1;
        }
        (remaining == 0 && rows > 0).then_some(rows)
    }

    /// Returns the pending image step, if one exists.
    pub(super) fn pending_image_step(&self) -> Option<ImageIngestStep> {
        self.req
            .multimodal_inputs
            .images
            .get(self.num_ingested_images)?
            .encoders
            .get(self.image_encoder_index)
            .map(|input| input.encoder)
    }
}
