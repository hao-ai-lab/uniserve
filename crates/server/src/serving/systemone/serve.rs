//! Serving one System One request on the block-diffusion engine.
//!
//! [`ServingRuntime::systemone`] validates the request body, resolves its
//! `x_images`, plans the readout on the blocking pool, and submits one engine
//! readout per planned prompt. Each readout prefills its prompt, with every
//! image's features written where its soft-token placeholders stand, and
//! denoises its canvas rows once; it answers with the candidates'
//! log-probabilities, which the plan assembles into the official response.
//!
//! When the prompts share a prefix of text, the first readout runs alone so
//! that the prefix cache holds the shared pages before the others are
//! admitted, which then prefill only their own remainders. Prompts with
//! images share no cached pages, since page identities cover token ids only,
//! and all of their readouts run concurrently.

use base64::Engine as _;
use uniserve_core::{
    CachePolicy, EngineCoreOutput, FinishReason, GenerationConstraint, GenerationRequest,
    ImageEncoderInput, ImageGenerationConfig, ImageIngestStep, ImageInput, ImageParams,
    MultimodalInputs, ReadoutRow, ReadoutSlot, RejectionKind, RequestId, SamplingParams,
};

use crate::serving::media;
use crate::serving::systemone::error::{LocItem, SystemOneError, ValidationIssue};
use crate::serving::systemone::plan::{ImageSize, ReadoutPrompt};
use crate::serving::systemone::{SystemOneRequest, SystemOneResponse};
use crate::serving::{ServeRequestId, ServingRuntime};

impl ServingRuntime {
    /// Returns whether this runtime serves `POST /v1/systemone`.
    pub fn serves_readout(&self) -> bool {
        self.readout.is_some()
    }

    /// Answers one System One request body.
    ///
    /// Engine requests are registered as `{request_id}-{prompt index}`.
    /// Dropping the returned future cancels every readout still running.
    ///
    /// # Errors
    ///
    /// - [`SystemOneError::Validation`] for an invalid body, an image that
    ///   cannot be fetched or decoded, a request that exceeds its token
    ///   limits, or a readout the engine refuses as unservable;
    /// - [`SystemOneError::ModelNotFound`] for another model name;
    /// - [`SystemOneError::Overloaded`] when the engine queue is full;
    /// - [`SystemOneError::Server`] when this runtime serves no readout or a
    ///   readout fails.
    pub async fn systemone(
        &self,
        request_id: &ServeRequestId,
        body: &[u8],
    ) -> Result<SystemOneResponse, SystemOneError> {
        let Some(encoder) = self.readout.clone() else {
            return Err(SystemOneError::Server(
                "this model serves no System One readout".to_owned(),
            ));
        };
        let request = SystemOneRequest::from_json(body)?;
        request.check_model(self.served_model_name())?;

        let images = self
            .images
            .fetch_all(request.images.clone())
            .await
            .map_err(|error| {
                SystemOneError::Validation(vec![ValidationIssue::value_error(
                    vec![
                        "body".into(),
                        "x_images".into(),
                        LocItem::Index(error.index),
                    ],
                    &error.source.to_string(),
                    None,
                )])
            })?;
        let sizes: Vec<ImageSize> = images
            .iter()
            .map(|image| ImageSize {
                width: image.width(),
                height: image.height(),
            })
            .collect();

        // Rendering and tokenizing the prompts is CPU-bound.
        let plan = tokio::task::spawn_blocking(move || encoder.plan(&request, &sizes))
            .await
            .map_err(|error| {
                SystemOneError::Server(format!("readout planning did not complete: {error}"))
            })??;

        let requests: Vec<GenerationRequest> = plan
            .prompts
            .iter()
            .map(|prompt| readout_request(prompt, &images))
            .collect();
        let shares_prefix =
            images.is_empty() && plan.shared_prefix_tokens > 0 && requests.len() > 1;
        let logprobs = self
            .run_readouts(request_id, requests, shares_prefix)
            .await?;
        plan.assemble(self.served_model_name(), &logprobs)
    }

    /// Runs every prompt's readout and returns their log-probabilities in
    /// prompt order.
    ///
    /// With `shares_prefix`, the first readout completes before the others
    /// are submitted, so they find the shared prefix in the prefix cache.
    async fn run_readouts(
        &self,
        request_id: &ServeRequestId,
        requests: Vec<GenerationRequest>,
        shares_prefix: bool,
    ) -> Result<Vec<f32>, SystemOneError> {
        let mut requests = requests.into_iter().enumerate();
        let mut logprobs = Vec::new();
        if shares_prefix && let Some((index, request)) = requests.next() {
            logprobs.extend(self.readout(request_id, index, request).await?);
        }
        let rest = futures::future::try_join_all(
            requests.map(|(index, request)| self.readout(request_id, index, request)),
        )
        .await?;
        logprobs.extend(rest.into_iter().flatten());
        Ok(logprobs)
    }

    /// Submits one prompt's readout and waits for its answer.
    async fn readout(
        &self,
        request_id: &ServeRequestId,
        index: usize,
        mut request: GenerationRequest,
    ) -> Result<Vec<f32>, SystemOneError> {
        let external = format!("{request_id}-{index}");
        request.request_id = self
            .engine
            .register_request(external.clone(), None)
            .map_err(|error| SystemOneError::Server(error.to_string()))?;
        let mut events = self
            .engine
            .submit_generation(external, request)
            .await
            .map_err(|error| SystemOneError::Server(error.to_string()))?;

        let mut answer = None;
        while let Some(event) = events.recv().await {
            match event {
                EngineCoreOutput::Readout { candidate_logprobs } => {
                    answer = Some(candidate_logprobs);
                }
                EngineCoreOutput::Finished {
                    reason: FinishReason::Completed,
                    ..
                } => {
                    return answer.ok_or_else(|| {
                        SystemOneError::Server("a readout finished without its answer".to_owned())
                    });
                }
                EngineCoreOutput::Finished { reason, .. } => {
                    return Err(SystemOneError::Server(format!(
                        "a readout ended as {reason:?}"
                    )));
                }
                EngineCoreOutput::Rejected {
                    kind: RejectionKind::Overloaded,
                    message,
                } => return Err(SystemOneError::Overloaded(message)),
                EngineCoreOutput::Rejected {
                    kind: RejectionKind::Invalid,
                    message,
                } => {
                    return Err(SystemOneError::Validation(vec![
                        ValidationIssue::value_error(vec!["body".into()], &message, None),
                    ]));
                }
                EngineCoreOutput::Error { message } => {
                    return Err(SystemOneError::Server(message));
                }
                _ => {}
            }
        }
        Err(SystemOneError::Server(
            "the engine closed a readout before it finished".to_owned(),
        ))
    }
}

/// Builds the engine readout of one planned prompt.
///
/// The engine prefills the prompt's text tokens and writes each image's
/// features at the position its placeholder run held, so the placeholders
/// leave the prompt and each image enters at the count of text tokens before
/// it. An image's soft tokens take one position each, and the vision encoder
/// writes exactly that many KV entries. `images` are the request's decoded
/// `x_images`, indexed by [`ImagePlacement::source`](super::ImagePlacement).
fn readout_request(prompt: &ReadoutPrompt, images: &[media::ImageInput]) -> GenerationRequest {
    let mut token_ids = Vec::with_capacity(prompt.token_ids.len());
    let mut inputs = Vec::with_capacity(prompt.images.len());
    let mut next = 0;
    for placement in &prompt.images {
        let offset = placement.offset as usize;
        token_ids.extend_from_slice(&prompt.token_ids[next..offset]);
        next = offset + placement.soft_tokens as usize;

        let image = &images[placement.source];
        inputs.push(ImageInput {
            hash: image.hash(),
            b64: base64::engine::general_purpose::STANDARD.encode(image.bytes()),
            position: u32::try_from(token_ids.len()).unwrap_or(u32::MAX),
            num_positions: placement.soft_tokens,
            encoders: vec![ImageEncoderInput {
                encoder: ImageIngestStep::VitEncode,
                num_kv_tokens: Some(placement.soft_tokens),
                max_kv_tokens: None,
            }],
        });
    }
    token_ids.extend_from_slice(&prompt.token_ids[next..]);

    GenerationRequest {
        // `ServingRuntime::readout` replaces it with the registered id.
        request_id: RequestId(0),
        prompt_token_ids: token_ids,
        negative_prompt_token_ids: Vec::new(),
        multimodal_inputs: MultimodalInputs { images: inputs },
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_und_tokens: 0,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy::default(),
        include_stop_token: false,
        image_generation: ImageGenerationConfig::default(),
        readout: prompt
            .rows
            .iter()
            .map(|row| ReadoutRow {
                token_ids: row.token_ids.clone(),
                slots: row
                    .slots
                    .iter()
                    .map(|slot| ReadoutSlot {
                        position: slot.position,
                        candidates: slot.candidates.clone(),
                    })
                    .collect(),
            })
            .collect(),
        canvas: None,
    }
}
