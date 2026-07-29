use base64::Engine as _;
use std::io::Cursor;
use uniserve_core::{
    ContextSegment as CoreContextSegment, GenerationBehaviorDescriptor,
    GenerationCachePolicyDescriptor, GenerationConstraint, GenerationRequest,
    GenerationResourceBounds, GenerationRuntimeCapabilities, ImageParams,
    ImageSegment as CoreImageSegment, RequestId, SamplingParams as EngineSamplingParams,
    SegmentPlacement, UndVisibility,
};
use uniserve_model_profile::dialect::{
    GenerationDialectProfile, PromptContext, PromptKind, resolution::resolve_resolution,
};

use self::defaults::*;
use crate::text::tokenizer::DynTokenizer;
use crate::{
    ContextRole, ContextSegment, ImageGenerationPolicy, ImageInput, ModelContext, ServeRequest,
};

mod defaults;
mod output;

#[derive(Debug)]
struct LoweredGenerationInput {
    prompt_ids: Vec<u32>,
    neg_prompt_ids: Vec<u32>,
    sampling: EngineSamplingParams,
    image: ImageParams,
    constraint: GenerationConstraint,
    mm_items: Vec<RenderedImage>,
    stop_token_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
struct RenderedImage {
    hash: u64,
    position: u32,
    b64: String,
}

pub(crate) struct DialectStreamProcessor {
    output: output::DialectOutputProcessor,
}

impl DialectStreamProcessor {
    pub(crate) fn new(
        tokenizer: DynTokenizer,
        profile: &GenerationDialectProfile,
        prompt_token_ids: &[u32],
        profile_reasoning: bool,
    ) -> Result<Self, uniserve_model_profile::reasoning::ReasoningError> {
        Ok(Self {
            output: output::DialectOutputProcessor::new(
                profile.output_filter.clone(),
                std::sync::Arc::clone(&tokenizer),
                prompt_token_ids,
                profile_reasoning,
            )?,
        })
    }

    pub(crate) fn push(&mut self, text: &str) -> output::DialectTextDelta {
        self.output.push(text)
    }
}

#[derive(Debug, Clone, Default)]
struct CompileInput {
    prompt: String,
    prompt_ids: Option<Vec<u32>>,
    constraint: Option<GenerationConstraint>,
    system_prompt: Option<String>,
    assistant_prefix: Option<String>,
    negative_prompt: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: Option<u64>,
    stop_token_ids: Vec<u32>,
    image_bias: Option<f32>,
    image: Option<ImageGenerationPolicy>,
    input_images: Vec<ImageInput>,
}

impl CompileInput {
    fn from_request(request: &ServeRequest) -> Result<Self, BuildError> {
        let mut input = Self {
            constraint: Some(request.generation.constraint),
            temperature: request.generation.temperature,
            top_p: request.generation.top_p,
            top_k: request.generation.top_k,
            seed: request.generation.image.seed.or_else(|| {
                request
                    .generation
                    .seed
                    .and_then(|value| value.try_into().ok())
            }),
            stop_token_ids: request.generation.stop_token_ids.clone(),
            image_bias: request.generation.image.image_bias,
            image: Some(request.generation.image.clone()),
            ..Self::default()
        };

        match &request.model_context {
            ModelContext::RawPrompt(prompt) => input.prompt = prompt.clone(),
            ModelContext::Segments(segments) => {
                for segment in segments {
                    match segment {
                        ContextSegment::Text { role, text } => match role {
                            ContextRole::System => append_text(&mut input.system_prompt, text),
                            ContextRole::Assistant => {
                                append_text(&mut input.assistant_prefix, text)
                            }
                            ContextRole::Developer | ContextRole::User | ContextRole::Tool => {
                                append_prompt(&mut input.prompt, text)
                            }
                        },
                        ContextSegment::Image(image) => input.input_images.push(image.clone()),
                        ContextSegment::TokenIds { token_ids, .. } => {
                            if input.prompt_ids.is_some() {
                                return Err(BuildError::new(
                                    "generation dialect requests accept one pre-tokenized context segment",
                                ));
                            }
                            input.prompt_ids = Some(token_ids.clone());
                        }
                    }
                }
            }
            ModelContext::TokenIds { token_ids, .. } => {
                input.prompt_ids = Some(token_ids.clone());
            }
            ModelContext::Chat { .. } => {
                return Err(BuildError::new(
                    "generation dialect requests require raw or segmented context",
                ));
            }
        }
        Ok(input)
    }

    fn image(&self) -> ImageGenerationPolicy {
        self.image.clone().unwrap_or_default()
    }

    fn negative_prompt(&self) -> String {
        self.negative_prompt
            .clone()
            .or_else(|| {
                self.image
                    .as_ref()
                    .and_then(|image| image.negative_prompt.clone())
            })
            .unwrap_or_default()
    }

    fn input_images(&self) -> Vec<ImageInput> {
        self.input_images.clone()
    }

    fn prompt_context(&self) -> PromptContext {
        PromptContext {
            prompt: self.prompt.clone(),
            system_prompt: self.system_prompt.clone(),
            assistant_prefix: self.assistant_prefix.clone(),
        }
    }
}

fn append_text(target: &mut Option<String>, text: &str) {
    match target {
        Some(value) if !value.is_empty() => {
            value.push('\n');
            value.push_str(text);
        }
        Some(value) => value.push_str(text),
        None => *target = Some(text.to_string()),
    }
}

fn append_prompt(target: &mut String, text: &str) {
    if !target.is_empty() {
        target.push('\n');
    }
    target.push_str(text);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildError {
    message: String,
}

impl BuildError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

/// CFG and resolution constants for text output with image context.
///
/// Context-image requests use a fixed, model-agnostic visual-reasoning
/// configuration that intentionally differs from the per-profile image
/// generation defaults (`GenerationDialectProfile::image_defaults`).
mod context_image_defaults {
    /// Text-guidance CFG scale for visual reasoning.
    pub(super) const CFG_TEXT_SCALE: f32 = 4.0;
    /// Image-guidance CFG scale for visual reasoning.
    pub(super) const CFG_IMG_SCALE: f32 = 2.0;
    /// CFG renorm strategy for visual reasoning.
    pub(super) const CFG_RENORM_TYPE: &str = "text_channel";
    /// CFG renorm floor for visual reasoning.
    pub(super) const CFG_RENORM_MIN: f32 = 0.0;
    /// CFG interval (lo, hi) for visual reasoning.
    pub(super) const CFG_INTERVAL: (f32, f32) = (0.0, 1.0);
    /// Square latent resolution used for visual reasoning images.
    pub(super) const RESOLUTION: u32 = 512;
}

struct GenerationRequestCompiler<'a> {
    tokenizer: DynTokenizer,
    profile: &'a GenerationDialectProfile,
}

impl<'a> GenerationRequestCompiler<'a> {
    fn new(tokenizer: DynTokenizer, profile: &'a GenerationDialectProfile) -> Self {
        Self { tokenizer, profile }
    }

    fn build(&self, body: &CompileInput) -> Result<LoweredGenerationInput, BuildError> {
        let constraint = body.constraint.unwrap_or_default();
        let input_images = body.input_images();
        let path = RequestPath::new(constraint, !input_images.is_empty());
        validate_prompt(body)?;
        if !self.profile.supports_constraint(constraint) {
            return Err(BuildError::new(format!(
                "constraint {} is not supported by this model profile",
                constraint.as_str()
            )));
        }
        if path.uses_feedback_ingest() {
            return self.build_und_with_images(body, constraint, &input_images);
        }

        let image = self.resolve_image_params(body, constraint)?;
        let negative_prompt = body.negative_prompt();
        let (prompt_ids, mm_items) = self.build_context_prompt(body, path.prompt, &input_images)?;
        let neg_prompt_ids = self
            .profile
            .build_negative_prompt_ids(&self.tokenizer, &negative_prompt)
            .map_err(|error| BuildError::new(error.to_string()))?;
        let sampling = self.resolve_sampling(body, constraint)?;

        Ok(LoweredGenerationInput {
            prompt_ids,
            neg_prompt_ids,
            sampling,
            image,
            constraint,
            mm_items,
            stop_token_ids: body.stop_token_ids.clone(),
        })
    }

    fn build_und_with_images(
        &self,
        body: &CompileInput,
        constraint: GenerationConstraint,
        input_images: &[ImageInput],
    ) -> Result<LoweredGenerationInput, BuildError> {
        if self.profile.context_markers_in_prompt() || body.prompt_ids.is_some() {
            return self.build_und_with_marked_images(body, constraint, input_images);
        }
        if input_images.is_empty() {
            return Err(BuildError::new(
                "und_only requests with context images require image data",
            ));
        }
        let first_image = input_images.first().ok_or_else(|| {
            BuildError::new("und_only requests with context images require image data")
        })?;
        let system = body
            .system_prompt
            .as_deref()
            .unwrap_or_else(|| self.profile.context_system_prompt());
        let mut sys_ids = self
            .profile
            .wrap_context_text(&self.tokenizer, system)
            .map_err(|error| BuildError::new(error.to_string()))?;
        let question_ids = self
            .profile
            .wrap_context_text(&self.tokenizer, &body.prompt)
            .map_err(|error| BuildError::new(error.to_string()))?;
        let image_position = first_image.placement.unwrap_or(sys_ids.len() as u32);
        sys_ids.extend_from_slice(&question_ids);

        let image_body = body.image();
        let steps = image_body.steps.unwrap_or(DEFAULT_STEPS);
        if steps == 0 {
            return Err(BuildError::new("image.steps must be positive"));
        }
        let max_images = image_body.max_images.unwrap_or(2);
        self.validate_max_images(max_images)?;
        let seed = image_body.seed.or(body.seed).or(Some(0));

        let image = ImageParams {
            steps,
            cfg_text_scale: context_image_defaults::CFG_TEXT_SCALE,
            cfg_img_scale: context_image_defaults::CFG_IMG_SCALE,
            cfg_renorm_type: context_image_defaults::CFG_RENORM_TYPE.into(),
            cfg_renorm_min: context_image_defaults::CFG_RENORM_MIN,
            cfg_interval: context_image_defaults::CFG_INTERVAL,
            timestep_shift: image_body
                .timestep_shift
                .unwrap_or(self.profile.image_defaults.timestep_shift),
            height: context_image_defaults::RESOLUTION,
            width: context_image_defaults::RESOLUTION,
            seed,
            negative_prompt: String::new(),
            max_images,
            image_prompts: image_body.prompts,
            retain_images: image_body.retain_images.unwrap_or(true),
        };
        Ok(LoweredGenerationInput {
            neg_prompt_ids: sys_ids.clone(),
            prompt_ids: sys_ids,
            sampling: self.resolve_sampling(body, constraint)?,
            image,
            constraint,
            mm_items: input_images
                .iter()
                .map(|image| mm_item(image, image.placement.unwrap_or(image_position)))
                .collect(),
            stop_token_ids: body.stop_token_ids.clone(),
        })
    }

    fn build_und_with_marked_images(
        &self,
        body: &CompileInput,
        constraint: GenerationConstraint,
        input_images: &[ImageInput],
    ) -> Result<LoweredGenerationInput, BuildError> {
        if input_images.is_empty() {
            return Err(BuildError::new(
                "und_only requests with context images require image data",
            ));
        }
        let (prompt_ids, mm_items) =
            self.build_context_prompt(body, PromptKind::UndWithImages, input_images)?;
        let negative_prompt = body.negative_prompt();
        Ok(LoweredGenerationInput {
            neg_prompt_ids: self
                .profile
                .build_negative_prompt_ids(&self.tokenizer, &negative_prompt)
                .map_err(|error| BuildError::new(error.to_string()))?,
            prompt_ids,
            sampling: self.resolve_sampling(body, constraint)?,
            image: self.resolve_image_params(body, constraint)?,
            constraint,
            mm_items,
            stop_token_ids: body.stop_token_ids.clone(),
        })
    }

    fn build_context_prompt(
        &self,
        body: &CompileInput,
        kind: PromptKind,
        input_images: &[ImageInput],
    ) -> Result<(Vec<u32>, Vec<RenderedImage>), BuildError> {
        if let Some(prompt_ids) = &body.prompt_ids {
            if input_images.is_empty() {
                return Ok((prompt_ids.clone(), Vec::new()));
            }
            if self.profile.context_markers_in_prompt() {
                let mut marker_positions = prompt_ids
                    .iter()
                    .enumerate()
                    .filter(|&(_, &token)| token == self.profile.controls.end_of_image)
                    .map(|(index, _)| index as u32);
                let mut mm_items = Vec::with_capacity(input_images.len());
                for image in input_images {
                    let position = marker_positions.next().ok_or_else(|| {
                        BuildError::new("pre-tokenized context lost an input-image marker")
                    })?;
                    mm_items.push(mm_item(image, position));
                }
                if marker_positions.next().is_some() {
                    return Err(BuildError::new(
                        "pre-tokenized context contains more image markers than image segments",
                    ));
                }
                return Ok((prompt_ids.clone(), mm_items));
            }
            let fallback_position = prompt_ids.len() as u32;
            let mm_items = input_images
                .iter()
                .map(|image| mm_item(image, image.placement.unwrap_or(fallback_position)))
                .collect();
            return Ok((prompt_ids.clone(), mm_items));
        }
        if input_images.is_empty() {
            let prompt_ids = self
                .profile
                .build_prompt_ids(&self.tokenizer, &body.prompt_context(), kind)
                .map_err(|error| BuildError::new(error.to_string()))?;
            return Ok((prompt_ids, Vec::new()));
        }
        let controls = &self.profile.controls;
        if self.profile.context_markers_in_prompt() {
            if controls.start_of_image_text.is_empty() || controls.end_of_image_text.is_empty() {
                return Err(BuildError::new(
                    "model profile declares no image marker tokens for context images",
                ));
            }
            let user_text = prompt_with_image_markers(
                &body.prompt,
                input_images.len(),
                &controls.start_of_image_text,
                &controls.end_of_image_text,
            );
            let prompt_ids = self
                .profile
                .build_prompt_ids_with_text(
                    &self.tokenizer,
                    &body.prompt_context(),
                    kind,
                    &user_text,
                )
                .map_err(|error| BuildError::new(error.to_string()))?;
            let mut marker_positions = prompt_ids
                .iter()
                .enumerate()
                .filter(|&(_, &token)| token == controls.end_of_image)
                .map(|(index, _)| index as u32);
            let mut mm_items = Vec::with_capacity(input_images.len());
            for image in input_images {
                let position = marker_positions.next().ok_or_else(|| {
                    BuildError::new("context image prompt lost its image markers")
                })?;
                mm_items.push(mm_item(image, position));
            }
            return Ok((prompt_ids, mm_items));
        }
        let prompt_ids = self
            .profile
            .build_prompt_ids(&self.tokenizer, &body.prompt_context(), kind)
            .map_err(|error| BuildError::new(error.to_string()))?;
        let fallback_position = prompt_ids.len() as u32;
        let mm_items = input_images
            .iter()
            .map(|image| mm_item(image, image.placement.unwrap_or(fallback_position)))
            .collect();
        Ok((prompt_ids, mm_items))
    }

    fn resolve_image_params(
        &self,
        body: &CompileInput,
        constraint: GenerationConstraint,
    ) -> Result<ImageParams, BuildError> {
        let image_body = body.image();
        let defaults = &self.profile.image_defaults;
        let resolution = resolve_resolution(
            &self.profile.resolution_policy,
            image_body
                .resolution
                .as_deref()
                .or(Some(defaults.resolution.as_str())),
            image_body.width,
            image_body.height,
        )
        .map_err(BuildError::new)?;
        let steps = image_body.steps.unwrap_or(defaults.steps);
        if steps == 0 {
            return Err(BuildError::new("image.steps must be positive"));
        }
        let cfg_interval = image_body
            .cfg_interval
            .map(|interval| (interval[0], interval[1]))
            .unwrap_or(defaults.cfg_interval);
        validate_cfg_interval(cfg_interval)?;

        let max_images = image_body.max_images.unwrap_or(defaults.max_images);
        self.validate_max_images(max_images)?;
        let negative_prompt = body.negative_prompt();
        Ok(ImageParams {
            steps,
            cfg_text_scale: finite_or(
                image_body.cfg_text_scale.unwrap_or(defaults.cfg_text_scale),
                "image.cfg_text_scale",
            )?,
            cfg_img_scale: finite_or(
                image_body.cfg_img_scale.unwrap_or(defaults.cfg_img_scale),
                "image.cfg_img_scale",
            )?,
            cfg_renorm_type: image_body
                .cfg_renorm_type
                .clone()
                .unwrap_or_else(|| defaults.cfg_renorm_type.clone()),
            cfg_renorm_min: finite_or(
                image_body.cfg_renorm_min.unwrap_or(defaults.cfg_renorm_min),
                "image.cfg_renorm_min",
            )?,
            cfg_interval,
            timestep_shift: finite_or(
                image_body.timestep_shift.unwrap_or(defaults.timestep_shift),
                "image.timestep_shift",
            )?,
            height: resolution.height,
            width: resolution.width,
            seed: image_body.seed.or(body.seed).or(defaults.seed),
            negative_prompt,
            max_images,
            image_prompts: image_body.prompts,
            retain_images: image_body
                .retain_images
                .unwrap_or(constraint != GenerationConstraint::GenOnly),
        })
    }

    fn resolve_sampling(
        &self,
        body: &CompileInput,
        constraint: GenerationConstraint,
    ) -> Result<EngineSamplingParams, BuildError> {
        let temperature = finite_or(
            body.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            "temperature",
        )?;
        if temperature < 0.0 {
            return Err(BuildError::new("temperature must be non-negative"));
        }
        let top_p = finite_or(body.top_p.unwrap_or(DEFAULT_TOP_P), "top_p")?;
        if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
            return Err(BuildError::new("top_p must be in (0, 1]"));
        }
        let image_bias = body.image_bias.unwrap_or(0.0);
        let logit_bias = if constraint == GenerationConstraint::Default
            && image_bias != 0.0
            && self.profile.controls.start_of_image != 0
        {
            vec![(self.profile.controls.start_of_image, image_bias)]
        } else {
            Vec::new()
        };
        Ok(EngineSamplingParams {
            temperature,
            top_p,
            top_k: body.top_k.unwrap_or(DEFAULT_TOP_K),
            seed: body.seed,
            logit_bias,
            ..EngineSamplingParams::default()
        })
    }

    fn validate_max_images(&self, value: u16) -> Result<(), BuildError> {
        if value == 0 {
            return Err(BuildError::new("image.max_images must be positive"));
        }
        if value > self.profile.image_defaults.max_images_limit {
            return Err(BuildError::new(format!(
                "image.max_images exceeds profile limit {}",
                self.profile.image_defaults.max_images_limit
            )));
        }
        Ok(())
    }
}

pub(crate) fn compile_generation_request(
    request: &ServeRequest,
    tokenizer: DynTokenizer,
    profile: &GenerationDialectProfile,
    capabilities: &GenerationRuntimeCapabilities,
    default_max_output_tokens: Option<u32>,
    max_model_tokens: u32,
) -> Result<GenerationRequest, BuildError> {
    let input = CompileInput::from_request(request)?;
    let mut lowered =
        GenerationRequestCompiler::new(std::sync::Arc::clone(&tokenizer), profile).build(&input)?;
    let prompt_tokens = u32::try_from(lowered.prompt_ids.len())
        .map_err(|_| BuildError::new("generation prompt exceeds the supported token count"))?;
    let mut policy = profile.generation_policy.clone();
    let behavior = GenerationBehaviorDescriptor::resolve(lowered.constraint, &policy);
    if behavior.generated_image_feedback
        && let Some(feedback) = policy.feedback.as_mut()
        && matches!(
            feedback.writeback,
            uniserve_core::FeedbackWriteback::DirectKv
        )
        && !capabilities
            .generated_image_commit
            .supports(feedback.commit)
    {
        feedback.commit = match feedback.commit {
            uniserve_core::CommitRecipe::CommitGen
                if capabilities.generated_image_commit.separate_writeback =>
            {
                uniserve_core::CommitRecipe::CommitGenThenWriteback
            }
            uniserve_core::CommitRecipe::CommitGenThenWriteback
                if capabilities.generated_image_commit.inline =>
            {
                uniserve_core::CommitRecipe::CommitGen
            }
            _ => {
                return Err(BuildError::new(
                    "runtime exposes no compatible generated-image commit mode",
                ));
            }
        };
    }
    let behavior = GenerationBehaviorDescriptor::resolve(lowered.constraint, &policy);
    let mut max_tokens = if behavior.und_decode {
        crate::text::resolve_max_tokens(
            request.generation.max_tokens,
            default_max_output_tokens,
            Some(max_model_tokens),
            prompt_tokens,
        )
        .map_err(|error| BuildError::new(error.to_string()))? as usize
    } else {
        0
    };
    lowered.sampling.ignore_eos = request.generation.ignore_eos;
    lowered.sampling.min_tokens = request.generation.min_tokens.unwrap_or(0) as usize;
    lowered.sampling.min_p = request.generation.min_p.unwrap_or(0.0);
    lowered.sampling.frequency_penalty = request.generation.frequency_penalty.unwrap_or(0.0);
    lowered.sampling.presence_penalty = request.generation.presence_penalty.unwrap_or(0.0);
    lowered.sampling.repetition_penalty = request.generation.repetition_penalty.unwrap_or(1.0);
    if let Some(request_bias) = &request.generation.logit_bias {
        let mut merged_bias = std::collections::BTreeMap::new();
        for (token_id, bias) in lowered.sampling.logit_bias.drain(..) {
            merged_bias.insert(token_id, bias);
        }
        for (&token_id, &bias) in request_bias {
            *merged_bias.entry(token_id).or_insert(0.0) += bias;
        }
        lowered.sampling.logit_bias = merged_bias.into_iter().collect();
    }
    lowered.sampling.return_logprobs =
        request.generation.logprobs.is_some() || request.generation.logprob_token_ids.is_some();
    lowered.sampling.n_logprobs = request
        .generation
        .logprobs
        .map_or(0, |count| if count < 0 { u32::MAX } else { count as u32 });
    lowered.sampling.return_prompt_logprobs = request.generation.prompt_logprobs.is_some();
    lowered.sampling.n_prompt_logprobs = request
        .generation
        .prompt_logprobs
        .map_or(0, |count| if count < 0 { u32::MAX } else { count as u32 });
    lowered.sampling.logprob_token_ids = request
        .generation
        .logprob_token_ids
        .clone()
        .unwrap_or_default();
    lowered.sampling.allowed_token_ids = request.generation.allowed_token_ids.clone();
    lowered.sampling.bad_words_ids = request
        .generation
        .bad_words
        .iter()
        .map(|word| {
            tokenizer
                .encode(word, false)
                .map_err(|error| BuildError::new(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut grammar_stop_token_ids = lowered.stop_token_ids.clone();
    grammar_stop_token_ids.push(profile.controls.eos);
    grammar_stop_token_ids.sort_unstable();
    grammar_stop_token_ids.dedup();
    let structured_output = request.generation.structured_output.as_ref();
    let grammar = crate::text::structured_output::compile_structured_output(
        structured_output,
        &*tokenizer,
        &grammar_stop_token_ids,
    )
    .map_err(|error| BuildError::new(error.to_string()))?;
    let lora_id = match &request.adapter {
        crate::AdapterSelection::Base => None,
        crate::AdapterSelection::Adapter { internal_id, .. } => Some(
            u32::try_from(*internal_id)
                .map_err(|_| BuildError::new("adapter id exceeds the engine id space"))?,
        ),
    };
    let cache =
        generation_cache_policy(&request.cache, lowered.sampling.prompt_logprobs_requested());
    let mut context = build_context_segments(&lowered, profile)?;
    for segment in &mut context {
        if let CoreContextSegment::Image { image, .. } = segment {
            image.hash = isolated_cache_key(image.hash, cache.isolation_key);
        }
    }
    let negative_context: Vec<CoreContextSegment> = (!lowered.neg_prompt_ids.is_empty())
        .then(|| CoreContextSegment::UndTokens {
            token_ids: lowered.neg_prompt_ids.clone(),
            visibility: UndVisibility::Internal,
        })
        .into_iter()
        .collect();
    let mut image = lowered.image;
    if policy.feedback.as_ref().is_some_and(|feedback| {
        matches!(
            feedback.writeback,
            uniserve_core::FeedbackWriteback::Reingest { .. }
        )
    }) {
        image.retain_images = false;
    }
    let mut resources = GenerationResourceBounds::conservative(
        &context,
        &negative_context,
        &behavior,
        &policy,
        &image,
        max_tokens,
        &cache,
        capabilities,
    )
    .map_err(|error| BuildError::new(error.to_string()))?;
    if resources.max_kv_tokens > max_model_tokens as usize && behavior.und_decode {
        let excess = resources.max_kv_tokens - max_model_tokens as usize;
        max_tokens = max_tokens.checked_sub(excess).filter(|value| *value > 0).ok_or_else(|| {
            BuildError::new(format!(
                "generation context requires at least {} KV tokens, exceeding the {max_model_tokens}-token runtime limit",
                resources.max_kv_tokens.saturating_sub(max_tokens)
            ))
        })?;
        resources = GenerationResourceBounds::conservative(
            &context,
            &negative_context,
            &behavior,
            &policy,
            &image,
            max_tokens,
            &cache,
            capabilities,
        )
        .map_err(|error| BuildError::new(error.to_string()))?;
    }
    if resources.max_kv_tokens > max_model_tokens as usize {
        return Err(BuildError::new(format!(
            "generation requires {} KV tokens, exceeding the {max_model_tokens}-token runtime limit",
            resources.max_kv_tokens
        )));
    }
    let request = GenerationRequest {
        request_id: RequestId(fnv1a(request.request_id.as_bytes())),
        context,
        negative_context,
        constraint: lowered.constraint,
        behavior,
        sampling: lowered.sampling,
        image,
        max_und_tokens: max_tokens,
        stop_strings: request.generation.stop_strings.clone(),
        stop_token_ids: lowered.stop_token_ids,
        priority: request.scheduling.priority,
        lora_id,
        grammar,
        cache,
        policy,
        resources,
    };
    request
        .validate()
        .map_err(|error| BuildError::new(error.to_string()))?;
    Ok(request)
}

fn generation_cache_policy(
    cache: &crate::CachePolicy,
    prompt_logprobs_requested: bool,
) -> GenerationCachePolicyDescriptor {
    GenerationCachePolicyDescriptor {
        read: !cache.bypass_read && !prompt_logprobs_requested,
        write: !cache.no_store,
        isolation_key: crate::cache_isolation_key(cache),
    }
}

fn isolated_cache_key(content_key: u64, isolation_key: Option<u64>) -> u64 {
    let Some(isolation_key) = isolation_key else {
        return content_key;
    };
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&content_key.to_le_bytes());
    bytes[8..].copy_from_slice(&isolation_key.to_le_bytes());
    fnv1a(&bytes)
}

fn build_context_segments(
    lowered: &LoweredGenerationInput,
    profile: &GenerationDialectProfile,
) -> Result<Vec<CoreContextSegment>, BuildError> {
    let mut images = lowered.mm_items.clone();
    images.sort_by_key(|image| image.position);
    let image_count = images.len();
    let mut segments = Vec::with_capacity(images.len().saturating_mul(2).saturating_add(1));
    let mut token_cursor = 0usize;
    for image in images {
        let position = (image.position as usize).min(lowered.prompt_ids.len());
        if position > token_cursor {
            segments.push(CoreContextSegment::UndTokens {
                token_ids: lowered.prompt_ids[token_cursor..position].to_vec(),
                visibility: UndVisibility::Internal,
            });
        }
        let ingest = if profile.image_ingest_requires_dimensions() {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&image.b64)
                .map_err(|error| BuildError::new(format!("invalid input image base64: {error}")))?;
            let dimensions = image::ImageReader::new(Cursor::new(bytes))
                .with_guessed_format()
                .map_err(|error| BuildError::new(format!("invalid input image data: {error}")))?
                .into_dimensions()
                .map_err(|error| BuildError::new(format!("invalid input image data: {error}")))?;
            profile
                .image_ingest_for_dimensions(dimensions.0, dimensions.1, image_count)
                .map_err(|error| BuildError::new(error.to_string()))?
        } else {
            profile.image_ingest.clone()
        };
        segments.push(CoreContextSegment::Image {
            image: CoreImageSegment {
                hash: image.hash,
                b64: image.b64,
                placement: SegmentPlacement::AtToken {
                    position: image.position,
                },
            },
            ingest,
        });
        token_cursor = position;
    }
    if token_cursor < lowered.prompt_ids.len() {
        segments.push(CoreContextSegment::UndTokens {
            token_ids: lowered.prompt_ids[token_cursor..].to_vec(),
            visibility: UndVisibility::Internal,
        });
    }
    if segments.is_empty() {
        segments.push(CoreContextSegment::UndTokens {
            token_ids: lowered.prompt_ids.clone(),
            visibility: UndVisibility::Internal,
        });
    }
    Ok(segments)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestPath {
    prompt: PromptKind,
    feedback_ingest: bool,
}

impl RequestPath {
    fn new(constraint: GenerationConstraint, has_input_images: bool) -> Self {
        match (constraint, has_input_images) {
            (GenerationConstraint::Default, _) => Self {
                prompt: PromptKind::Default,
                feedback_ingest: false,
            },
            (GenerationConstraint::UndOnly, true) => Self {
                prompt: PromptKind::UndWithImages,
                feedback_ingest: true,
            },
            (GenerationConstraint::UndOnly, false) => Self {
                prompt: PromptKind::Und,
                feedback_ingest: false,
            },
            (GenerationConstraint::GenOnly, _) => Self {
                prompt: PromptKind::Gen,
                feedback_ingest: false,
            },
        }
    }

    fn uses_feedback_ingest(self) -> bool {
        self.feedback_ingest
    }
}

fn validate_prompt(body: &CompileInput) -> Result<(), BuildError> {
    if body.prompt.trim().is_empty()
        && body
            .prompt_ids
            .as_ref()
            .is_none_or(|token_ids| token_ids.is_empty())
    {
        return Err(BuildError::new("generation requires a non-empty prompt"));
    }
    Ok(())
}

fn prompt_with_image_markers(
    prompt: &str,
    image_count: usize,
    start_marker: &str,
    end_marker: &str,
) -> String {
    let mut user_text = String::new();
    if image_count == 1 {
        user_text.push_str(start_marker);
        user_text.push_str(end_marker);
        user_text.push('\n');
    } else {
        for index in 0..image_count {
            user_text.push_str(&format!(
                "Image-{}:{}{}\n",
                index + 1,
                start_marker,
                end_marker
            ));
        }
    }
    user_text.push_str(prompt);
    user_text
}

fn mm_item(image: &ImageInput, position: u32) -> RenderedImage {
    RenderedImage {
        hash: fnv1a(image.b64.as_bytes()),
        position,
        b64: image.b64.clone(),
    }
}

fn validate_cfg_interval(value: (f32, f32)) -> Result<(), BuildError> {
    let (lo, hi) = value;
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err(BuildError::new(
            "image.cfg_interval must be a finite ordered pair",
        ));
    }
    Ok(())
}

fn finite_or(value: f32, name: &str) -> Result<f32, BuildError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(BuildError::new(format!("{name} must be finite")))
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::ModalityPolicy;
    use crate::text::tokenizer::{DynTokenizer, Tokenizer};
    use base64::Engine as _;
    use uniserve_model_profile::dialect::resolve_generation_dialect_for_model;

    use super::*;

    fn test_png_b64(width: u32, height: u32) -> String {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("PNG header");
            writer
                .write_image_data(&vec![0; (width * height) as usize])
                .expect("PNG pixels");
        }
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    #[derive(Debug)]
    struct SenseNovaTokenizer;

    impl Tokenizer for SenseNovaTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> crate::text::tokenizer::Result<Vec<u32>> {
            // Added tokens encode atomically (as the real tokenizer does);
            // everything else byte-encodes.
            const MARKERS: [(&str, u32); 2] = [("<img>", 151670), ("</img>", 151671)];
            let mut ids = Vec::new();
            let mut rest = text;
            'outer: while !rest.is_empty() {
                for (marker, id) in MARKERS {
                    if let Some(stripped) = rest.strip_prefix(marker) {
                        ids.push(id);
                        rest = stripped;
                        continue 'outer;
                    }
                }
                let mut chars = rest.chars();
                let ch = chars.next().expect("non-empty");
                let mut buf = [0u8; 4];
                ids.extend(ch.encode_utf8(&mut buf).bytes().map(u32::from));
                rest = chars.as_str();
            }
            Ok(ids)
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> crate::text::tokenizer::Result<String> {
            Ok(
                String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                    .into_owned(),
            )
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "<img>" => Some(151670),
                "</img>" => Some(151671),
                "<|im_start|>" => Some(151644),
                "<|im_end|>" => Some(151645),
                _ => None,
            }
        }
    }

    fn test_capabilities() -> GenerationRuntimeCapabilities {
        GenerationRuntimeCapabilities {
            supports_understanding: true,
            supports_vision_encode: true,
            supports_latent_encode: true,
            supports_image_generation: true,
            supports_commit_writeback: true,
            max_latent_units: 65_536,
            latent_downsample: 16,
            max_vae_grid_tokens: 4_096,
            max_vit_grid_tokens: 2_048,
            commit_marker_tokens: 2,
            max_cfg_branches: 3,
            scratch_capacity_tokens: 65_536,
            scratch_block_size: 64,
            encoder_cache_entries: 256,
            generated_image_commit: uniserve_core::GeneratedImageCommitCapabilities {
                inline: true,
                separate_writeback: true,
            },
        }
    }

    #[test]
    fn sensenova_defaults_match_official_path() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        assert_eq!(profile.id, "sensenova-u1");
        let body = CompileInput {
            prompt: "Generate a travel guide covering Sonoma, Sequoia, Tahoe, and the Golden Gate."
                .into(),
            ..Default::default()
        };
        let request = GenerationRequestCompiler::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert_eq!(request.constraint, GenerationConstraint::Default);
        assert_eq!(request.image.width, 2048);
        assert_eq!(request.image.height, 1152);
        assert_eq!(request.image.steps, 50);
        assert_eq!(request.image.cfg_text_scale, 4.0);
        assert_eq!(request.image.cfg_img_scale, 1.0);
        assert_eq!(request.image.timestep_shift, 3.0);
        assert_eq!(request.image.seed, Some(42));
        assert_eq!(request.image.max_images, 4);
        assert!(request.image.retain_images);
        assert!(request.image.image_prompts.is_empty());
        let rendered = request
            .prompt_ids
            .iter()
            .filter_map(|id| u8::try_from(*id).ok())
            .map(char::from)
            .collect::<String>();
        assert!(rendered.contains("MUST interleave text with generated images"));
        assert!(!rendered.contains("MAY place generated images"));
    }

    #[test]
    fn sensenova_context_images_build_marked_mm_items() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let body = CompileInput {
            prompt: "Describe this image in detail.".into(),
            constraint: Some(GenerationConstraint::UndOnly),
            input_images: vec![ImageInput {
                b64: "aGVsbG8=".into(),
                placement: None,
            }],
            ..Default::default()
        };
        let request = GenerationRequestCompiler::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert_eq!(request.constraint, GenerationConstraint::UndOnly);
        assert_eq!(request.mm_items.len(), 1);
        let position = request.mm_items[0].position as usize;
        // The encode gap sits exactly between the in-prompt markers.
        assert_eq!(
            request.prompt_ids[position], 151671,
            "position is the </img> token"
        );
        assert_eq!(
            request.prompt_ids[position - 1],
            151670,
            "preceded by <img>"
        );
        // No CFG branch is needed for a text-only answer.
        assert!(request.neg_prompt_ids.is_empty());
        // Context-image requests still use the profile resolution bucket.
        assert_eq!(request.image.width, 2048);
        assert_eq!(request.image.height, 1152);
    }

    #[test]
    fn sensenova_und_only_multi_image_positions_are_ordered() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let body = CompileInput {
            prompt: "Compare the two images.".into(),
            constraint: Some(GenerationConstraint::UndOnly),
            input_images: vec![
                ImageInput {
                    b64: "aQ==".into(),
                    placement: None,
                },
                ImageInput {
                    b64: "ag==".into(),
                    placement: None,
                },
            ],
            ..Default::default()
        };
        let request = GenerationRequestCompiler::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert_eq!(request.mm_items.len(), 2);
        let first = request.mm_items[0].position as usize;
        let second = request.mm_items[1].position as usize;
        assert!(first < second);
        assert_eq!(request.prompt_ids[first], 151671);
        assert_eq!(request.prompt_ids[second], 151671);
        assert_ne!(request.mm_items[0].hash, request.mm_items[1].hash);
    }

    #[test]
    fn non_marker_und_only_preserves_every_context_image_in_request_order() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("bagel", &*tok)
            .expect("profile resolution")
            .expect("BAGEL profile");
        let body = CompileInput {
            prompt: "Compare the two images.".into(),
            constraint: Some(GenerationConstraint::UndOnly),
            input_images: vec![
                ImageInput {
                    b64: "aQ==".into(),
                    placement: None,
                },
                ImageInput {
                    b64: "ag==".into(),
                    placement: None,
                },
            ],
            ..Default::default()
        };

        let request = GenerationRequestCompiler::new(tok, &profile)
            .build(&body)
            .expect("compile multi-image understanding request");

        assert_eq!(request.mm_items.len(), 2);
        assert_eq!(request.mm_items[0].position, request.mm_items[1].position);
        assert_ne!(request.mm_items[0].hash, request.mm_items[1].hash);
    }

    #[test]
    fn canonical_lowering_covers_every_constraint_with_and_without_image_context() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");

        for constraint in [
            GenerationConstraint::Default,
            GenerationConstraint::UndOnly,
            GenerationConstraint::GenOnly,
        ] {
            for with_image in [false, true] {
                let mut request = ServeRequest::text(
                    format!("{constraint:?}-{with_image}"),
                    "Render this request",
                );
                if with_image {
                    request.model_context = ModelContext::Segments(vec![
                        ContextSegment::Text {
                            role: ContextRole::User,
                            text: "Render this request".into(),
                        },
                        ContextSegment::Image(ImageInput {
                            b64: test_png_b64(512, 512),
                            placement: None,
                        }),
                    ]);
                }
                request.generation.constraint = constraint;
                request.generation.max_tokens = Some(8);
                request.generation.image.max_images = Some(1);
                request.modalities = ModalityPolicy {
                    input_text: true,
                    input_image: with_image,
                    output_text: constraint != GenerationConstraint::GenOnly,
                    output_image: constraint != GenerationConstraint::UndOnly,
                };

                let lowered = compile_generation_request(
                    &request,
                    Arc::clone(&tok),
                    &profile,
                    &test_capabilities(),
                    None,
                    32_768,
                )
                .expect("canonical lowering");
                lowered.validate().expect("valid generation request");
                assert_eq!(lowered.constraint, constraint);
                assert_eq!(lowered.context_image_count(), usize::from(with_image));
                if with_image {
                    let image_index = lowered
                        .context
                        .iter()
                        .position(|segment| matches!(segment, CoreContextSegment::Image { .. }))
                        .expect("ordered image segment");
                    assert!(image_index > 0, "prompt tokens precede the image marker");
                    assert!(
                        lowered.context[image_index + 1..]
                            .iter()
                            .any(|segment| matches!(segment, CoreContextSegment::UndTokens { .. })),
                        "prompt tokens follow the image marker"
                    );
                }
                match constraint {
                    GenerationConstraint::Default => {
                        assert!(lowered.behavior.und_decode);
                        assert!(lowered.behavior.emits_und());
                        assert!(lowered.behavior.gen_output);
                        assert!(!lowered.behavior.start_gen_after_context);
                    }
                    GenerationConstraint::UndOnly => {
                        assert!(lowered.behavior.und_decode);
                        assert!(lowered.behavior.emits_und());
                        assert!(!lowered.behavior.gen_output);
                    }
                    GenerationConstraint::GenOnly => {
                        assert!(!lowered.behavior.und_decode);
                        assert!(!lowered.behavior.emits_und());
                        assert!(lowered.behavior.gen_output);
                        assert!(lowered.behavior.start_gen_after_context);
                        assert_eq!(lowered.max_und_tokens, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn canonical_lowering_resolves_profile_and_context_output_limits() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let mut request = ServeRequest::text("bounded-output", "unused");
        request.model_context = ModelContext::TokenIds {
            token_ids: vec![10, 11, 12, 13],
            tokenizer: crate::TokenizerReference::RuntimeProfile,
        };
        request.generation.constraint = GenerationConstraint::UndOnly;

        let profile_default = compile_generation_request(
            &request,
            Arc::clone(&tok),
            &profile,
            &test_capabilities(),
            Some(7),
            100,
        )
        .expect("profile default output limit");
        assert_eq!(profile_default.max_und_tokens, 7);

        request.generation.max_tokens = Some(20);
        let context_capped =
            compile_generation_request(&request, tok, &profile, &test_capabilities(), Some(7), 10)
                .expect("remaining context output limit");
        assert_eq!(context_capped.max_und_tokens, 6);
        assert_eq!(context_capped.resources.max_kv_tokens, 10);
    }

    #[test]
    fn canonical_lowering_resolves_direct_kv_commit_to_runtime_topology() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let mut request = ServeRequest::text("commit-topology", "Render a travel image");
        request.generation.constraint = GenerationConstraint::Default;
        request.generation.max_tokens = Some(8);
        request.generation.image.max_images = Some(1);
        request.modalities.output_image = true;

        let mut inline = test_capabilities();
        inline.generated_image_commit = uniserve_core::GeneratedImageCommitCapabilities {
            inline: true,
            separate_writeback: false,
        };
        let inline_request =
            compile_generation_request(&request, Arc::clone(&tok), &profile, &inline, None, 32_768)
                .expect("inline commit plan");
        assert_eq!(
            inline_request
                .policy
                .feedback
                .as_ref()
                .map(|feedback| feedback.commit),
            Some(uniserve_core::CommitRecipe::CommitGen)
        );
        assert!(
            !inline_request
                .behavior
                .capability_needs(&inline_request.policy, std::iter::empty())
                .commit_writeback
        );

        let mut separate = test_capabilities();
        separate.generated_image_commit = uniserve_core::GeneratedImageCommitCapabilities {
            inline: false,
            separate_writeback: true,
        };
        let separate_request = compile_generation_request(
            &request,
            Arc::clone(&tok),
            &profile,
            &separate,
            None,
            32_768,
        )
        .expect("separate commit plan");
        assert_eq!(
            separate_request
                .policy
                .feedback
                .as_ref()
                .map(|feedback| feedback.commit),
            Some(uniserve_core::CommitRecipe::CommitGenThenWriteback)
        );
        assert!(
            separate_request
                .behavior
                .capability_needs(&separate_request.policy, std::iter::empty())
                .commit_writeback
        );

        let mut unsupported = test_capabilities();
        unsupported.generated_image_commit = Default::default();
        let error = compile_generation_request(&request, tok, &profile, &unsupported, None, 32_768)
            .expect_err("missing commit mode must fail compilation");
        assert!(
            error
                .message()
                .contains("no compatible generated-image commit mode")
        );
    }

    #[test]
    fn unsupported_sensenova_preview_resolution_is_rejected() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let body = CompileInput {
            prompt: "paint".into(),
            constraint: Some(GenerationConstraint::GenOnly),
            image: Some(ImageGenerationPolicy {
                resolution: Some("preview".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(
            GenerationRequestCompiler::new(tok, &profile)
                .build(&body)
                .is_err()
        );
    }

    #[test]
    fn client_image_values_override_profile_defaults() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let body = CompileInput {
            prompt: "paint".into(),
            constraint: Some(GenerationConstraint::GenOnly),
            image: Some(ImageGenerationPolicy {
                resolution: Some("1:1".into()),
                steps: Some(12),
                cfg_text_scale: Some(3.0),
                max_images: Some(2),
                seed: Some(7),
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = GenerationRequestCompiler::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert_eq!((request.image.width, request.image.height), (1536, 1536));
        assert_eq!(request.image.steps, 12);
        assert_eq!(request.image.cfg_text_scale, 3.0);
        assert_eq!(request.image.max_images, 2);
        assert_eq!(request.image.seed, Some(7));
        assert!(!request.image.retain_images);
        assert!(request.image.image_prompts.is_empty());
    }

    #[test]
    fn gen_only_respects_explicit_retain_images() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let body = CompileInput {
            prompt: "paint".into(),
            constraint: Some(GenerationConstraint::GenOnly),
            image: Some(ImageGenerationPolicy {
                retain_images: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = GenerationRequestCompiler::new(tok, &profile)
            .build(&body)
            .unwrap();
        assert!(request.image.retain_images);
    }

    #[test]
    fn explicit_image_prompts_are_preserved() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let body = CompileInput {
            prompt: "Generate a travel guide".into(),
            image: Some(ImageGenerationPolicy {
                prompts: vec!["custom visual prompt".into()],
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = GenerationRequestCompiler::new(tok, &profile)
            .build(&body)
            .unwrap();

        assert_eq!(request.image.image_prompts, vec!["custom visual prompt"]);
    }

    #[test]
    fn invalid_interval_is_rejected() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let body = CompileInput {
            prompt: "paint".into(),
            constraint: Some(GenerationConstraint::GenOnly),
            image: Some(ImageGenerationPolicy {
                cfg_interval: Some([0.8, 0.2]),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(
            GenerationRequestCompiler::new(tok, &profile)
                .build(&body)
                .is_err()
        );
    }

    #[test]
    fn cfg_interval_can_cover_the_full_timestep_domain() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let body = CompileInput {
            prompt: "paint".into(),
            constraint: Some(GenerationConstraint::GenOnly),
            image: Some(ImageGenerationPolicy {
                cfg_interval: Some([-1.0, 2.0]),
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = GenerationRequestCompiler::new(tok, &profile)
            .build(&body)
            .unwrap();

        assert_eq!(request.image.cfg_interval, (-1.0, 2.0));
    }

    #[test]
    fn canonical_request_bounds_input_and_generated_feedback_kv() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("sensenova-u1", &*tok)
            .expect("profile resolution")
            .expect("SenseNova profile");
        let mut request = ServeRequest::text("resource-envelope", "paint");
        request.model_context = ModelContext::Segments(vec![
            ContextSegment::Text {
                role: ContextRole::User,
                text: "paint".into(),
            },
            ContextSegment::Image(ImageInput {
                b64: test_png_b64(512, 512),
                placement: None,
            }),
        ]);
        request.generation.constraint = GenerationConstraint::Default;
        request.generation.max_tokens = Some(10);
        request.generation.image.resolution = Some("1:1".into());
        request.generation.image.max_images = Some(2);
        request.modalities.input_image = true;
        request.modalities.output_image = true;
        request.cache.namespace = Some("tenant-a".into());
        request.cache.no_store = true;

        let capabilities = test_capabilities();
        let lowered =
            compile_generation_request(&request, tok, &profile, &capabilities, None, 32_768)
                .expect("lower request");
        assert!(!lowered.cache.write);
        assert!(lowered.cache.isolation_key.is_some());
        let expected = lowered
            .resources
            .context_tokens
            .saturating_add(lowered.max_und_tokens)
            .saturating_add(256)
            .saturating_add(
                capabilities
                    .max_vae_grid_tokens
                    .saturating_add(capabilities.commit_marker_tokens)
                    .saturating_mul(2) as usize,
            );

        assert!(lowered.behavior.generated_image_feedback);
        assert_eq!(lowered.resources.max_kv_tokens, expected);
        assert_eq!(lowered.resources.encoder_cache_keys.len(), 1);
    }

    #[test]
    fn bagel_i2t_reserves_the_exact_image_kv_envelope() {
        let tok: DynTokenizer = Arc::new(SenseNovaTokenizer);
        let profile = resolve_generation_dialect_for_model("bagel", &*tok)
            .expect("profile resolution")
            .expect("BAGEL profile");
        let mut request = ServeRequest::text("bagel-i2t-resources", "describe this image");
        request.model_context = ModelContext::Segments(vec![
            ContextSegment::Text {
                role: ContextRole::User,
                text: "describe this image".into(),
            },
            ContextSegment::Image(ImageInput {
                b64: test_png_b64(512, 512),
                placement: None,
            }),
        ]);
        request.generation.constraint = GenerationConstraint::UndOnly;
        request.generation.max_tokens = Some(256);
        request.modalities.input_image = true;

        let lowered =
            compile_generation_request(&request, tok, &profile, &test_capabilities(), None, 32_768)
                .expect("lower BAGEL I2T request");
        let expected_tokens = lowered.resources.context_tokens + 256 + 1_026 + 1_371;
        let global_cap_tokens = lowered.resources.context_tokens
            + 256
            + test_capabilities().max_vae_grid_tokens as usize
            + test_capabilities().max_vit_grid_tokens as usize;

        assert_eq!(lowered.resources.max_kv_tokens, expected_tokens);
        assert!(lowered.resources.max_kv_tokens.div_ceil(64) < global_cap_tokens.div_ceil(64));
        let image_ingest = lowered
            .context
            .iter()
            .find_map(|segment| match segment {
                CoreContextSegment::Image { ingest, .. } => Some(ingest),
                CoreContextSegment::UndTokens { .. } => None,
            })
            .expect("input image segment");
        assert_eq!(
            image_ingest.step_kv_tokens,
            vec![
                uniserve_core::ImageKvEffect::Exact { tokens: 1_026 },
                uniserve_core::ImageKvEffect::Exact { tokens: 1_371 },
            ]
        );
    }
}
