//! DiffusionGemma chat generation lowered into block-diffusion requests.
//!
//! A chat request renders with the checkpoint's Gemma-4 template, which
//! writes one `<|image|>` per image part. Each expands into the image's
//! soft-token run ([`ControlTokens::expand_images`]), and the engine writes
//! the image's vision features where that run stands ([`engine_images`]).
//! The reply is generated in canvases of the checkpoint's `canvas_length`
//! tokens under the server's block-diffusion sampling ([`canvas_sampling`]),
//! and Gemma-4 parsers split its thought channel and tool calls
//! ([`Gemma4ChatOutputProcessor`]). Token-sampling controls have no meaning
//! for a denoised canvas, so a request that sets one is refused
//! ([`refuse_token_sampling_controls`]).
//!
//! [`ControlTokens::expand_images`]: crate::profile::diffusion_gemma::ControlTokens::expand_images

use std::sync::Arc;

use base64::Engine as _;
use uniserve_core::{
    CanvasSampling, GenerationRequest, ImageEncoderInput, ImageIngestStep, ImageInput,
    MultimodalInputs,
};

use crate::profile::diffusion_gemma::{DiffusionGemmaProfile, ImagePlacement};
use crate::serving::chat::{ChatOutputProcessor, Gemma4ChatOutputProcessor};
use crate::serving::text::TextDecodeOptions;
use crate::serving::{
    InputProcessor, OutputProcessorPolicy, PromptInput, SamplingConfig, ServeError, ServeRequestId,
    StopConfig, TokenizeError, media,
};

/// Splits a prompt whose images stand as soft-token runs into the engine's
/// prompt tokens and image inputs.
///
/// Each run leaves the prompt, and its image enters at the count of prompt
/// tokens before it: the engine writes the image's vision features there,
/// one KV entry and one position per soft token. `images` are indexed by
/// [`ImagePlacement::source`], and `placements` are in prompt order.
pub(crate) fn engine_images(
    token_ids: &[u32],
    placements: &[ImagePlacement],
    images: &[media::ImageInput],
) -> (Vec<u32>, Vec<ImageInput>) {
    let mut prompt = Vec::with_capacity(token_ids.len());
    let mut inputs = Vec::with_capacity(placements.len());
    let mut next = 0;
    for placement in placements {
        let offset = placement.offset as usize;
        prompt.extend_from_slice(&token_ids[next..offset]);
        next = offset + placement.soft_tokens as usize;

        let image = &images[placement.source];
        inputs.push(ImageInput {
            hash: image.hash(),
            b64: base64::engine::general_purpose::STANDARD.encode(image.bytes()),
            position: u32::try_from(prompt.len()).unwrap_or(u32::MAX),
            num_positions: placement.soft_tokens,
            encoders: vec![ImageEncoderInput {
                encoder: ImageIngestStep::VitEncode,
                num_kv_tokens: Some(placement.soft_tokens),
                max_kv_tokens: None,
            }],
        });
    }
    prompt.extend_from_slice(&token_ids[next..]);
    (prompt, inputs)
}

/// Returns the block-diffusion sampling of every canvas the server
/// generates: the checkpoint's canvas length under its sampler defaults, as
/// the server's configuration overrides them.
pub(crate) fn canvas_sampling(profile: &DiffusionGemmaProfile) -> CanvasSampling {
    let denoising = &profile.denoising;
    CanvasSampling {
        canvas_length: profile.canvas_length,
        max_steps: denoising.max_denoising_steps,
        entropy_bound: denoising.entropy_bound,
        t_min: denoising.t_min,
        t_max: denoising.t_max,
        confidence_threshold: denoising.confidence_threshold,
        stability_threshold: denoising.stability_threshold,
    }
}

/// Why block diffusion defines no temperature or truncation control.
const SCHEDULED_SAMPLING: &str = "block diffusion samples every canvas position from the full \
     vocabulary at the temperature schedule the server configures";
/// Why block diffusion defines no token penalty.
const NO_PENALTIES: &str = "block diffusion applies no token penalties";
/// Why block diffusion defines no token constraint.
const UNCONSTRAINED: &str = "block diffusion does not constrain canvas tokens";
/// Why block diffusion reports no log-probabilities.
const NO_LOGPROBS: &str = "committed canvases carry no token log-probabilities";
/// Why block diffusion defines no end-of-sequence control.
const FIRST_EOS: &str = "a canvas ends the reply at its first end-of-sequence token";

/// Refuses a request that sets a token-sampling control, naming the first.
///
/// Block diffusion samples whole canvases under the server's sampling, so
/// temperature, truncation, penalties, logit bias, token constraints,
/// log-probabilities, a minimum length, and ignoring EOS have no meaning
/// there; a request that sets one is refused rather than served without it.
///
/// # Errors
///
/// Returns [`ServeError::UnsupportedSamplingControl`] for the first control
/// the request sets.
pub(crate) fn refuse_token_sampling_controls(
    request_id: &ServeRequestId,
    sampling: &SamplingConfig,
    stop: &StopConfig,
) -> Result<(), ServeError> {
    let controls = [
        (
            "temperature",
            sampling.temperature.is_some(),
            SCHEDULED_SAMPLING,
        ),
        ("top_p", sampling.top_p.is_some(), SCHEDULED_SAMPLING),
        ("top_k", sampling.top_k.is_some(), SCHEDULED_SAMPLING),
        ("min_p", sampling.min_p.is_some(), SCHEDULED_SAMPLING),
        (
            "frequency_penalty",
            sampling.frequency_penalty.is_some(),
            NO_PENALTIES,
        ),
        (
            "presence_penalty",
            sampling.presence_penalty.is_some(),
            NO_PENALTIES,
        ),
        (
            "repetition_penalty",
            sampling.repetition_penalty.is_some(),
            NO_PENALTIES,
        ),
        ("logit_bias", stop.logit_bias.is_some(), UNCONSTRAINED),
        (
            "allowed_token_ids",
            stop.allowed_token_ids.is_some(),
            UNCONSTRAINED,
        ),
        ("bad_words", !stop.bad_words.is_empty(), UNCONSTRAINED),
        ("logprobs", stop.logprobs.is_some(), NO_LOGPROBS),
        (
            "prompt_logprobs",
            stop.prompt_logprobs.is_some(),
            NO_LOGPROBS,
        ),
        ("min_tokens", sampling.min_tokens.is_some(), FIRST_EOS),
        ("ignore_eos", sampling.ignore_eos, FIRST_EOS),
    ];
    match controls.into_iter().find(|(_, set, _)| *set) {
        Some((control, _, reason)) => Err(ServeError::UnsupportedSamplingControl {
            request_id: request_id.clone(),
            control,
            reason,
        }),
        None => Ok(()),
    }
}

impl InputProcessor {
    /// Renders a DiffusionGemma chat prompt with its images and fills the
    /// request's tokens, images, length, seed, and block-diffusion sampling.
    ///
    /// The reply may use whole canvases of the context the prompt leaves;
    /// `max_completion_tokens` then truncates the canvas that reaches it. An
    /// unseeded request draws a fresh random seed, so its canvases' draws
    /// are independent of every other request's.
    ///
    /// # Errors
    ///
    /// Returns `TokenizeError::Invalid` for a text prompt, an image with no
    /// area, request text that encodes image tokens of its own, or a prompt
    /// that leaves no room for one canvas; chat validation, rendering,
    /// tokenizer, and parser construction errors propagate.
    pub(super) fn preprocess_diffusion_gemma_input(
        &self,
        profile: &DiffusionGemmaProfile,
        prompt: PromptInput,
        images: &[media::ImageInput],
        sampling: &SamplingConfig,
        generation: &mut GenerationRequest,
        decode: &mut TextDecodeOptions,
    ) -> Result<OutputProcessorPolicy, TokenizeError> {
        let PromptInput::Chat(mut chat) = prompt else {
            return Err(TokenizeError::Invalid(
                "DiffusionGemma generates replies to chat prompts only".to_owned(),
            ));
        };
        chat.decode_options = decode.clone();
        chat.validate()?;
        let processor = Gemma4ChatOutputProcessor::new(
            &mut chat,
            Arc::clone(&self.tokenizer),
            self.parse_reasoning,
        )?;
        let rendered = self
            .renderer
            .as_ref()
            .ok_or(crate::serving::chat::Error::MissingChatTemplate)?
            .render(&chat)?;
        let rendered = self.tokenizer.encode(&rendered, false)?;

        // Each image's soft-token count follows the Gemma-4 image processor.
        let soft_tokens = images
            .iter()
            .map(|image| profile.images.soft_tokens(image.width(), image.height()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(TokenizeError::Invalid)?;
        let (expanded, placements) = profile
            .tokens
            .expand_images(&rendered, &soft_tokens)
            .map_err(|mismatch| TokenizeError::Invalid(mismatch.to_string()))?;

        // Canvases occupy whole blocks of positions after the prompt, so the
        // reply may use only the whole canvases that fit the context.
        let prompt_len = u32::try_from(expanded.len()).unwrap_or(u32::MAX);
        let max_model_len = self.config.max_model_tokens();
        let canvas = profile.canvas_length;
        let room = max_model_len.saturating_sub(prompt_len) / canvas * canvas;
        if room == 0 {
            return Err(TokenizeError::Invalid(format!(
                "this model's maximum context length is {max_model_len} tokens; the prompt's \
                 {prompt_len} tokens leave no room for one {canvas}-token canvas"
            )));
        }
        let max_tokens = crate::serving::text::resolve_max_tokens(
            sampling.max_tokens,
            self.config.sampling_defaults.max_output_tokens,
            Some(prompt_len + room),
            prompt_len,
        )?;

        let (prompt_token_ids, inputs) = engine_images(&expanded, &placements, images);
        generation.prompt_token_ids = prompt_token_ids;
        generation.multimodal_inputs = MultimodalInputs { images: inputs };
        generation.max_und_tokens = max_tokens as usize;
        generation.include_stop_token = decode.include_stop_str_in_output;
        generation.sampling.seed = Some(
            generation
                .sampling
                .seed
                .unwrap_or_else(|| uuid::Uuid::new_v4().as_u64_pair().0),
        );
        generation.canvas = Some(canvas_sampling(profile));
        decode.skip_special_tokens = chat.decode_options.skip_special_tokens;
        Ok(OutputProcessorPolicy::Chat(ChatOutputProcessor::Gemma4(
            processor,
        )))
    }
}

#[cfg(test)]
mod tests {
    //! DiffusionGemma chat replies generated in canvases, through real
    //! preprocessing, the block-diffusion engine, and chat output assembly.
    //!
    //! The engine simulator stops each canvas after a few steps and fills it with
    //! synthetic text tokens (ids in `1000..6000`) that end in EOS after
    //! `TEXT_LEN` tokens. The tokenizer decodes every synthetic token to its own
    //! word, so each delta's text follows from its token ids.

    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::collections::{BTreeSet, HashMap};
    use std::sync::Arc;

    use crate::engine_client::EngineClient;
    use crate::profile::diffusion_gemma::{
        ControlTokens, DenoisingDefaults, DiffusionGemmaProfile, PatchBudget,
    };
    use crate::profile::tokenizer::HuggingFaceTokenizer;
    use crate::profile::{ModelConfig, ModelParameters, SamplingDefaults};
    use crate::serving::chat::{ChatTemplateContentFormatOption, HfChatRenderer};
    use crate::serving::media::{ImageFetchPolicy, ImageFetcher};
    use crate::serving::{
        FinishStatus, InputProcessor, RequestOutput, ServedSamplingControl, ServingRuntime,
    };
    use futures::StreamExt;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::{AddedToken, Tokenizer};
    use uniserve_core::ModelDtype;
    use uniserve_engine::{EngineConfig, SimEngine, SimExecutor};

    /// Tokens of one canvas.
    const CANVAS: u32 = 16;
    /// Synthetic reply tokens before the simulator's EOS.
    const TEXT_LEN: usize = 40;

    /// Builds a DiffusionGemma serving runtime over the simulator.
    ///
    /// The tokenizer maps ASCII code points to themselves and every larger id
    /// below 6000 to the word `w<id>`, followed by the control tokens. The chat
    /// template renders the first message's content.
    fn runtime() -> (ServingRuntime, Arc<HuggingFaceTokenizer>) {
        let vocabulary = (0_u32..6_000)
            .map(|id| {
                let token = if id < 128 {
                    char::from_u32(id).unwrap().to_string()
                } else {
                    format!("w{id}")
                };
                (token, id)
            })
            .collect::<Vocab>();
        let mut tokenizer = Tokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocabulary, Vec::new())
                .build()
                .unwrap(),
        );
        tokenizer.add_special_tokens(
            &["<|image>", "<|image|>", "<image|>"].map(|token| AddedToken::from(token, true)),
        );
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();
        let tokenizer = Arc::new(HuggingFaceTokenizer::new(&path).unwrap());
        let id = |token: &str| tokenizer.token_to_id(token).unwrap();

        let profile = DiffusionGemmaProfile {
            canvas_length: CANVAS,
            tokens: ControlTokens {
                mask: 4,
                pad: 0,
                turn_end: 5,
                image: id("<|image|>"),
                image_start: id("<|image>"),
                image_end: id("<image|>"),
            },
            images: PatchBudget {
                patch_size: 16,
                pooling_kernel_size: 3,
                max_soft_tokens: 280,
            },
            denoising: DenoisingDefaults {
                max_denoising_steps: 4,
                entropy_bound: 0.1,
                t_min: 0.4,
                t_max: 0.8,
                confidence_threshold: 0.005,
                stability_threshold: 1,
            },
            quantized: false,
        };
        let model = ModelConfig {
            served_name: "sim-model".to_string(),
            parameters: ModelParameters::DiffusionGemma(profile),
            sampling_defaults: SamplingDefaults::default(),
            max_model_tokens: Some(4096),
            primary_eos_token_id: None,
            eos_token_ids: BTreeSet::new(),
        };

        let mut worker = SimEngine::new();
        worker.set_text_len(TEXT_LEN);
        let mut config = EngineConfig::sim("sim-model");
        config.runtime_family = model.runtime_family();
        config.generation_limits = model.generation_limits(ModelDtype::BFloat16);
        let client = Arc::new(
            EngineClient::connect_with_executor(config, Box::new(SimExecutor::new(worker)))
                .unwrap(),
        );
        let processor = InputProcessor::new(
            model,
            Arc::clone(&tokenizer),
            Some(
                HfChatRenderer::new(
                    Some("{{ messages[0].content }}".to_string()),
                    HashMap::new(),
                    ChatTemplateContentFormatOption::String,
                )
                .unwrap(),
            ),
            crate::serving::WorkerCapabilities {
                limits: client.generation_limits(),
                sampling_controls: ServedSamplingControl::ALL.to_vec(),
                max_model_tokens: 4096,
            },
            None,
            false,
        )
        .unwrap();

        let images = ImageFetcher::new(ImageFetchPolicy::default()).unwrap();
        (
            ServingRuntime::new(processor, client, images, false),
            tokenizer,
        )
    }

    /// One chat reply as the chat stream published it.
    struct Reply {
        /// Visible text deltas, in order.
        texts: Vec<String>,
        /// Token-id deltas, in order.
        token_ids: Vec<Vec<u32>>,
        finish: FinishStatus,
        /// Visible output tokens the usage reports.
        tokens: u32,
    }

    /// Runs one chat request with its token ids returned. The chat stream
    /// publishes parsed text and token ids as separate deltas.
    async fn chat(runtime: &ServingRuntime, id: &str, max_completion_tokens: u32) -> Reply {
        let request = serde_json::from_value(serde_json::json!({
            "model": "sim-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": max_completion_tokens,
            "return_token_ids": true,
        }))
        .unwrap();
        let mut stream = runtime.generate_chat(id.into(), request).await.unwrap();
        let (mut texts, mut token_ids, mut usage, mut finish) =
            (Vec::new(), Vec::new(), None, None);
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                RequestOutput::TextDelta {
                    text,
                    token_ids: ids,
                    ..
                } => {
                    if !text.is_empty() {
                        texts.push(text);
                    }
                    if !ids.is_empty() {
                        token_ids.push(ids);
                    }
                }
                RequestOutput::Usage {
                    visible_output_tokens,
                    ..
                } => usage = Some(visible_output_tokens),
                RequestOutput::Finished { reason, .. } => finish = Some(reason),
                _ => {}
            }
        }
        Reply {
            texts,
            token_ids,
            finish: finish.unwrap(),
            tokens: usage.unwrap(),
        }
    }

    /// Asserts that the reply's text deltas decode its token-id deltas one
    /// for one, and that those are synthetic reply tokens; returns the
    /// deltas' token counts.
    fn token_counts(tokenizer: &HuggingFaceTokenizer, reply: &Reply) -> Vec<usize> {
        let decoded: Vec<String> = reply
            .token_ids
            .iter()
            .map(|ids| tokenizer.decode(ids, false).unwrap())
            .collect();
        assert_eq!(reply.texts, decoded);
        assert!(
            reply
                .token_ids
                .iter()
                .flatten()
                .all(|id| (1_000..6_000).contains(id))
        );
        reply.token_ids.iter().map(Vec::len).collect()
    }

    /// Each committed canvas reaches the chat stream as one delta, and the reply
    /// ends at the canvas holding EOS.
    #[tokio::test]
    async fn a_chat_reply_streams_one_delta_per_canvas_until_eos() {
        let (runtime, tokenizer) = runtime();
        let reply = chat(&runtime, "blocks", 100).await;

        assert_eq!(token_counts(&tokenizer, &reply), [16, 16, 8]);
        assert!(
            matches!(reply.finish, FinishStatus::Stop { .. }),
            "{:?}",
            reply.finish
        );
        assert_eq!(reply.tokens, TEXT_LEN as u32);
        runtime.shutdown().await.unwrap();
    }

    /// `max_completion_tokens` truncates the canvas that reaches it.
    #[tokio::test]
    async fn max_completion_tokens_truncates_the_reply_inside_a_canvas() {
        let (runtime, tokenizer) = runtime();
        let reply = chat(&runtime, "truncated", 20).await;

        assert_eq!(token_counts(&tokenizer, &reply), [16, 4]);
        assert_eq!(reply.finish, FinishStatus::Length);
        assert_eq!(reply.tokens, 20);
        runtime.shutdown().await.unwrap();
    }
}
