#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Model-profile prompt preprocessing and generation-policy integration behavior.

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use tempfile::tempdir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
use uniserve_core::{GenerationLimits, ImageIngestStep};
use uniserve_server::profile::assets::ResolvedModelFiles;
use uniserve_server::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use uniserve_server::profile::{ModelConfig, ModelDescription};
use uniserve_server::serving::chat::{ChatTemplateContentFormatOption, HfChatRenderer};
use uniserve_server::serving::{InputProcessor, ServeRequestId};

/// Base64 payload of a 1x1 PNG image.
const PNG_1X1: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
/// Minimal ChatML template: one `<|im_start|>role\ncontent<|im_end|>` turn per
/// message, then an open assistant turn when a generation prompt is requested.
const CHAT_TEMPLATE: &str = "{%- for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
/// Special tokens added to the synthetic tokenizer: ChatML turn delimiters,
/// the Bagel (`<|vision_*|>`) and SenseNova (`<img>`) image markers, and
/// reasoning and answer delimiters.
const SPECIAL_TOKENS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<img>",
    "</img>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
];

/// Resolves `description` over a synthetic checkpoint with worker limits that
/// cover every generation feature; see `try_resolved_model`.
fn resolved_model(
    description: ModelDescription,
    model_type: &str,
) -> (tempfile::TempDir, DynTokenizer, InputProcessor) {
    try_resolved_model(description, model_type, runtime_limits()).unwrap()
}

/// Writes a synthetic checkpoint to a temporary directory and binds an
/// `InputProcessor` for `description` over it with the given worker limits.
///
/// The tokenizer maps each ASCII code point from 1 to 127 to one token whose
/// ID is the code point (no merges), plus `SPECIAL_TOKENS`, so outside the
/// special tokens one character is one prompt token. `generation_config.json`
/// sets a 128-token default generation length. The returned directory owns
/// the checkpoint files.
///
/// # Errors
///
/// Returns the `InputProcessor::new` error, for example when `limits` do not
/// cover a feature the profile needs. Other setup failures panic.
fn try_resolved_model(
    description: ModelDescription,
    model_type: &str,
    limits: GenerationLimits,
) -> uniserve_server::serving::Result<(tempfile::TempDir, DynTokenizer, InputProcessor)> {
    let directory = tempdir().unwrap();

    let mut vocab = Vocab::from_iter([("<unk>".to_string(), 0_u32)]);
    for codepoint in 1_u32..=127 {
        vocab.insert(char::from_u32(codepoint).unwrap().to_string(), codepoint);
    }
    let tokenizer_model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();
    let mut tokenizer_builder = TokenizerBuilder::new(tokenizer_model);
    tokenizer_builder.add_special_tokens(
        &SPECIAL_TOKENS
            .iter()
            .map(|token| AddedToken::from(*token, true))
            .collect::<Vec<_>>(),
    );
    let tokenizer_path = directory.path().join("tokenizer.json");
    tokenizer_builder.save(&tokenizer_path, false).unwrap();

    let config_path = directory.path().join("config.json");
    fs::write(
        &config_path,
        format!(
            r#"{{"model_type":"{model_type}","max_position_embeddings":4096,"num_attention_heads":8}}"#
        ),
    )
    .unwrap();
    let tokenizer_config_path = directory.path().join("tokenizer_config.json");
    fs::write(
        &tokenizer_config_path,
        format!(
            "{{\"eos_token\":\"<|im_end|>\",\"chat_template\":{}}}",
            serde_json::to_string(CHAT_TEMPLATE).unwrap()
        ),
    )
    .unwrap();
    let generation_config_path = directory.path().join("generation_config.json");
    fs::write(
        &generation_config_path,
        r#"{"eos_token_id":2,"max_new_tokens":128}"#,
    )
    .unwrap();
    let files = ResolvedModelFiles {
        tokenizer_path,
        tokenizer_config_path: Some(tokenizer_config_path),
        generation_config_path: Some(generation_config_path),
        preprocessor_config_path: None,
        chat_template_path: None,
        config_path: Some(config_path),
    };
    // Bagel reads its latent position grid side from the primary checkpoint's
    // safetensors header; a header-only file declaring the published 64x64
    // table suffices.
    if description == ModelDescription::Bagel {
        let header = serde_json::to_vec(&serde_json::json!({
            "latent_pos_embed.pos_embed": {
                "dtype": "BF16", "shape": [4096, 1], "data_offsets": [0, 8192]
            }
        }))
        .unwrap();
        let mut checkpoint = (header.len() as u64).to_le_bytes().to_vec();
        checkpoint.extend(header);
        checkpoint.resize(checkpoint.len() + 8192, 0);
        fs::write(directory.path().join("ema.safetensors"), checkpoint).unwrap();
    }

    let tokenizer: DynTokenizer =
        Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap());
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    // A diffusers pipeline checkpoint has no root `config.json`, so MiniMax H3
    // is built by `ModelConfig::from_pipeline`, here with the server's default
    // 15-second maximum video duration.
    let config = match description {
        ModelDescription::MiniMaxH3 => {
            ModelConfig::from_pipeline(description.id(), description, 15.0, Some(4096)).unwrap()
        }
        _ => tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(ModelConfig::from_files(
                description.id(),
                directory.path().to_str().unwrap(),
                &files,
                None,
                tokenizer.as_ref(),
            ))
            .unwrap(),
    };
    let model = InputProcessor::new(
        config,
        Arc::clone(&tokenizer),
        Some(renderer),
        uniserve_server::serving::WorkerCapabilities {
            limits,
            sampling_controls: uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
            max_model_tokens: 4096,
            denoise_steps: 4,
        },
        true,
    )?;
    Ok((directory, tokenizer, model))
}

/// Worker limits that cover every generation feature with capacities no test
/// request reaches.
fn runtime_limits() -> GenerationLimits {
    GenerationLimits {
        features: uniserve_core::GenerationFeatures::all(),
        max_latent_units: 1_000_000,
        latent_downsample: 16,
        max_vae_grid_tokens: 1_000_000,
        max_vit_grid_tokens: 1_000_000,
        max_latent_feature_bytes: u64::MAX,
        max_vision_feature_bytes: u64::MAX,
        commit_marker_tokens: 2,
        max_cfg_branches: 3,
        encoder_cache_entries: 32,
    }
}

/// Chat request with one input image between two text parts.
///
/// The leading text contains a literal `</img>`, which the synthetic tokenizer
/// encodes as the SenseNova end-of-image token. The top-level `seed` (9) and
/// the `image_config` seed (17) differ so tests can tell which one wins.
fn image_chat_request(model: &str) -> uniserve_server::openai::ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "literal </img> before "},
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{PNG_1X1}")}},
            {"type": "text", "text": " after"}
        ]}],
        "seed": 9,
        "image_config": {"seed": 17}
    }))
    .unwrap()
}

/// `InputProcessor::new` refuses to bind a profile whose worker limits lack a
/// feature the profile's configured branches need, and names the missing
/// feature by its `GenerationFeatures::name` diagnostic. SenseNova appears
/// twice because it needs both ViT encoding and image generation.
#[test]
fn model_resolution_requires_every_configured_runtime_branch() {
    type ChangeLimits = fn(&mut GenerationLimits);
    let cases: [(ModelDescription, &'static str, &'static str, ChangeLimits); 4] = [
        (
            ModelDescription::Qwen3,
            "qwen3",
            "runtime_und_execution",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(uniserve_core::GenerationFeatures::UNDERSTANDING);
            },
        ),
        (
            ModelDescription::SenseNova,
            "neo_chat",
            "runtime_vit_encode",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(uniserve_core::GenerationFeatures::VISION_ENCODE);
            },
        ),
        (
            ModelDescription::Bagel,
            "bagel",
            "runtime_vae_encode",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(uniserve_core::GenerationFeatures::LATENT_ENCODE);
            },
        ),
        (
            ModelDescription::SenseNova,
            "neo_chat",
            "runtime_gen_denoise",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(uniserve_core::GenerationFeatures::IMAGE_GENERATION);
            },
        ),
    ];

    for (description, model_type, required, remove) in cases {
        let mut limits = runtime_limits();
        remove(&mut limits);
        let error = match try_resolved_model(description, model_type, limits) {
            Ok(_) => panic!("incomplete worker limits must fail model resolution"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(required), "got: {error}");
    }
}

/// The literal `</img>` in the user text also encodes to the end-of-image
/// token, so the prompt holds two of them. The image's position is the index
/// of the end-of-image token of the marker rendered for its slot (the second
/// occurrence), not the first one found in the text.
#[test]
fn sensenova_places_the_input_image_at_its_rendered_slot() {
    let (_directory, tokenizer, model) = resolved_model(ModelDescription::SenseNova, "neo_chat");
    let request = image_chat_request(model.served_model_name());
    let (generation, response) = model
        .preprocess_chat_request(ServeRequestId::new("image-params"), request)
        .unwrap();

    // The `image_config` seed takes precedence over the text sampling seed.
    assert_eq!(generation.sampling.seed, Some(17));
    assert_eq!(generation.image.seed, Some(17));

    let end_image = tokenizer.token_to_id("</img>").unwrap();
    let marker_positions = response
        .prompt_token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (*token == end_image).then_some(index as u32))
        .collect::<Vec<_>>();
    assert_eq!(marker_positions.len(), 2);
    let image = &generation.multimodal_inputs.images[0];
    let (position, steps) = (
        image.position,
        image
            .encoders
            .iter()
            .map(|input| input.encoder)
            .collect::<Vec<_>>(),
    );
    assert_eq!(position, marker_positions[1]);
    assert_eq!(steps, &[ImageIngestStep::VitEncode]);

    // Feedback for the default 16:9 canvas (2048x1152) is a 64x36 ViT grid at
    // 32 pixels per token plus one marker token.
    assert_eq!(
        generation
            .image_generation
            .feedback_encoders
            .iter()
            .map(|input| input.num_kv_tokens)
            .collect::<Vec<_>>(),
        vec![Some(2305)]
    );
}

#[test]
fn bagel_places_the_input_image_between_surrounding_chat_text() {
    let (_directory, _tokenizer, model) = resolved_model(ModelDescription::Bagel, "bagel");
    let request = image_chat_request(model.served_model_name());
    let (generation, response) = model
        .preprocess_chat_request(ServeRequestId::new("image-params"), request)
        .unwrap();
    assert_eq!(generation.sampling.seed, Some(17));
    assert_eq!(generation.image.seed, Some(17));
    let image = &generation.multimodal_inputs.images[0];
    let (position, steps) = (
        image.position,
        image
            .encoders
            .iter()
            .map(|input| input.encoder)
            .collect::<Vec<_>>(),
    );

    // Bagel removes the slot and places the image at the token count of the
    // text before it, so text on both sides keeps it strictly inside the
    // prompt. The Bagel profile configures a VAE and a ViT encoder input for
    // each input image, in that order.
    assert!(position > 0);
    assert!((position as usize) < response.prompt_token_ids.len());
    assert_eq!(
        steps,
        &[ImageIngestStep::VaeEncode, ImageIngestStep::VitEncode]
    );
}

#[test]
fn minimax_video_preprocessing_preserves_tokens_seed_and_frame_alignment() {
    let (_directory, tokenizer, model) = resolved_model(ModelDescription::MiniMaxH3, "minimax_h3");
    let prompt = "exact token sequence";
    let expected = tokenizer.encode(prompt, false).unwrap();

    let request = model
        .preprocess_video_request(
            &ServeRequestId::new("video"),
            uniserve_server::openai::VideoGenerationRequest {
                model: "minimax_h3".to_string(),
                prompt: prompt.to_string(),
                seconds: Some(1.0),
                seed: 17,
            },
        )
        .unwrap();

    // The prompt is tokenized without chat framing. One second is 24 frames
    // at 24 fps, which rounds up to the next count of the form `5 + 17k`:
    // 39 frames, decoded as two 17-frame video units plus a 5-frame tail.
    assert_eq!(request.prompt_token_ids, expected);
    assert_eq!(request.sampling.seed, 17);
    assert_eq!(request.sampling.num_frames, 39);
    assert_eq!(request.sampling.video_units, 2);

    // Rejections: a model name other than the served one, a whitespace-only
    // prompt, and a duration that is not finite and positive.
    let invalid = |model_name: &str, prompt: &str, seconds| {
        model
            .preprocess_video_request(
                &ServeRequestId::new("video"),
                uniserve_server::openai::VideoGenerationRequest {
                    model: model_name.to_string(),
                    prompt: prompt.to_string(),
                    seconds: Some(seconds),
                    seed: 17,
                },
            )
            .unwrap_err()
    };
    assert_eq!(
        invalid("another-model", prompt, 1.0),
        uniserve_server::openai::ApiError::ModelNotFound {
            model: "another-model".to_string()
        },
    );
    assert!(matches!(
        invalid("minimax_h3", "  ", 1.0),
        uniserve_server::openai::ApiError::InvalidRequest {
            param: Some("prompt"),
            ..
        },
    ));
    for seconds in [0.0, -1.0, f64::NAN] {
        assert!(matches!(
            invalid("minimax_h3", prompt, seconds),
            uniserve_server::openai::ApiError::InvalidRequest { .. },
        ));
    }
}

/// `InputProcessor::new` replaces the checkpoint's context length with the
/// worker's `max_model_tokens`, and text preprocessing enforces that bound.
#[test]
fn worker_context_capacity_limits_preprocessed_requests() {
    let (_directory, tokenizer, loaded) = resolved_model(ModelDescription::Qwen3, "qwen3");
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    let processor = InputProcessor::new(
        loaded.config().clone(),
        tokenizer,
        Some(renderer),
        uniserve_server::serving::WorkerCapabilities {
            limits: runtime_limits(),
            sampling_controls: uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
            max_model_tokens: 8,
            denoise_steps: 0,
        },
        true,
    )
    .unwrap();

    assert_eq!(processor.config().max_model_tokens, Some(8));
    let request = uniserve_server::serving::TextPromptRequest::new(
        "context-capacity",
        "This prompt exceeds eight tokens",
    );
    assert!(matches!(
        processor.preprocess_text_request(request),
        Err(uniserve_server::serving::ServeError::Tokenize {
            source: uniserve_server::serving::TokenizeError::Text(
                uniserve_server::serving::text::Error::PromptTooLong {
                    max_model_len: 8,
                    ..
                }
            ),
            ..
        })
    ));
}

/// Defaults fill only omitted controls: an explicit zero temperature stays
/// zero rather than taking the default 1.0, while an explicit zero completion
/// length is rejected rather than replaced by the 128-token checkpoint default.
#[test]
fn sampling_defaults_preserve_explicit_zero_controls() {
    let (_directory, _tokenizer, model) = resolved_model(ModelDescription::Qwen3, "qwen3");
    let mut request: uniserve_server::openai::ChatCompletionRequest =
        serde_json::from_value(serde_json::json!({
            "model": "qwen3", "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    let (default, _) = model
        .preprocess_chat_request(ServeRequestId::new("sampling-defaults"), request.clone())
        .unwrap();
    assert_eq!(default.max_und_tokens, 128);
    assert_eq!(default.sampling.temperature, 1.0);

    request.temperature = Some(0.0);
    let (greedy, _) = model
        .preprocess_chat_request(ServeRequestId::new("sampling-defaults"), request.clone())
        .unwrap();
    assert_eq!(greedy.sampling.temperature, 0.0);
    assert_eq!(greedy.max_und_tokens, 128);

    request.max_completion_tokens = Some(0);
    assert!(matches!(
        model.preprocess_chat_request(ServeRequestId::new("sampling-defaults"), request),
        Err(uniserve_server::openai::ApiError::InvalidRequest { .. })
    ));
}

/// The image API produces an image-only (`GenOnly`) request that keeps the
/// requested canvas, steps, seed, guidance scale, and negative prompt.
/// SenseNova accepts only its resolution buckets, and 1536x1536 is one of
/// them.
#[test]
fn image_api_preserves_requested_dimensions_seed_and_guidance() {
    for (description, model_type, width, height) in [
        (ModelDescription::SenseNova, "neo_chat", 1536, 1536),
        (ModelDescription::Bagel, "bagel", 512, 512),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        let request: uniserve_server::openai::ImageGenerationRequest =
            serde_json::from_value(serde_json::json!({
                "model": description.id(),
                "prompt": "a blue bird",
                "negative_prompt": "blur",
                "size": format!("{width}x{height}"),
                "steps": 4,
                "seed": 17,
                "guidance_scale": 7.5,
                "n": 1
            }))
            .unwrap();
        let (generation, _) = processor
            .preprocess_image_request(ServeRequestId::new("image-api"), request)
            .unwrap();
        assert_eq!(
            generation.constraint,
            uniserve_core::GenerationConstraint::GenOnly
        );
        assert_eq!(
            (generation.image.width, generation.image.height),
            (width, height)
        );
        assert_eq!(generation.image.steps, 4);
        assert_eq!(generation.image.max_images, 1);
        assert_eq!(generation.image.seed, Some(17));
        assert_eq!(generation.sampling.seed, Some(17));
        assert_eq!(generation.image.cfg_text_scale, 7.5);
        assert_eq!(generation.image.negative_prompt, "blur");
    }
}

/// SenseNova bounds each input image by its share of a pixel budget, so the
/// same image preprocessed under a different bound is a different encoder
/// product. Its encoder-cache identity changes once five input images lower
/// the bound, while requests whose images keep the same bound share it.
#[test]
fn sensenova_image_cache_identity_follows_its_pixel_bound() {
    let (_directory, _tokenizer, processor) =
        resolved_model(ModelDescription::SenseNova, "neo_chat");
    let first_image_hash = |count: usize| {
        let content = (0..count)
            .map(|_| {
                serde_json::json!({
                    "type": "image_url",
                    "image_url": {"url": format!("data:image/png;base64,{PNG_1X1}")}
                })
            })
            .chain([serde_json::json!({"type": "text", "text": "compare"})])
            .collect::<Vec<_>>();
        let request: uniserve_server::openai::ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "sensenova",
                "messages": [{"role": "user", "content": content}]
            }))
            .unwrap();
        let (generation, _) = processor
            .preprocess_chat_request(ServeRequestId::new("image-bound"), request)
            .unwrap();
        generation.multimodal_inputs.images[0].hash
    };

    // One to four input images keep the 2048x2048 bound; a fifth lowers it.
    assert_eq!(first_image_hash(1), first_image_hash(4));
    assert_ne!(first_image_hash(4), first_image_hash(5));
}

/// Bagel's learned latent position table covers 64 latent patches per side
/// in the synthetic checkpoint, 1024 pixels at the 16-pixel latent stride. A
/// canvas with a longer side, which would index past a table row or past the
/// table, is an invalid request like other unsupported sizes, while a side
/// of exactly 64 patches is accepted.
#[test]
fn bagel_rejects_a_canvas_side_beyond_its_latent_position_table() {
    let (_directory, _tokenizer, processor) = resolved_model(ModelDescription::Bagel, "bagel");
    let request = |size: &str| -> uniserve_server::openai::ImageGenerationRequest {
        serde_json::from_value(serde_json::json!({
            "model": "bagel",
            "prompt": "a banner",
            "size": size
        }))
        .unwrap()
    };

    for size in ["1040x16", "16x1040"] {
        let Err(error) =
            processor.preprocess_image_request(ServeRequestId::new("latent-side"), request(size))
        else {
            panic!("{size} was admitted");
        };
        assert!(
            matches!(
                error,
                uniserve_server::openai::ApiError::InvalidRequest { .. }
            ),
            "{size}: {error:?}"
        );
    }
    for size in ["1024x16", "16x1024"] {
        let (generation, _) = processor
            .preprocess_image_request(ServeRequestId::new("latent-side"), request(size))
            .unwrap();
        assert_eq!(
            format!("{}x{}", generation.image.width, generation.image.height),
            size
        );
    }
}

/// Cache namespace and salt select a stable isolation partition, the bypass
/// flags disable prefix-cache reads and writes without changing the
/// partition, and prompt logprobs disable reads only, because a prefix-cache
/// hit would skip the prompt positions whose logprobs are requested.
#[test]
fn cache_controls_preserve_isolation_and_prompt_logprob_requirements() {
    use uniserve_server::serving::TextPromptRequest;

    for (description, model_type) in [
        (ModelDescription::Qwen3, "qwen3"),
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        let mut request = TextPromptRequest::new("cache-controls", "hello");

        let (shared, _) = processor.preprocess_text_request(request.clone()).unwrap();
        assert_eq!(shared.cache.isolation_key, None);
        assert!(shared.cache.read && shared.cache.write);

        // The same namespace and salt always produce the same key.
        request.cache_namespace = Some("ab".to_string());
        request.cache_salt = Some("c".to_string());
        let (isolated, _) = processor.preprocess_text_request(request.clone()).unwrap();
        assert!(isolated.cache.isolation_key.is_some());
        let (repeat, _) = processor.preprocess_text_request(request.clone()).unwrap();
        assert_eq!(repeat.cache.isolation_key, isolated.cache.isolation_key);

        // Namespace and salt are separate coordinates, even when their
        // concatenated text is identical.
        request.cache_namespace = Some("a".to_string());
        request.cache_salt = Some("bc".to_string());
        let (other, _) = processor.preprocess_text_request(request.clone()).unwrap();
        assert_ne!(other.cache.isolation_key, isolated.cache.isolation_key);

        request.bypass_cache_read = true;
        request.no_cache_store = true;
        let (disabled, response) = processor.preprocess_text_request(request.clone()).unwrap();
        assert!(!disabled.cache.read && !disabled.cache.write);
        assert!(!response.cache.read_enabled && !response.cache.write_enabled);
        assert_eq!(disabled.cache.isolation_key, other.cache.isolation_key);

        request.bypass_cache_read = false;
        request.no_cache_store = false;
        request.stop.prompt_logprobs = Some(0);
        let (logprobs, response) = processor.preprocess_text_request(request).unwrap();
        assert!(!logprobs.cache.read && logprobs.cache.write);
        assert!(!response.cache.read_enabled && response.cache.write_enabled);
        assert!(response.prompt_logprobs_requested);
    }
}

/// A streamed chat response has no field for prompt logprobs. A streamed
/// request with `prompt_logprobs: 0`, which the route accepts like vLLM does,
/// therefore requests no prompt scoring and keeps prefix-cache reads, while
/// the same buffered request still requests them.
#[test]
fn streamed_chat_does_not_request_undeliverable_prompt_logprobs() {
    for (description, model_type) in [
        (ModelDescription::Qwen3, "qwen3"),
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        for stream in [true, false] {
            let request: uniserve_server::openai::ChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": processor.served_model_name(),
                    "messages": [{"role": "user", "content": "hello"}],
                    "modalities": ["text"],
                    "stream": stream,
                    "prompt_logprobs": 0
                }))
                .unwrap();

            let (generation, response) = processor
                .preprocess_chat_request(ServeRequestId::new("stream-prompt-logprobs"), request)
                .unwrap();

            assert_eq!(
                response.prompt_logprobs_requested, !stream,
                "{model_type} stream={stream}"
            );
            assert_eq!(
                generation.sampling.prompt_logprobs_requested(),
                !stream,
                "{model_type} stream={stream}"
            );
            assert_eq!(
                generation.cache.read, stream,
                "{model_type} stream={stream}"
            );
        }
    }
}

/// Negative seeds are accepted and reinterpreted as their two's-complement
/// `u64` value, and a logprob count of `-1` requests every candidate
/// (`u32::MAX`) while values below `-1` are rejected, for every token model.
#[test]
fn sampling_controls_have_the_same_meaning_across_token_models() {
    for (description, model_type) in [
        (ModelDescription::Qwen3, "qwen3"),
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _, model) = resolved_model(description, model_type);
        for seed in [-1, -2] {
            let mut input = uniserve_server::serving::TextPromptRequest::new("sampling", "hello");
            input.sampling.seed = Some(seed);
            input.stop.logprobs = Some(-1);
            let (request, _) = model.preprocess_text_request(input.clone()).unwrap();
            assert_eq!(request.sampling.seed, Some(seed as u64));
            assert_eq!(request.sampling.n_logprobs, u32::MAX);
            input.stop.logprobs = Some(-2);
            assert!(model.preprocess_text_request(input).is_err());
        }
    }
}

/// An omitted duration resolves to the default advertised by
/// `video_capabilities` (the lesser of 5 seconds and the configured maximum,
/// here 2 seconds) and yields the same sampling as requesting it explicitly.
#[test]
fn omitted_video_duration_uses_the_advertised_model_default() {
    let (_directory, tokenizer, loaded) = resolved_model(ModelDescription::MiniMaxH3, "minimax_h3");
    let mut config = loaded.config().clone();
    config.parameters = uniserve_server::profile::ModelParameters::MiniMaxH3 {
        max_video_seconds: 2.0,
        num_inference_steps: 4,
    };
    let model = InputProcessor::new(
        config,
        tokenizer,
        None,
        uniserve_server::serving::WorkerCapabilities {
            limits: runtime_limits(),
            sampling_controls: uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
            max_model_tokens: 4096,
            denoise_steps: 4,
        },
        true,
    )
    .unwrap();

    let input: uniserve_server::openai::VideoGenerationRequest =
        serde_json::from_value(serde_json::json!({
            "model": "minimax_h3", "prompt": "a river"
        }))
        .unwrap();
    assert_eq!(model.video_capabilities()["default_seconds"], 2.0);
    let id = ServeRequestId::new("duration");
    let implicit = model.preprocess_video_request(&id, input.clone()).unwrap();
    let explicit = model
        .preprocess_video_request(
            &id,
            uniserve_server::openai::VideoGenerationRequest {
                seconds: Some(2.0),
                ..input
            },
        )
        .unwrap();
    assert_eq!(implicit.sampling, explicit.sampling);
}
