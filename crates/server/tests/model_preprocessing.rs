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

const PNG_1X1: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
const CHAT_TEMPLATE: &str = "{%- for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
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

fn resolved_model(
    description: ModelDescription,
    model_type: &str,
) -> (tempfile::TempDir, DynTokenizer, InputProcessor) {
    try_resolved_model(description, model_type, runtime_limits()).unwrap()
}

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
    let tokenizer: DynTokenizer =
        Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap());
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    let config =
        ModelConfig::from_files(description.id(), &files, None, tokenizer.as_ref()).unwrap();
    let model = InputProcessor::new(
        config,
        Arc::clone(&tokenizer),
        Some(renderer),
        limits,
        uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
        4096,
        0,
        true,
    )?;
    Ok((directory, tokenizer, model))
}

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

#[test]
fn sensenova_places_the_input_image_at_its_rendered_slot() {
    let (_directory, tokenizer, model) = resolved_model(ModelDescription::SenseNova, "neo_chat");
    let request = image_chat_request(model.served_model_name());
    let (generation, response) = model
        .preprocess_chat_request(ServeRequestId::new("image-params"), request)
        .unwrap();
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
                seconds: 1.0,
                seed: 17,
            },
        )
        .unwrap();

    assert_eq!(request.prompt_token_ids, expected);
    assert_eq!(request.sampling.seed, 17);
    assert_eq!(request.sampling.num_frames, 39);
    assert_eq!(request.sampling.num_decode_chunks, 2);

    let invalid = |model_name: &str, prompt: &str, seconds| {
        model
            .preprocess_video_request(
                &ServeRequestId::new("video"),
                uniserve_server::openai::VideoGenerationRequest {
                    model: model_name.to_string(),
                    prompt: prompt.to_string(),
                    seconds,
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
        runtime_limits(),
        uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
        8,
        0,
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
