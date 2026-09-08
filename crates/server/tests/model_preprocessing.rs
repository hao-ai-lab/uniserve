#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Model-profile prompt preprocessing and generation-policy integration behavior.

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use tempfile::tempdir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
use uniserve_core::{
    ContextSegment, GenerationLimits, ImageIngestStep, ImageKvEffect, SegmentPosition,
};
use uniserve_server::profile::assets::ResolvedModelFiles;
use uniserve_server::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use uniserve_server::profile::{ModelDescription, ProfileOverrides};
use uniserve_server::serving::chat::{
    ChatContentPart, ChatMessage, ChatTemplateContentFormatOption, HfChatRenderer,
};
use uniserve_server::serving::{GenerateReqInput, ResolvedAssets, ResolvedModel, ServeRequestId};

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
) -> (tempfile::TempDir, DynTokenizer, ResolvedModel) {
    try_resolved_model(description, model_type, runtime_limits()).unwrap()
}

fn try_resolved_model(
    description: ModelDescription,
    model_type: &str,
    limits: GenerationLimits,
) -> uniserve_server::serving::Result<(tempfile::TempDir, DynTokenizer, ResolvedModel)> {
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
    let assets = ResolvedAssets::from_files(
        description,
        description.id(),
        &files,
        &ProfileOverrides::default(),
        Arc::clone(&tokenizer),
        renderer,
    )
    .unwrap();
    let model = ResolvedModel::resolve(
        assets,
        limits,
        uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
        4096,
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

fn image_chat_request() -> GenerateReqInput {
    GenerateReqInput::chat(
        "image-params",
        vec![ChatMessage::user(vec![
            ChatContentPart::text("literal </img> before "),
            ChatContentPart::image_url(format!("data:image/png;base64,{PNG_1X1}")),
            ChatContentPart::text(" after"),
        ])],
    )
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
    let request = image_chat_request();
    model.validate_request(&request).unwrap();
    let tokenized = model.tokenize(request).unwrap();
    let end_image = tokenizer.token_to_id("</img>").unwrap();
    let marker_positions = tokenized
        .prompt_token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (*token == end_image).then_some(index as u32))
        .collect::<Vec<_>>();
    assert_eq!(marker_positions.len(), 2);
    let (params, steps) = tokenized
        .request
        .context
        .iter()
        .find_map(|segment| match segment {
            ContextSegment::Image { image, ingest } => Some((image.position, &ingest.steps)),
            ContextSegment::UndTokens { .. } => None,
        })
        .unwrap();
    assert_eq!(
        params,
        SegmentPosition::AtToken {
            position: marker_positions[1]
        }
    );
    assert_eq!(steps, &[ImageIngestStep::VitEncode]);
    assert_eq!(
        tokenized
            .request
            .policy
            .feedback
            .unwrap()
            .ingest
            .step_kv_tokens,
        vec![ImageKvEffect::Exact { tokens: 2305 }]
    );
}

#[test]
fn bagel_places_the_input_image_between_surrounding_chat_text() {
    let (_directory, _tokenizer, model) = resolved_model(ModelDescription::Bagel, "bagel");
    let request = image_chat_request();
    model.validate_request(&request).unwrap();
    let tokenized = model.tokenize(request).unwrap();
    let (params, steps) = tokenized
        .request
        .context
        .iter()
        .find_map(|segment| match segment {
            ContextSegment::Image { image, ingest } => Some((image.position, &ingest.steps)),
            ContextSegment::UndTokens { .. } => None,
        })
        .unwrap();
    let SegmentPosition::AtToken { position } = params else {
        panic!("Bagel chat image must have a token position")
    };
    assert!(position > 0);
    assert!((position as usize) < tokenized.prompt_token_ids.len());
    assert_eq!(
        steps,
        &[ImageIngestStep::VaeEncode, ImageIngestStep::VitEncode]
    );
}

#[test]
fn minimax_video_geometry_carries_the_admitted_token_sequence() {
    let (_directory, tokenizer, model) = resolved_model(ModelDescription::MiniMaxH3, "minimax_h3");
    let prompt = "exact token sequence";
    let expected = tokenizer.encode(prompt, false).unwrap();

    let (geometry, prompt_token_ids) = model
        .resolve_video_request_geometry(&ServeRequestId::new("video"), prompt, 1.0)
        .unwrap();

    assert_eq!(prompt_token_ids, expected);
    assert_eq!(geometry.prompt_tokens as usize, prompt_token_ids.len());
}
