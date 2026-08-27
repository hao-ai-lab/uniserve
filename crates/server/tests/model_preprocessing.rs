#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use tempfile::tempdir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
use uniserve_core::{
    ContextSegment, GenerationRuntimeCapabilities, ImageIngestStep, ImageKvEffect, SegmentPlacement,
};
use uniserve_server::profile::assets::ResolvedModelFiles;
use uniserve_server::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use uniserve_server::profile::{ModelDescription, ProfileDeploymentConfig};
use uniserve_server::serving::chat::{
    ChatContentPart, ChatMessage, ChatTemplateContentFormatOption, HfChatRenderer,
};
use uniserve_server::serving::{GenerateReqInput, ResolvedAssets, ResolvedModel};

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
    try_resolved_model(description, model_type, runtime_capabilities()).unwrap()
}

fn try_resolved_model(
    description: ModelDescription,
    model_type: &str,
    capabilities: GenerationRuntimeCapabilities,
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
        &ProfileDeploymentConfig::default(),
        Arc::clone(&tokenizer),
        renderer,
    )
    .unwrap();
    let model = ResolvedModel::resolve(
        assets,
        capabilities,
        uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
        4096,
        true,
    )?;
    Ok((directory, tokenizer, model))
}

fn runtime_capabilities() -> GenerationRuntimeCapabilities {
    GenerationRuntimeCapabilities {
        supports_understanding: true,
        supports_vision_encode: true,
        supports_latent_encode: true,
        supports_image_generation: true,
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
        "image-placement",
        vec![ChatMessage::user(vec![
            ChatContentPart::text("literal </img> before "),
            ChatContentPart::image_url(format!("data:image/png;base64,{PNG_1X1}")),
            ChatContentPart::text(" after"),
        ])],
    )
}

#[test]
fn model_resolution_requires_every_configured_runtime_branch() {
    type RemoveCapability = fn(&mut GenerationRuntimeCapabilities);
    let cases: [(
        ModelDescription,
        &'static str,
        &'static str,
        RemoveCapability,
    ); 4] = [
        (
            ModelDescription::Qwen3,
            "qwen3",
            "runtime_und_execution",
            |capabilities: &mut GenerationRuntimeCapabilities| {
                capabilities.supports_understanding = false;
            },
        ),
        (
            ModelDescription::SenseNova,
            "neo_chat",
            "runtime_vit_encode",
            |capabilities: &mut GenerationRuntimeCapabilities| {
                capabilities.supports_vision_encode = false;
            },
        ),
        (
            ModelDescription::Bagel,
            "bagel",
            "runtime_vae_encode",
            |capabilities: &mut GenerationRuntimeCapabilities| {
                capabilities.supports_latent_encode = false;
            },
        ),
        (
            ModelDescription::SenseNova,
            "neo_chat",
            "runtime_gen_denoise",
            |capabilities: &mut GenerationRuntimeCapabilities| {
                capabilities.supports_image_generation = false;
            },
        ),
    ];

    for (description, model_type, required, remove) in cases {
        let mut capabilities = runtime_capabilities();
        remove(&mut capabilities);
        let error = match try_resolved_model(description, model_type, capabilities) {
            Ok(_) => panic!("incomplete worker capabilities must fail model resolution"),
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
    let (placement, steps) = tokenized
        .request
        .context
        .iter()
        .find_map(|segment| match segment {
            ContextSegment::Image { image, ingest } => Some((image.placement, &ingest.steps)),
            ContextSegment::UndTokens { .. } => None,
        })
        .unwrap();
    assert_eq!(
        placement,
        SegmentPlacement::AtToken {
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
    let (placement, steps) = tokenized
        .request
        .context
        .iter()
        .find_map(|segment| match segment {
            ContextSegment::Image { image, ingest } => Some((image.placement, &ingest.steps)),
            ContextSegment::UndTokens { .. } => None,
        })
        .unwrap();
    let SegmentPlacement::AtToken { position } = placement else {
        panic!("Bagel chat image must have a token position")
    };
    assert!(position > 0);
    assert!((position as usize) < tokenized.prompt_token_ids.len());
    assert_eq!(
        steps,
        &[ImageIngestStep::VaeEncode, ImageIngestStep::VitEncode]
    );
}
