#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use tempfile::tempdir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
use uniserve_core::{
    ContextSegment, GenerationRuntimeCapabilities, ImageIngestStep, SegmentPlacement,
};
use uniserve_model_profile::assets::ResolvedModelFiles;
use uniserve_model_profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use uniserve_model_profile::{ModelDescription, ModelProfile, ProfileDeploymentConfig};
use uniserve_serving::chat::{
    ChatContentPart, ChatMessage, ChatTemplateContentFormatOption, HfChatRenderer,
};
use uniserve_serving::{GenerateReqInput, ResolvedModel};

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
        r#"{"eos_token_id":[2,3],"temperature":0.37,"top_p":0.73,"top_k":17,"min_p":0.11,"repetition_penalty":1.2,"max_new_tokens":128}"#,
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
    let profile = ModelProfile::resolve(
        description,
        description.id(),
        &files,
        &ProfileDeploymentConfig::default(),
        tokenizer.as_ref(),
    )
    .unwrap();
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    let capabilities = GenerationRuntimeCapabilities {
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
        scratch_capacity_tokens: 1_000_000,
        scratch_block_size: 32,
        encoder_cache_entries: 32,
    };
    let model = ResolvedModel::resolve(
        profile,
        Arc::clone(&tokenizer),
        renderer,
        capabilities,
        4096,
    )
    .unwrap();
    (directory, tokenizer, model)
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

fn configured_descriptions() -> [(ModelDescription, &'static str); 3] {
    [
        (ModelDescription::Qwen3, "qwen3"),
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ]
}

#[test]
fn configured_descriptions_share_canonical_sampling_lowering() {
    let mut expected = None;
    for (description, model_type) in configured_descriptions() {
        let (_directory, _tokenizer, model) = resolved_model(description, model_type);
        let mut request = if description == ModelDescription::Qwen3 {
            GenerateReqInput::text(description.id(), "sample this")
        } else {
            let mut request = image_chat_request();
            request.request_id = description.id().into();
            request
        };
        request.sampling.seed = Some(23);
        request.sampling.min_tokens = Some(7);
        request.sampling.frequency_penalty = Some(0.4);
        request.sampling.presence_penalty = Some(-0.2);
        request.stop.stop_token_ids = vec![9, 3, 9];
        request.stop.bad_words = vec!["blocked".to_string(), "blocked".to_string()];
        request.stop.allowed_token_ids = Some(vec![12, 10, 12]);
        request.stop.logprobs = Some(-1);
        request.stop.prompt_logprobs = Some(2);
        request.stop.logprob_token_ids = Some(vec![8, 6, 8]);
        request.stop.logit_bias = Some(HashMap::from([(7, 0.5), (4, -0.25)]));

        let tokenized = model.tokenize(request).unwrap();
        let sampling = &tokenized.request.sampling;
        assert_eq!(sampling.temperature, 0.37);
        assert_eq!(sampling.top_p, 0.73);
        assert_eq!(sampling.top_k, 17);
        assert_eq!(sampling.min_p, 0.11);
        assert_eq!(sampling.repetition_penalty, 1.2);
        assert_eq!(sampling.seed, Some(23));
        assert_eq!(sampling.min_tokens, 7);
        assert_eq!(sampling.frequency_penalty, 0.4);
        assert_eq!(sampling.presence_penalty, -0.2);
        assert_eq!(sampling.allowed_token_ids, Some(vec![10, 12]));
        assert_eq!(sampling.logprob_token_ids, vec![6, 8]);
        assert_eq!(sampling.logit_bias, vec![(4, -0.25), (7, 0.5)]);
        assert_eq!(sampling.n_logprobs, u32::MAX);
        assert_eq!(sampling.n_prompt_logprobs, 2);
        assert_eq!(tokenized.request.stop_token_ids, vec![2, 3, 9]);
        assert_eq!(tokenized.request.max_und_tokens, 128);

        if let Some(expected) = &expected {
            assert_eq!(sampling, expected);
        } else {
            expected = Some(sampling.clone());
        }
    }
}

#[test]
fn configured_descriptions_reject_invalid_sampling_before_submission() {
    for (description, model_type) in configured_descriptions() {
        let (_directory, _tokenizer, model) = resolved_model(description, model_type);
        let mut request = GenerateReqInput::text(description.id(), "sample this");
        request.sampling.seed = Some(-1);
        let error = model.tokenize(request).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("seed must be a non-negative integer")
        );

        let mut request = GenerateReqInput::text(description.id(), "sample this");
        request.stop.prompt_logprobs = Some(-2);
        let error = model.tokenize(request).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("prompt_logprobs must be non-negative or -1")
        );
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
