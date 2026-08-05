//! Tokenize oracle: one `GenerateReqInput` -> one description-owned `tokenize`
//! call -> one validated `TokenizedGenerateReqInput` per configured behavior.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{
    MAX_MODEL_TOKENS, chat_input, i2t_input, resolve_bagel, resolve_qwen3, resolve_sensenova,
    resolve_unsupported_text, t2i_input, text_input,
};
use uniserve_core::GenerationConstraint;
use uniserve_serving::chat::ChatMessage;
use uniserve_serving::{OutputContract, OutputProcessorPolicy, SamplingConfig};

#[test]
fn unsupported_text_family_fails_closed_resolution() {
    let Err(error) = resolve_unsupported_text() else {
        panic!("unsupported text model must fail");
    };
    assert!(error.to_string().contains("unsupported-text-family"));
}

#[test]
fn qwen3_text_lowers_to_und_only_request() {
    let model = resolve_qwen3();
    let tokenized = model
        .tokenize(text_input("q-text", "hello"))
        .expect("tokenize");

    assert_eq!(tokenized.request.constraint, GenerationConstraint::UndOnly);
    assert_eq!(
        tokenized.prompt_token_ids,
        b"hello".iter().map(|b| *b as u32).collect::<Vec<_>>()
    );
    assert!(tokenized.request.lora_id.is_none());
    assert!(tokenized.request.grammar.is_none());
    assert!(matches!(
        tokenized.output_processor,
        OutputProcessorPolicy::None
    ));
    tokenized
        .request
        .validate()
        .expect("valid generation request");
}

#[test]
fn qwen3_chat_renders_and_attaches_chat_processor() {
    let model = resolve_qwen3();
    let request = chat_input("q-chat", vec![ChatMessage::user("Say hi")]);
    let tokenized = model.tokenize(request).expect("tokenize");

    // The chat template wraps the message, so the prompt is longer than the raw text.
    assert!(tokenized.prompt_token_ids.len() > "Say hi".len());
    assert_eq!(tokenized.request.constraint, GenerationConstraint::UndOnly);
    assert!(matches!(
        tokenized.output_processor,
        OutputProcessorPolicy::Chat(_)
    ));
    tokenized
        .request
        .validate()
        .expect("valid generation request");
}

#[test]
fn qwen3_sampling_merges_user_values() {
    let model = resolve_qwen3();
    let mut request = text_input("q-sampling", "hello");
    request.sampling = SamplingConfig {
        temperature: Some(0.2),
        top_p: Some(0.5),
        top_k: Some(7),
        max_tokens: Some(16),
        ..SamplingConfig::default()
    };
    let tokenized = model.tokenize(request).expect("tokenize");
    assert_eq!(tokenized.request.sampling.temperature, 0.2);
    assert_eq!(tokenized.request.sampling.top_p, 0.5);
    assert_eq!(tokenized.request.sampling.top_k, 7);
    assert_eq!(tokenized.request.max_und_tokens, 16);
}

#[test]
fn qwen3_output_contract_gates_token_ids() {
    let model = resolve_qwen3();
    let mut request = text_input("q-tokens", "hi");
    request.output = OutputContract::Tokens;
    let tokenized = model.tokenize(request).expect("tokenize");
    assert!(tokenized.emit_token_ids);

    let tokenized_default = model
        .tokenize(text_input("q-visible", "hi"))
        .expect("tokenize");
    assert!(!tokenized_default.emit_token_ids);
}

#[test]
fn qwen3_rejects_image_modalities_before_submission() {
    let model = resolve_qwen3();
    let t2i = t2i_input("q-img-out", "draw a cat");
    assert!(model.validate_request(&t2i).is_err());
    let i2t = i2t_input("q-img-in", "describe");
    assert!(model.validate_request(&i2t).is_err());
}

#[test]
fn qwen3_prompt_exceeding_context_is_rejected() {
    let model = resolve_qwen3();
    let prompt = "x".repeat((MAX_MODEL_TOKENS as usize) + 8);
    let error = model.tokenize(text_input("q-long", &prompt)).unwrap_err();
    assert!(matches!(
        error,
        uniserve_serving::ServeError::Tokenize { .. }
    ));
}

#[test]
fn sensenova_t2i_derives_gen_only_and_dialect_filter() {
    let model = resolve_sensenova();
    let request = t2i_input("sn-t2i", "a serene mountain lake at dawn");
    model.validate_request(&request).expect("valid t2i request");
    let tokenized = model.tokenize(request).expect("tokenize");

    assert_eq!(tokenized.request.constraint, GenerationConstraint::GenOnly);
    assert!(tokenized.request.image.steps > 0);
    assert!(tokenized.request.image.width > 0 && tokenized.request.image.height > 0);
    assert!(matches!(
        tokenized.output_processor,
        OutputProcessorPolicy::DialectFilter(_)
    ));
    assert!(tokenized.request.lora_id.is_none() && tokenized.request.grammar.is_none());
    tokenized
        .request
        .validate()
        .expect("valid generation request");
}

#[test]
fn sensenova_i2t_derives_und_only_with_image_segment() {
    let model = resolve_sensenova();
    let request = i2t_input("sn-i2t", "Describe this image in detail.");
    model.validate_request(&request).expect("valid i2t request");
    let tokenized = model.tokenize(request).expect("tokenize");

    assert_eq!(tokenized.request.constraint, GenerationConstraint::UndOnly);
    assert_eq!(tokenized.request.context_image_count(), 1);
    assert!(matches!(
        tokenized.output_processor,
        OutputProcessorPolicy::DialectFilter(_)
    ));
    tokenized
        .request
        .validate()
        .expect("valid generation request");
}

#[test]
fn bagel_i2t_derives_und_only_with_image_segment() {
    let model = resolve_bagel();
    let request = i2t_input("bagel-i2t", "What is in this picture?");
    model.validate_request(&request).expect("valid i2t request");
    let tokenized = model.tokenize(request).expect("tokenize");

    assert_eq!(tokenized.request.constraint, GenerationConstraint::UndOnly);
    assert_eq!(tokenized.request.context_image_count(), 1);
    assert!(matches!(
        tokenized.output_processor,
        OutputProcessorPolicy::DialectFilter(_)
    ));
    tokenized
        .request
        .validate()
        .expect("valid generation request");
}

#[test]
fn tokenize_is_deterministic() {
    let model = resolve_sensenova();
    let first = model
        .tokenize(t2i_input("det", "identical prompt"))
        .expect("tokenize");
    let second = model
        .tokenize(t2i_input("det", "identical prompt"))
        .expect("tokenize");
    assert_eq!(first.prompt_token_ids, second.prompt_token_ids);
    assert_eq!(first.request.constraint, second.request.constraint);
    assert_eq!(first.request.sampling, second.request.sampling);
    assert_eq!(first.request.image, second.request.image);
}
