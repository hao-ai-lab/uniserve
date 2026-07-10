use std::collections::HashMap;

use serde::Deserialize;
use uniserve_core::GenerationConstraint;
use uniserve_openai_types::Normalizable;
use validator::Validate;

#[derive(Debug, Clone, Deserialize, Default, Validate)]
#[serde(deny_unknown_fields)]
pub struct NativeGenerateBody {
    #[serde(default)]
    pub context: Vec<NativeContextSegment>,
    #[serde(default)]
    pub constraint: Option<GenerationConstraint>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub adapter: Option<String>,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub data_parallel_rank: Option<u32>,
    #[serde(default)]
    pub deadline_ms: Option<u64>,
    #[serde(default)]
    pub trace_context: HashMap<String, String>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub min_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default)]
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[serde(default)]
    pub allowed_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub bad_words: Vec<String>,
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
    #[serde(default)]
    pub image_bias: Option<f32>,
    #[serde(default)]
    pub image: Option<NativeImageBody>,
}

impl Normalizable for NativeGenerateBody {}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum NativeContextSegment {
    Text {
        role: NativeContextRole,
        text: String,
    },
    TokenIds {
        token_ids: Vec<u32>,
        tokenizer_fingerprint: String,
    },
    Image {
        b64: String,
        #[serde(default)]
        placement: Option<u32>,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NativeContextRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct NativeImageBody {
    #[serde(default)]
    pub resolution: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub steps: Option<u16>,
    #[serde(default)]
    pub cfg_text_scale: Option<f32>,
    #[serde(default)]
    pub cfg_img_scale: Option<f32>,
    #[serde(default)]
    pub cfg_interval: Option<[f32; 2]>,
    #[serde(default)]
    pub cfg_renorm_type: Option<String>,
    #[serde(default)]
    pub cfg_renorm_min: Option<f32>,
    #[serde(default)]
    pub timestep_shift: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub max_images: Option<u16>,
    #[serde(default)]
    pub prompts: Vec<String>,
    #[serde(default)]
    pub retain_images: Option<bool>,
}

impl NativeGenerateBody {
    pub fn image(&self) -> NativeImageBody {
        self.image.clone().unwrap_or_default()
    }

    pub fn negative_prompt(&self) -> String {
        self.negative_prompt
            .clone()
            .or_else(|| {
                self.image
                    .as_ref()
                    .and_then(|image| image.negative_prompt.clone())
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_rejects_unknown_generation_selector() {
        let error = serde_json::from_str::<NativeGenerateBody>(
            r#"{"context":[{"type":"text","role":"user","text":"draw a city"}],"mode":"image"}"#,
        )
        .expect_err("unknown generation selectors must be rejected");

        assert!(error.to_string().contains("unknown field `mode`"));
    }

    #[test]
    fn body_rejects_noncanonical_image_fields() {
        for body in [
            r#"{"context":[{"type":"text","role":"user","text":"draw a city"}],"input_image_b64":"AA=="}"#,
            r#"{"context":[{"type":"text","role":"user","text":"draw a city"}],"image":{"image_prompts":["skyline"]}}"#,
        ] {
            serde_json::from_str::<NativeGenerateBody>(body)
                .expect_err("alternate native image fields must be rejected");
        }
    }

    #[test]
    fn body_preserves_ordered_context_and_runtime_controls() {
        let body: NativeGenerateBody = serde_json::from_str(
            r#"{
                "context": [
                    {"type":"text","role":"system","text":"policy"},
                    {"type":"image","b64":"aQ==","placement":3},
                    {"type":"text","role":"user","text":"describe"}
                ],
                "profile_id":"sensenova-u1",
                "adapter":"travel-style",
                "grammar":"root ::= 'ok'",
                "priority":-4,
                "data_parallel_rank":2,
                "deadline_ms":5000,
                "trace_context":{"traceparent":"00-ab-cd-01"}
            }"#,
        )
        .expect("canonical native request");

        assert_eq!(body.context.len(), 3);
        assert_eq!(body.profile_id.as_deref(), Some("sensenova-u1"));
        assert_eq!(body.adapter.as_deref(), Some("travel-style"));
        assert_eq!(body.grammar.as_deref(), Some("root ::= 'ok'"));
        assert_eq!(body.priority, -4);
        assert_eq!(body.data_parallel_rank, Some(2));
        assert_eq!(body.deadline_ms, Some(5000));
    }

    #[test]
    fn body_rejects_request_owned_image_token_spans() {
        let error = serde_json::from_str::<NativeGenerateBody>(
            r#"{"context":[{"type":"image","b64":"aQ==","token_span_hint":16}]}"#,
        )
        .expect_err("image token spans are owned by the resolved model profile");

        assert!(
            error
                .to_string()
                .contains("unknown field `token_span_hint`")
        );
    }
}
