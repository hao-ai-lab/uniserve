use thiserror::Error;
use uniserve_serving::{
    AdapterSelection, CachePolicy, ContextRole, ContextSegment, GenerationPolicy,
    ImageGenerationPolicy, ImageInput, ModalityPolicy, ModelContext, RequestMetadata,
    SchedulingPolicy, ServeRequest, ServeRequestId, StructuredOutputIntent,
};

use super::schema::{NativeContextRole, NativeContextSegment, NativeGenerateBody};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeRequestResolution {
    pub profile_id: String,
    pub requested_adapter: Option<String>,
    pub adapter: AdapterSelection,
}

impl NativeRequestResolution {
    pub fn base(profile_id: impl Into<String>) -> Self {
        Self {
            profile_id: profile_id.into(),
            requested_adapter: None,
            adapter: AdapterSelection::Base,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NativeAdapterError {
    #[error("generation requires non-empty ordered context")]
    EmptyContext,
    #[error("requested profile `{requested}` resolved to `{resolved}`")]
    ProfileMismatch { requested: String, resolved: String },
    #[error("requested adapter does not match router resolution")]
    AdapterMismatch,
    #[error("max_tokens exceeds the supported range")]
    MaxTokensOutOfRange,
    #[error("seed exceeds the supported range")]
    SeedOutOfRange,
}

pub fn into_serve_request(
    request_id: impl Into<ServeRequestId>,
    body: NativeGenerateBody,
    resolution: NativeRequestResolution,
    metadata: RequestMetadata,
) -> Result<ServeRequest, NativeAdapterError> {
    if body.context.is_empty() {
        return Err(NativeAdapterError::EmptyContext);
    }
    if let Some(requested) = body.profile_id.as_ref()
        && requested != &resolution.profile_id
    {
        return Err(NativeAdapterError::ProfileMismatch {
            requested: requested.clone(),
            resolved: resolution.profile_id,
        });
    }
    if body.adapter != resolution.requested_adapter {
        return Err(NativeAdapterError::AdapterMismatch);
    }

    let constraint = body.constraint.unwrap_or_default();
    let image = body.image();
    let negative_prompt = body.negative_prompt();
    let input_text = body
        .context
        .iter()
        .any(|segment| !matches!(segment, NativeContextSegment::Image { .. }));
    let input_image = body
        .context
        .iter()
        .any(|segment| matches!(segment, NativeContextSegment::Image { .. }));
    let segments = body.context.into_iter().map(ContextSegment::from).collect();

    let max_tokens = body
        .max_tokens
        .map(u32::try_from)
        .transpose()
        .map_err(|_| NativeAdapterError::MaxTokensOutOfRange)?;
    let seed = body
        .seed
        .map(i64::try_from)
        .transpose()
        .map_err(|_| NativeAdapterError::SeedOutOfRange)?;

    Ok(ServeRequest {
        request_id: request_id.into(),
        model_context: ModelContext::Segments(segments),
        generation: GenerationPolicy {
            constraint,
            temperature: body.temperature,
            top_p: body.top_p,
            top_k: body.top_k,
            seed,
            max_tokens,
            min_tokens: body.min_tokens,
            min_p: body.min_p,
            frequency_penalty: body.frequency_penalty,
            presence_penalty: body.presence_penalty,
            repetition_penalty: body.repetition_penalty,
            stop_token_ids: body.stop_token_ids,
            ignore_eos: body.ignore_eos,
            logit_bias: body.logit_bias,
            allowed_token_ids: body.allowed_token_ids,
            bad_words: body.bad_words,
            structured_output: body.grammar.map(StructuredOutputIntent::Grammar),
            image: ImageGenerationPolicy {
                resolution: image.resolution,
                width: image.width,
                height: image.height,
                steps: image.steps,
                cfg_text_scale: image.cfg_text_scale,
                cfg_img_scale: image.cfg_img_scale,
                cfg_interval: image.cfg_interval,
                cfg_renorm_type: image.cfg_renorm_type,
                cfg_renorm_min: image.cfg_renorm_min,
                timestep_shift: image.timestep_shift,
                seed: image.seed.or(body.seed),
                negative_prompt: Some(negative_prompt),
                max_images: image.max_images,
                prompts: image.prompts,
                retain_images: image.retain_images,
                image_bias: body.image_bias,
            },
            ..GenerationPolicy::default()
        },
        modalities: ModalityPolicy {
            input_text,
            input_image,
            output_text: constraint != uniserve_core::GenerationConstraint::GenOnly,
            output_image: constraint != uniserve_core::GenerationConstraint::UndOnly,
        },
        adapter: resolution.adapter,
        cache: CachePolicy::default(),
        scheduling: SchedulingPolicy {
            priority: body.priority,
            data_parallel_rank: body.data_parallel_rank,
            deadline_ms: body.deadline_ms,
            trace_context: body.trace_context,
        },
        metadata,
    })
}

impl From<NativeContextSegment> for ContextSegment {
    fn from(value: NativeContextSegment) -> Self {
        match value {
            NativeContextSegment::Text { role, text } => Self::Text {
                role: role.into(),
                text,
            },
            NativeContextSegment::TokenIds {
                token_ids,
                tokenizer_fingerprint,
            } => Self::TokenIds {
                token_ids,
                tokenizer_fingerprint,
            },
            NativeContextSegment::Image { b64, placement } => {
                Self::Image(ImageInput { b64, placement })
            }
        }
    }
}

impl From<NativeContextRole> for ContextRole {
    fn from(value: NativeContextRole) -> Self {
        match value {
            NativeContextRole::System => Self::System,
            NativeContextRole::Developer => Self::Developer,
            NativeContextRole::User => Self::User,
            NativeContextRole::Assistant => Self::Assistant,
            NativeContextRole::Tool => Self::Tool,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use uniserve_core::GenerationConstraint;

    use super::*;
    use crate::native::schema::{NativeContextRole, NativeContextSegment};

    #[test]
    fn ordered_native_context_lowers_to_semantic_segments() {
        let request = into_serve_request(
            "req",
            NativeGenerateBody {
                context: vec![
                    NativeContextSegment::Text {
                        role: NativeContextRole::System,
                        text: "system".to_string(),
                    },
                    NativeContextSegment::Image {
                        b64: "aQ==".to_string(),
                        placement: Some(3),
                    },
                    NativeContextSegment::Text {
                        role: NativeContextRole::User,
                        text: "describe".to_string(),
                    },
                    NativeContextSegment::Text {
                        role: NativeContextRole::Assistant,
                        text: "prefix".to_string(),
                    },
                ],
                constraint: Some(GenerationConstraint::UndOnly),
                ..NativeGenerateBody::default()
            },
            NativeRequestResolution::base("profile"),
            RequestMetadata::default(),
        )
        .unwrap();

        let ModelContext::Segments(segments) = request.model_context else {
            panic!("native adapter must produce ordered semantic segments");
        };
        assert_eq!(segments.len(), 4);
        assert!(matches!(
            segments[0],
            ContextSegment::Text {
                role: ContextRole::System,
                ..
            }
        ));
        assert!(matches!(
            segments[2],
            ContextSegment::Text {
                role: ContextRole::User,
                ..
            }
        ));
        assert!(matches!(segments[1], ContextSegment::Image(_)));
        assert!(matches!(
            segments[3],
            ContextSegment::Text {
                role: ContextRole::Assistant,
                ..
            }
        ));
        assert!(!request.modalities.output_image);
    }

    #[test]
    fn native_sampling_controls_lower_without_loss() {
        let request = into_serve_request(
            "req",
            NativeGenerateBody {
                context: vec![NativeContextSegment::Text {
                    role: NativeContextRole::User,
                    text: "continue".to_string(),
                }],
                min_tokens: Some(5),
                min_p: Some(0.12),
                frequency_penalty: Some(0.25),
                presence_penalty: Some(-0.5),
                repetition_penalty: Some(1.1),
                ignore_eos: true,
                logit_bias: Some(HashMap::from([(151670, 80.0)])),
                allowed_token_ids: Some(vec![7, 11]),
                bad_words: vec!["blocked".to_string()],
                ..NativeGenerateBody::default()
            },
            NativeRequestResolution::base("profile"),
            RequestMetadata::default(),
        )
        .unwrap();

        assert_eq!(request.generation.min_tokens, Some(5));
        assert_eq!(request.generation.min_p, Some(0.12));
        assert_eq!(request.generation.frequency_penalty, Some(0.25));
        assert_eq!(request.generation.presence_penalty, Some(-0.5));
        assert_eq!(request.generation.repetition_penalty, Some(1.1));
        assert!(request.generation.ignore_eos);
        assert_eq!(
            request.generation.logit_bias,
            Some(HashMap::from([(151670, 80.0)]))
        );
        assert_eq!(request.generation.allowed_token_ids, Some(vec![7, 11]));
        assert_eq!(request.generation.bad_words, vec!["blocked"]);
    }

    #[test]
    fn native_profile_adapter_grammar_and_scheduling_use_router_resolution() {
        let adapter = AdapterSelection::Adapter {
            name: "travel-style".to_string(),
            internal_id: 7,
            path: "/models/travel-style".to_string(),
            load_inplace: false,
            is_3d_lora_weight: false,
        };
        let request = into_serve_request(
            "req",
            NativeGenerateBody {
                context: vec![NativeContextSegment::Text {
                    role: NativeContextRole::User,
                    text: "continue".to_string(),
                }],
                profile_id: Some("sensenova-u1".to_string()),
                adapter: Some("travel-style".to_string()),
                grammar: Some("root ::= 'ok'".to_string()),
                priority: -4,
                data_parallel_rank: Some(2),
                deadline_ms: Some(5000),
                trace_context: HashMap::from([(
                    "traceparent".to_string(),
                    "00-ab-cd-01".to_string(),
                )]),
                ..NativeGenerateBody::default()
            },
            NativeRequestResolution {
                profile_id: "sensenova-u1".to_string(),
                requested_adapter: Some("travel-style".to_string()),
                adapter: adapter.clone(),
            },
            RequestMetadata::default(),
        )
        .expect("resolved native request");

        assert_eq!(request.adapter, adapter);
        assert!(matches!(
            request.generation.structured_output,
            Some(StructuredOutputIntent::Grammar(ref grammar)) if grammar == "root ::= 'ok'"
        ));
        assert_eq!(request.scheduling.priority, -4);
        assert_eq!(request.scheduling.data_parallel_rank, Some(2));
        assert_eq!(request.scheduling.deadline_ms, Some(5000));
        assert_eq!(
            request
                .scheduling
                .trace_context
                .get("traceparent")
                .map(String::as_str),
            Some("00-ab-cd-01")
        );
    }

    #[test]
    fn native_profile_must_match_router_resolution() {
        let error = into_serve_request(
            "req",
            NativeGenerateBody {
                context: vec![NativeContextSegment::Text {
                    role: NativeContextRole::User,
                    text: "continue".to_string(),
                }],
                profile_id: Some("other".to_string()),
                ..NativeGenerateBody::default()
            },
            NativeRequestResolution::base("sensenova-u1"),
            RequestMetadata::default(),
        )
        .expect_err("profile mismatch");

        assert!(matches!(error, NativeAdapterError::ProfileMismatch { .. }));
    }
}
