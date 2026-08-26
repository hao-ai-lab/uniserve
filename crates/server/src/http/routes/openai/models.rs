use std::sync::Arc;

use crate::openai::{
    ListModelsResponse, ModelCapabilities, ModelEndpoint, ModelFeature, ModelModality, ModelObject,
    ModelSamplingControl, ServedModelIdentity,
};
use crate::serving::{ServedEndpoint, ServedFeature, ServedModality, ServedSamplingControl};
use axum::Json;
use axum::extract::State;

use crate::AppState;

use crate::http::utils::unix_timestamp;

/// Return all configured served model names in OpenAI `list models` format.
pub(crate) async fn list_models(State(state): State<Arc<AppState>>) -> Json<ListModelsResponse> {
    // OpenAI clients expect `created` to be a real Unix timestamp; report when
    // this listing was produced rather than the epoch.
    let created = unix_timestamp() as i64;
    let model = state.runtime().model();
    let identity = model.served_identity();
    let declared = model.served_capabilities();
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.served_model_name().to_string(),
            object: "model".to_string(),
            created,
            owned_by: "uniserve".to_string(),
            identity: ServedModelIdentity {
                profile_id: identity.profile_id.clone(),
                description_id: identity.description_id.clone(),
                config_fingerprint: identity.config_fingerprint.clone(),
            },
            capabilities: ModelCapabilities {
                endpoints: declared.endpoints.into_iter().map(endpoint).collect(),
                input_modalities: declared
                    .input_modalities
                    .into_iter()
                    .map(modality)
                    .collect(),
                output_modalities: declared
                    .output_modalities
                    .into_iter()
                    .map(modality)
                    .collect(),
                features: declared.features.into_iter().map(feature).collect(),
                sampling_controls: declared
                    .sampling_controls
                    .into_iter()
                    .map(sampling_control)
                    .collect(),
            },
        }],
    })
}

fn endpoint(value: ServedEndpoint) -> ModelEndpoint {
    match value {
        ServedEndpoint::ChatCompletions => ModelEndpoint::ChatCompletions,
        ServedEndpoint::ImageGenerations => ModelEndpoint::ImageGenerations,
        ServedEndpoint::VideoGenerations => ModelEndpoint::VideoGenerations,
    }
}

fn modality(value: ServedModality) -> ModelModality {
    match value {
        ServedModality::Text => ModelModality::Text,
        ServedModality::Image => ModelModality::Image,
        ServedModality::Video => ModelModality::Video,
        ServedModality::Audio => ModelModality::Audio,
    }
}

fn feature(value: ServedFeature) -> ModelFeature {
    match value {
        ServedFeature::Streaming => ModelFeature::Streaming,
        ServedFeature::Usage => ModelFeature::Usage,
        ServedFeature::Logprobs => ModelFeature::Logprobs,
        ServedFeature::Reasoning => ModelFeature::Reasoning,
        ServedFeature::ToolCalling => ModelFeature::ToolCalling,
        ServedFeature::RepeatedInterleave => ModelFeature::RepeatedInterleave,
    }
}

fn sampling_control(value: ServedSamplingControl) -> ModelSamplingControl {
    match value {
        ServedSamplingControl::Greedy => ModelSamplingControl::Greedy,
        ServedSamplingControl::Temperature => ModelSamplingControl::Temperature,
        ServedSamplingControl::TopK => ModelSamplingControl::TopK,
        ServedSamplingControl::TopP => ModelSamplingControl::TopP,
        ServedSamplingControl::MinP => ModelSamplingControl::MinP,
        ServedSamplingControl::RepetitionPenalty => ModelSamplingControl::RepetitionPenalty,
        ServedSamplingControl::FrequencyPenalty => ModelSamplingControl::FrequencyPenalty,
        ServedSamplingControl::PresencePenalty => ModelSamplingControl::PresencePenalty,
        ServedSamplingControl::LogitBias => ModelSamplingControl::LogitBias,
        ServedSamplingControl::AllowedTokenIds => ModelSamplingControl::AllowedTokenIds,
        ServedSamplingControl::BadWords => ModelSamplingControl::BadWords,
        ServedSamplingControl::MinTokens => ModelSamplingControl::MinTokens,
        ServedSamplingControl::Logprobs => ModelSamplingControl::Logprobs,
        ServedSamplingControl::StopTokenIds => ModelSamplingControl::StopTokenIds,
        ServedSamplingControl::Eos => ModelSamplingControl::Eos,
        ServedSamplingControl::StopStrings => ModelSamplingControl::StopStrings,
    }
}
