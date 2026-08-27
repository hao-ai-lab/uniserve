use std::sync::Arc;

use crate::openai::{ListModelsResponse, ModelCapabilities, ModelObject, ServedModelIdentity};
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
                served_name: identity.served_name.clone(),
                description: identity.description,
                fingerprint: identity.fingerprint.clone(),
            },
            capabilities: ModelCapabilities {
                endpoints: declared.endpoints,
                input_modalities: declared.input_modalities,
                output_modalities: declared.output_modalities,
                features: declared.features,
                sampling_controls: declared.sampling_controls,
            },
        }],
    })
}
