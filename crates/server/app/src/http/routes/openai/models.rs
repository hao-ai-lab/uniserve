use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use uniserve_protocol_adapters::openai::{ListModelsResponse, ModelObject};

use crate::AppState;

use crate::http::utils::unix_timestamp;

/// Return all configured served model names in OpenAI `list models` format.
pub(crate) async fn list_models(State(state): State<Arc<AppState>>) -> Json<ListModelsResponse> {
    // OpenAI clients expect `created` to be a real Unix timestamp; report when
    // this listing was produced rather than the epoch.
    let created = unix_timestamp() as i64;
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.served_model_name().to_string(),
            object: "model".to_string(),
            created,
            owned_by: "uniserve".to_string(),
        }],
    })
}
