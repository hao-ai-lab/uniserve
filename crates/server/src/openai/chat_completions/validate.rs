//! Configuration-dependent validation for chat-completion requests.
//!
//! Field ranges and cross-field rules that need no configuration are declared
//! on `ChatCompletionRequest` and run by the route's `ValidatedJson`
//! extractor. This module holds the served-model check and the
//! `prompt_logprobs` checks, including the rule that depends on `stream`.

use crate::openai::error::{ApiError, bail_invalid_request};
use crate::openai::types::ChatCompletionRequest;
use crate::openai::utils::{check_model_served, check_prompt_logprobs_bound};

/// Validates relationships that depend on the configured serving route.
///
/// Rejects a `model` other than `served_model_name` with
/// [`ApiError::ModelNotFound`], and rejects with [`ApiError::InvalidRequest`]
/// a `prompt_logprobs` value below `-1` or a positive or `-1` value on a
/// streamed request. `ServingRuntime::generate_chat` calls this before
/// scheduling preprocessing, and `InputProcessor::preprocess_chat_request`
/// repeats it, so callers that preprocess directly get the same checks.
pub fn validate_request_compat(
    request: &ChatCompletionRequest,
    served_model_name: &str,
) -> Result<(), ApiError> {
    check_model_served(&request.model, served_model_name)?;

    if let Some(prompt_logprobs) = request.prompt_logprobs {
        check_prompt_logprobs_bound(prompt_logprobs, "prompt_logprobs")?;

        // Stream chunks (`ChatCompletionStreamResponse`) have no
        // `prompt_logprobs` field, so a streamed response could not deliver
        // them. A streamed `0` is accepted, as vLLM accepts it, and
        // `preprocess_chat_request` drops it.
        if request.stream && (prompt_logprobs > 0 || prompt_logprobs == -1) {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "prompt_logprobs are not available when stream=true."
            );
        }
    }

    Ok(())
}
