//! Deployment-dependent validation for chat-completion requests.

use crate::openai::error::{ApiError, bail_invalid_request};
use crate::openai::types::ChatCompletionRequest;
use crate::openai::utils::{check_model_served, check_prompt_logprobs_bound};

/// Validates relationships that depend on the configured serving route.
pub fn validate_request_compat(
    request: &ChatCompletionRequest,
    served_model_name: &str,
) -> Result<(), ApiError> {
    check_model_served(&request.model, served_model_name)?;

    if let Some(prompt_logprobs) = request.prompt_logprobs {
        check_prompt_logprobs_bound(prompt_logprobs, "prompt_logprobs")?;
        if request.stream && (prompt_logprobs > 0 || prompt_logprobs == -1) {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "prompt_logprobs are not available when stream=true."
            );
        }
    }

    Ok(())
}
