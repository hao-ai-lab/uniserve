use crate::openai::error::{ApiError, bail_invalid_request};
use crate::openai::types::{ChatCompletionRequest, ChatMessage, Tool};
use crate::openai::utils::{check_model_served, check_prompt_logprobs_bound};

/// Validate relationships that depend on the configured serving route.
pub fn validate_request_compat(
    request: &ChatCompletionRequest,
    served_model_names: &[String],
) -> Result<(), ApiError> {
    check_model_served(&request.model, served_model_names)?;

    if let Some(prompt_logprobs) = request.prompt_logprobs {
        check_prompt_logprobs_bound(prompt_logprobs, "prompt_logprobs")?;
        if request.stream && (prompt_logprobs > 0 || prompt_logprobs == -1) {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "prompt_logprobs are not available when stream=true."
            );
        }
    }

    if let Some(tools) = request.tools.as_deref() {
        validate_function_tools(tools, "tools")?;
    }
    for message in &request.messages {
        if let ChatMessage::Developer {
            tools: Some(tools), ..
        } = message
        {
            validate_function_tools(tools, "messages[].tools")?;
        }
    }
    Ok(())
}

fn validate_function_tools(tools: &[Tool], param: &'static str) -> Result<(), ApiError> {
    if tools.iter().any(|tool| tool.tool_type != "function") {
        bail_invalid_request!(param = param, "Only function tools are supported.");
    }
    Ok(())
}
