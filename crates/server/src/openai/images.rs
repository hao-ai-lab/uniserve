use crate::serving::{
    GenerateReqInput, ImageGenControls, ModalitySelection, OutputContract, PromptInput,
    SchedulingBounds, ServeEvent, ServeRequestId,
};
use futures::{Stream, StreamExt as _};

use crate::openai::error::{ApiError, serve_error_to_api};
use crate::openai::types::{GeneratedImageData, ImageGenerationRequest, ImageGenerationResponse};
use crate::openai::utils::{ResolvedRequestContext, check_model_served};

/// Lower one image-generation wire request into the sole generate admission
/// value.
pub fn lower_image_generation_request(
    request: ImageGenerationRequest,
    served_model_name: &str,
    context: ResolvedRequestContext,
) -> Result<GenerateReqInput, ApiError> {
    if request.prompt.trim().is_empty() {
        return Err(ApiError::invalid_request(
            "prompt must not be empty".to_string(),
            Some("prompt"),
        ));
    }
    if request.n != 1 {
        return Err(ApiError::invalid_request(
            "n must be 1 for the configured image-generation route".to_string(),
            Some("n"),
        ));
    }
    if let Some(model) = request.model.as_deref() {
        check_model_served(model, served_model_name)?;
    }
    if request.steps == Some(0) {
        return Err(ApiError::invalid_request(
            "steps must be positive".to_string(),
            Some("steps"),
        ));
    }
    let (width, height) = request
        .size
        .as_deref()
        .map(parse_size)
        .transpose()?
        .map_or((None, None), |(width, height)| (Some(width), Some(height)));
    let request_id = format!("img-{}", context.request_id);
    Ok(GenerateReqInput {
        stream: false,
        prompt: PromptInput::Text(request.prompt),
        modalities: ModalitySelection::Image,
        negative_text: request.negative_prompt,
        image_gen: Some(ImageGenControls {
            width,
            height,
            steps: request.steps,
            cfg_text_scale: request.guidance_scale,
            cfg_img_scale: request.image_guidance_scale,
            cfg_interval: request.cfg_interval,
            cfg_renorm_type: request.cfg_norm,
            timestep_shift: request.timestep_shift,
            seed: request.seed,
            max_images: Some(1),
            ..ImageGenControls::default()
        }),
        output: OutputContract::VisibleText,
        scheduling: SchedulingBounds {
            trace_context: context.trace_context.into_iter().collect(),
            ..SchedulingBounds::default()
        },
        ..GenerateReqInput::text(ServeRequestId::from(request_id), String::new())
    })
}

fn parse_size(size: &str) -> Result<(u32, u32), ApiError> {
    let Some((width, height)) = size.split_once('x') else {
        return Err(ApiError::invalid_request(
            "size must use WIDTHxHEIGHT syntax".to_string(),
            Some("size"),
        ));
    };
    if height.contains('x') {
        return Err(ApiError::invalid_request(
            "size must use WIDTHxHEIGHT syntax".to_string(),
            Some("size"),
        ));
    }
    let width = width.parse::<u32>().ok().filter(|value| *value > 0);
    let height = height.parse::<u32>().ok().filter(|value| *value > 0);
    width.zip(height).ok_or_else(|| {
        ApiError::invalid_request(
            "size dimensions must be positive integers".to_string(),
            Some("size"),
        )
    })
}

pub async fn collect_image_generation(
    stream: impl Stream<Item = crate::serving::Result<ServeEvent>> + Send,
    created: u64,
) -> Result<ImageGenerationResponse, ApiError> {
    futures::pin_mut!(stream);
    let mut data = Vec::new();
    let mut finished = false;
    while let Some(event) = stream.next().await {
        match event.map_err(serve_error_to_api)? {
            ServeEvent::ImageDone {
                height,
                width,
                bytes,
                sha256,
                pixels_png_b64,
                ..
            } => data.push(GeneratedImageData {
                b64_json: required_image_field(pixels_png_b64, "PNG artifact")?,
                revised_prompt: None,
                height: required_image_field(height, "height")?,
                width: required_image_field(width, "width")?,
                bytes: required_image_field(bytes, "byte count")?,
                sha256: required_image_field(sha256, "SHA-256")?,
            }),
            ServeEvent::Finished { reason, .. } => {
                if matches!(reason, crate::serving::FinishStatus::Error) {
                    return Err(ApiError::server_error(
                        "image generation terminated with an execution error".to_string(),
                    ));
                }
                finished = true;
                break;
            }
            ServeEvent::Rejected { message, .. } => {
                return Err(ApiError::invalid_request(message, None));
            }
            ServeEvent::Failed { message, .. } => return Err(ApiError::server_error(message)),
            ServeEvent::Cancelled { .. } => {
                return Err(ApiError::server_error(
                    "image generation request was cancelled".to_string(),
                ));
            }
            ServeEvent::Aborted { .. } => {
                return Err(ApiError::server_error(
                    "image generation request was aborted".to_string(),
                ));
            }
            ServeEvent::Accepted { .. }
            | ServeEvent::Scheduled { .. }
            | ServeEvent::PublicCommit { .. }
            | ServeEvent::TextDelta { .. }
            | ServeEvent::InternalTextDelta { .. }
            | ServeEvent::ReasoningDelta { .. }
            | ServeEvent::OutputBlockStart { .. }
            | ServeEvent::OutputBlockEnd { .. }
            | ServeEvent::ToolCallStart { .. }
            | ServeEvent::ToolCallArgumentsDelta { .. }
            | ServeEvent::ToolCallEnd { .. }
            | ServeEvent::ImageBegin { .. }
            | ServeEvent::ImageStep { .. }
            | ServeEvent::ImageCommit { .. }
            | ServeEvent::Usage { .. } => {}
        }
    }
    if !finished || data.is_empty() {
        return Err(ApiError::server_error(
            "image generation completed without a committed image".to_string(),
        ));
    }
    Ok(ImageGenerationResponse { created, data })
}

fn required_image_field<T>(value: Option<T>, field: &str) -> Result<T, ApiError> {
    value.ok_or_else(|| {
        ApiError::server_error(format!(
            "image generation completed without required {field} metadata"
        ))
    })
}
