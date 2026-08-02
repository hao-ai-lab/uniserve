use futures::{Stream, StreamExt as _};
use uniserve_core::GenerationConstraint;
use uniserve_openai_types::{
    GeneratedImageData, ImageGenerationRequest, ImageGenerationResponse, ImageOutputFormat,
    ImageResponseFormat,
};
use uniserve_serving::{
    ImageGenerationPolicy, ModalityPolicy, RequestMetadata, SchedulingPolicy, ServeEvent,
    ServeRequest,
};

use crate::openai::error::{ApiError, serve_error_to_api};
use crate::openai::lora::LoraModelResolution;
use crate::openai::utils::{ResolvedRequestContext, check_model_served};

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedImageGeneration {
    pub request_id: String,
    pub response_model: String,
    pub serve_request: ServeRequest,
}

pub fn prepare_image_generation_request(
    request: ImageGenerationRequest,
    lora_resolution: &LoraModelResolution,
    context: ResolvedRequestContext,
) -> Result<PreparedImageGeneration, ApiError> {
    if request.prompt.trim().is_empty() {
        return Err(ApiError::invalid_request(
            "prompt must not be empty".to_string(),
            Some("prompt"),
        ));
    }
    if request.n != 1 {
        return Err(ApiError::invalid_request(
            "this runtime supports exactly one image per request".to_string(),
            Some("n"),
        ));
    }
    if let Some(model) = request.model.as_deref() {
        check_model_served(model, &lora_resolution.model_names)?;
    }
    reject_unsupported_option(request.quality.as_ref(), "quality")?;
    reject_unsupported_option(request.style.as_ref(), "style")?;
    reject_unsupported_option(request.background.as_ref(), "background")?;
    reject_unsupported_option(request.moderation.as_ref(), "moderation")?;
    if matches!(request.response_format, Some(ImageResponseFormat::Url)) {
        return Err(ApiError::invalid_request(
            "response_format=url is unavailable; use b64_json".to_string(),
            Some("response_format"),
        ));
    }
    if matches!(
        request.output_format,
        Some(ImageOutputFormat::Jpeg | ImageOutputFormat::Webp)
    ) {
        return Err(ApiError::invalid_request(
            "only PNG image output is supported".to_string(),
            Some("output_format"),
        ));
    }
    if request.steps.is_some()
        && request.num_inference_steps.is_some()
        && request.steps != request.num_inference_steps
    {
        return Err(ApiError::invalid_request(
            "steps and num_inference_steps must match when both are provided".to_string(),
            Some("num_inference_steps"),
        ));
    }
    let steps = request.steps.or(request.num_inference_steps);
    if steps == Some(0) {
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
    let response_model = lora_resolution.response_model();
    if response_model.is_empty() {
        return Err(ApiError::server_error(
            "image generation has no resolved response model".to_string(),
        ));
    }
    let mut serve_request = ServeRequest::text(request_id.clone(), request.prompt);
    serve_request.generation.constraint = GenerationConstraint::GenOnly;
    serve_request.generation.image = ImageGenerationPolicy {
        width,
        height,
        steps,
        cfg_text_scale: request.guidance_scale,
        cfg_img_scale: request.image_guidance_scale,
        cfg_interval: request.cfg_interval,
        cfg_renorm_type: request.cfg_norm,
        timestep_shift: request.timestep_shift,
        seed: request.seed,
        negative_prompt: request.negative_prompt,
        max_images: Some(1),
        ..ImageGenerationPolicy::default()
    };
    serve_request.modalities = ModalityPolicy {
        input_text: true,
        input_image: false,
        output_text: false,
        output_image: true,
    };
    serve_request.adapter = lora_resolution.adapter.clone();
    serve_request.scheduling = SchedulingPolicy {
        data_parallel_rank: context.data_parallel_rank,
        trace_context: context.trace_context,
        ..SchedulingPolicy::default()
    };
    serve_request.metadata = RequestMetadata {
        tenant: request.user,
        route: Some("/v1/images/generations".to_string()),
        protocol_adapter: Some("openai_images".to_string()),
    };

    Ok(PreparedImageGeneration {
        request_id,
        response_model,
        serve_request,
    })
}

fn reject_unsupported_option<T>(value: Option<&T>, param: &'static str) -> Result<(), ApiError> {
    if value.is_some() {
        return Err(ApiError::invalid_request(
            format!("{param} is not supported by this image runtime"),
            Some(param),
        ));
    }
    Ok(())
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
    stream: impl Stream<Item = uniserve_serving::Result<ServeEvent>> + Send,
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
                if matches!(reason, uniserve_serving::FinishStatus::Error) {
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

#[cfg(test)]
mod tests {
    use futures::stream;
    use uniserve_serving::{CandidateId, FinishStatus};

    use super::*;

    fn resolution() -> LoraModelResolution {
        LoraModelResolution {
            model_names: vec!["image-model".to_string()],
            adapter: uniserve_serving::AdapterSelection::Base,
        }
    }

    #[test]
    fn request_validation_owns_model_size_and_output_controls() {
        let request: ImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "prompt": "draw",
            "model": "missing",
            "size": "1024-by-1024"
        }))
        .unwrap();
        assert!(matches!(
            prepare_image_generation_request(
                request,
                &resolution(),
                ResolvedRequestContext::default(),
            ),
            Err(ApiError::ModelNotFound { .. })
        ));

        let unsupported: ImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "prompt": "draw",
            "model": "image-model",
            "response_format": "url"
        }))
        .unwrap();
        assert!(matches!(
            prepare_image_generation_request(
                unsupported,
                &resolution(),
                ResolvedRequestContext::default(),
            ),
            Err(ApiError::InvalidRequest {
                param: Some("response_format"),
                ..
            })
        ));
    }

    #[test]
    fn request_lowers_to_canonical_image_generation() {
        let request: ImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "prompt": "draw",
            "model": "image-model",
            "size": "640x480",
            "steps": 12,
            "num_inference_steps": 12,
            "seed": 7,
            "negative_prompt": "blur",
            "guidance_scale": 4.0,
            "image_guidance_scale": 1.25,
            "cfg_norm": "none",
            "cfg_interval": [0.1, 0.9],
            "timestep_shift": 3.0
        }))
        .unwrap();
        let prepared = prepare_image_generation_request(
            request,
            &resolution(),
            ResolvedRequestContext {
                request_id: "abc".to_string(),
                ..ResolvedRequestContext::default()
            },
        )
        .unwrap();

        assert_eq!(prepared.request_id, "img-abc");
        assert_eq!(prepared.response_model, "image-model");
        assert_eq!(
            prepared.serve_request.generation.constraint,
            GenerationConstraint::GenOnly
        );
        assert_eq!(prepared.serve_request.generation.image.width, Some(640));
        assert_eq!(prepared.serve_request.generation.image.height, Some(480));
        assert_eq!(prepared.serve_request.generation.image.steps, Some(12));
        assert_eq!(
            prepared.serve_request.generation.image.cfg_text_scale,
            Some(4.0)
        );
        assert_eq!(
            prepared.serve_request.generation.image.cfg_img_scale,
            Some(1.25)
        );
        assert_eq!(
            prepared
                .serve_request
                .generation
                .image
                .cfg_renorm_type
                .as_deref(),
            Some("none")
        );
        assert_eq!(
            prepared.serve_request.generation.image.cfg_interval,
            Some([0.1, 0.9])
        );
        assert_eq!(
            prepared.serve_request.generation.image.timestep_shift,
            Some(3.0)
        );
        assert_eq!(prepared.serve_request.generation.image.seed, Some(7));
        assert!(prepared.serve_request.modalities.output_image);
        assert!(!prepared.serve_request.modalities.output_text);
    }

    #[test]
    fn request_rejects_conflicting_step_aliases() {
        let request: ImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "prompt": "draw",
            "steps": 12,
            "num_inference_steps": 13
        }))
        .unwrap();

        assert!(matches!(
            prepare_image_generation_request(
                request,
                &resolution(),
                ResolvedRequestContext::default(),
            ),
            Err(ApiError::InvalidRequest {
                param: Some("num_inference_steps"),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn response_requires_complete_image_metadata_and_terminal_completion() {
        let complete = stream::iter(vec![
            Ok(ServeEvent::ImageDone {
                candidate_id: CandidateId::PRIMARY,
                image_id: "0".to_string(),
                width: Some(64),
                height: Some(32),
                bytes: Some(3),
                sha256: Some("hash".to_string()),
                pixels_png_b64: Some("cG5n".to_string()),
                elapsed_us: 1,
            }),
            Ok(ServeEvent::Finished {
                candidate_id: CandidateId::PRIMARY,
                reason: FinishStatus::Stop { cause: None },
                finish_detail: Some("image_done".to_string()),
            }),
        ]);
        let response = collect_image_generation(complete, 9).await.unwrap();
        assert_eq!(response.created, 9);
        assert_eq!(response.data[0].width, 64);
        assert_eq!(response.data[0].height, 32);
        assert_eq!(response.data[0].b64_json, "cG5n");

        let incomplete = stream::iter(vec![Ok(ServeEvent::ImageDone {
            candidate_id: CandidateId::PRIMARY,
            image_id: "0".to_string(),
            width: None,
            height: Some(32),
            bytes: Some(3),
            sha256: Some("hash".to_string()),
            pixels_png_b64: Some("cG5n".to_string()),
            elapsed_us: 1,
        })]);
        assert!(matches!(
            collect_image_generation(incomplete, 9).await,
            Err(ApiError::ServerError { .. })
        ));
    }
}
