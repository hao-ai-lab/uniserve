//! HTTP handler for synchronous OpenAI-compatible video generation.
//!
//! Completed video bytes retain the engine-owned immutable mapping for each
//! active response and retained video job.

use std::sync::Arc;

use crate::openai::VideoGenerationRequest;
use crate::openai::serve_error_to_api;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use uniserve_core::{EngineCoreOutput, FinishReason};

use crate::AppState;

use crate::http::utils::resolve_request_id;
use crate::openai::ApiError;

/// Validates and streams one completed video artifact synchronously.
pub(crate) async fn videos_sync(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    VideoBody(body): VideoBody,
) -> Response {
    let started_at = std::time::Instant::now();

    let base_id = resolve_request_id(&headers);
    let request_id = crate::serving::ServeRequestId::new(format!("vid-{base_id}"));
    let submission = match state
        .runtime()
        .model()
        .preprocess_video_request(&request_id, body)
    {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    let mut stream = match state
        .engine()
        .submit_media(request_id.into_inner(), submission)
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return serve_error_to_api(crate::serving::ServeError::Engine(error)).into_response();
        }
    };

    // The synchronous endpoint consumes lifecycle events until the runtime
    // confirms completion and supplies exactly one artifact descriptor.
    let mut artifact = None;
    loop {
        match stream.next().await {
            Some(EngineCoreOutput::Artifact(value)) => artifact = Some(value),
            Some(EngineCoreOutput::Finished {
                reason: FinishReason::Completed,
                ..
            }) => break,
            Some(EngineCoreOutput::Finished { reason, .. }) => {
                return ApiError::server_error(format!(
                    "video generation ended without an artifact: {reason:?}"
                ))
                .into_response();
            }
            Some(EngineCoreOutput::Rejected { message }) => {
                return ApiError::invalid_request(message, None).into_response();
            }
            Some(
                EngineCoreOutput::Error { message }
                | EngineCoreOutput::ArtifactUnavailable { message },
            ) => {
                return ApiError::server_error(message).into_response();
            }
            Some(EngineCoreOutput::Scheduled { .. } | EngineCoreOutput::MediaProgress { .. }) => {}
            Some(_) => {
                return ApiError::server_error(
                    "video runtime emitted an incompatible event".to_string(),
                )
                .into_response();
            }
            None => {
                return ApiError::server_error("video generation task stopped".to_string())
                    .into_response();
            }
        }
    }

    let Some(artifact) = artifact else {
        return ApiError::server_error("video generation produced no artifact".to_string())
            .into_response();
    };

    let media = artifact.media;
    let length = media.len();

    let generation_ms = started_at.elapsed().as_secs_f64() * 1_000.0;
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, &artifact.content_type)
        .header(header::CONTENT_LENGTH, length)
        .header(
            "server-timing",
            format!("generation;dur={generation_ms:.1}"),
        )
        .body(media_body(media))
        .unwrap_or_else(|error| {
            ApiError::server_error(format!("failed to construct media response: {error}"))
                .into_response()
        })
}

/// JSON and multipart share the same strict typed request and capability validation.
pub(crate) struct VideoBody(pub VideoGenerationRequest);

impl<S: Send + Sync> axum::extract::FromRequest<S> for VideoBody {
    type Rejection = Response;

    async fn from_request(
        request: axum::extract::Request,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        let content_type = request
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        if content_type.starts_with("multipart/form-data") {
            let mut multipart = axum::extract::Multipart::from_request(request, state)
                .await
                .map_err(|error| {
                    ApiError::invalid_request(error.to_string(), None).into_response()
                })?;
            let mut fields = serde_json::Map::new();
            while let Some(field) = multipart.next_field().await.map_err(|error| {
                ApiError::invalid_request(error.to_string(), None).into_response()
            })? {
                let name = field.name().unwrap_or("").to_owned();
                if !["model", "prompt", "seconds", "seed"].contains(&name.as_str())
                    || field.file_name().is_some()
                {
                    return Err(ApiError::invalid_request(
                        format!("unsupported video field {name:?}; this checkpoint accepts text-to-video-and-audio only"),
                        None,
                    ).into_response());
                }
                if fields.contains_key(&name) {
                    return Err(ApiError::invalid_request(
                        format!("duplicate video field {name:?}"),
                        None,
                    )
                    .into_response());
                }
                let value = field.text().await.map_err(|error| {
                    ApiError::invalid_request(error.to_string(), None).into_response()
                })?;
                let value = match name.as_str() {
                    "seconds" => serde_json::to_value(value.parse::<f64>().map_err(|_| {
                        ApiError::invalid_request("seconds must be numeric", Some("seconds"))
                            .into_response()
                    })?)
                    .unwrap_or_default(),
                    "seed" => serde_json::Value::from(value.parse::<u64>().map_err(|_| {
                        ApiError::invalid_request("seed must be an unsigned integer", Some("seed"))
                            .into_response()
                    })?),
                    _ => serde_json::Value::String(value),
                };
                fields.insert(name, value);
            }
            serde_json::from_value(serde_json::Value::Object(fields))
                .map(Self)
                .map_err(|error| ApiError::invalid_request(error.to_string(), None).into_response())
        } else if content_type
            .split(';')
            .next()
            .is_some_and(|value| value.trim() == "application/json")
        {
            let axum::Json(value) =
                axum::Json::<VideoGenerationRequest>::from_request(request, state)
                    .await
                    .map_err(|error| {
                        ApiError::invalid_request(error.to_string(), None).into_response()
                    })?;
            Ok(Self(value))
        } else {
            Err((
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                axum::Json(serde_json::json!({"error": {
                    "code": "unsupported_media_type",
                    "message": "use application/json or multipart/form-data"
                }})),
            )
                .into_response())
        }
    }
}

pub(crate) async fn videos_create(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    VideoBody(body): VideoBody,
) -> Response {
    use crate::video_jobs::{VideoFailure, VideoJob, timestamp};
    let requested_seconds = body.seconds;
    let base_id = resolve_request_id(&headers);
    let request_id = crate::serving::ServeRequestId::new(format!("vid-{base_id}"));
    let submission = match state
        .runtime()
        .model()
        .preprocess_video_request(&request_id, body)
    {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    // Public IDs are server-generated; caller request-ID headers cannot collide with retained jobs.
    let id = format!("video_{}", uuid::Uuid::new_v4().simple());
    let record = VideoJob {
        id: id.clone(),
        object: "video",
        model: state.served_model_name().to_owned(),
        created_at: timestamp(),
        completed_at: None,
        expires_at: None,
        seconds: requested_seconds,
        actual_seconds: f64::from(submission.sampling.num_frames) / 24.0,
        status: "queued",
        phase: "queued".to_owned(),
        completed_steps: 0,
        total_steps: submission.sampling.num_inference_steps,
        error: None,
    };
    let cancellation = match state.videos.insert(record.clone()) {
        Ok(token) => token,
        Err(message) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                axum::Json(serde_json::json!({"error": {
                    "code": "video_job_capacity_exceeded", "message": message
                }})),
            )
                .into_response();
        }
    };
    // No await separates reservation and detachment. Dropping this HTTP response cannot abort the job.
    tokio::spawn(async move {
        let result = async {
            let mut stream = state
                .engine()
                .submit_media(id.clone(), submission)
                .await
                .map_err(|error| VideoFailure {
                    code: "submission_failed",
                    message: error.to_string(),
                })?;
            let mut artifact = None;
            let mut cancelled = false;
            loop {
                let event = tokio::select! {
                    event = stream.next() => event,
                    _ = cancellation.cancelled(), if !cancelled => {
                        stream.cancel();
                        cancelled = true;
                        continue;
                    }
                };
                match event {
                    Some(EngineCoreOutput::Scheduled { .. }) => {
                        state.videos.progress(&id, "encoding", 0)
                    }
                    Some(EngineCoreOutput::MediaProgress {
                        phase,
                        completed_steps,
                    }) => state.videos.progress(&id, &phase, completed_steps),
                    Some(EngineCoreOutput::Artifact(value)) => {
                        artifact = Some(value.media);
                    }
                    Some(EngineCoreOutput::Finished {
                        reason: FinishReason::Completed,
                        ..
                    }) => {
                        return artifact.ok_or(VideoFailure {
                            code: "missing_artifact",
                            message: "generation completed without an artifact".to_owned(),
                        });
                    }
                    Some(EngineCoreOutput::Finished { reason, .. }) => {
                        return Err(VideoFailure {
                            code: "generation_terminated",
                            message: format!("generation ended: {reason:?}"),
                        });
                    }
                    Some(EngineCoreOutput::ArtifactUnavailable { message }) => {
                        return Err(VideoFailure {
                            code: "artifact_unavailable",
                            message,
                        });
                    }
                    Some(
                        EngineCoreOutput::Rejected { message }
                        | EngineCoreOutput::Error { message },
                    ) => {
                        return Err(VideoFailure {
                            code: "generation_failed",
                            message,
                        });
                    }
                    None => {
                        return Err(VideoFailure {
                            code: "generation_stopped",
                            message: "video runtime closed before completion".to_owned(),
                        });
                    }
                    _ => {
                        return Err(VideoFailure {
                            code: "invalid_runtime_event",
                            message: "video runtime emitted an incompatible event".to_owned(),
                        });
                    }
                }
            }
        }
        .await;
        state.videos.finish(&id, result);
    });
    (StatusCode::OK, axum::Json(record)).into_response()
}

fn missing_video() -> Response {
    (
        StatusCode::NOT_FOUND,
        axum::Json(serde_json::json!({"error": {
            "code": "video_not_found", "message": "video does not exist or has expired"
        }})),
    )
        .into_response()
}

pub(crate) async fn videos_list(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(
        serde_json::json!({"object": "list", "data": state.videos.list(), "has_more": false}),
    )
    .into_response()
}

pub(crate) async fn videos_get(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    state
        .videos
        .get(&id)
        .map(|record| axum::Json(record).into_response())
        .unwrap_or_else(missing_video)
}

pub(crate) async fn videos_delete(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if state.videos.delete(&id) {
        axum::Json(serde_json::json!({"id": id, "object": "video.deleted", "deleted": true}))
            .into_response()
    } else {
        missing_video()
    }
}

pub(crate) async fn videos_content(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let Some(media) = state.videos.content(&id) else {
        return if state.videos.get(&id).is_some() {
            ApiError::conflict("video has no completed content").into_response()
        } else {
            missing_video()
        };
    };
    let length = media.len();
    (
        [
            (header::CONTENT_TYPE, "video/mp4".to_owned()),
            (header::CONTENT_LENGTH, length.to_string()),
        ],
        media_body(media),
    )
        .into_response()
}

/// Retains the immutable mapping while yielding bounded, independently owned body chunks.
fn media_body<M: AsRef<[u8]> + Send + Sync + 'static>(media: Arc<M>) -> Body {
    let chunks = futures::stream::try_unfold((media, 0_usize), |(media, offset)| async move {
        let bytes = media.as_ref().as_ref();
        if offset == bytes.len() {
            Ok::<_, std::convert::Infallible>(None)
        } else {
            let count = (bytes.len() - offset).min(64 * 1024);
            let chunk = axum::body::Bytes::copy_from_slice(&bytes[offset..offset + count]);
            Ok(Some((chunk, (media, offset + count))))
        }
    });
    Body::from_stream(chunks)
}

pub(crate) async fn capabilities(State(state): State<Arc<AppState>>) -> Response {
    let mut value = serde_json::json!({"model": state.served_model_name()});
    value["video"] = state.runtime().model().video_capabilities();
    value["video_jobs"] = serde_json::json!({
        "max_jobs": crate::video_jobs::MAX_VIDEO_JOBS,
        "max_retained_bytes": crate::video_jobs::MAX_VIDEO_BYTES,
        "retention_seconds": crate::video_jobs::VIDEO_RETENTION.as_secs(),
        "restart_behavior": "jobs and retained content are removed",
    });
    axum::Json(value).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::FromRequest;

    #[tokio::test]
    async fn json_and_multipart_normalize_to_the_same_request() {
        let json = axum::extract::Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                r#"{"model":"FastH3","prompt":"A river","seconds":5.5,"seed":42}"#,
            ))
            .unwrap();
        let multipart = axum::extract::Request::builder().header(header::CONTENT_TYPE, "multipart/form-data; boundary=clip")
            .body(Body::from("--clip\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nFastH3\r\n--clip\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nA river\r\n--clip\r\nContent-Disposition: form-data; name=\"seconds\"\r\n\r\n5.5\r\n--clip\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n42\r\n--clip--\r\n")).unwrap();
        let left = VideoBody::from_request(json, &()).await.unwrap().0;
        let right = VideoBody::from_request(multipart, &()).await.unwrap().0;
        assert_eq!(left, right);
        assert_eq!(left.seconds, 5.5);
        assert_eq!(left.seed, 42);
    }

    #[tokio::test]
    async fn unsupported_conditioning_is_rejected_in_both_formats() {
        let json = axum::extract::Request::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                r#"{"model":"FastH3","prompt":"A river","input_reference":"image.png"}"#,
            ))
            .unwrap();
        let multipart = axum::extract::Request::builder().header(header::CONTENT_TYPE, "multipart/form-data; boundary=clip")
            .body(Body::from("--clip\r\nContent-Disposition: form-data; name=\"input_reference\"; filename=\"image.png\"\r\n\r\nimage\r\n--clip--\r\n")).unwrap();
        for request in [json, multipart] {
            let response = match VideoBody::from_request(request, &()).await {
                Ok(_) => panic!("conditioning was accepted"),
                Err(response) => response,
            };
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }
}
