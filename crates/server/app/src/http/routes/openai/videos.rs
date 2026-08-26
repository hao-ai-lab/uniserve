use std::path::PathBuf;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use tokio::io::AsyncReadExt as _;
use uniserve_engine_gateway::MediaEvent;
use uniserve_protocol_adapters::openai::VideoGenerationRequest;
use uniserve_protocol_adapters::openai::serve_error_to_api;
use uniserve_protocol_adapters::openai::videos::lower_video_generation_request;

use crate::AppState;
use crate::http::error::ApiError;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::resolve_request_context;

struct MediaFile {
    path: PathBuf,
}

impl Drop for MediaFile {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(path = %self.path.display(), %error, "failed to remove media spool file");
        }
    }
}

pub(crate) async fn videos_sync(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<VideoGenerationRequest>,
) -> Response {
    let context = resolve_request_context(&headers);
    let path = state
        .media_spool()
        .join(format!("{}.mp4", uuid::Uuid::new_v4().simple()));
    let guard = MediaFile { path: path.clone() };
    let input = match lower_video_generation_request(
        body,
        state.served_model_names(),
        context,
        path.to_string_lossy().into_owned(),
    ) {
        Ok(input) => input,
        Err(error) => return ApiError::from(error).into_response(),
    };
    let mut stream = match state.runtime().generate_video(input).await {
        Ok(stream) => stream,
        Err(error) => return ApiError::from(serve_error_to_api(error)).into_response(),
    };
    let (mut event_tx, event_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        tokio::select! {
            event = stream.next() => {
                let _ = event_tx.send((event, guard));
            }
            _ = event_tx.closed() => {
                stream.cancel();
                let _ = stream.next().await;
            }
        }
    });
    let (event, guard) = match event_rx.await {
        Ok(result) => result,
        Err(_) => {
            return ApiError::server_error("video generation task stopped".to_string())
                .into_response();
        }
    };
    match event {
        Some(MediaEvent::Completed { bytes }) => {
            let file = match tokio::fs::File::open(&path).await {
                Ok(file) => file,
                Err(error) => {
                    return ApiError::server_error(format!(
                        "failed to open generated media: {error}"
                    ))
                    .into_response();
                }
            };
            let body_stream =
                futures::stream::try_unfold((file, guard), |(mut file, guard)| async move {
                    let mut buffer = vec![0_u8; 64 * 1024];
                    let count = file.read(&mut buffer).await?;
                    if count == 0 {
                        Ok::<_, std::io::Error>(None)
                    } else {
                        buffer.truncate(count);
                        Ok::<_, std::io::Error>(Some((Bytes::from(buffer), (file, guard))))
                    }
                });
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "video/mp4")
                .header(header::CONTENT_LENGTH, bytes)
                .body(Body::from_stream(body_stream))
                .unwrap_or_else(|error| {
                    ApiError::server_error(format!("failed to construct media response: {error}"))
                        .into_response()
                })
        }
        Some(MediaEvent::Rejected { message }) => {
            ApiError::invalid_request(message, None).into_response()
        }
        Some(MediaEvent::Failed { message }) => ApiError::server_error(message).into_response(),
        Some(MediaEvent::Aborted) | None => {
            ApiError::server_error("video generation was aborted".to_string()).into_response()
        }
    }
}
