//! HTTP handlers for OpenAI-compatible video generation.
//!
//! Two submission routes share the request extractor `VideoBody`:
//!
//! - `POST /v1/videos/sync` (`videos_sync`) waits for generation and responds
//!   with the video bytes.
//! - `POST /v1/videos` (`videos_create`) returns a job record at once and
//!   drives generation in a detached task. `GET /v1/videos`,
//!   `GET`/`DELETE /v1/videos/{id}`, and `GET /v1/videos/{id}/content` list,
//!   read, delete (cancelling a running job), and download jobs held in
//!   `crate::video_jobs::VideoJobs`.
//!
//! Both submission routes take one of the job slots of `VideoJobs` before
//! submitting, so `MAX_VIDEO_JOBS` bounds retained jobs and in-flight
//! synchronous requests together.
//!
//! `GET /v1/capabilities` (`capabilities`) reports the model's video limits
//! and the job-store bounds.
//!
//! Completed video bytes arrive as `ArtifactEvent::media`, the read-only
//! shared-memory mapping (`SharedMedia`) the engine claims from the worker's
//! publication. Responses and retained jobs hold that mapping by `Arc`, and
//! `media_body` streams from it without gathering the video into one buffer.

use std::sync::Arc;

use crate::openai::VideoGenerationRequest;
use crate::openai::serve_error_to_api;
use crate::serving::{FinishStatus, RequestOutput};
use axum::Extension;
use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::StreamExt as _;
use uniserve_core::SharedMedia;

use crate::AppState;
use crate::video_jobs::JobSlot;

use crate::http::middleware::RequestId;
use crate::openai::ApiError;

/// Validates and streams one completed video artifact synchronously.
///
/// Responds with the artifact bytes, its content type, and a `Server-Timing`
/// `generation` entry in milliseconds from handler entry (after body
/// extraction) until the runtime reported completion. The request holds one
/// of the job slots the asynchronous route uses from before submission until
/// its response body has been sent or dropped, so a full job store answers
/// `429 Too Many Requests` exactly as `videos_create` does. Submission errors,
/// stream errors, and engine rejections map through `ApiError` (`400` or `503`
/// for a rejection); a failed or non-`Stop` finish, an unexpected event, a
/// closed stream, or a missing artifact is a `500`.
pub(crate) async fn videos_sync(
    State(state): State<Arc<AppState>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    VideoBody(body): VideoBody,
) -> Response {
    let started_at = std::time::Instant::now();

    let request_id = crate::serving::ServeRequestId::new(format!("vid-{base_id}"));

    // As in `videos_create`, the slot is claimed before submission; it is
    // released on any early return and otherwise moves into the response
    // body with the artifact.
    let slot = match state.videos.reserve() {
        Ok(slot) => slot,
        Err(message) => return job_capacity_exceeded(message),
    };

    let mut stream = match state.runtime().generate_video(request_id, body).await {
        Ok(stream) => stream,
        Err(error) => return error.into_response(),
    };

    // The synchronous endpoint consumes lifecycle events until the runtime
    // confirms completion and supplies exactly one artifact descriptor.
    // Acceptance, usage, scheduling, and progress events are ignored; any other
    // event, including `Cancelled` and `Aborted`, is answered as an
    // incompatible runtime event.
    let mut artifact = None;
    loop {
        let event = match stream.next().await.transpose() {
            Ok(event) => event,
            Err(error) => return serve_error_to_api(error).into_response(),
        };
        match event {
            Some(RequestOutput::Artifact(value)) => artifact = Some(value),
            Some(RequestOutput::Finished {
                reason: FinishStatus::Stop { .. },
                ..
            }) => break,
            Some(RequestOutput::Finished { reason, .. }) => {
                return ApiError::server_error(format!(
                    "video generation ended without an artifact: {reason:?}"
                ))
                .into_response();
            }
            Some(RequestOutput::Rejected { kind, message, .. }) => {
                return ApiError::rejected(kind, message).into_response();
            }
            Some(RequestOutput::Failed { message, .. }) => {
                return ApiError::server_error(message).into_response();
            }
            Some(
                RequestOutput::Accepted { .. }
                | RequestOutput::Usage { .. }
                | RequestOutput::Scheduled { .. }
                | RequestOutput::MediaProgress { .. },
            ) => {}
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

    let length = artifact.media.len();
    let media = Arc::new(SlottedMedia {
        media: artifact.media,
        _slot: slot,
    });

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

/// Video request extractor accepting `application/json` or
/// `multipart/form-data`.
///
/// Both content types deserialize into the same `VideoGenerationRequest`, whose
/// `deny_unknown_fields` rejects unknown JSON fields. The multipart path admits
/// only the text fields `model`, `prompt`, `seconds`, and `seed`, rejects file
/// parts and repeated fields, and parses `seconds` as a finite number and
/// `seed` as an unsigned integer before deserializing. Every rejection is
/// `400 Bad Request` except an unsupported content type, which is
/// `415 Unsupported Media Type`. Semantic checks (served model, prompt,
/// duration) happen later, in `InputProcessor::video_sampling` and
/// `InputProcessor::preprocess_video_request`.
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
            // Multipart values are all text, so the accepted fields are
            // assembled into a JSON object with numeric `seconds` and `seed`
            // and deserialized exactly like a JSON body.
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
                    // `f64` parsing also accepts `inf`, `NaN`, and overflowing
                    // literals, which a JSON number cannot represent. They are
                    // refused here, as the JSON body parser refuses them.
                    "seconds" => value
                        .parse::<f64>()
                        .ok()
                        .and_then(serde_json::Number::from_f64)
                        .map(serde_json::Value::Number)
                        .ok_or_else(|| {
                            ApiError::invalid_request(
                                "seconds must be a finite number",
                                Some("seconds"),
                            )
                            .into_response()
                        })?,
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

/// Creates an asynchronous video job and returns its `queued` record.
///
/// Duration and sampling are resolved before submission and fill the record's
/// `seconds`, `actual_seconds`, `num_frames`, and `total_steps`. A full job store answers
/// `429 Too Many Requests`; resolution and submission errors map through
/// `ApiError`. Once the record is inserted, a detached task consumes
/// the runtime stream and publishes progress and the final result through
/// `VideoJobs`, so the job continues after the HTTP response is sent or
/// dropped.
pub(crate) async fn videos_create(
    State(state): State<Arc<AppState>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    VideoBody(body): VideoBody,
) -> Response {
    use crate::video_jobs::{VideoFailure, VideoJob, timestamp};

    let request_id = crate::serving::ServeRequestId::new(format!("vid-{base_id}"));
    let (requested_seconds, sampling) =
        match state
            .runtime()
            .model()
            .video_sampling(&request_id, body.seconds, body.seed)
        {
            Ok(options) => options,
            Err(error) => return error.into_response(),
        };

    // Job capacity is claimed before submission, so a request refused for it
    // never reaches the engine. The slot is released if this handler is
    // dropped before the job record owns it.
    let slot = match state.videos.reserve() {
        Ok(slot) => slot,
        Err(message) => return job_capacity_exceeded(message),
    };

    let mut stream = match state
        .runtime()
        .generate_video(request_id.clone(), body)
        .await
    {
        Ok(stream) => stream,
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
        actual_seconds: f64::from(sampling.num_frames) / f64::from(crate::serving::VIDEO_FPS),
        num_frames: sampling.num_frames,
        status: "queued",
        phase: "queued".to_owned(),
        completed_steps: 0,
        total_steps: sampling.num_inference_steps,
        error: None,
    };
    let cancellation = match state.videos.insert(record.clone(), slot) {
        Ok(token) => token,
        Err(message) => return job_capacity_exceeded(message),
    };

    // No await separates `insert` from `tokio::spawn`, so every inserted record
    // has a task driving it. The task owns the stream, the job's
    // `JobReservation` (and with it the job slot), and an `Arc<AppState>`: the
    // slot stays occupied while either the record or the task exists, so a
    // deleted job holds it until the task ends, and `AppState::shutdown` waits
    // for the task.
    tokio::spawn(async move {
        let result = async {
            let mut artifact = None;
            let mut cancelled = false;
            loop {
                // Cancellation (from `VideoJobs::delete` or `cancel_all`) is
                // forwarded to the runtime once. The token stays cancelled, so
                // the `cancelled` flag disables the branch afterwards and the
                // loop keeps draining the stream until its terminal event. A
                // failed runtime cancel ends the task with
                // `cancellation_failed` instead.
                let event = tokio::select! {
                    event = stream.next() => event,
                    _ = cancellation.cancelled(), if !cancelled => {
                        state.runtime().cancel(request_id.clone()).await.map_err(|error| VideoFailure { code: "cancellation_failed", message: error.to_string() })?;
                        cancelled = true;
                        continue;
                    }
                };
                let event = event.transpose().map_err(|error| VideoFailure { code: "generation_failed", message: error.to_string() })?;
                match event {
                    // Scheduling is reported as the `encoding` phase; later
                    // phases arrive as `MediaProgress`.
                    Some(RequestOutput::Scheduled { .. }) => {
                        state.videos.progress(&id, "encoding", 0)
                    }
                    Some(RequestOutput::MediaProgress {
                        phase,
                        completed_steps,
                    }) => state.videos.progress(&id, &phase, completed_steps),
                    Some(RequestOutput::Artifact(value)) => {
                        artifact = Some(value.media);
                    }
                    Some(RequestOutput::Finished {
                        reason: FinishStatus::Stop { .. },
                        ..
                    }) => {
                        return artifact.ok_or(VideoFailure {
                            code: "missing_artifact",
                            message: "generation completed without an artifact".to_owned(),
                        });
                    }
                    Some(RequestOutput::Finished { reason, .. }) => {
                        return Err(VideoFailure {
                            code: "generation_terminated",
                            message: format!("generation ended: {reason:?}"),
                        });
                    }
                    // A rejection reports the error code the synchronous
                    // route answers the same rejection with.
                    Some(RequestOutput::Rejected { kind, message, .. }) => {
                        let error = ApiError::rejected(kind, message);
                        return Err(VideoFailure {
                            code: error.code(),
                            message: error.to_error_response().error.message,
                        });
                    }
                    Some(RequestOutput::Failed { message, .. }) => {
                        return Err(VideoFailure { code: "generation_failed", message });
                    }
                    Some(RequestOutput::Cancelled { .. } | RequestOutput::Aborted { .. }) => {
                        return Err(VideoFailure { code: "generation_terminated", message: "video generation cancelled".to_owned() });
                    }
                    Some(RequestOutput::Accepted { .. } | RequestOutput::Usage { .. }) => {}
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

        // `finish` fails a successful result whose bytes do not fit the
        // remaining retained-byte budget, and discards the result if the
        // record was deleted.
        state.videos.finish(&id, result);
    });

    (StatusCode::OK, axum::Json(record)).into_response()
}

/// Builds the `429 Too Many Requests` response for a job-store refusal.
fn job_capacity_exceeded(message: &str) -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        axum::Json(serde_json::json!({"error": {
            "code": "video_job_capacity_exceeded", "message": message
        }})),
    )
        .into_response()
}

/// Builds the `404 Not Found` response for an unknown job id. Expired jobs are
/// removed from `VideoJobs`, so they answer the same way.
fn missing_video() -> Response {
    (
        StatusCode::NOT_FOUND,
        axum::Json(serde_json::json!({"error": {
            "code": "video_not_found", "message": "video does not exist or has expired"
        }})),
    )
        .into_response()
}

/// Lists every retained job, newest first. The list is returned whole (at most
/// `MAX_VIDEO_JOBS` records), so `has_more` is always `false`.
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

/// Removes a job record and cancels its generation if it is still running.
///
/// The response does not wait for the cancellation to drain; the job slot is
/// released once the detached generation task has also ended.
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

/// Streams a completed job's video as `video/mp4`.
///
/// Answers `409 Conflict` for a job without content (queued, running, or
/// failed) and `404 Not Found` for an unknown id. The response body holds the
/// `RetainedMedia`, so the mapping and its share of the retained-byte budget
/// stay reserved until the download ends, even if the job is deleted or
/// expires meanwhile.
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

/// A synchronous response's video together with the job slot its request
/// holds.
///
/// The response body owns it (through `media_body`), so the slot, like the
/// mapping, is released once the body has been sent or dropped.
struct SlottedMedia {
    media: Arc<SharedMedia>,
    _slot: JobSlot,
}

impl AsRef<[u8]> for SlottedMedia {
    fn as_ref(&self) -> &[u8] {
        self.media.as_bytes()
    }
}

/// Streams `media` as a body of copied chunks of at most 64 KiB.
///
/// The stream state owns the `Arc`, so the backing mapping, and for
/// `RetainedMedia` its retained-byte permit or for `SlottedMedia` its job
/// slot, lives until the body completes or is dropped.
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

/// Reports the served model, its video limits
/// (`InputProcessor::video_capabilities`, `null` for a model without video),
/// and the job-store bounds from `crate::video_jobs`.
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
    use crate::http::test_support::{SERVED_MODEL, post_json, send, sim_state};
    use crate::profile::ModelParameters;
    use axum::extract::FromRequest;

    fn video_state() -> AppState {
        sim_state(ModelParameters::MiniMaxH3 {
            max_video_seconds: 15.0,
            num_inference_steps: 4,
        })
    }

    fn video_request() -> serde_json::Value {
        serde_json::json!({"model": SERVED_MODEL, "prompt": "A river", "seconds": 5})
    }

    /// Both video submission routes draw on the same job slots: with every
    /// slot taken, each answers with the same `429`, and a synchronous request
    /// gives its slot back once its response is complete.
    #[tokio::test]
    async fn synchronous_videos_share_the_job_slot_bound() {
        let state = Arc::new(video_state());
        let router = crate::http::build_router(Arc::clone(&state));
        let slots = (0..crate::video_jobs::MAX_VIDEO_JOBS)
            .map(|_| state.videos.reserve().unwrap())
            .collect::<Vec<_>>();

        let (async_status, _, async_body) =
            send(&router, post_json("/v1/videos", &video_request())).await;
        let (sync_status, _, sync_body) =
            send(&router, post_json("/v1/videos/sync", &video_request())).await;

        assert_eq!(async_status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(async_body["error"]["code"], "video_job_capacity_exceeded");
        assert_eq!((sync_status, sync_body), (async_status, async_body));

        // With a slot free the request reaches the engine, which refuses it
        // because the simulator has no video media components.
        drop(slots);
        let (status, _, _) = send(&router, post_json("/v1/videos/sync", &video_request())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let slots = (0..crate::video_jobs::MAX_VIDEO_JOBS)
            .map(|_| state.videos.reserve())
            .collect::<Vec<_>>();
        assert!(slots.iter().all(Result::is_ok));
    }

    /// An engine rejection reaches an asynchronous job with the error code the
    /// synchronous route answers the same rejection with.
    #[tokio::test]
    async fn a_rejected_job_reports_the_synchronous_error_code() {
        let router = crate::http::build_router(Arc::new(video_state()));

        // The simulator has no video media components, so the engine rejects
        // the request as invalid on both routes.
        let (sync_status, _, sync_body) =
            send(&router, post_json("/v1/videos/sync", &video_request())).await;
        assert_eq!(sync_status, StatusCode::BAD_REQUEST);

        let (status, _, job) = send(&router, post_json("/v1/videos", &video_request())).await;
        assert_eq!(status, StatusCode::OK);
        let uri = format!("/v1/videos/{}", job["id"].as_str().unwrap());
        let failed = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let request = axum::extract::Request::builder()
                    .uri(&uri)
                    .body(Body::empty())
                    .unwrap();
                let (_, _, job) = send(&router, request).await;
                if job["status"] == "failed" {
                    return job;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();

        assert_eq!(failed["error"]["code"], sync_body["error"]["code"]);
    }

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
        assert_eq!(left.seconds, Some(5.5));
        assert_eq!(left.seed, 42);
    }

    /// A duration Rust parses as a float but JSON cannot carry (infinite, NaN,
    /// or overflowing) is a bad request, as it is for a JSON body, rather than
    /// a request for the default duration.
    #[tokio::test]
    async fn a_non_finite_multipart_duration_is_rejected() {
        for seconds in ["inf", "-infinity", "NaN", "1e400"] {
            let multipart = axum::extract::Request::builder()
                .header(header::CONTENT_TYPE, "multipart/form-data; boundary=clip")
                .body(Body::from(format!(
                    "--clip\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nFastH3\r\n\
                     --clip\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nA river\r\n\
                     --clip\r\nContent-Disposition: form-data; name=\"seconds\"\r\n\r\n{seconds}\r\n\
                     --clip--\r\n"
                )))
                .unwrap();

            match VideoBody::from_request(multipart, &()).await {
                Ok(VideoBody(body)) => panic!("seconds={seconds} was accepted as {body:?}"),
                Err(response) => assert_eq!(response.status(), StatusCode::BAD_REQUEST),
            }
        }
    }

    #[tokio::test]
    async fn unsupported_conditioning_is_rejected_in_both_formats() {
        // The JSON body is refused by the request schema's unknown-field check,
        // the multipart body by the field allowlist and file-part check.
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
