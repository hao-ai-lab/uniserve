//! MiniMax-H3 condition ingestion: media sources, probing, request planning
//! and the conditioner presentation.
//!
//! A MiniMax-H3 request must be sized before admission: the engine reserves
//! the conditioner's token count and the denoiser's condition rows, so both
//! are known exactly before any pixel is decoded. One request flows through
//! four stages, which [`VideoIngest::prepare`] runs in order:
//!
//! 1. [`plan::check_request`] applies every rule that needs no media
//!    (task, roles, counts, target), so a malformed request is rejected
//!    before anything is fetched.
//! 2. [`sources::MediaFetcher`] resolves each `conditions[i].uri` under the
//!    server's media policy (`data:`, `file://` under the media directory,
//!    `http(s)://` with size, time and address limits) into an owned buffer.
//! 3. [`probe::MediaProber`] reads the facts the plan needs: an image's
//!    displayed size from its header and EXIF orientation; a video's display
//!    aspect, frame rate, decoded frame count and soundtrack, and an audio
//!    file's sample rate and decoded sample count, through `ffprobe`.
//! 4. [`plan::plan_request`] resolves the canvas, the frame counts, how each
//!    condition is prepared, the Qwen3-VL vision grids and the condition rows;
//!    [`presentation::present`] tokenizes the conditioner's presentation and
//!    tags every token for the denoiser's AdaLN modulation.
//!
//! The rules are those of `uniserve_models/minimax_h3/processing.py`; both
//! implementations are tested against
//! `tests/python/fixtures/minimax_h3_plan.json`, generated from the diffusers
//! reference pipeline.
//!
//! Every rejection is a [`VideoInputError::Invalid`] naming the request field
//! at fault, `conditions[i]` for a condition; it converts into an OpenAI
//! `invalid_request_error`. Failures of the server's own resources (a missing
//! `ffprobe`, scratch storage) are [`VideoInputError::Internal`].

pub mod plan;
pub mod presentation;
pub mod probe;
pub mod service;
pub mod sources;

pub use service::{PreparedVideo, VideoService};

use std::fmt;

use crate::openai::ApiError;
use crate::profile::tokenizer::DynTokenizer;

use plan::{ConditionSpec, PlanLimits, RequestPlan, Target, VisionConfig};
use presentation::Presentation;
use probe::MediaProber;
use sources::{FetchedMedia, MediaFetcher};
use uniserve_core::VideoTask;

/// The request field a rejection names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestField {
    /// `task`.
    Task,
    /// `prompt`.
    Prompt,
    /// `target.short_edge`.
    TargetShortEdge,
    /// `target.aspect_ratio`.
    TargetAspectRatio,
    /// `target.duration_seconds`.
    TargetDuration,
    /// The condition list as a whole: counts and keyframe order.
    Conditions,
    /// One condition, by its position in the request.
    Condition(usize),
}

impl RequestField {
    /// The OpenAI error `param` of the field: the field path, with a
    /// condition reported as `conditions` (the message names its index).
    pub const fn param(self) -> &'static str {
        match self {
            Self::Task => "task",
            Self::Prompt => "prompt",
            Self::TargetShortEdge => "target.short_edge",
            Self::TargetAspectRatio => "target.aspect_ratio",
            Self::TargetDuration => "target.duration_seconds",
            Self::Conditions | Self::Condition(_) => "conditions",
        }
    }
}

impl fmt::Display for RequestField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Condition(index) => write!(formatter, "conditions[{index}]"),
            field => formatter.write_str(field.param()),
        }
    }
}

/// A conditioned video request the server does not serve, or could not
/// ingest.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VideoInputError {
    /// The request breaks a rule; the same request fails again on retry.
    #[error("{field}: {message}")]
    Invalid {
        /// The field at fault.
        field: RequestField,
        /// What is wrong with it.
        message: String,
    },
    /// The server failed while ingesting valid input, for example because
    /// `ffprobe` could not run or scratch storage failed.
    #[error("{0}")]
    Internal(String),
}

impl VideoInputError {
    /// Rejects `field` with `message`.
    pub fn invalid(field: RequestField, message: impl Into<String>) -> Self {
        Self::Invalid {
            field,
            message: message.into(),
        }
    }

    /// Rejects `conditions[index]` with `message`.
    pub fn condition(index: usize, message: impl Into<String>) -> Self {
        Self::invalid(RequestField::Condition(index), message)
    }

    /// Reports a server-side ingestion failure.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    /// The field a rejection names; `None` for a server failure.
    pub const fn field(&self) -> Option<RequestField> {
        match self {
            Self::Invalid { field, .. } => Some(*field),
            Self::Internal(_) => None,
        }
    }
}

impl From<VideoInputError> for ApiError {
    /// A rejection becomes a 400 `invalid_request_error` whose message starts
    /// with the field path; a server failure becomes a 500 `server_error`.
    fn from(error: VideoInputError) -> Self {
        match &error {
            VideoInputError::Invalid { field, .. } => {
                ApiError::invalid_request(error.to_string(), Some(field.param()))
            }
            VideoInputError::Internal(message) => {
                ApiError::server_error(format!("video condition ingestion failed: {message}"))
            }
        }
    }
}

/// One condition of a request, as the request states it.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionInput<'a> {
    /// The condition's type, role, keyframe index and start offset.
    pub spec: ConditionSpec,
    /// Where the media comes from.
    pub uri: &'a str,
}

/// The fields of a MiniMax-H3 request that ingestion reads.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoRequestInput<'a> {
    /// The request's task.
    pub task: VideoTask,
    /// The prompt, presented after the conditions.
    pub prompt: &'a str,
    /// The requested output.
    pub target: Target<'a>,
    /// The conditions, in request order.
    pub conditions: Vec<ConditionInput<'a>>,
}

/// A request ready for admission: its sizes, its presentation and the media
/// bytes of its conditions.
#[derive(Debug, Clone)]
pub struct PreparedVideoRequest {
    /// Canvas, frame counts, condition preparation and rows.
    pub plan: RequestPlan,
    /// The conditioner's token ids and their AdaLN tags.
    pub presentation: Presentation,
    /// The fetched media of each condition, in request order.
    pub media: Vec<FetchedMedia>,
}

/// Ingests MiniMax-H3 requests for one served checkpoint.
///
/// One instance is shared by all requests; [`VideoIngest::prepare`] takes
/// `&self` and runs concurrently.
pub struct VideoIngest {
    fetcher: MediaFetcher,
    prober: MediaProber,
    vision: VisionConfig,
    limits: PlanLimits,
    tokenizer: DynTokenizer,
}

impl VideoIngest {
    /// Combines the media policy, the prober, the checkpoint's vision
    /// processor geometry and serving limits, and its tokenizer.
    pub fn new(
        fetcher: MediaFetcher,
        prober: MediaProber,
        vision: VisionConfig,
        limits: PlanLimits,
        tokenizer: DynTokenizer,
    ) -> Self {
        Self {
            fetcher,
            prober,
            vision,
            limits,
            tokenizer,
        }
    }

    /// Validates, fetches, probes, plans and presents one request.
    ///
    /// The request-only rules run before any fetch. Fetches and probes of
    /// different conditions run concurrently; the first failure ends the
    /// request.
    ///
    /// # Errors
    ///
    /// Returns [`VideoInputError::Invalid`] naming the field at fault for a
    /// request the server does not serve, and [`VideoInputError::Internal`]
    /// when the server's own ingestion resources fail.
    pub async fn prepare(
        &self,
        request: &VideoRequestInput<'_>,
    ) -> Result<PreparedVideoRequest, VideoInputError> {
        let specs: Vec<ConditionSpec> = request
            .conditions
            .iter()
            .map(|condition| condition.spec.clone())
            .collect();
        plan::check_request(request.task, &request.target, &specs, &self.limits)?;

        let sources: Vec<(plan::ConditionType, &str)> = request
            .conditions
            .iter()
            .map(|condition| (condition.spec.condition_type, condition.uri))
            .collect();
        let media = self.fetcher.fetch_all(&sources).await?;
        let probes =
            futures::future::try_join_all(media.iter().enumerate().map(|(index, fetched)| {
                self.prober
                    .probe(index, specs[index].condition_type, fetched.bytes())
            }))
            .await?;

        let plan = plan::plan_request(
            request.task,
            &request.target,
            &specs,
            &probes,
            &self.vision,
            &self.limits,
        )?;
        let presentation = presentation::present(&self.tokenizer, &plan, request.prompt)?;
        Ok(PreparedVideoRequest {
            plan,
            presentation,
            media,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use axum::Router;
    use axum::routing::get;
    use base64::Engine as _;
    use image::ImageEncoder as _;

    use super::plan::tests::{fixture, limits, vision};
    use super::plan::{ConditionRole, ConditionSpec, ConditionType, Prepared, Target};
    use super::presentation::tests::character_tokenizer;
    use super::presentation::{TEXT_TAG, VIDEO_TAG};
    use super::probe::{MediaProber, ProbeConfig};
    use super::sources::{MediaFetcher, MediaLimits, MediaPolicy, RemoteMediaPolicy};
    use super::{ConditionInput, RequestField, VideoIngest, VideoInputError, VideoRequestInput};
    use crate::openai::ApiError;
    use uniserve_core::VideoTask;

    fn ingest(scratch: &std::path::Path) -> VideoIngest {
        let fetcher = MediaFetcher::new(MediaPolicy {
            media_directory: None,
            remote: RemoteMediaPolicy {
                allow_private_addresses: true,
                ..RemoteMediaPolicy::default()
            },
            limits: MediaLimits::new(64 << 20),
        })
        .unwrap();
        // Image conditions are probed in process; `ffprobe` is never run.
        let prober = MediaProber::new(ProbeConfig {
            ffprobe: scratch.join("ffprobe"),
            timeout: Duration::from_secs(5),
            scratch_directory: scratch.to_path_buf(),
            max_concurrent_probes: 2,
        });
        VideoIngest::new(
            fetcher,
            prober,
            vision(&fixture()),
            limits(),
            Arc::new(character_tokenizer()),
        )
    }

    fn png_data_uri(width: u32, height: u32) -> String {
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(
                &vec![90; (width * height * 3) as usize],
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(encoded)
        )
    }

    const REFERENCE: ConditionSpec = ConditionSpec {
        condition_type: ConditionType::Image,
        role: ConditionRole::Reference,
        frame_index: None,
        start_seconds: None,
    };

    /// A request is fetched, probed, planned and presented, and keeps its
    /// media bytes for the worker.
    #[tokio::test]
    async fn a_request_is_prepared_end_to_end() {
        let scratch = tempfile::tempdir().unwrap();
        let ingest = ingest(scratch.path());
        let uri = png_data_uri(160, 90);
        let request = VideoRequestInput {
            task: VideoTask::Ref2va,
            prompt: "a fox",
            target: Target {
                short_edge: 768,
                aspect_ratio: "auto",
                duration_seconds: Some(5.0),
            },
            conditions: vec![ConditionInput {
                spec: REFERENCE,
                uri: &uri,
            }],
        };
        let prepared = ingest.prepare(&request).await.unwrap();

        // A 16:9 image reference is encoded at 3648x2048: 7296 rows and as
        // many vision tokens.
        let plan = &prepared.plan;
        assert_eq!((plan.canvas.width, plan.canvas.height), (1344, 768));
        assert_eq!(plan.num_frames, 124);
        let Prepared::Image(size) = plan.conditions[0].prepared else {
            panic!("not an image reference");
        };
        assert_eq!((size.width, size.height), (3648, 2048));
        assert_eq!(plan.condition_video_rows(), 7296);

        let label = "<Picture 1>: ".chars().count();
        let tags = &prepared.presentation.tags;
        assert_eq!(tags.len(), label + 7296 + 2 + "a fox".len());
        assert!(tags[..label].iter().all(|&tag| tag == TEXT_TAG));
        assert!(
            tags[label..label + 7298]
                .iter()
                .all(|&tag| tag == VIDEO_TAG)
        );
        assert!(tags[label + 7298..].iter().all(|&tag| tag == TEXT_TAG));
        assert_eq!(prepared.media.len(), 1);
        assert!(prepared.media[0].bytes.starts_with(b"\x89PNG"));
    }

    /// A request that breaks a request-only rule is rejected before any of
    /// its media is fetched.
    #[tokio::test]
    async fn request_rules_apply_before_fetching() {
        let fetches = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&fetches);
        let router = Router::new().route(
            "/image.png",
            get(move || {
                counter.fetch_add(1, Ordering::Relaxed);
                async { Vec::<u8>::new() }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        let scratch = tempfile::tempdir().unwrap();
        let ingest = ingest(scratch.path());
        let uri = format!("http://{address}/image.png");
        let request = VideoRequestInput {
            task: VideoTask::Ref2va,
            prompt: "a fox",
            target: Target {
                short_edge: 768,
                aspect_ratio: "auto",
                duration_seconds: Some(3.0),
            },
            conditions: vec![ConditionInput {
                spec: REFERENCE,
                uri: &uri,
            }],
        };
        let error = ingest.prepare(&request).await.unwrap_err();
        assert_eq!(error.field(), Some(RequestField::TargetDuration));
        assert_eq!(fetches.load(Ordering::Relaxed), 0);
    }

    /// A rejection is an invalid request that names the condition at fault
    /// in its message and the condition list as its parameter; a server
    /// failure is a server error.
    #[test]
    fn errors_map_to_openai_categories() {
        let rejected: ApiError = VideoInputError::condition(2, "is not an image").into();
        assert_eq!(
            rejected,
            ApiError::invalid_request("conditions[2]: is not an image", Some("conditions"))
        );

        let rejected: ApiError =
            VideoInputError::invalid(RequestField::TargetDuration, "is required").into();
        assert_eq!(
            rejected,
            ApiError::invalid_request(
                "target.duration_seconds: is required",
                Some("target.duration_seconds")
            )
        );

        let failed: ApiError = VideoInputError::internal("ffprobe did not start").into();
        assert_eq!(failed.code(), "server_error");
    }
}
