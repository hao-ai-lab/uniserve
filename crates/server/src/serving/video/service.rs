//! The MiniMax-H3 video service of one deployment.
//!
//! [`VideoService`] binds what the deployment's denoiser serves, as the
//! worker reports it in its startup handshake ([`VideoDenoiserInfo`]), to the
//! server's media policy and the checkpoint's vision processor. It prepares
//! every video request for the engine: the fields that only restate the
//! served contract are checked first, ingestion ([`VideoIngest`]) then plans
//! and presents the request, and the result becomes one `DiffusionRequest`.
//! [`VideoService::capabilities`] describes the same contract for
//! `GET /v1/capabilities`.

use serde_json::{Value, json};
use uniserve_core::{Canvas, DiffusionRequest, DiffusionSamplingParams, RequestId, VideoTask};
use uniserve_engine::VideoDenoiserInfo;

use super::plan::{
    self, CANVAS_MAX_PIXELS, CANVAS_MULTIPLE, CANVAS_SHORT_EDGE, ConditionSpec, MAX_ASPECT_RATIO,
    MAX_AUDIO_REFERENCES, MAX_IMAGE_REFERENCES, MAX_REFERENCES, MAX_VIDEO_REFERENCES,
    MIN_ASPECT_RATIO, NAMED_ASPECT_RATIOS, PlanLimits, Target, VisionConfig,
};
use super::probe::{MediaProber, ProbeConfig};
use super::sources::{MediaFetcher, MediaLimits, MediaPolicy, RemoteMediaPolicy};
use super::{ConditionInput, VideoIngest, VideoRequestInput};
use crate::config::VideoMediaSettings;
use crate::openai::{ApiError, VideoCondition, VideoGenerationRequest};
use crate::profile::tokenizer::DynTokenizer;
use crate::serving::{
    MAX_VIDEO_SECONDS, MIN_VIDEO_SECONDS, ServeError, ServeRequestId, VIDEO_FPS, video_frame_count,
};

/// The only numerical quality the video service runs.
const LOSSLESS: &str = "lossless";

/// The fields of the video request body, in schema order.
pub(crate) const REQUEST_FIELDS: [&str; 16] = [
    "model",
    "prompt",
    "task",
    "conditions",
    "target",
    "seed",
    "num_inference_steps",
    "flow_shift",
    "audio_flow_shift",
    "num_outputs_per_prompt",
    "n",
    "quality",
    "seconds",
    "size",
    "width",
    "height",
];

/// A video request ready for submission.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedVideo {
    /// The engine request; its identifier is a placeholder the runtime
    /// replaces with the one it reserves.
    pub request: DiffusionRequest,
    /// The requested duration in seconds, before frame alignment.
    pub duration_seconds: f64,
    /// The generated canvas.
    pub canvas: Canvas,
}

/// Prepares the video requests of one deployment.
pub struct VideoService {
    denoiser: VideoDenoiserInfo,
    tasks: Vec<VideoTask>,
    max_video_seconds: f64,
    media: VideoMediaSettings,
    ingest: VideoIngest,
}

impl VideoService {
    /// Binds the served denoiser, the checkpoint's vision processor, the
    /// deployment's duration capacity and media policy, and the tokenizer.
    ///
    /// # Errors
    ///
    /// Fails when the handshake names a task this server does not know or
    /// is otherwise invalid, or when the media fetcher cannot be built.
    pub fn new(
        denoiser: VideoDenoiserInfo,
        vision: VisionConfig,
        max_video_seconds: f64,
        media: &VideoMediaSettings,
        tokenizer: DynTokenizer,
    ) -> anyhow::Result<Self> {
        denoiser
            .validate()
            .map_err(|error| anyhow::anyhow!("invalid video denoiser handshake: {error}"))?;
        let tasks = denoiser
            .tasks
            .iter()
            .map(|name| {
                VideoTask::from_name(name)
                    .ok_or_else(|| anyhow::anyhow!("the worker serves unknown video task {name:?}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let fetcher = MediaFetcher::new(MediaPolicy {
            media_directory: media.media_directory.clone(),
            remote: RemoteMediaPolicy {
                enabled: media.remote_media,
                ..RemoteMediaPolicy::default()
            },
            limits: MediaLimits::new(media.max_request_bytes),
        })?;
        let prober = MediaProber::new(ProbeConfig::new(media.ffprobe.clone()));
        let limits = PlanLimits {
            tasks: tasks.clone(),
            max_video_seconds,
            // An empty list means the denoiser generates every canvas of the
            // canvas rule.
            canvases: (!denoiser.canvases.is_empty()).then(|| denoiser.canvases.clone()),
        };
        Ok(Self {
            ingest: VideoIngest::new(fetcher, prober, vision, limits, tokenizer),
            denoiser,
            tasks,
            max_video_seconds,
            media: media.clone(),
        })
    }

    /// The denoiser network evaluations of every request: one per schedule
    /// interval.
    pub fn num_inference_steps(&self) -> u32 {
        self.denoiser.schedule_points - 1
    }

    /// Describes the served video contract for `GET /v1/capabilities`.
    ///
    /// Reports the tasks and their condition rules, the canvas rule and the
    /// canvases it yields here, the duration range, the fixed schedule (its
    /// sigma points, the form `num_inference_steps` restates), the prompt and
    /// sequence capacities, and the media sources and size caps.
    pub fn capabilities(&self, max_prompt_tokens: u32) -> Value {
        let served = |size: Canvas| {
            self.denoiser.canvases.is_empty() || self.denoiser.canvases.contains(&size)
        };
        // The named ratios whose canvas this denoiser generates; `auto` is
        // 16:9 for t2va and ref2va.
        let named: Vec<String> = NAMED_ASPECT_RATIOS
            .iter()
            .filter(|(width, height)| {
                plan::canvas(f64::from(*width), f64::from(*height)).is_ok_and(served)
            })
            .map(|(width, height)| format!("{width}:{height}"))
            .collect();
        let limits = MediaLimits::new(self.media.max_request_bytes);
        let conditions: serde_json::Map<String, Value> = self
            .tasks
            .iter()
            .map(|task| {
                let rules = match task {
                    VideoTask::T2va => json!({"conditions": []}),
                    VideoTask::Fl2va => json!({
                        "keyframe_frame_indices": [[0], [-1], [0, -1]],
                        "aspect_ratio": "auto or W:H",
                    }),
                    VideoTask::Ref2va => json!({
                        "max_images": MAX_IMAGE_REFERENCES,
                        "max_videos": MAX_VIDEO_REFERENCES,
                        "max_audios": MAX_AUDIO_REFERENCES,
                        "max_references": MAX_REFERENCES,
                        "keyframe_frame_indices": [[], [0], [-1], [0, -1]],
                    }),
                };
                (task.as_str().to_owned(), rules)
            })
            .collect();
        json!({
            "tasks": self.denoiser.tasks,
            "task_conditions": conditions,
            "canvas": {
                "short_edge": CANVAS_SHORT_EDGE,
                "multiple": CANVAS_MULTIPLE,
                "max_pixels": CANVAS_MAX_PIXELS,
                "aspect_ratios": named,
                "free_aspect_range": [MIN_ASPECT_RATIO, MAX_ASPECT_RATIO],
                "canvases": (!self.denoiser.canvases.is_empty()).then_some(&self.denoiser.canvases),
            },
            "fps": VIDEO_FPS,
            "min_seconds": MIN_VIDEO_SECONDS,
            "max_seconds": self.max_video_seconds,
            "model_max_seconds": MAX_VIDEO_SECONDS,
            "schedule": {
                "num_inference_steps": self.denoiser.schedule_points,
                "flow_shift": self.denoiser.video_shift,
                "audio_flow_shift": self.denoiser.audio_shift,
            },
            "max_prompt_tokens": max_prompt_tokens,
            "max_sequence_rows": self.denoiser.max_sequence_rows,
            "media": {
                "data": true,
                "http": self.media.remote_media,
                "file": self.media.media_directory.is_some(),
                "max_request_bytes": limits.total_bytes,
                "max_image_bytes": limits.image_bytes,
                "max_video_bytes": limits.video_bytes,
                "max_audio_bytes": limits.audio_bytes,
            },
            "request_fields": REQUEST_FIELDS,
        })
    }

    /// Validates, ingests and sizes one video request.
    ///
    /// The schedule, output-count and quality fields are checked before any
    /// media is fetched; the fields the SGLang client restates (`seconds`,
    /// `size`, `width`, `height`) are checked against the duration and
    /// canvas the plan resolves.
    ///
    /// # Errors
    ///
    /// Returns `invalid_request` naming the field at fault, and a server
    /// error when ingestion's own resources fail.
    pub async fn prepare(
        &self,
        request_id: &ServeRequestId,
        request: &VideoGenerationRequest,
        max_prompt_tokens: u32,
    ) -> Result<PreparedVideo, ApiError> {
        self.check_contract(request)?;

        let conditions: Vec<ConditionInput<'_>> =
            request.conditions.iter().map(condition_input).collect();
        let prepared = self
            .ingest
            .prepare(&VideoRequestInput {
                task: request.task,
                prompt: &request.prompt,
                target: Target {
                    short_edge: request.target.short_edge,
                    aspect_ratio: &request.target.aspect_ratio,
                    duration_seconds: request.target.duration_seconds,
                },
                conditions,
            })
            .await?;
        let plan = &prepared.plan;
        check_restated(
            request,
            plan.canvas,
            plan.num_frames,
            self.max_video_seconds,
        )?;

        // The engine request carries the generated video alone: a served task
        // with conditions is planned here but has no engine form to take it.
        if !plan.conditions.is_empty() {
            return Err(ApiError::invalid_request(
                format!(
                    "{} conditions are not executed by this deployment",
                    plan.task.as_str()
                ),
                Some("conditions"),
            ));
        }

        let prompt_token_ids = prepared.presentation.token_ids;
        if prompt_token_ids.len() > max_prompt_tokens as usize {
            return Err(crate::openai::serve_error_to_api(
                ServeError::ContextLengthExceeded {
                    request_id: request_id.clone(),
                    prompt_tokens: prompt_token_ids.len(),
                    max_tokens: max_prompt_tokens,
                },
            ));
        }
        Ok(PreparedVideo {
            request: DiffusionRequest {
                request_id: RequestId(0),
                prompt_token_ids,
                priority: 0,
                sampling: DiffusionSamplingParams {
                    num_frames: plan.num_frames,
                    // Each H3 video media unit decodes one 17-frame temporal
                    // latent window past the first 5 frames.
                    video_units: (plan.num_frames - 5) / 17,
                    num_inference_steps: self.num_inference_steps(),
                    seed: request.seed,
                    width: plan.canvas.width,
                    height: plan.canvas.height,
                },
            },
            duration_seconds: plan.duration_seconds,
            canvas: plan.canvas,
        })
    }

    /// Checks the fields that may only restate the served contract: the
    /// schedule (the checkpoint's own, D5), one output per prompt and the
    /// lossless quality.
    fn check_contract(&self, request: &VideoGenerationRequest) -> Result<(), ApiError> {
        let restate = |param: &'static str, given: String, served: String| {
            ApiError::invalid_request(
                format!(
                    "{param} must be omitted or equal the served schedule's {served}, got {given}"
                ),
                Some(param),
            )
        };
        if let Some(steps) = request.num_inference_steps
            && steps != self.denoiser.schedule_points
        {
            return Err(restate(
                "num_inference_steps",
                steps.to_string(),
                self.denoiser.schedule_points.to_string(),
            ));
        }
        for (param, given, served) in [
            ("flow_shift", request.flow_shift, self.denoiser.video_shift),
            (
                "audio_flow_shift",
                request.audio_flow_shift,
                self.denoiser.audio_shift,
            ),
        ] {
            if let Some(given) = given
                && given != served
            {
                return Err(restate(param, given.to_string(), served.to_string()));
            }
        }
        for (param, count) in [
            ("num_outputs_per_prompt", request.num_outputs_per_prompt),
            ("n", request.n),
        ] {
            if count.is_some_and(|count| count != 1) {
                return Err(ApiError::invalid_request(
                    format!("{param} accepts only 1"),
                    Some(param),
                ));
            }
        }
        if request
            .quality
            .as_deref()
            .is_some_and(|quality| quality != LOSSLESS)
        {
            return Err(ApiError::invalid_request(
                format!("quality accepts only {LOSSLESS:?}"),
                Some("quality"),
            ));
        }
        Ok(())
    }
}

/// The ingestion input of one request condition.
fn condition_input(condition: &VideoCondition) -> ConditionInput<'_> {
    ConditionInput {
        spec: ConditionSpec {
            condition_type: condition.condition_type,
            role: condition.role,
            frame_index: condition.frame_index,
            start_seconds: condition.start_time_seconds,
        },
        uri: &condition.uri,
    }
}

/// Checks the duration and canvas fields the SGLang client restates.
///
/// `seconds` agrees when it aligns to the same frame count as the target's
/// duration; `size` (`WxH`), `width` and `height` agree when they name the
/// planned canvas.
fn check_restated(
    request: &VideoGenerationRequest,
    canvas: Canvas,
    num_frames: u32,
    max_video_seconds: f64,
) -> Result<(), ApiError> {
    if let Some(seconds) = request.seconds
        && video_frame_count(seconds, max_video_seconds).ok() != Some(num_frames)
    {
        return Err(ApiError::invalid_request(
            format!(
                "seconds={seconds} disagrees with the target duration, which generates \
                 {num_frames} frames"
            ),
            Some("seconds"),
        ));
    }
    let size = format!("{}x{}", canvas.width, canvas.height);
    if let Some(given) = &request.size
        && *given != size
    {
        return Err(ApiError::invalid_request(
            format!("size={given:?} disagrees with the target canvas {size}"),
            Some("size"),
        ));
    }
    for (param, given, planned) in [
        ("width", request.width, canvas.width),
        ("height", request.height, canvas.height),
    ] {
        if let Some(given) = given
            && given != planned
        {
            return Err(ApiError::invalid_request(
                format!("{param}={given} disagrees with the target canvas {size}"),
                Some(param),
            ));
        }
    }
    Ok(())
}
