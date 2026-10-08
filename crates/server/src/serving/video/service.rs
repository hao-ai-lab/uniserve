//! The MiniMax-H3 video service of one deployment.
//!
//! [`VideoService`] binds what the deployment's denoiser serves, as the
//! worker reports it in its startup handshake ([`VideoDenoiserInfo`]), to the
//! server's media policy, sequence capacity and the checkpoint's vision
//! processor. It prepares every video request for the engine: the fields
//! that only restate the served contract are checked first, ingestion
//! ([`VideoIngest`]) then plans and presents the request, the packed
//! sequence is bounded, each condition's fetched media is published to
//! shared memory for the worker's media reader, and the result becomes one
//! `DiffusionRequest` describing every condition.
//! [`VideoService::capabilities`] describes the same contract for
//! `GET /v1/capabilities`.

use std::sync::Arc;

use serde_json::{Value, json};
use uniserve_core::{
    Canvas, DiffusionRequest, DiffusionSamplingParams, MediaSource, RequestId, VideoTask,
};
use uniserve_engine::VideoDenoiserInfo;

use super::plan::{
    self, CANVAS_MAX_PIXELS, CANVAS_MULTIPLE, ConditionSpec, MAX_ASPECT_RATIO,
    MAX_AUDIO_REFERENCES, MAX_IMAGE_REFERENCES, MAX_REFERENCES, MAX_VIDEO_REFERENCES,
    MIN_ASPECT_RATIO, NAMED_ASPECT_RATIOS, PlanLimits, RequestPlan, Target, VisionConfig,
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
    max_condition_rows: u32,
    /// Condition units one latent encoding round covers, which reference
    /// images are banded to fill (`RequestPlan::image_bands`).
    latent_encoding_lane: u32,
    media: VideoMediaSettings,
    ingest: VideoIngest,
}

impl VideoService {
    /// Binds the served denoiser, the checkpoint's vision processor, the
    /// deployment's duration and condition-row capacities, the condition
    /// units its latent encoder covers in one round, its media policy, and
    /// the tokenizer.
    ///
    /// # Errors
    ///
    /// Fails when the handshake names a task this server does not know or
    /// is otherwise invalid, or when the media fetcher cannot be built.
    pub fn new(
        denoiser: VideoDenoiserInfo,
        vision: VisionConfig,
        max_video_seconds: f64,
        max_condition_rows: u32,
        latent_encoding_lane: u32,
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
            max_condition_rows,
            latent_encoding_lane: latent_encoding_lane.max(1),
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
    /// Reports the tasks and their condition rules, the canvas rule, the
    /// short edges and named aspect ratios the deployment serves and the
    /// canvas each pair resolves to, the duration range, the fixed schedule
    /// (its sigma points, the form `num_inference_steps` restates), the
    /// prompt and sequence capacities, and the media sources and size caps.
    pub fn capabilities(&self, max_prompt_tokens: u32) -> Value {
        // The named ratios served at any short edge, in their canonical
        // order, and each served pair's canvas; `auto` is 16:9 for t2va and
        // ref2va.
        let served = plan::named_canvases(self.ingest.limits());
        let named: Vec<String> = NAMED_ASPECT_RATIOS
            .iter()
            .map(|(width, height)| format!("{width}:{height}"))
            .filter(|ratio| served.iter().any(|(_, served, _)| served == ratio))
            .collect();
        let mut sizes = serde_json::Map::new();
        for (short_edge, aspect_ratio, size) in served {
            let entry = sizes
                .entry(short_edge.to_string())
                .or_insert_with(|| json!({}));
            entry[aspect_ratio] = json!({"width": size.width, "height": size.height});
        }
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
                "short_edges": self.ingest.limits().short_edges(),
                "multiple": CANVAS_MULTIPLE,
                "max_pixels": CANVAS_MAX_PIXELS,
                "aspect_ratios": named,
                "sizes": sizes,
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
            "max_condition_rows": self.max_condition_rows,
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
    /// canvas the plan resolves. The presentation must fit the prompt
    /// capacity (`--max-model-len`), the condition rows the condition
    /// capacity (`--max-condition-rows`), and the packed sequence (text,
    /// conditions, generated audio and video) the denoiser's sequence
    /// capacity. Each condition's fetched media is then published to shared
    /// memory, which the request holds until the engine retires it.
    ///
    /// # Errors
    ///
    /// Returns `invalid_request` naming the field at fault, or the rows the
    /// request needs and the capacity it exceeds, and a server error when
    /// ingestion's own resources or the media publication fail.
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
        self.check_rows(plan, prompt_token_ids.len())?;

        // Each condition's media reaches the worker's media reader through
        // shared memory on this host.
        let image_bands = plan.image_bands(self.latent_encoding_lane);
        let mut media = Vec::with_capacity(prepared.media.len());
        let mut conditions = Vec::with_capacity(prepared.media.len());
        for (condition, fetched) in plan.conditions.iter().zip(&prepared.media) {
            let source = MediaSource::publish(fetched.bytes()).map_err(|error| {
                ApiError::server_error(format!(
                    "publishing conditions[{}] for the worker failed: {error}",
                    condition.index
                ))
            })?;
            conditions.push(condition.describe(plan.canvas, source.locator(), image_bands));
            media.push(Arc::new(source));
        }
        Ok(PreparedVideo {
            request: DiffusionRequest {
                request_id: RequestId(0),
                task: plan.task,
                prompt_token_ids,
                text_tags: prepared.presentation.tags,
                conditions,
                media,
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

    /// Bounds a planned request's denoiser rows.
    ///
    /// The condition rows, counted as the denoiser packs them (densely, or
    /// in the whole tiles its handshake declares), may not exceed the
    /// deployment's condition capacity, and the packed sequence (the
    /// presentation's text rows, the condition rows and the generated audio
    /// and video rows, not counting alignment padding) may not exceed the
    /// denoiser's sequence capacity, when its checkpoint has one.
    ///
    /// # Errors
    ///
    /// Returns `invalid_request` for `conditions` naming the rows the request
    /// needs and the capacity, and for a deployment's condition capacity the
    /// option that raises it; and for a keyframe a tile-packing denoiser
    /// holds no rows for.
    fn check_rows(&self, plan: &RequestPlan, text_rows: usize) -> Result<(), ApiError> {
        let Some(condition_rows) = plan.condition_rows(self.denoiser.condition_tiles) else {
            return Err(ApiError::invalid_request(
                "conditions: this denoiser packs references in region tiles and holds no keyframes",
                Some("conditions"),
            ));
        };
        if condition_rows > u64::from(self.max_condition_rows) {
            return Err(ApiError::invalid_request(
                format!(
                    "conditions: the conditions take {condition_rows} denoiser rows, more than \
                     the {} this deployment serves; serve with a larger --max-condition-rows",
                    self.max_condition_rows
                ),
                Some("conditions"),
            ));
        }
        let sequence_rows = text_rows as u64
            + condition_rows
            + u64::from(plan.target_audio_rows())
            + u64::from(plan.target_video_rows());
        if let Some(capacity) = self.denoiser.max_sequence_rows
            && sequence_rows > u64::from(capacity)
        {
            return Err(ApiError::invalid_request(
                format!(
                    "conditions: the request packs {sequence_rows} denoiser rows, more than the \
                     checkpoint's {capacity}"
                ),
                Some("conditions"),
            ));
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use base64::Engine as _;
    use image::ImageEncoder as _;
    use uniserve_core::{
        Canvas, ConditionMedia, ConditionRole, ConditionVision, ImageFit, VideoTask, VisionGrid,
    };
    use uniserve_engine::{ConditionTiles, VideoDenoiserInfo};

    use super::super::plan::tests::{Request, fixture, vision};
    use super::super::presentation::tests::character_tokenizer;
    use super::VideoService;
    use crate::config::VideoMediaSettings;
    use crate::openai::{ApiError, VideoGenerationRequest};
    use crate::serving::ServeRequestId;

    /// A base denoiser serving every task, with an optional checkpoint
    /// sequence capacity, and a deployment condition capacity.
    fn service(max_condition_rows: u32, max_sequence_rows: Option<u32>) -> VideoService {
        packed_service(max_condition_rows, max_sequence_rows, None)
    }

    /// A denoiser serving every task that packs conditions as `tiles`
    /// prescribes, on a latent encoder of one rank.
    fn packed_service(
        max_condition_rows: u32,
        max_sequence_rows: Option<u32>,
        condition_tiles: Option<ConditionTiles>,
    ) -> VideoService {
        encoding_service(max_condition_rows, max_sequence_rows, condition_tiles, 1)
    }

    /// A denoiser serving every task whose latent encoder covers
    /// `latent_encoding_lane` condition units in one round.
    fn encoding_service(
        max_condition_rows: u32,
        max_sequence_rows: Option<u32>,
        condition_tiles: Option<ConditionTiles>,
        latent_encoding_lane: u32,
    ) -> VideoService {
        VideoService::new(
            VideoDenoiserInfo {
                tasks: ["t2va", "fl2va", "ref2va"].map(str::to_owned).to_vec(),
                schedule_points: 50,
                video_shift: 12.0,
                audio_shift: 3.0,
                canvases: Vec::new(),
                max_sequence_rows,
                condition_tiles,
            },
            vision(&fixture()),
            15.0,
            max_condition_rows,
            latent_encoding_lane,
            &VideoMediaSettings::default(),
            Arc::new(character_tokenizer()),
        )
        .unwrap()
    }

    /// The FastH3 OmniRef reference denoiser's condition tiles: 128 rows, a
    /// video tile of 4 latent frames by 4 by 8 tokens.
    const REGION_TILES: ConditionTiles = ConditionTiles {
        rows: 128,
        video: [4, 4, 8],
    };

    /// The plan of the vectors request named `name`.
    fn planned(name: &str) -> super::RequestPlan {
        let fixture = fixture();
        let case = fixture["requests"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap();
        Request::parse(case).plan(&vision(&fixture)).unwrap()
    }

    fn refusal(error: ApiError) -> String {
        match error {
            ApiError::InvalidRequest { message, param } => {
                assert_eq!(param, Some("conditions"));
                message
            }
            other => panic!("not refused: {other:?}"),
        }
    }

    /// A tile-packing denoiser's conditions take the rows of their whole
    /// tiles. The video-plus-audio request (a 37-frame 1344x768 reference
    /// video whose token grid is 37 x 24 x 42, its 414-row soundtrack, and
    /// a 240-row audio reference) takes 37,296 + 414 + 240 = 37,950 rows
    /// densely, but 10 * 6 * 6 + 4 + 2 = 366 tiles of 128 rows, 46,848, in
    /// 4x4x8-token video tiles: admitted at exactly that capacity, refused
    /// one row below it, while a dense denoiser still counts 37,950.
    #[test]
    fn tile_packed_conditions_are_bounded_by_their_tiles() {
        let plan = planned("ref2va_video_audio");
        let text_rows = 1024;

        assert_eq!(service(37_950, None).check_rows(&plan, text_rows), Ok(()));
        let message = refusal(
            service(37_949, None)
                .check_rows(&plan, text_rows)
                .unwrap_err(),
        );
        assert!(message.contains("37950"), "{message}");

        let packed = |capacity| packed_service(capacity, None, Some(REGION_TILES));
        assert_eq!(packed(46_848).check_rows(&plan, text_rows), Ok(()));
        let message = refusal(packed(46_847).check_rows(&plan, text_rows).unwrap_err());
        assert!(
            message.contains("46848")
                && message.contains("46847")
                && message.contains("--max-condition-rows"),
            "{message}"
        );
    }

    /// The checkpoint's sequence bound counts the tile-packed condition
    /// rows: the video-plus-audio request with 1024 text rows packs 1024 +
    /// 46,848 condition rows + 414 + 37,296 generated rows, 85,582 in all,
    /// and is refused one row below that bound, naming both counts.
    #[test]
    fn the_sequence_bound_counts_tile_packed_rows() {
        let plan = planned("ref2va_video_audio");
        let bounded = |bound| packed_service(1 << 17, Some(bound), Some(REGION_TILES));
        assert_eq!(bounded(85_582).check_rows(&plan, 1024), Ok(()));
        let message = refusal(bounded(85_581).check_rows(&plan, 1024).unwrap_err());
        assert!(
            message.contains("85582") && message.contains("85581"),
            "{message}"
        );
    }

    /// A keyframe has no rows in a region packing, so a tile-packing
    /// denoiser refuses it while a dense one admits it.
    #[test]
    fn tile_packing_refuses_keyframes() {
        let plan = planned("fl2va_first_auto");
        assert_eq!(service(1 << 17, None).check_rows(&plan, 1024), Ok(()));
        let message = refusal(
            packed_service(1 << 17, None, Some(REGION_TILES))
                .check_rows(&plan, 1024)
                .unwrap_err(),
        );
        assert!(message.contains("keyframes"), "{message}");
    }

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(
                &vec![90; (width * height * 3) as usize],
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        encoded
    }

    fn data_uri(bytes: &[u8]) -> String {
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    fn request(body: serde_json::Value) -> VideoGenerationRequest {
        serde_json::from_value(body).unwrap()
    }

    /// Reads a published object's bytes by its locator, as the worker's
    /// media reader does.
    fn published(name: &str, bytes: u64) -> Vec<u8> {
        let path = std::ffi::CString::new(format!("/{name}")).unwrap();
        // SAFETY: path is a valid NUL-terminated POSIX shm name.
        let descriptor = unsafe { libc::shm_open(path.as_ptr(), libc::O_RDONLY, 0) };
        assert!(descriptor >= 0, "the media is not published");
        let mut read = vec![0_u8; bytes as usize];
        // SAFETY: descriptor is open and `read` holds `bytes` writable bytes.
        let count = unsafe { libc::read(descriptor, read.as_mut_ptr().cast(), read.len()) };
        // SAFETY: descriptor is open.
        unsafe { libc::close(descriptor) };
        assert_eq!(count, bytes as isize);
        read
    }

    /// A reference image becomes one condition the worker can read and
    /// encode: its bytes published under its locator, resized to its 2048
    /// short edge, read by the conditioner as one image block, and encoded
    /// in one band of its patch rows per unit of a latent encoding round.
    /// The presentation's tags travel with its tokens.
    #[tokio::test]
    async fn a_reference_image_is_described_for_the_worker() {
        let image = png(160, 90);
        let prepared = encoding_service(1 << 17, None, None, 4)
            .prepare(
                &ServeRequestId::new("reference"),
                &request(serde_json::json!({
                    "model": "minimax_h3",
                    "prompt": "a fox",
                    "task": "ref2va",
                    "conditions": [{"type": "image", "uri": data_uri(&image), "role": "reference"}],
                    "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
                })),
                1 << 16,
            )
            .await
            .unwrap();
        let request = &prepared.request;
        assert_eq!(request.task, VideoTask::Ref2va);
        assert_eq!(request.text_tags.len(), request.prompt_token_ids.len());
        assert_eq!(request.validate(), Ok(()));

        let [condition] = request.conditions.as_slice() else {
            panic!("one condition");
        };
        let size = Canvas {
            width: 3648,
            height: 2048,
        };
        assert_eq!(condition.role, ConditionRole::Reference);
        assert_eq!(
            condition.media,
            ConditionMedia::Image(ImageFit {
                resized: size,
                left: 0,
                top: 0,
                size,
            })
        );
        assert_eq!(
            condition.vision,
            Some(ConditionVision {
                grid: VisionGrid {
                    t: 1,
                    h: 128,
                    w: 228
                },
                tokens: 7296,
                frame_indices: Vec::new(),
            })
        );
        // 64 patch rows of 114 patches, 16 rows to each of four bands.
        assert_eq!(condition.latent_units, [16 * 114; 4]);
        assert_eq!(condition.audio_rows, 0);
        assert_eq!(
            published(&condition.source.name, condition.source.bytes),
            image
        );
    }

    /// The first keyframe is stretched onto the canvas its own aspect sets
    /// and a second one cover-cropped: each is one still image at the
    /// canvas, anchoring its end of the video, and one latent unit on any
    /// latent encoder.
    #[tokio::test]
    async fn keyframes_are_fitted_to_the_canvas() {
        let prepared = encoding_service(1 << 17, None, None, 4)
            .prepare(
                &ServeRequestId::new("keyframes"),
                &request(serde_json::json!({
                    "model": "minimax_h3",
                    "prompt": "a fox",
                    "task": "fl2va",
                    "conditions": [
                        {"type": "image", "uri": data_uri(&png(160, 90)), "role": "keyframe", "frame_index": 0},
                        {"type": "image", "uri": data_uri(&png(90, 90)), "role": "keyframe", "frame_index": -1},
                    ],
                    "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5.0},
                })),
                1 << 16,
            )
            .await
            .unwrap();
        let canvas = prepared.canvas;
        let [first, last] = prepared.request.conditions.as_slice() else {
            panic!("two keyframes");
        };
        assert_eq!(first.role, ConditionRole::FirstFrame);
        assert_eq!(
            first.media,
            ConditionMedia::Image(ImageFit {
                resized: canvas,
                left: 0,
                top: 0,
                size: canvas,
            })
        );
        assert_eq!(last.role, ConditionRole::LastFrame);
        // A square image covers the 1344x768 canvas at 1344x1344, centred.
        assert_eq!(
            last.media,
            ConditionMedia::Image(ImageFit {
                resized: Canvas {
                    width: 1344,
                    height: 1344,
                },
                left: 0,
                top: 288,
                size: canvas,
            })
        );
        assert_eq!(first.latent_units, [1008]);
        assert_eq!(last.latent_units, [1008]);
        assert!(first.vision.is_some() && last.vision.is_some());
    }

    /// Condition rows beyond the deployment's capacity, and a packed
    /// sequence beyond the checkpoint's, are refused naming both counts.
    #[tokio::test]
    async fn rows_beyond_the_capacities_are_refused() {
        let body = || {
            request(serde_json::json!({
                "model": "minimax_h3",
                "prompt": "a fox",
                "task": "ref2va",
                "conditions": [{"type": "image", "uri": data_uri(&png(160, 90)), "role": "reference"}],
                "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
            }))
        };
        let id = ServeRequestId::new("rows");
        let refused = |error: ApiError| match error {
            ApiError::InvalidRequest { message, param } => {
                assert_eq!(param, Some("conditions"));
                message
            }
            other => panic!("not refused: {other:?}"),
        };

        let error = service(7_000, None)
            .prepare(&id, &body(), 1 << 16)
            .await
            .unwrap_err();
        let message = refused(error);
        assert!(
            message.contains("7296")
                && message.contains("7000")
                && message.contains("--max-condition-rows"),
            "{message}"
        );

        // 7296 condition rows, 2 * 207 generated audio rows, 37 * 1008
        // generated video rows and the presentation's text rows.
        let error = service(1 << 17, Some(40_000))
            .prepare(&id, &body(), 1 << 16)
            .await
            .unwrap_err();
        let message = refused(error);
        assert!(message.contains("40000"), "{message}");
    }
}
