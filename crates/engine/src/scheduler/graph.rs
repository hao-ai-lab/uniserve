//! The call graph of one video request.
//!
//! Admission builds a [`VideoGraph`] from the request's task and conditions:
//! which media calls the request runs, the products each one publishes with
//! their exact leading extents, and which calls read them. A `t2va` request
//! is text encoding, latent preparation, the denoising ladder, both decoders,
//! both encoders and the muxer. A conditioned request first reads its
//! condition media on a host rank (`MediaReading`), encodes the vision blocks
//! the presentation names (`VisionEncoding`) ahead of the text encoding that
//! splices them in, and encodes its visual conditions by temporal units and
//! its audio tracks separately (`LatentEncoding`); latent preparation then
//! reads the text features and every condition latent.
//!
//! Products are named as the loaded components declare them, so admission
//! finds each product's output on the component serving its call by name.
//! The graph holds sizes only: it never inspects media.

use uniserve_core::{DiffusionRequest, VideoTask};
use uniserve_worker_ipc::MediaCall;

/// Product names, as the worker's components declare their outputs.
pub(crate) mod product {
    /// The media reader's RGB24 pixels of every visual condition, `[pixels,
    /// 3]` bytes in request order, each condition's frames whole.
    pub(crate) const CONDITION_PIXELS: &str = "condition_pixels";
    /// The media reader's stereo PCM of every audio track, `[samples, 2]` in
    /// request order.
    pub(crate) const CONDITION_SAMPLES: &str = "condition_samples";
    /// The media reader's vision processor patches of every vision block,
    /// `[patches, features]` in presentation order.
    pub(crate) const VISION_PIXELS: &str = "vision_pixels";
    /// The vision encoder's token features, one row per vision placeholder.
    pub(crate) const VISION_FEATURES: &str = "vision_features";
    /// The text encoder's prompt features, one row per presentation token.
    pub(crate) const CONDITIONING: &str = "conditioning";
    /// The latent encoder's visual condition rows in request order.
    pub(crate) const CONDITION_VIDEO_LATENTS: &str = "condition_video_latents";
    /// The latent encoder's audio condition rows in request order.
    pub(crate) const CONDITION_AUDIO_LATENTS: &str = "condition_audio_latents";
    /// The denoiser's final video latents.
    pub(crate) const VIDEO_LATENTS: &str = "video_latents";
    /// The denoiser's final audio latents.
    pub(crate) const AUDIO_LATENTS: &str = "audio_latents";
    /// The video decoder's RGB media units.
    pub(crate) const VIDEO_UNITS: &str = "video_units";
    /// The video codec's encoded media unit rows.
    pub(crate) const ENCODED_UNITS: &str = "encoded_units";
    /// The audio decoder's PCM track.
    pub(crate) const AUDIO_SAMPLES: &str = "audio_samples";
}

/// One product a request reserves at admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphProduct {
    /// The call that publishes it.
    pub(crate) call: MediaCall,
    /// The name its component declares it under.
    pub(crate) name: &'static str,
    /// The request's exact extent of the product's device-sized axis, or
    /// `None` for a product whose every extent is static.
    pub(crate) extent: Option<u32>,
}

/// The media calls of one video request and the products connecting them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VideoGraph {
    calls: Vec<MediaCall>,
    products: Vec<GraphProduct>,
    /// Denoiser rows of each temporal unit the visual latent encoding
    /// covers, condition by condition in request order.
    condition_units: Vec<u32>,
    /// Whether any condition carries an audio track to encode.
    condition_audio: bool,
}

/// The calls whose consumption order is fixed by kind, as one request's
/// graph connects them: each producing call with its reading calls.
const EDGES: [(MediaCall, &[MediaCall]); 11] = [
    (
        MediaCall::MediaReading,
        &[MediaCall::VisionEncoding, MediaCall::LatentEncoding],
    ),
    (MediaCall::VisionEncoding, &[MediaCall::TextEncoding]),
    (MediaCall::TextEncoding, &[MediaCall::LatentPreparation]),
    (MediaCall::LatentEncoding, &[MediaCall::LatentPreparation]),
    (MediaCall::LatentPreparation, &[MediaCall::Denoising]),
    (
        MediaCall::Denoising,
        &[
            MediaCall::Denoising,
            MediaCall::VideoDecoding,
            MediaCall::AudioDecoding,
        ],
    ),
    (MediaCall::VideoDecoding, &[MediaCall::VideoEncoding]),
    (MediaCall::VideoEncoding, &[MediaCall::Muxing]),
    (MediaCall::AudioDecoding, &[MediaCall::AudioEncoding]),
    (MediaCall::AudioEncoding, &[]),
    (MediaCall::Muxing, &[]),
];

/// The media calls every video request runs, whatever its task.
const GENERATION_CALLS: [MediaCall; 8] = [
    MediaCall::TextEncoding,
    MediaCall::LatentPreparation,
    MediaCall::Denoising,
    MediaCall::VideoDecoding,
    MediaCall::AudioDecoding,
    MediaCall::VideoEncoding,
    MediaCall::AudioEncoding,
    MediaCall::Muxing,
];

/// The media calls a request of `task` runs: the generation calls, and for a
/// conditioned task the calls reading and encoding its conditions. Every
/// keyframe and every `ref2va` request presents vision blocks, so a
/// conditioned task always encodes vision.
pub(crate) fn task_calls(task: VideoTask) -> Vec<MediaCall> {
    let conditions: &[MediaCall] = match task {
        VideoTask::T2va => &[],
        VideoTask::Fl2va | VideoTask::Ref2va => &[
            MediaCall::MediaReading,
            MediaCall::VisionEncoding,
            MediaCall::LatentEncoding,
        ],
    };
    conditions
        .iter()
        .chain(GENERATION_CALLS.iter())
        .copied()
        .collect()
}

impl VideoGraph {
    /// Builds the graph of a request that passed `DiffusionRequest::validate`.
    ///
    /// # Errors
    ///
    /// Returns a message when a product extent does not fit the protocol's
    /// `u32` dimensions.
    pub(crate) fn new(request: &DiffusionRequest) -> Result<Self, String> {
        let extent = |name: &str, value: u64| {
            u32::try_from(value)
                .map_err(|_| format!("the request's {name} exceed the protocol's extents"))
        };
        let conditions = &request.conditions;
        let pixels: u64 = conditions
            .iter()
            .map(|condition| condition.pixel_bytes() / 3)
            .sum();
        let samples: u64 = conditions
            .iter()
            .filter_map(|condition| condition.media.audio())
            .map(|track| track.samples)
            .sum();
        let patches: u64 = conditions
            .iter()
            .filter_map(|condition| condition.vision.as_ref())
            .map(|vision| vision.grid.patches())
            .sum();
        let tokens: u64 = conditions
            .iter()
            .filter_map(|condition| condition.vision.as_ref())
            .map(|vision| u64::from(vision.tokens))
            .sum();
        let video_rows: u64 = conditions.iter().map(|c| c.video_rows()).sum();
        let audio_rows: u64 = conditions.iter().map(|c| u64::from(c.audio_rows)).sum();

        let mut calls = Vec::new();
        let mut products = Vec::new();
        let mut produce = |call: MediaCall, name: &'static str, extent: Option<u32>| {
            products.push(GraphProduct { call, name, extent });
        };
        if request.task != VideoTask::T2va {
            calls.push(MediaCall::MediaReading);
            if pixels > 0 {
                produce(
                    MediaCall::MediaReading,
                    product::CONDITION_PIXELS,
                    Some(extent("condition pixels", pixels)?),
                );
            }
            if samples > 0 {
                produce(
                    MediaCall::MediaReading,
                    product::CONDITION_SAMPLES,
                    Some(extent("condition samples", samples)?),
                );
            }
            if patches > 0 {
                calls.push(MediaCall::VisionEncoding);
                produce(
                    MediaCall::MediaReading,
                    product::VISION_PIXELS,
                    Some(extent("vision patches", patches)?),
                );
                produce(
                    MediaCall::VisionEncoding,
                    product::VISION_FEATURES,
                    Some(extent("vision tokens", tokens)?),
                );
            }
            calls.push(MediaCall::LatentEncoding);
            if video_rows > 0 {
                produce(
                    MediaCall::LatentEncoding,
                    product::CONDITION_VIDEO_LATENTS,
                    Some(extent("condition video rows", video_rows)?),
                );
            }
            if audio_rows > 0 {
                produce(
                    MediaCall::LatentEncoding,
                    product::CONDITION_AUDIO_LATENTS,
                    Some(extent("condition audio rows", audio_rows)?),
                );
            }
        }
        calls.extend(GENERATION_CALLS);
        produce(
            MediaCall::TextEncoding,
            product::CONDITIONING,
            Some(extent(
                "prompt tokens",
                request.prompt_token_ids.len() as u64,
            )?),
        );
        produce(MediaCall::Denoising, product::VIDEO_LATENTS, None);
        produce(MediaCall::Denoising, product::AUDIO_LATENTS, None);
        produce(
            MediaCall::VideoDecoding,
            product::VIDEO_UNITS,
            Some(request.sampling.video_units),
        );
        produce(
            MediaCall::VideoEncoding,
            product::ENCODED_UNITS,
            Some(request.sampling.video_units),
        );
        produce(MediaCall::AudioDecoding, product::AUDIO_SAMPLES, None);

        Ok(Self {
            calls,
            products,
            condition_units: conditions
                .iter()
                .flat_map(|condition| condition.latent_units.iter().copied())
                .collect(),
            condition_audio: audio_rows > 0,
        })
    }

    /// The media calls the request runs, in graph order.
    pub(crate) fn calls(&self) -> &[MediaCall] {
        &self.calls
    }

    /// Whether the request runs calls of this kind.
    pub(crate) fn runs(&self, call: MediaCall) -> bool {
        self.calls.contains(&call)
    }

    /// The calls of this request that read the products of a call of kind
    /// `call`; empty for a call the request does not run.
    pub(crate) fn consumers(&self, call: MediaCall) -> Vec<MediaCall> {
        if !self.runs(call) {
            return Vec::new();
        }
        EDGES
            .iter()
            .find(|(producer, _)| *producer == call)
            .map(|(_, readers)| {
                readers
                    .iter()
                    .copied()
                    .filter(|reader| self.runs(*reader))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Every product the request reserves at admission.
    pub(crate) fn products(&self) -> &[GraphProduct] {
        &self.products
    }

    /// The product a call publishes under `name`.
    pub(crate) fn product(&self, name: &str) -> Option<&GraphProduct> {
        self.products.iter().find(|product| product.name == name)
    }

    /// Whether the request reads condition media before encoding.
    pub(crate) fn reads_media(&self) -> bool {
        self.runs(MediaCall::MediaReading)
    }

    /// Whether the presentation holds vision blocks to encode.
    pub(crate) fn encodes_vision(&self) -> bool {
        self.runs(MediaCall::VisionEncoding)
    }

    /// Denoiser rows of each visual condition unit, in encoding order.
    pub(crate) fn condition_units(&self) -> &[u32] {
        &self.condition_units
    }

    /// Whether any condition carries an audio track to encode.
    pub(crate) fn encodes_condition_audio(&self) -> bool {
        self.condition_audio
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use uniserve_core::{
        AudioClip, Canvas, ConditionMedia, ConditionRole, ConditionVision, DiffusionSamplingParams,
        ImageFit, MediaSource, RequestId, VideoClip, VideoCondition, VisionGrid,
    };

    use super::*;

    fn request(task: VideoTask, conditions: Vec<VideoCondition>) -> DiffusionRequest {
        DiffusionRequest {
            request_id: RequestId(1),
            task,
            prompt_token_ids: vec![1; 40],
            text_tags: vec![1; 40],
            conditions,
            media: Vec::new(),
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: 124,
                video_units: 7,
                num_inference_steps: 49,
                seed: 42,
                width: 1344,
                height: 768,
            },
        }
    }

    fn source() -> Arc<MediaSource> {
        Arc::new(MediaSource::publish(b"media").unwrap())
    }

    fn canvas() -> Canvas {
        Canvas {
            width: 1344,
            height: 768,
        }
    }

    /// A text request runs the generation calls alone and reserves only the
    /// prompt features, the latents and the media output products.
    #[test]
    fn a_text_request_runs_the_generation_calls_alone() {
        let graph = VideoGraph::new(&request(VideoTask::T2va, Vec::new())).unwrap();
        assert!(!graph.reads_media());
        assert!(!graph.runs(MediaCall::LatentEncoding));
        assert_eq!(
            graph.consumers(MediaCall::TextEncoding),
            [MediaCall::LatentPreparation]
        );
        let names = graph
            .products()
            .iter()
            .map(|product| product.name)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                product::CONDITIONING,
                product::VIDEO_LATENTS,
                product::AUDIO_LATENTS,
                product::VIDEO_UNITS,
                product::ENCODED_UNITS,
                product::AUDIO_SAMPLES,
            ]
        );
        assert_eq!(
            graph.product(product::CONDITIONING).unwrap().extent,
            Some(40)
        );
    }

    /// A video reference with its soundtrack is read, its vision blocks
    /// encoded before the text, its clips encoded by units and its
    /// soundtrack separately; the products carry the request's exact sizes.
    #[test]
    fn a_reference_request_reads_and_encodes_its_conditions() {
        let media = source();
        let soundtrack = AudioClip {
            sample_rate: 48_000,
            start_sample: 0,
            source_samples: 248_000,
            samples: 165_334,
        };
        let condition = VideoCondition {
            role: ConditionRole::Reference,
            source: media.locator(),
            media: ConditionMedia::Video {
                clip: VideoClip {
                    canvas: canvas(),
                    start_frame: 0,
                    frames: 124,
                    vae_frames: 124,
                },
                soundtrack: Some(soundtrack),
            },
            vision: Some(ConditionVision {
                grid: VisionGrid { t: 6, h: 42, w: 24 },
                tokens: 1512,
                frame_indices: vec![0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
            }),
            latent_units: [vec![5040; 7], vec![2016]].concat(),
            audio_rows: 414,
        };
        let image = VideoCondition {
            role: ConditionRole::Reference,
            source: media.locator(),
            media: ConditionMedia::Image(ImageFit {
                resized: canvas(),
                left: 0,
                top: 0,
                size: canvas(),
            }),
            vision: Some(ConditionVision {
                grid: VisionGrid { t: 1, h: 48, w: 84 },
                tokens: 1008,
                frame_indices: Vec::new(),
            }),
            latent_units: vec![1008],
            audio_rows: 0,
        };
        let graph = VideoGraph::new(&request(VideoTask::Ref2va, vec![condition, image])).unwrap();

        assert_eq!(
            graph.calls()[..3],
            [
                MediaCall::MediaReading,
                MediaCall::VisionEncoding,
                MediaCall::LatentEncoding
            ]
        );
        assert_eq!(
            graph.consumers(MediaCall::MediaReading),
            [MediaCall::VisionEncoding, MediaCall::LatentEncoding]
        );
        assert_eq!(
            graph.consumers(MediaCall::VisionEncoding),
            [MediaCall::TextEncoding]
        );
        assert_eq!(
            graph.consumers(MediaCall::LatentEncoding),
            [MediaCall::LatentPreparation]
        );
        assert_eq!(graph.condition_units().len(), 9);
        assert!(graph.encodes_condition_audio());

        let extent = |name| graph.product(name).unwrap().extent.unwrap();
        assert_eq!(extent(product::CONDITION_PIXELS), (124 + 1) * 1344 * 768);
        assert_eq!(extent(product::CONDITION_SAMPLES), 165_334);
        assert_eq!(extent(product::VISION_PIXELS), 6 * 42 * 24 + 48 * 84);
        assert_eq!(extent(product::VISION_FEATURES), 1512 + 1008);
        assert_eq!(
            extent(product::CONDITION_VIDEO_LATENTS),
            7 * 5040 + 2016 + 1008
        );
        assert_eq!(extent(product::CONDITION_AUDIO_LATENTS), 414);
    }

    /// An audio reference adds only audio products: no pixels, no vision.
    #[test]
    fn an_audio_reference_adds_no_visual_products() {
        let media = source();
        let audio = VideoCondition {
            role: ConditionRole::Reference,
            source: media.locator(),
            media: ConditionMedia::Audio(AudioClip {
                sample_rate: 32_000,
                start_sample: 0,
                source_samples: 165_600,
                samples: 165_600,
            }),
            vision: None,
            latent_units: Vec::new(),
            audio_rows: 414,
        };
        let graph = VideoGraph::new(&request(VideoTask::Ref2va, vec![audio])).unwrap();
        assert!(!graph.encodes_vision());
        assert!(graph.product(product::CONDITION_PIXELS).is_none());
        assert!(graph.product(product::CONDITION_VIDEO_LATENTS).is_none());
        assert_eq!(
            graph.consumers(MediaCall::MediaReading),
            [MediaCall::LatentEncoding]
        );
    }
}
