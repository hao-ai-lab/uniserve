#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Model-profile prompt preprocessing and generation-policy integration behavior.

use std::collections::HashMap;
use std::fs;
use std::io::{Read as _, Write as _};
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use base64::Engine as _;
use tempfile::tempdir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};
use uniserve_core::{GenerationLimits, GenerationRequest, ImageIngestStep};
use uniserve_engine::VideoDenoiserInfo;
use uniserve_server::VideoMediaSettings;
use uniserve_server::openai::VideoGenerationRequest;
use uniserve_server::profile::assets::ResolvedModelFiles;
use uniserve_server::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use uniserve_server::profile::{ModelConfig, ModelDescription};
use uniserve_server::serving::chat::{ChatTemplateContentFormatOption, HfChatRenderer};
use uniserve_server::serving::media::{
    ImageFetchError, ImageFetchPolicy, ImageFetcher, ImageInput, ImageListError,
};
use uniserve_server::serving::video::VideoService;
use uniserve_server::serving::video::plan::VisionConfig;
use uniserve_server::serving::{
    InputProcessor, ResponseOptions, ServeRequestId, ServedFeature, chat_image_urls,
};

/// Base64 payload of a 1x1 PNG image.
const PNG_1X1: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
/// Minimal ChatML template: one `<|im_start|>role\ncontent<|im_end|>` turn per
/// message, then an open assistant turn when a generation prompt is requested.
const CHAT_TEMPLATE: &str = "{%- for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
/// Special tokens added to the synthetic tokenizer: ChatML turn delimiters,
/// the Bagel (`<|vision_*|>`) and SenseNova (`<img>`) image markers, and
/// reasoning and answer delimiters.
const SPECIAL_TOKENS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<img>",
    "</img>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
];

/// What a released base denoiser reports in its handshake: text-to-video
/// over every canvas of the canvas rule, on the 50-point schedule.
fn base_denoiser() -> VideoDenoiserInfo {
    VideoDenoiserInfo {
        tasks: vec!["t2va".to_owned()],
        schedule_points: 50,
        video_shift: 12.0,
        audio_shift: 3.0,
        canvases: Vec::new(),
        max_sequence_rows: None,
        condition_tiles: None,
        max_condition_rows: 0,
    }
}

/// The video service of `denoiser` with a `max_video_seconds` capacity, the
/// released checkpoint's Qwen3-VL processor geometry, a one-rank latent
/// encoder and the default media policy.
fn video_service(
    denoiser: VideoDenoiserInfo,
    max_video_seconds: f64,
    tokenizer: &DynTokenizer,
) -> VideoService {
    VideoService::new(
        denoiser,
        VisionConfig {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_min_pixels: 65_536,
            image_max_pixels: 16_777_216,
            video_min_pixels: 4_096,
            video_max_pixels: 25_165_824,
        },
        max_video_seconds,
        1,
        &VideoMediaSettings::default(),
        Arc::clone(tokenizer),
    )
    .unwrap()
}

/// A t2va request body with `fields` merged over a 5-second 16:9 target.
fn video_request(fields: serde_json::Value) -> VideoGenerationRequest {
    let mut body = serde_json::json!({
        "model": "minimax_h3",
        "prompt": "exact token sequence",
        "task": "t2va",
        "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
    });
    body.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    serde_json::from_value(body).unwrap()
}

/// Resolves `description` over a synthetic checkpoint with worker limits that
/// cover every generation feature; see `try_resolved_model`.
fn resolved_model(
    description: ModelDescription,
    model_type: &str,
) -> (tempfile::TempDir, DynTokenizer, InputProcessor) {
    try_resolved_model(description, model_type, runtime_limits()).unwrap()
}

/// Writes a synthetic checkpoint to a temporary directory and binds an
/// `InputProcessor` for `description` over it with the given worker limits.
///
/// The tokenizer maps each ASCII code point from 1 to 127 to one token whose
/// ID is the code point (no merges), plus `SPECIAL_TOKENS`, so outside the
/// special tokens one character is one prompt token. `generation_config.json`
/// sets a 128-token default generation length. The returned directory owns
/// the checkpoint files.
///
/// # Errors
///
/// Returns the `InputProcessor::new` error, for example when `limits` do not
/// cover a feature the profile needs. Other setup failures panic.
fn try_resolved_model(
    description: ModelDescription,
    model_type: &str,
    limits: GenerationLimits,
) -> uniserve_server::serving::Result<(tempfile::TempDir, DynTokenizer, InputProcessor)> {
    let directory = tempdir().unwrap();

    let mut vocab = Vocab::from_iter([("<unk>".to_string(), 0_u32)]);
    for codepoint in 1_u32..=127 {
        vocab.insert(char::from_u32(codepoint).unwrap().to_string(), codepoint);
    }
    let tokenizer_model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();
    let mut tokenizer_builder = TokenizerBuilder::new(tokenizer_model);
    tokenizer_builder.add_special_tokens(
        &SPECIAL_TOKENS
            .iter()
            .map(|token| AddedToken::from(*token, true))
            .collect::<Vec<_>>(),
    );
    let tokenizer_path = directory.path().join("tokenizer.json");
    tokenizer_builder.save(&tokenizer_path, false).unwrap();

    let config_path = directory.path().join("config.json");
    fs::write(
        &config_path,
        format!(
            r#"{{"model_type":"{model_type}","max_position_embeddings":4096,"num_attention_heads":8}}"#
        ),
    )
    .unwrap();
    let tokenizer_config_path = directory.path().join("tokenizer_config.json");
    fs::write(
        &tokenizer_config_path,
        format!(
            "{{\"eos_token\":\"<|im_end|>\",\"chat_template\":{}}}",
            serde_json::to_string(CHAT_TEMPLATE).unwrap()
        ),
    )
    .unwrap();
    let generation_config_path = directory.path().join("generation_config.json");
    fs::write(
        &generation_config_path,
        r#"{"eos_token_id":2,"max_new_tokens":128}"#,
    )
    .unwrap();
    let files = ResolvedModelFiles {
        tokenizer_path,
        tokenizer_config_path: Some(tokenizer_config_path),
        generation_config_path: Some(generation_config_path),
        preprocessor_config_path: None,
        chat_template_path: None,
        config_path: Some(config_path),
    };
    // Bagel reads its latent position grid side from the primary checkpoint's
    // safetensors header; a header-only file declaring the published 64x64
    // table suffices.
    if description == ModelDescription::Bagel {
        let header = serde_json::to_vec(&serde_json::json!({
            "latent_pos_embed.pos_embed": {
                "dtype": "BF16", "shape": [4096, 1], "data_offsets": [0, 8192]
            }
        }))
        .unwrap();
        let mut checkpoint = (header.len() as u64).to_le_bytes().to_vec();
        checkpoint.extend(header);
        checkpoint.resize(checkpoint.len() + 8192, 0);
        fs::write(directory.path().join("ema.safetensors"), checkpoint).unwrap();
    }

    let tokenizer: DynTokenizer =
        Arc::new(HuggingFaceTokenizer::new(&files.tokenizer_path).unwrap());
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    // A diffusers pipeline checkpoint has no root `config.json`, so MiniMax H3
    // is built by `ModelConfig::from_pipeline`, here with the server's default
    // 15-second maximum video duration.
    let config = match description {
        ModelDescription::MiniMaxH3 => {
            ModelConfig::from_pipeline(description.id(), description, 15.0, Some(4096), None)
                .unwrap()
        }
        _ => tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(ModelConfig::from_files(
                description.id(),
                directory.path().to_str().unwrap(),
                &files,
                None,
                tokenizer.as_ref(),
            ))
            .unwrap(),
    };
    let video = (description == ModelDescription::MiniMaxH3)
        .then(|| video_service(base_denoiser(), 15.0, &tokenizer));
    let model = InputProcessor::new(
        config,
        Arc::clone(&tokenizer),
        Some(renderer),
        uniserve_server::serving::WorkerCapabilities {
            limits,
            sampling_controls: uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
            max_model_tokens: 4096,
        },
        video,
        true,
    )?;
    Ok((directory, tokenizer, model))
}

/// Worker limits that cover every generation feature with capacities no test
/// request reaches.
fn runtime_limits() -> GenerationLimits {
    GenerationLimits {
        features: uniserve_core::GenerationFeatures::all(),
        max_latent_units: 1_000_000,
        latent_downsample: 16,
        max_vae_grid_tokens: 1_000_000,
        max_vit_grid_tokens: 1_000_000,
        max_latent_feature_bytes: u64::MAX,
        max_vision_feature_bytes: u64::MAX,
        commit_marker_tokens: 2,
        max_cfg_branches: 3,
        encoder_cache_entries: 32,
    }
}

/// Data URL of the `PNG_1X1` image.
fn png_data_url() -> String {
    format!("data:image/png;base64,{PNG_1X1}")
}

/// Chat request with one input image, referenced by `image_url`, between two
/// text parts.
///
/// The leading text contains a literal `</img>`, which the synthetic tokenizer
/// encodes as the SenseNova end-of-image token. The top-level `seed` (9) and
/// the `image_config` seed (17) differ so tests can tell which one wins.
fn image_chat_request(
    model: &str,
    image_url: &str,
) -> uniserve_server::openai::ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "literal </img> before "},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": " after"}
        ]}],
        "seed": 9,
        "image_config": {"seed": 17}
    }))
    .unwrap()
}

/// Runs `ImageFetcher::fetch_all` to completion on a current-thread runtime.
fn resolve_images(
    fetcher: &ImageFetcher,
    urls: Vec<String>,
) -> Result<Vec<ImageInput>, ImageListError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(fetcher.fetch_all(urls))
}

/// Resolves a chat request's input images under the default image policy
/// and preprocesses the request with them, as `ServingRuntime::generate_chat`
/// does.
fn preprocess_image_chat(
    processor: &InputProcessor,
    request_id: &str,
    request: uniserve_server::openai::ChatCompletionRequest,
) -> (GenerationRequest, ResponseOptions) {
    let fetcher = ImageFetcher::new(ImageFetchPolicy::default()).unwrap();
    let images = resolve_images(&fetcher, chat_image_urls(&request)).unwrap();
    processor
        .preprocess_chat_request(ServeRequestId::new(request_id), request, images)
        .unwrap()
}

/// HTTP server on an ephemeral 127.0.0.1 port that answers every request
/// with one `image/png` body from a background thread.
struct ImageServer {
    address: SocketAddr,
    connections: Arc<AtomicUsize>,
}

impl ImageServer {
    /// Starts serving `body`.
    fn start(body: Vec<u8>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let accepted = Arc::clone(&connections);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else {
                    return;
                };
                accepted.fetch_add(1, Ordering::SeqCst);

                // Consume the request head, then answer and close.
                let mut head = Vec::new();
                let mut byte = [0_u8; 1];
                while !head.ends_with(b"\r\n\r\n") && matches!(stream.read(&mut byte), Ok(1)) {
                    head.push(byte[0]);
                }
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(&body);
            }
        });
        Self {
            address,
            connections,
        }
    }

    /// Number of connections accepted so far.
    fn connections(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }
}

/// An http image URL prepares exactly the image input its equivalent data URL
/// does, for both omni profiles: the same prompt tokens, encoder-cache
/// identity, worker payload, position, and encoders.
#[test]
fn image_urls_prepare_the_same_image_input_as_data_urls() {
    let png = base64::engine::general_purpose::STANDARD
        .decode(PNG_1X1)
        .unwrap();
    let server = ImageServer::start(png);
    // The server is on loopback, which only the private-destination policy reaches.
    let fetcher = ImageFetcher::new(ImageFetchPolicy {
        allow_private: true,
        ..ImageFetchPolicy::default()
    })
    .unwrap();

    for (description, model_type) in [
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        let prepare = |image_url: &str| {
            let request = image_chat_request(processor.served_model_name(), image_url);
            let images = resolve_images(&fetcher, chat_image_urls(&request)).unwrap();
            processor
                .preprocess_chat_request(ServeRequestId::new("image-source"), request, images)
                .unwrap()
        };

        let (from_url, url_response) = prepare(&format!("http://{}/image.png", server.address));
        let (from_data, data_response) = prepare(&png_data_url());

        assert_eq!(
            url_response.prompt_token_ids,
            data_response.prompt_token_ids
        );
        assert_eq!(from_url.multimodal_inputs.images.len(), 1, "{model_type}");
        assert_eq!(from_url.multimodal_inputs, from_data.multimodal_inputs);
    }
}

/// Under the default image policy, a chat image URL on a loopback host is
/// refused before any connection is made, however the host is written.
#[test]
fn loopback_image_urls_are_refused_by_default() {
    let png = base64::engine::general_purpose::STANDARD
        .decode(PNG_1X1)
        .unwrap();
    let server = ImageServer::start(png);
    let fetcher = ImageFetcher::new(ImageFetchPolicy::default()).unwrap();
    let port = server.address.port();

    for host in ["127.0.0.1", "localhost", "[::1]"] {
        let request = image_chat_request("sensenova", &format!("http://{host}:{port}/image.png"));
        let error = resolve_images(&fetcher, chat_image_urls(&request)).unwrap_err();

        assert_eq!(error.index, 0);
        assert!(
            matches!(error.source, ImageFetchError::NonPublicAddress),
            "{host}: {error}"
        );
    }
    assert_eq!(server.connections(), 0);
}

/// `InputProcessor::new` refuses to bind a profile whose worker limits lack a
/// feature the profile's configured branches need, and names the missing
/// feature by its `GenerationFeatures::name` diagnostic. SenseNova appears
/// twice because it needs both ViT encoding and image generation.
#[test]
fn model_resolution_requires_every_configured_runtime_branch() {
    type ChangeLimits = fn(&mut GenerationLimits);
    let cases: [(ModelDescription, &'static str, &'static str, ChangeLimits); 4] = [
        (
            ModelDescription::Qwen3,
            "qwen3",
            "runtime_und_execution",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(uniserve_core::GenerationFeatures::UNDERSTANDING);
            },
        ),
        (
            ModelDescription::SenseNova,
            "neo_chat",
            "runtime_vit_encode",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(uniserve_core::GenerationFeatures::VISION_ENCODE);
            },
        ),
        (
            ModelDescription::Bagel,
            "bagel",
            "runtime_vae_encode",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(uniserve_core::GenerationFeatures::LATENT_ENCODE);
            },
        ),
        (
            ModelDescription::SenseNova,
            "neo_chat",
            "runtime_gen_denoise",
            |limits: &mut GenerationLimits| {
                limits
                    .features
                    .remove(uniserve_core::GenerationFeatures::IMAGE_GENERATION);
            },
        ),
    ];

    for (description, model_type, required, remove) in cases {
        let mut limits = runtime_limits();
        remove(&mut limits);
        let error = match try_resolved_model(description, model_type, limits) {
            Ok(_) => panic!("incomplete worker limits must fail model resolution"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(required), "got: {error}");
    }
}

/// The literal `</img>` in the user text also encodes to the end-of-image
/// token, so the prompt holds two of them. The image's position is the index
/// of the end-of-image token of the marker rendered for its slot (the second
/// occurrence), not the first one found in the text.
#[test]
fn sensenova_places_the_input_image_at_its_rendered_slot() {
    let (_directory, tokenizer, model) = resolved_model(ModelDescription::SenseNova, "neo_chat");
    let request = image_chat_request(model.served_model_name(), &png_data_url());
    let (generation, response) = preprocess_image_chat(&model, "image-params", request);

    // The `image_config` seed takes precedence over the text sampling seed.
    assert_eq!(generation.sampling.seed, Some(17));
    assert_eq!(generation.image.seed, Some(17));

    let end_image = tokenizer.token_to_id("</img>").unwrap();
    let marker_positions = response
        .prompt_token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (*token == end_image).then_some(index as u32))
        .collect::<Vec<_>>();
    assert_eq!(marker_positions.len(), 2);
    let image = &generation.multimodal_inputs.images[0];
    let (position, steps) = (
        image.position,
        image
            .encoders
            .iter()
            .map(|input| input.encoder)
            .collect::<Vec<_>>(),
    );
    assert_eq!(position, marker_positions[1]);
    assert_eq!(steps, &[ImageIngestStep::VitEncode]);

    // Feedback for the default 16:9 canvas (2048x1152) is a 64x36 ViT grid at
    // 32 pixels per token plus one marker token.
    assert_eq!(
        generation
            .image_generation
            .feedback_encoders
            .iter()
            .map(|input| input.num_kv_tokens)
            .collect::<Vec<_>>(),
        vec![Some(2305)]
    );
}

#[test]
fn bagel_places_the_input_image_between_surrounding_chat_text() {
    let (_directory, _tokenizer, model) = resolved_model(ModelDescription::Bagel, "bagel");
    let request = image_chat_request(model.served_model_name(), &png_data_url());
    let (generation, response) = preprocess_image_chat(&model, "image-params", request);
    assert_eq!(generation.sampling.seed, Some(17));
    assert_eq!(generation.image.seed, Some(17));
    let image = &generation.multimodal_inputs.images[0];
    let (position, steps) = (
        image.position,
        image
            .encoders
            .iter()
            .map(|input| input.encoder)
            .collect::<Vec<_>>(),
    );

    // Bagel removes the slot and places the image at the token count of the
    // text before it, so text on both sides keeps it strictly inside the
    // prompt. The Bagel profile configures a VAE and a ViT encoder input for
    // each input image, in that order.
    assert!(position > 0);
    assert!((position as usize) < response.prompt_token_ids.len());
    assert_eq!(
        steps,
        &[ImageIngestStep::VaeEncode, ImageIngestStep::VitEncode]
    );
}

#[tokio::test]
async fn minimax_video_preprocessing_preserves_tokens_seed_and_frame_alignment() {
    let (_directory, tokenizer, model) = resolved_model(ModelDescription::MiniMaxH3, "minimax_h3");
    let id = ServeRequestId::new("video");
    let prepare = |fields: serde_json::Value| {
        let request = video_request(fields);
        let model = &model;
        let id = &id;
        async move { model.preprocess_video_request(id, &request).await }
    };

    // The t2va presentation is the prompt tokenized without chat framing.
    // The seed defaults to the reference's 42, and the denoiser evaluates
    // the schedule's 49 intervals. Five seconds are 120 frames at 24 fps,
    // which extend upward to the next count of the form `17n + 5`: 124
    // frames, decoded as seven 17-frame units plus a tail.
    let accepted = prepare(serde_json::json!({})).await.unwrap();
    assert_eq!(
        accepted.request.prompt_token_ids,
        tokenizer.encode("exact token sequence", false).unwrap()
    );
    let sampling = accepted.request.sampling;
    assert_eq!(sampling.seed, 42);
    assert_eq!(sampling.num_inference_steps, 49);
    assert_eq!((sampling.num_frames, sampling.video_units), (124, 7));
    assert_eq!((sampling.width, sampling.height), (1344, 768));
    assert_eq!(accepted.duration_seconds, 5.0);
    let seeded = prepare(serde_json::json!({"seed": 17})).await.unwrap();
    assert_eq!(seeded.request.sampling.seed, 17);

    // The admitted interval is [4, 15] seconds inclusive. Fractional
    // durations round half to even before alignment: 5.1875 s is 124.5
    // frames, which rounds to 124 and keeps 124 output frames, where rounding
    // away from zero would reach 125 and extend to 141. 7.3 s is 175.2 frames
    // and 175 is already aligned.
    for (seconds, frames) in [
        (4.0, 107),
        (4.25, 107),
        (5.1875, 124),
        (7.3, 175),
        (10.0, 243),
        (14.0, 345),
        (15.0, 362),
    ] {
        let target = serde_json::json!({
            "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": seconds}
        });
        let sampling = prepare(target).await.unwrap().request.sampling;
        assert_eq!(sampling.num_frames, frames, "{seconds} s");
        assert_eq!(sampling.video_units, (frames - 5) / 17, "{seconds} s");
    }

    // The canvas follows the named aspect ratio; `auto` is 16:9.
    for (ratio, canvas) in [
        ("auto", (1344, 768)),
        ("21:9", (1536, 672)),
        ("4:3", (1024, 768)),
        ("1:1", (768, 768)),
        ("3:4", (768, 1024)),
        ("9:16", (768, 1344)),
    ] {
        let target = serde_json::json!({
            "target": {"short_edge": 768, "aspect_ratio": ratio, "duration_seconds": 5.0}
        });
        let prepared = prepare(target).await.unwrap();
        assert_eq!(
            (prepared.canvas.width, prepared.canvas.height),
            canvas,
            "{ratio}"
        );
    }

    // Fields that restate the served contract are accepted when they agree:
    // the schedule's 50 sigma points and shifts, one output, the lossless
    // quality, and the SGLang client's duration and canvas.
    let restated = prepare(serde_json::json!({
        "num_inference_steps": 50, "flow_shift": 12.0, "audio_flow_shift": 3.0,
        "num_outputs_per_prompt": 1, "n": 1, "quality": "lossless",
        "seconds": 5, "size": "1344x768", "width": 1344, "height": 768,
    }))
    .await
    .unwrap();
    assert_eq!(restated.request, accepted.request);

    // Each disagreement is refused and names its field.
    for (fields, param) in [
        (
            serde_json::json!({"num_inference_steps": 9}),
            "num_inference_steps",
        ),
        (serde_json::json!({"flow_shift": 10.0}), "flow_shift"),
        (
            serde_json::json!({"audio_flow_shift": 1.0}),
            "audio_flow_shift",
        ),
        (
            serde_json::json!({"num_outputs_per_prompt": 2}),
            "num_outputs_per_prompt",
        ),
        (serde_json::json!({"n": 4}), "n"),
        (serde_json::json!({"quality": "fast"}), "quality"),
        (serde_json::json!({"seconds": 10}), "seconds"),
        (serde_json::json!({"size": "768x1344"}), "size"),
        (serde_json::json!({"width": 768}), "width"),
        (serde_json::json!({"height": 1344}), "height"),
        (serde_json::json!({"prompt": "  "}), "prompt"),
        (serde_json::json!({"task": "fl2va"}), "task"),
    ] {
        let error = prepare(fields.clone()).await.unwrap_err();
        assert!(
            matches!(
                error,
                uniserve_server::openai::ApiError::InvalidRequest { param: Some(name), .. }
                    if name == param
            ),
            "{fields}: {error:?}"
        );
    }

    // The target alone sets the duration and canvas: a t2va request needs a
    // duration within the finite [4, 15] second range and the 768 short edge.
    for target in [
        serde_json::json!({"short_edge": 768, "aspect_ratio": "16:9"}),
        serde_json::json!({"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 3.99}),
        serde_json::json!({"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 15.01}),
        serde_json::json!({"short_edge": 720, "aspect_ratio": "16:9", "duration_seconds": 5.0}),
        serde_json::json!({"short_edge": 768, "aspect_ratio": "2:1", "duration_seconds": 5.0}),
    ] {
        let error = prepare(serde_json::json!({"target": target}))
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                uniserve_server::openai::ApiError::InvalidRequest { param: Some(name), .. }
                    if name.starts_with("target.")
            ),
            "{target}: {error:?}"
        );
    }

    assert_eq!(
        prepare(serde_json::json!({"model": "another-model"}))
            .await
            .unwrap_err(),
        uniserve_server::openai::ApiError::ModelNotFound {
            model: "another-model".to_string()
        },
    );
}

/// A FastH3 DMD deployment serves the training buckets its workers prepared,
/// on the export's own schedule: other canvases are refused, and the
/// capabilities report the canvases, the short edges and named ratios they
/// serve, the schedule and the deployment's duration capacity.
#[tokio::test]
async fn video_denoiser_handshake_bounds_requests_and_capabilities() {
    let (_directory, tokenizer, loaded) = resolved_model(ModelDescription::MiniMaxH3, "minimax_h3");
    let canvas = |width, height| uniserve_core::Canvas { width, height };
    // 768p 16:9, then 480p 21:9 and 16:9.
    let denoiser = VideoDenoiserInfo {
        tasks: vec!["t2va".to_owned()],
        schedule_points: 9,
        video_shift: 10.0,
        audio_shift: 3.0,
        canvases: vec![canvas(1344, 768), canvas(992, 416), canvas(832, 480)],
        max_sequence_rows: None,
        condition_tiles: None,
        max_condition_rows: 0,
    };
    let processor = |max_video_seconds: f64| {
        let mut config = loaded.config().clone();
        config.parameters =
            uniserve_server::profile::ModelParameters::MiniMaxH3 { max_video_seconds };
        InputProcessor::new(
            config,
            Arc::clone(&tokenizer),
            None,
            uniserve_server::serving::WorkerCapabilities {
                limits: runtime_limits(),
                sampling_controls: uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
                max_model_tokens: 4096,
            },
            Some(video_service(
                denoiser.clone(),
                max_video_seconds,
                &tokenizer,
            )),
            true,
        )
    };
    for capacity in [0.0, 3.5, 15.5, f64::NAN] {
        assert!(processor(capacity).is_err(), "{capacity} s");
    }

    let model = processor(10.0).unwrap();
    let capabilities = model.video_capabilities();
    assert_eq!(capabilities["tasks"], serde_json::json!(["t2va"]));
    assert_eq!(capabilities["min_seconds"], 4.0);
    assert_eq!(capabilities["max_seconds"], 10.0);
    assert_eq!(capabilities["model_max_seconds"], 15.0);
    assert_eq!(capabilities["fps"], 24);
    assert_eq!(
        capabilities["canvas"]["canvases"],
        serde_json::json!([
            {"width": 1344, "height": 768},
            {"width": 992, "height": 416},
            {"width": 832, "height": 480},
        ])
    );
    assert_eq!(
        capabilities["canvas"]["short_edges"],
        serde_json::json!([768, 480])
    );
    assert_eq!(
        capabilities["canvas"]["aspect_ratios"],
        serde_json::json!(["21:9", "16:9"])
    );
    assert_eq!(
        capabilities["canvas"]["sizes"],
        serde_json::json!({
            "768": {"16:9": {"width": 1344, "height": 768}},
            "480": {
                "21:9": {"width": 992, "height": 416},
                "16:9": {"width": 832, "height": 480},
            },
        })
    );
    assert_eq!(
        capabilities["schedule"],
        serde_json::json!({"num_inference_steps": 9, "flow_shift": 10.0, "audio_flow_shift": 3.0})
    );
    assert_eq!(capabilities["max_prompt_tokens"], 4096);

    let id = ServeRequestId::new("capacity");
    let prepared = model
        .preprocess_video_request(
            &id,
            &video_request(serde_json::json!({
                "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 10.0}
            })),
        )
        .await
        .unwrap();
    assert_eq!(prepared.request.sampling.num_frames, 243);
    assert_eq!(prepared.request.sampling.num_inference_steps, 8);
    assert_eq!(prepared.canvas, canvas(1344, 768));

    // A 480 short edge names the 480p training bucket of the ratio.
    let narrow = model
        .preprocess_video_request(
            &id,
            &video_request(serde_json::json!({
                "target": {"short_edge": 480, "aspect_ratio": "21:9", "duration_seconds": 5.0}
            })),
        )
        .await
        .unwrap();
    assert_eq!(narrow.canvas, canvas(992, 416));
    assert_eq!(
        (
            narrow.request.sampling.width,
            narrow.request.sampling.height
        ),
        (992, 416)
    );

    // A duration beyond the capacity, an unprepared bucket at either short
    // edge, and an unserved short edge are refused.
    for target in [
        serde_json::json!({"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 10.5}),
        serde_json::json!({"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 5.0}),
        serde_json::json!({"short_edge": 768, "aspect_ratio": "21:9", "duration_seconds": 5.0}),
        serde_json::json!({"short_edge": 480, "aspect_ratio": "1:1", "duration_seconds": 5.0}),
        serde_json::json!({"short_edge": 720, "aspect_ratio": "16:9", "duration_seconds": 5.0}),
    ] {
        assert!(
            model
                .preprocess_video_request(
                    &id,
                    &video_request(serde_json::json!({"target": target}))
                )
                .await
                .is_err(),
            "{target}"
        );
    }

    // Only a video checkpoint has a video service, and it needs one.
    let mut config = loaded.config().clone();
    config.parameters = uniserve_server::profile::ModelParameters::MiniMaxH3 {
        max_video_seconds: 10.0,
    };
    assert!(
        InputProcessor::new(
            config,
            Arc::clone(&tokenizer),
            None,
            uniserve_server::serving::WorkerCapabilities {
                limits: runtime_limits(),
                sampling_controls: Vec::new(),
                max_model_tokens: 4096,
            },
            None,
            true,
        )
        .is_err()
    );
}

/// `InputProcessor::new` replaces the checkpoint's context length with the
/// worker's `max_model_tokens`, and text preprocessing enforces that bound.
#[test]
fn worker_context_capacity_limits_preprocessed_requests() {
    let (_directory, tokenizer, loaded) = resolved_model(ModelDescription::Qwen3, "qwen3");
    let renderer = HfChatRenderer::new(
        Some(CHAT_TEMPLATE.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    let processor = InputProcessor::new(
        loaded.config().clone(),
        tokenizer,
        Some(renderer),
        uniserve_server::serving::WorkerCapabilities {
            limits: runtime_limits(),
            sampling_controls: uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
            max_model_tokens: 8,
        },
        None,
        true,
    )
    .unwrap();

    assert_eq!(processor.config().max_model_tokens, Some(8));
    let request = uniserve_server::serving::TextPromptRequest::new(
        "context-capacity",
        "This prompt exceeds eight tokens",
    );
    assert!(matches!(
        processor.preprocess_text_request(request),
        Err(uniserve_server::serving::ServeError::Tokenize {
            source: uniserve_server::serving::TokenizeError::Text(
                uniserve_server::serving::text::Error::PromptTooLong {
                    max_model_len: 8,
                    ..
                }
            ),
            ..
        })
    ));
}

/// Defaults fill only omitted controls: an explicit zero temperature stays
/// zero rather than taking the default 1.0, while an explicit zero completion
/// length is rejected rather than replaced by the 128-token checkpoint default.
#[test]
fn sampling_defaults_preserve_explicit_zero_controls() {
    let (_directory, _tokenizer, model) = resolved_model(ModelDescription::Qwen3, "qwen3");
    let mut request: uniserve_server::openai::ChatCompletionRequest =
        serde_json::from_value(serde_json::json!({
            "model": "qwen3", "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
    let (default, _) = model
        .preprocess_chat_request(
            ServeRequestId::new("sampling-defaults"),
            request.clone(),
            Vec::new(),
        )
        .unwrap();
    assert_eq!(default.max_und_tokens, 128);
    assert_eq!(default.sampling.temperature, 1.0);

    request.temperature = Some(0.0);
    let (greedy, _) = model
        .preprocess_chat_request(
            ServeRequestId::new("sampling-defaults"),
            request.clone(),
            Vec::new(),
        )
        .unwrap();
    assert_eq!(greedy.sampling.temperature, 0.0);
    assert_eq!(greedy.max_und_tokens, 128);

    request.max_completion_tokens = Some(0);
    assert!(matches!(
        model.preprocess_chat_request(
            ServeRequestId::new("sampling-defaults"),
            request,
            Vec::new()
        ),
        Err(uniserve_server::openai::ApiError::InvalidRequest { .. })
    ));
}

/// The image API produces an image-only (`GenOnly`) request that keeps the
/// requested canvas, steps, seed, guidance scale, and negative prompt.
/// SenseNova accepts only its resolution buckets, and 1536x1536 is one of
/// them.
#[test]
fn image_api_preserves_requested_dimensions_seed_and_guidance() {
    for (description, model_type, width, height) in [
        (ModelDescription::SenseNova, "neo_chat", 1536, 1536),
        (ModelDescription::Bagel, "bagel", 512, 512),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        let request: uniserve_server::openai::ImageGenerationRequest =
            serde_json::from_value(serde_json::json!({
                "model": description.id(),
                "prompt": "a blue bird",
                "negative_prompt": "blur",
                "size": format!("{width}x{height}"),
                "steps": 4,
                "seed": 17,
                "guidance_scale": 7.5,
                "n": 1
            }))
            .unwrap();
        let (generation, _) = processor
            .preprocess_image_request(ServeRequestId::new("image-api"), request)
            .unwrap();
        assert_eq!(
            generation.constraint,
            uniserve_core::GenerationConstraint::GenOnly
        );
        assert_eq!(
            (generation.image.width, generation.image.height),
            (width, height)
        );
        assert_eq!(generation.image.steps, 4);
        assert_eq!(generation.image.max_images, 1);
        assert_eq!(generation.image.seed, Some(17));
        assert_eq!(generation.sampling.seed, Some(17));
        assert_eq!(generation.image.cfg_text_scale, 7.5);
        assert_eq!(generation.image.negative_prompt, "blur");
    }
}

/// SenseNova bounds each input image by its share of a pixel budget, so the
/// same image preprocessed under a different bound is a different encoder
/// product. Its encoder-cache identity changes once five input images lower
/// the bound, while requests whose images keep the same bound share it.
#[test]
fn sensenova_image_cache_identity_follows_its_pixel_bound() {
    let (_directory, _tokenizer, processor) =
        resolved_model(ModelDescription::SenseNova, "neo_chat");
    let first_image_hash = |count: usize| {
        let content = (0..count)
            .map(|_| {
                serde_json::json!({
                    "type": "image_url",
                    "image_url": {"url": png_data_url()}
                })
            })
            .chain([serde_json::json!({"type": "text", "text": "compare"})])
            .collect::<Vec<_>>();
        let request: uniserve_server::openai::ChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "sensenova",
                "messages": [{"role": "user", "content": content}]
            }))
            .unwrap();
        let (generation, _) = preprocess_image_chat(&processor, "image-bound", request);
        generation.multimodal_inputs.images[0].hash
    };

    // One to four input images keep the 2048x2048 bound; a fifth lowers it.
    assert_eq!(first_image_hash(1), first_image_hash(4));
    assert_ne!(first_image_hash(4), first_image_hash(5));
}

/// Bagel's learned latent position table covers 64 latent patches per side
/// in the synthetic checkpoint, 1024 pixels at the 16-pixel latent stride. A
/// canvas with a longer side, which would index past a table row or past the
/// table, is an invalid request like other unsupported sizes, while a side
/// of exactly 64 patches is accepted.
#[test]
fn bagel_rejects_a_canvas_side_beyond_its_latent_position_table() {
    let (_directory, _tokenizer, processor) = resolved_model(ModelDescription::Bagel, "bagel");
    let request = |size: &str| -> uniserve_server::openai::ImageGenerationRequest {
        serde_json::from_value(serde_json::json!({
            "model": "bagel",
            "prompt": "a banner",
            "size": size
        }))
        .unwrap()
    };

    for size in ["1040x16", "16x1040"] {
        let Err(error) =
            processor.preprocess_image_request(ServeRequestId::new("latent-side"), request(size))
        else {
            panic!("{size} was admitted");
        };
        assert!(
            matches!(
                error,
                uniserve_server::openai::ApiError::InvalidRequest { .. }
            ),
            "{size}: {error:?}"
        );
    }
    for size in ["1024x16", "16x1024"] {
        let (generation, _) = processor
            .preprocess_image_request(ServeRequestId::new("latent-side"), request(size))
            .unwrap();
        assert_eq!(
            format!("{}x{}", generation.image.width, generation.image.height),
            size
        );
    }
}

/// Cache namespace and salt select a stable isolation partition, the bypass
/// flags disable prefix-cache reads and writes without changing the
/// partition, and prompt logprobs disable reads only, because a prefix-cache
/// hit would skip the prompt positions whose logprobs are requested.
#[test]
fn cache_controls_preserve_isolation_and_prompt_logprob_requirements() {
    use uniserve_server::serving::TextPromptRequest;

    for (description, model_type) in [
        (ModelDescription::Qwen3, "qwen3"),
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        let mut request = TextPromptRequest::new("cache-controls", "hello");

        let (shared, _) = processor.preprocess_text_request(request.clone()).unwrap();
        assert_eq!(shared.cache.isolation_key, None);
        assert!(shared.cache.read && shared.cache.write);

        // The same namespace and salt always produce the same key.
        request.cache_namespace = Some("ab".to_string());
        request.cache_salt = Some("c".to_string());
        let (isolated, _) = processor.preprocess_text_request(request.clone()).unwrap();
        assert!(isolated.cache.isolation_key.is_some());
        let (repeat, _) = processor.preprocess_text_request(request.clone()).unwrap();
        assert_eq!(repeat.cache.isolation_key, isolated.cache.isolation_key);

        // Namespace and salt are separate coordinates, even when their
        // concatenated text is identical.
        request.cache_namespace = Some("a".to_string());
        request.cache_salt = Some("bc".to_string());
        let (other, _) = processor.preprocess_text_request(request.clone()).unwrap();
        assert_ne!(other.cache.isolation_key, isolated.cache.isolation_key);

        request.bypass_cache_read = true;
        request.no_cache_store = true;
        let (disabled, response) = processor.preprocess_text_request(request.clone()).unwrap();
        assert!(!disabled.cache.read && !disabled.cache.write);
        assert!(!response.cache.read_enabled && !response.cache.write_enabled);
        assert_eq!(disabled.cache.isolation_key, other.cache.isolation_key);

        request.bypass_cache_read = false;
        request.no_cache_store = false;
        request.stop.prompt_logprobs = Some(0);
        let (logprobs, response) = processor.preprocess_text_request(request).unwrap();
        assert!(!logprobs.cache.read && logprobs.cache.write);
        assert!(!response.cache.read_enabled && response.cache.write_enabled);
        assert!(response.prompt_logprobs_requested);
    }
}

/// A streamed chat response has no field for prompt logprobs. A streamed
/// request with `prompt_logprobs: 0`, which the route accepts like vLLM does,
/// therefore requests no prompt scoring and keeps prefix-cache reads, while
/// the same buffered request still requests them.
#[test]
fn streamed_chat_does_not_request_undeliverable_prompt_logprobs() {
    for (description, model_type) in [
        (ModelDescription::Qwen3, "qwen3"),
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _tokenizer, processor) = resolved_model(description, model_type);
        for stream in [true, false] {
            let request: uniserve_server::openai::ChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": processor.served_model_name(),
                    "messages": [{"role": "user", "content": "hello"}],
                    "modalities": ["text"],
                    "stream": stream,
                    "prompt_logprobs": 0
                }))
                .unwrap();

            let (generation, response) = processor
                .preprocess_chat_request(
                    ServeRequestId::new("stream-prompt-logprobs"),
                    request,
                    Vec::new(),
                )
                .unwrap();

            assert_eq!(
                response.prompt_logprobs_requested, !stream,
                "{model_type} stream={stream}"
            );
            assert_eq!(
                generation.sampling.prompt_logprobs_requested(),
                !stream,
                "{model_type} stream={stream}"
            );
            assert_eq!(
                generation.cache.read, stream,
                "{model_type} stream={stream}"
            );
        }
    }
}

/// Qwen3 chat template shipped by Qwen3 and Qwen3-MoE checkpoints; it asks for
/// JSON tool calls inside `<tool_call>` lines.
const QWEN3_TEMPLATE: &str = include_str!("templates/qwen3.jinja");
/// Tool instruction in the style of Qwen3-Coder templates, which ask for XML
/// `<function=...>` calls instead of JSON objects.
const XML_CALL_TEMPLATE: &str = "{%- if tools -%}<|im_start|>system\n# Tools\n<tools>{%- for tool in tools -%}{{ tool | tojson }}{%- endfor -%}</tools>\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter>\nvalue\n</parameter>\n</function>\n</tool_call><|im_end|>\n{%- endif -%}{%- for message in messages -%}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";

/// Rebinds a resolved Qwen3-family model to `template`.
fn with_template(model_type: &str, template: &str) -> InputProcessor {
    let (_directory, tokenizer, loaded) = resolved_model(ModelDescription::Qwen3, model_type);
    let renderer = HfChatRenderer::new(
        Some(template.to_string()),
        HashMap::new(),
        ChatTemplateContentFormatOption::String,
    )
    .unwrap();
    InputProcessor::new(
        loaded.config().clone(),
        tokenizer,
        Some(renderer),
        uniserve_server::serving::WorkerCapabilities {
            limits: runtime_limits(),
            sampling_controls: uniserve_server::serving::ServedSamplingControl::ALL.to_vec(),
            max_model_tokens: 4096,
        },
        None,
        true,
    )
    .unwrap()
}

/// Dense and mixture-of-experts Qwen3 checkpoints serve tool calls only when
/// their chat template asks for the JSON calls the Qwen3 parser reads; a
/// template asking for another call format refuses requests with tools
/// instead of returning their calls as unparsed text.
#[test]
fn qwen3_family_declares_tool_calling_only_for_json_calls() {
    let tool_request = serde_json::json!({
        "model": "qwen3",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"type": "function", "function": {
            "name": "weather",
            "parameters": {"type": "object", "properties": {}}
        }}]
    });
    for model_type in ["qwen3", "qwen3_moe"] {
        let json_calls = with_template(model_type, QWEN3_TEMPLATE);
        assert!(
            json_calls
                .support()
                .features
                .contains(&ServedFeature::ToolCalling),
            "{model_type}"
        );
        json_calls
            .preprocess_chat_request(
                ServeRequestId::new("json-tools"),
                serde_json::from_value(tool_request.clone()).unwrap(),
                Vec::new(),
            )
            .unwrap();

        for template in [XML_CALL_TEMPLATE, CHAT_TEMPLATE] {
            let other = with_template(model_type, template);
            assert!(
                !other
                    .support()
                    .features
                    .contains(&ServedFeature::ToolCalling),
                "{model_type}"
            );
            let error = other
                .preprocess_chat_request(
                    ServeRequestId::new("other-tools"),
                    serde_json::from_value(tool_request.clone()).unwrap(),
                    Vec::new(),
                )
                .err()
                .unwrap();
            assert_eq!(error.status_code().as_u16(), 400, "{model_type}");
            let message = error.to_error_response().error.message;
            assert!(message.contains("tool_calling"), "{message}");
        }
    }
}

/// Negative seeds are accepted and reinterpreted as their two's-complement
/// `u64` value, and a logprob count of `-1` requests every candidate
/// (`u32::MAX`) while values below `-1` are rejected, for every token model.
#[test]
fn sampling_controls_have_the_same_meaning_across_token_models() {
    for (description, model_type) in [
        (ModelDescription::Qwen3, "qwen3"),
        (ModelDescription::SenseNova, "neo_chat"),
        (ModelDescription::Bagel, "bagel"),
    ] {
        let (_directory, _, model) = resolved_model(description, model_type);
        for seed in [-1, -2] {
            let mut input = uniserve_server::serving::TextPromptRequest::new("sampling", "hello");
            input.sampling.seed = Some(seed);
            input.stop.logprobs = Some(-1);
            let (request, _) = model.preprocess_text_request(input.clone()).unwrap();
            assert_eq!(request.sampling.seed, Some(seed as u64));
            assert_eq!(request.sampling.n_logprobs, u32::MAX);
            input.stop.logprobs = Some(-2);
            assert!(model.preprocess_text_request(input).is_err());
        }
    }
}
