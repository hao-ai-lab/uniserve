//! Demonstrates concurrent text and image requests through the simulated engine.
//!
//! The example exercises scheduling and lifecycle events without requiring a
//! model worker or GPU. Three text-only and two image-only requests share one
//! scheduler thread over `SimExecutor`; the run passes when every request
//! finishes within the deadline, each text request emits at least one text
//! token, and each image request emits at least one image. The process exits
//! with status 1 on failure.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use uniserve_core::EngineCoreOutput;
use uniserve_core::{
    FeedbackNextToken, FeedbackSource, GenerationConstraint, GenerationLimits, GenerationRequest,
    ImageEncoderInput, ImageGenerationConfig, ImageIngestStep, ImageParams, ImageTrigger,
    RequestId, SamplingParams,
};
use uniserve_engine::{EngineHandle, Scheduler, SimEngine, SimExecutor, SpecialTokenIds};

fn main() {
    let ctrl = SpecialTokenIds::default();
    let executor = Box::new(SimExecutor::new(SimEngine::new()));
    let sched = Scheduler::new(executor, ctrl, 32).unwrap();

    // The scheduler runs on its own thread; this thread submits through the
    // handle and drains each request's event receiver, as a server would.
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(cmd_tx);
    let jh = thread::spawn(move || sched.run(cmd_rx));

    let mut rxs: HashMap<RequestId, (String, uniserve_engine::EventRx)> = HashMap::new();
    let mut next_id = 1u64;
    let mut mk = |constraint: GenerationConstraint| {
        let id = RequestId(next_id);
        next_id += 1;
        let policy = ImageGenerationConfig {
            trigger: ImageTrigger::Token { token_id: 1000 },
            requires_text_for_image: false,
            feedback_source: Some(FeedbackSource::DeviceProduct),
            feedback_next_token: FeedbackNextToken::EndOfImage,
            num_feedback_positions: 2,
            feedback_encoders: vec![ImageEncoderInput {
                encoder: ImageIngestStep::VitEncode,
                num_kv_tokens: None,
                max_kv_tokens: None,
            }],
            sample_feedback_continuation: true,
        };
        let prompt_token_ids = vec![1, 2, 3];

        let image = ImageParams {
            steps: 6,
            ..Default::default()
        };
        let cache = Default::default();
        // These limits only check the request locally before submission; the
        // scheduler resolves its limits separately against the executor.
        let limits = GenerationLimits {
            features: uniserve_core::GenerationFeatures::UNDERSTANDING
                | uniserve_core::GenerationFeatures::IMAGE_GENERATION,
            max_latent_units: 64,
            latent_downsample: 16,
            max_vae_grid_tokens: 64,
            max_vit_grid_tokens: 64,
            max_latent_feature_bytes: 1 << 20,
            max_vision_feature_bytes: 1 << 20,
            commit_marker_tokens: 2,
            max_cfg_branches: 3,
            encoder_cache_entries: 256,
        };
        let request = GenerationRequest {
            request_id: id,
            prompt_token_ids,
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint,
            sampling: SamplingParams::default(),
            image,
            max_und_tokens: 20,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache,
            image_generation: policy,
            readout: Vec::new(),
        };
        request
            .validate_resources(&limits)
            .expect("bounded simulation request");
        (id, request)
    };

    for _ in 0..3 {
        let (id, request) = mk(GenerationConstraint::UndOnly);
        let rx = handle.submit(request).unwrap();
        rxs.insert(id, ("text".to_string(), rx));
    }
    for _ in 0..2 {
        let (id, request) = mk(GenerationConstraint::GenOnly);
        let rx = handle.submit(request).unwrap();
        rxs.insert(id, ("image".to_string(), rx));
    }

    // Poll every event receiver until each request has reported `Finished`
    // once, or the deadline passes. After the kind label, the tuple counts
    // text tokens and images and records whether `Finished` was seen.
    let mut done = 0usize;
    let total = rxs.len();
    let mut counts: HashMap<RequestId, (String, usize, usize, bool)> = HashMap::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while done < total && std::time::Instant::now() < deadline {
        for (id, (kind, rx)) in rxs.iter_mut() {
            while let Ok(ev) = rx.try_recv() {
                let e = counts.entry(*id).or_insert((kind.clone(), 0, 0, false));
                match ev {
                    EngineCoreOutput::TextToken { .. } => e.1 += 1,
                    EngineCoreOutput::ImageDone { .. } => e.2 += 1,
                    EngineCoreOutput::Finished { .. } if !e.3 => {
                        e.3 = true;
                        done += 1;
                    }
                    _ => {}
                }
            }
        }
        thread::sleep(Duration::from_millis(2));
    }

    let mut ok = done == total;
    for (id, (kind, toks, imgs, fin)) in &counts {
        println!(
            "req {:?} [{}]: text_tokens={} images={} finished={}",
            id, kind, toks, imgs, fin
        );
        if kind == "text" && *toks == 0 {
            ok = false;
        }
        if kind == "image" && *imgs == 0 {
            ok = false;
        }
    }
    handle.shutdown();
    let _ = jh.join();
    println!(
        "\nRUST SIM CONTROL-PLANE: {}",
        if ok { "PASS" } else { "FAIL" }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
