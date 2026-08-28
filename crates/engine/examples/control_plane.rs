//! GPU-free control-plane test in Rust: drive the Scheduler with
//! SimEngine over concurrent text + image requests; check lifecycle events.
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use uniserve_core::GenerationEvent;
use uniserve_core::{
    ContextSegment, FeedbackNextToken, FeedbackSource, GeneratedImageFeedbackRecipe,
    GenerationBehaviorDescriptor, GenerationConstraint, GenerationLimits,
    GenerationPolicyDescriptor, GenerationRequest, GenerationResourceBounds, ImageIngestRecipe,
    ImageKvEffect, ImageParams, RequestId, SamplingParams, TriggerPolicyDescriptor, UndVisibility,
};
use uniserve_engine::{ControlTokens, EngineHandle, Scheduler, SimEngine, SimExecutor};

fn main() {
    let ctrl = ControlTokens::default();
    let executor = Box::new(SimExecutor::new(SimEngine::new()));
    let sched = Scheduler::new(executor, ctrl, 32);
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(cmd_tx);
    let jh = thread::spawn(move || sched.run(cmd_rx));

    let mut rxs: HashMap<RequestId, (String, uniserve_engine::EventRx)> = HashMap::new();
    let mut next_id = 1u64;
    let mut mk = |constraint: GenerationConstraint| {
        let id = RequestId(next_id);
        next_id += 1;
        let policy = GenerationPolicyDescriptor {
            trigger: TriggerPolicyDescriptor::Token { token_id: 1000 },
            gen_only_start: uniserve_core::GenOnlyStartPolicyDescriptor::Immediate,
            feedback: Some(GeneratedImageFeedbackRecipe {
                source: FeedbackSource::DeviceProduct,
                next_und_token: FeedbackNextToken::EndOfImage,
                ingest: ImageIngestRecipe::vit_only(2, ImageKvEffect::WorkerDefined),
                sample_continuation: true,
            }),
            ..GenerationPolicyDescriptor::default()
        };
        let context = vec![ContextSegment::UndTokens {
            token_ids: vec![1, 2, 3],
            visibility: UndVisibility::Internal,
        }];
        let behavior = GenerationBehaviorDescriptor::resolve(constraint, &policy);
        let image = ImageParams {
            steps: 6,
            ..Default::default()
        };
        let cache = Default::default();
        let resources =
            GenerationResourceBounds::conservative(uniserve_core::GenerationResources {
                context: &context,
                negative_context: &[],
                behavior: &behavior,
                policy: &policy,
                image: &image,
                max_und_tokens: 20,
                cache: &cache,
                limits: &GenerationLimits {
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
                },
            })
            .expect("bounded simulation request");
        let request = GenerationRequest {
            request_id: id,
            context,
            negative_context: Vec::new(),
            constraint,
            behavior,
            sampling: SamplingParams::default(),
            image,
            max_und_tokens: 20,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache,
            policy,
            resources,
        };
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

    // collect until every request is Finished
    let mut done = 0usize;
    let total = rxs.len();
    let mut counts: HashMap<RequestId, (String, usize, usize, bool)> = HashMap::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while done < total && std::time::Instant::now() < deadline {
        for (id, (kind, rx)) in rxs.iter_mut() {
            while let Ok(ev) = rx.try_recv() {
                let e = counts.entry(*id).or_insert((kind.clone(), 0, 0, false));
                match ev {
                    GenerationEvent::TextToken { .. } => e.1 += 1,
                    GenerationEvent::ImageDone { .. } => e.2 += 1,
                    GenerationEvent::Finished { .. } if !e.3 => {
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
