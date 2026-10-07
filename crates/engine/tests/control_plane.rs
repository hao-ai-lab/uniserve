#![allow(clippy::unwrap_used, clippy::expect_used)]

//! End-to-end engine-loop lifecycle behavior over the GPU-free simulator.
//!
//! Each test drives a real `Scheduler` against `SimExecutor`/`SimEngine`,
//! either on a spawned thread through `Scheduler::run` or stepped on the test
//! thread with `Scheduler::step`, and observes behavior through `EngineHandle`
//! event streams, `SchedulerStats` counters, and the scheduler's public
//! accessors.
//!
//! Simulator conventions the tests rely on:
//! - For request id `r`, the greedy token at text index `n` is
//!   `1000 + (7r + n) % 5000` (1007, 1008, ... for request 1) until
//!   `SimEngine::set_text_len` tokens have been produced; after that the
//!   synthetic EOS (151645, the first EOS of `SpecialTokenIds::default`)
//!   dominates the logits. The index restarts at zero when a generated image
//!   is fed back into the request's context.
//! - `SimEngine::set_queue_depth` bounds the batches the scheduler may keep
//!   unresolved. Depth 1 is the serial oracle that the depth-invariance tests
//!   compare depth 2 against.
//! - The KV page size is 64 tokens, one unit per page (`WorkerInfo::default`),
//!   unless a test calls `SimEngine::set_page_tokens` or `set_groups`, and
//!   generated images default to 512x512 (`ImageParams::default`).
//! - The `general` request gauges and the `kv_cache` and `encoder` counters in
//!   `SchedulerStats` are snapshots that `Scheduler::publish_cache_stats`
//!   refreshes as the loop runs, not when an event is sent, so they can trail
//!   the event stream. The `prefix` counters are recorded at admission.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TryRecvError;
use uniserve_worker_ipc::{ForwardMode, MediaCall};

use uniserve_core::{EngineCoreOutput, FinishReason};
use uniserve_core::{
    FeedbackNextToken, FeedbackSource, GenerationConstraint, GenerationLimits, GenerationRequest,
    ImageEncoderInput, ImageGenerationConfig, ImageIngestStep, ImageInput, ImageParams,
    ImageTrigger, MultimodalInputs, RequestId, SamplingParams,
};
use uniserve_engine::{
    BatchEvent, BatchResult, EVENT_BUFFER_CAPACITY, EngineHandle, ExecutionBatch, Executor,
    ExecutorError, ExecutorInfo, ExecutorSubmitError, Scheduler, SchedulingPolicy, SimEngine,
    SimExecutor, SpecialTokenIds, WorkerId,
};

/// Default special tokens, whose first EOS id equals the simulator's synthetic
/// EOS, so the simulator's end of text finishes requests with `Eos`.
fn ctrl() -> SpecialTokenIds {
    SpecialTokenIds::default()
}

/// Advertised generation features follow the worker's supported calls: image
/// generation needs the complete latent-preparation, denoising, and
/// image-decoding path (video decoding does not substitute), and each encoder
/// call enables only its own feature.
#[test]
fn generation_capabilities_require_complete_paths_and_distinct_encoders() {
    use uniserve_core::GenerationFeatures;
    use uniserve_worker_ipc::CallKind;

    let image_path = vec![
        CallKind::Media(MediaCall::LatentPreparation),
        CallKind::Media(MediaCall::Denoising),
        CallKind::Media(MediaCall::ImageDecoding),
    ];
    let cases = [
        (
            vec![CallKind::Media(MediaCall::VisionEncoding)],
            GenerationFeatures::VISION_ENCODE,
        ),
        (
            vec![CallKind::Media(MediaCall::LatentEncoding)],
            GenerationFeatures::LATENT_ENCODE,
        ),
        (
            vec![
                CallKind::Media(MediaCall::VisionEncoding),
                CallKind::Media(MediaCall::LatentEncoding),
            ],
            GenerationFeatures::VISION_ENCODE | GenerationFeatures::LATENT_ENCODE,
        ),
        (image_path, GenerationFeatures::IMAGE_GENERATION),
        (
            vec![
                CallKind::Media(MediaCall::Denoising),
                CallKind::Media(MediaCall::ImageDecoding),
            ],
            GenerationFeatures::empty(),
        ),
        (
            vec![
                CallKind::Media(MediaCall::LatentPreparation),
                CallKind::Media(MediaCall::ImageDecoding),
            ],
            GenerationFeatures::empty(),
        ),
        (
            vec![
                CallKind::Media(MediaCall::LatentPreparation),
                CallKind::Media(MediaCall::Denoising),
                CallKind::Media(MediaCall::VideoDecoding),
            ],
            GenerationFeatures::empty(),
        ),
    ];
    for (calls, expected) in cases {
        let mut sim = SimEngine::new();
        sim.mut_info_for_test().supported_calls = vec![
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::Decode),
        ];
        sim.mut_info_for_test().supported_calls.extend(calls);
        let scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
        assert_eq!(
            scheduler.generation_limits().features,
            GenerationFeatures::UNDERSTANDING | expected,
        );
    }
}

/// Encoder feature byte limits clamp to the smaller of the worker's encoder
/// cache entry size and buffer pool size, and the encoder cache entry count
/// clamps to the worker's advertised entries.
#[test]
fn encoder_products_obey_worker_entry_capacity() {
    let mut sim = SimEngine::new();
    let info = sim.mut_info_for_test();
    info.buffer_pool_bytes = 8 << 30;
    info.encoder_entry_bytes = 128 << 20;
    info.encoder_cache_entries = 64;
    let scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    let limits = scheduler.generation_limits();
    assert_eq!(limits.max_vision_feature_bytes, 128 << 20);
    assert_eq!(limits.max_latent_feature_bytes, 128 << 20);
    assert_eq!(limits.encoder_cache_entries, 64);
}

fn text_input(token_ids: Vec<u32>) -> (Vec<u32>, MultimodalInputs) {
    (token_ids, MultimodalInputs::default())
}

/// Builds a prompt with one context image between `before` and `after`.
///
/// The image sits at prompt-token position `before.len()`, the boundary
/// between `before` and `after`, occupies `logical_positions` positions, and
/// contributes `physical_tokens` KV tokens through a single ViT encode. `hash` is the image identity from which encoder-cache keys are
/// derived; the payload is a placeholder.
fn image_input(
    mut before: Vec<u32>,
    after: Vec<u32>,
    hash: u64,
    logical_positions: u32,
    physical_tokens: u32,
) -> (Vec<u32>, MultimodalInputs) {
    let position = before.len() as u32;
    before.extend(after);
    (
        before,
        MultimodalInputs {
            images: vec![ImageInput {
                hash,
                b64: "aW1hZ2U=".to_string(),
                position,
                num_positions: logical_positions,
                encoders: vec![ImageEncoderInput {
                    encoder: ImageIngestStep::VitEncode,
                    num_kv_tokens: Some(physical_tokens),
                    max_kv_tokens: None,
                }],
            }],
        },
    )
}

/// Builds a unified generation request with a fixed image-generation policy.
///
/// The policy triggers on token 1000, lets image-only requests start image
/// computation without first generating internal text, and, for
/// Default-constraint requests, feeds each generated image back through one
/// ViT encode. Tests that need a different trigger use `with_trigger` or edit
/// `image_generation` afterwards.
///
/// # Panics
///
/// Panics when the request fails `GenerationRequest::validate_resources`
/// against the fixed limits below, so every test starts from a well-formed
/// request. The scheduler validates again against its own resolved limits.
fn generation_request(
    request_id: RequestId,
    (prompt_token_ids, multimodal_inputs): (Vec<u32>, MultimodalInputs),
    sampling: SamplingParams,
    image: ImageParams,
    constraint: GenerationConstraint,
    max_und_tokens: usize,
) -> GenerationRequest {
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

    let cache = Default::default();
    let limits = GenerationLimits {
        features: uniserve_core::GenerationFeatures::UNDERSTANDING
            | uniserve_core::GenerationFeatures::VISION_ENCODE
            | uniserve_core::GenerationFeatures::IMAGE_GENERATION,
        max_latent_units: 65_536,
        latent_downsample: 16,
        max_vae_grid_tokens: 65_536,
        max_vit_grid_tokens: 8_192,
        max_latent_feature_bytes: 256 << 20,
        max_vision_feature_bytes: 256 << 20,
        commit_marker_tokens: 2,
        max_cfg_branches: 3,
        encoder_cache_entries: 256,
    };
    let request = GenerationRequest {
        request_id,
        prompt_token_ids,
        multimodal_inputs,
        negative_prompt_token_ids: Vec::new(),
        constraint,
        sampling,
        image,
        max_und_tokens,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache,
        image_generation: policy,
    };
    request
        .validate_resources(&limits)
        .expect("bounded simulation request");
    request
}

fn with_trigger(mut request: GenerationRequest, trigger: ImageTrigger) -> GenerationRequest {
    request.image_generation.trigger = trigger;

    request
}

struct Collected {
    text: usize,
    images: usize,
    finished: bool,
    reason: Option<FinishReason>,
}

/// Runs the given (constraint, count) requests through a fresh scheduler at `depth`,
/// returning per-request collected counts.
///
/// Request ids are assigned from 1 in case order. A request that emits no event
/// before the 20-second deadline has no entry in the result, and one that emits
/// events but no terminal event has `finished == false`.
fn batch_requests(
    depth: u32,
    policy: SchedulingPolicy,
    cases: &[(GenerationConstraint, usize)],
) -> HashMap<RequestId, Collected> {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(depth);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::with_policy(executor, ctrl(), 32, policy).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut rxs: HashMap<RequestId, uniserve_engine::EventRx> = HashMap::new();
    let mut id = 1u64;
    for (constraint, n) in cases {
        for _ in 0..*n {
            let req = generation_request(
                RequestId(id),
                text_input(vec![1, 2, 3, 4, 5]),
                SamplingParams::default(),
                ImageParams {
                    steps: 4,
                    ..Default::default()
                },
                *constraint,
                16,
            );
            let erx = handle.submit(req).unwrap();
            rxs.insert(RequestId(id), erx);
            id += 1;
        }
    }

    let total = rxs.len();
    let mut out: HashMap<RequestId, Collected> = HashMap::new();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut done = 0;
    while done < total && Instant::now() < deadline {
        for (rid, erx) in rxs.iter_mut() {
            while let Ok(ev) = erx.try_recv() {
                let c = out.entry(*rid).or_insert(Collected {
                    text: 0,
                    images: 0,
                    finished: false,
                    reason: None,
                });
                match ev {
                    EngineCoreOutput::TextToken { .. } => c.text += 1,
                    EngineCoreOutput::ImageDone { .. } => c.images += 1,
                    EngineCoreOutput::Finished { reason, .. } if !c.finished => {
                        c.finished = true;
                        c.reason = Some(reason);
                        done += 1;
                    }
                    _ => {}
                }
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = jh.join();
    out
}

#[test]
fn text_and_image_requests_complete() {
    let out = batch_requests(
        2,
        SchedulingPolicy::Fcfs,
        &[
            (GenerationConstraint::UndOnly, 3),
            (GenerationConstraint::GenOnly, 2),
        ],
    );
    assert_eq!(out.len(), 5);
    for (id, c) in &out {
        assert!(c.finished, "req {id:?} did not finish");
        // Ids follow case order, so 1-3 are the Und-only text requests.
        if id.0 <= 3 {
            assert!(c.text > 0, "text req {id:?} produced no tokens");
        } else {
            assert_eq!(c.images, 1, "image req {id:?} produced {} images", c.images);
        }
    }
}

/// Cancelling a resident image request reports `Cancelled` and releases its
/// resources, so an image request submitted behind it still completes. The
/// latent pool (64 usable pages of 64 units) and buffer pool are reduced from
/// the simulator defaults, and the first request runs `ImageParams::MAX_STEPS`
/// denoising steps so it is still resident when the second arrives.
#[test]
fn cancellation_releases_latent_admission_for_a_waiting_image() {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(4);
    sim.mut_info_for_test().latent_page_units = 64;
    sim.mut_info_for_test().latent_pages = 65;
    sim.mut_info_for_test().buffer_pool_bytes = 16 << 20;
    sim.mut_info_for_test().max_batch_calls = 1024;
    let scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let thread = thread::spawn(move || scheduler.run(rx));

    let mut first = handle
        .submit(generation_request(
            RequestId(1),
            text_input(vec![4, 5, 6]),
            SamplingParams::default(),
            ImageParams {
                steps: ImageParams::MAX_STEPS,
                ..Default::default()
            },
            GenerationConstraint::GenOnly,
            0,
        ))
        .unwrap();
    let begin_deadline = Instant::now() + Duration::from_secs(10);
    let mut began = false;
    while !began && Instant::now() < begin_deadline {
        match first.try_recv() {
            Ok(EngineCoreOutput::ImageBegin { .. }) => began = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert!(began, "resident image did not enter generation");

    let mut second = handle
        .submit(generation_request(
            RequestId(2),
            text_input(vec![4, 5, 6]),
            SamplingParams::default(),
            ImageParams {
                steps: 2,
                ..Default::default()
            },
            GenerationConstraint::GenOnly,
            0,
        ))
        .unwrap();
    handle.cancel(RequestId(1));

    let mut reasons = HashMap::new();
    let mut second_images = 0;
    let deadline = Instant::now() + Duration::from_secs(15);
    while reasons.len() < 2 && Instant::now() < deadline {
        for (id, receiver) in [(RequestId(1), &mut first), (RequestId(2), &mut second)] {
            while let Ok(event) = receiver.try_recv() {
                match event {
                    EngineCoreOutput::ImageDone { .. } if id == RequestId(2) => second_images += 1,
                    EngineCoreOutput::Finished { reason, .. } => {
                        reasons.insert(id, reason);
                    }
                    _ => {}
                }
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = thread.join();

    assert_eq!(reasons.get(&RequestId(1)), Some(&FinishReason::Cancelled));
    assert!(reasons.contains_key(&RequestId(2)));
    assert_eq!(second_images, 1);
}

/// Image requests waiting for the one flow-prefix request row do not starve
/// the younger request that holds it.
///
/// With four request slots, three requests run and one row is left for a
/// multi-branch guidance prefix. The younger request 3 has a short prompt, so
/// it reaches denoising first and takes that row while the older requests 1
/// and 2 are still prefilling their chunked prompts. When those reach
/// denoising, neither can allocate a prefix until request 3 finishes and
/// releases the row. One denoising step costs 1024 latent units times two
/// guidance branches, so the two waiting requests together cost the whole
/// pass budget. At queue depth one every step of request 3 resolves before
/// its next step is planned, so each pass finds the denoiser free and tries
/// the older requests first.
#[test]
fn flow_prefix_holder_progresses_while_older_images_wait() {
    const STEPS: u16 = 12;

    let mut sim = SimEngine::new();
    sim.set_queue_depth(1);
    sim.mut_info_for_test().request_slots = 4;
    let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    scheduler.set_token_budget(4096);
    scheduler.set_long_prefill_threshold(64);
    assert_eq!(scheduler.config().max_num_seqs, 3);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let thread = thread::spawn(move || scheduler.run(rx));

    let image = ImageParams {
        steps: STEPS,
        ..Default::default()
    };
    assert_eq!(image.cfg_branch_count(), 2);
    let long_prompt: Vec<u32> = (0..640).map(|index| 10 + index % 500).collect();
    let prompts = [
        (RequestId(1), long_prompt.clone()),
        (RequestId(2), long_prompt),
        (RequestId(3), vec![4, 5, 6]),
    ];
    let mut receivers = Vec::new();
    for (id, prompt) in prompts {
        let events = handle
            .submit(generation_request(
                id,
                text_input(prompt),
                SamplingParams::default(),
                image.clone(),
                GenerationConstraint::GenOnly,
                0,
            ))
            .unwrap();
        receivers.push((id, events));
    }

    let mut reasons = HashMap::new();
    let mut images = HashMap::new();
    let deadline = Instant::now() + Duration::from_secs(20);
    while reasons.len() < receivers.len() && Instant::now() < deadline {
        for (id, receiver) in receivers.iter_mut() {
            while let Ok(event) = receiver.try_recv() {
                match event {
                    EngineCoreOutput::ImageDone { .. } => *images.entry(*id).or_insert(0) += 1,
                    EngineCoreOutput::Finished { reason, .. } => {
                        reasons.insert(*id, reason);
                    }
                    _ => {}
                }
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = thread.join();

    for (id, _) in &receivers {
        assert_eq!(
            reasons.get(id),
            Some(&FinishReason::ImageDone),
            "request {id:?} did not finish its image"
        );
        assert_eq!(images.get(id), Some(&1));
    }
}

/// Image generation exposes every declared denoise step in order before one committed image and terminal completion.
#[test]
fn image_events_cover_declared_denoise_steps() {
    const STEPS: u16 = 3;
    let scheduler =
        Scheduler::new(Box::new(SimExecutor::new(SimEngine::new())), ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));

    let mut events = handle
        .submit(generation_request(
            RequestId(1),
            text_input(vec![4, 5, 6]),
            SamplingParams::default(),
            ImageParams {
                steps: STEPS,
                ..Default::default()
            },
            GenerationConstraint::GenOnly,
            0,
        ))
        .unwrap();

    let mut begin = None;
    let mut steps = Vec::new();
    let mut commits = 0;
    let mut image_done = None;
    let mut finished = None;
    let deadline = Instant::now() + Duration::from_secs(15);
    while finished.is_none() && Instant::now() < deadline {
        match events.try_recv() {
            Ok(EngineCoreOutput::ImageBegin {
                image_id,
                height,
                width,
                steps,
            }) => begin = Some((image_id, height, width, steps)),
            Ok(EngineCoreOutput::ImageStep { image_id, step }) => steps.push((image_id, step)),
            Ok(EngineCoreOutput::ImageCommit { image_id }) => {
                assert_eq!(image_id, 1);
                commits += 1;
            }
            Ok(EngineCoreOutput::ImageDone {
                image_id,
                height,
                width,
                ..
            }) => image_done = Some((image_id, height, width)),
            Ok(EngineCoreOutput::Finished { reason, images, .. }) => {
                finished = Some((reason, images))
            }
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = scheduler_thread.join();

    assert_eq!(begin, Some((1, 512, 512, STEPS)));
    assert_eq!(steps, (1..=STEPS).map(|step| (1, step)).collect::<Vec<_>>());
    assert_eq!(commits, 1);
    let (image_id, height, width) = image_done.expect("image completion event");
    assert_eq!((image_id, height, width), (1, 512, 512));
    assert_eq!(finished, Some((FinishReason::ImageDone, 1)));
}

#[test]
fn scheduler_clamps_max_batch_to_worker_info() {
    let mut sim = SimEngine::new();
    sim.mut_info_for_test().max_batch_calls = 3;
    let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();

    assert_eq!(sched.config().max_batch, 3);
}

/// Single-worker results must be identical regardless of queue depth:
/// the same prompts produce the same per-request token counts at depth 1 and 2.
#[test]
fn queue_depth_is_token_identical() {
    let d1 = batch_requests(
        1,
        SchedulingPolicy::Fcfs,
        &[(GenerationConstraint::UndOnly, 4)],
    );
    let d2 = batch_requests(
        2,
        SchedulingPolicy::Fcfs,
        &[(GenerationConstraint::UndOnly, 4)],
    );
    for id in d1.keys() {
        assert_eq!(
            d1[id].text, d2[id].text,
            "req {id:?} token count differs by depth"
        );
        assert_eq!(d1[id].reason, d2[id].reason);
    }
}

/// Per-domain call accounting balances over a request's lifetime: every
/// launched call completes and returns its credit, and the stats reporter's
/// snapshot shows the same balance.
#[test]
fn call_window_metrics_record_the_full_lifecycle() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = uniserve_engine::EngineHandle::new(command_tx);

    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    sim.set_text_len(6);
    let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    let request = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3, 4, 5]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        16,
    );
    let mut events = handle.submit(request).unwrap();
    let mut finished = false;
    for _ in 0..512 {
        scheduler.step(&commands);
        while let Ok(event) = events.try_recv() {
            if matches!(event, EngineCoreOutput::Finished { .. }) {
                finished = true;
            }
        }
        if finished {
            break;
        }
    }
    assert!(finished, "request finished");
    // Keep stepping after the terminal event so the counters are compared only
    // once the loop has drained.
    for _ in 0..32 {
        scheduler.step(&commands);
    }

    let active_domains = [
        &scheduler.stats.domains.prefill,
        &scheduler.stats.domains.decode,
        &scheduler.stats.domains.flow,
    ]
    .into_iter()
    .filter(|domain| domain.launched_calls.load(Ordering::Relaxed) > 0)
    .collect::<Vec<_>>();
    assert!(!active_domains.is_empty(), "domain launches recorded");
    for domain in &active_domains {
        assert_eq!(domain.active_credits.load(Ordering::Relaxed), 0);
        assert_eq!(
            domain.launched_calls.load(Ordering::Relaxed),
            domain.completed_calls.load(Ordering::Relaxed)
        );
        assert_eq!(
            domain.launched_calls.load(Ordering::Relaxed),
            domain.reclaimed_credits.load(Ordering::Relaxed)
        );
        assert!(domain.completed_batches.load(Ordering::Relaxed) > 0);
    }
    let mut reporter = uniserve_engine::SchedulerStatsReporter::default();
    let snapshot = reporter.snapshot(&scheduler.stats);
    let decoded_active = snapshot
        .domain_stats
        .iter()
        .filter(|domain| domain.launched_calls > 0)
        .collect::<Vec<_>>();
    assert_eq!(decoded_active.len(), active_domains.len());
    assert!(decoded_active.iter().all(|domain| {
        domain.active_credits == 0
            && domain.launched_calls == domain.completed_calls
            && domain.launched_calls == domain.reclaimed_credits
    }));
}

/// Runs one text request by stepping a fresh scheduler on the test thread and
/// returns its emitted token ids.
///
/// `depth` is the simulator queue depth: 1 gives the serial oracle, and 2 lets
/// the scheduler submit a successor before its predecessor's token has been
/// observed on the host.
fn relay_run(
    sampling: SamplingParams,
    stop_token_ids: Vec<u32>,
    max_und_tokens: usize,
    text_len: usize,
    depth: u32,
) -> Vec<u32> {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = uniserve_engine::EngineHandle::new(command_tx);

    let mut sim = SimEngine::new();
    sim.set_queue_depth(depth);
    sim.set_text_len(text_len);
    let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    let mut request = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3, 4, 5]),
        sampling,
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        max_und_tokens,
    );
    request.stop_token_ids = stop_token_ids;
    let mut events = handle.submit(request).unwrap();
    let mut tokens = Vec::new();
    let mut finished = false;
    for _ in 0..512 {
        scheduler.step(&commands);
        while let Ok(event) = events.try_recv() {
            match event {
                EngineCoreOutput::TextToken { id, .. } => tokens.push(id),
                EngineCoreOutput::Finished { .. } => finished = true,
                _ => {}
            }
        }
        if finished {
            break;
        }
    }
    assert!(finished, "processor request reached a terminal event");

    tokens
}

/// Sampling processors (penalties, logprobs, minimum-token floors, stop tokens,
/// and allowlists) yield the same tokens when successors are launched ahead of
/// host observation at depth 2 as in the serial depth-one run.
#[test]
fn generalized_processor_successors_match_depth_one_before_observation() {
    let cases: Vec<(&str, SamplingParams, Vec<u32>)> = vec![
        (
            "repetition+frequency+presence penalties",
            SamplingParams {
                temperature: 0.9,
                top_k: 40,
                seed: Some(7),
                repetition_penalty: 1.3,
                frequency_penalty: 0.7,
                presence_penalty: 0.4,
                ..SamplingParams::default()
            },
            Vec::new(),
        ),
        (
            "requested logprobs",
            SamplingParams {
                temperature: 0.7,
                seed: Some(11),
                n_logprobs: 5,
                ..SamplingParams::default()
            },
            Vec::new(),
        ),
        (
            "minimum-token floor over stop tokens",
            SamplingParams {
                temperature: 0.0,
                min_tokens: 6,
                ..SamplingParams::default()
            },
            vec![1_002, 1_003],
        ),
        (
            "greedy penalties with logprobs and a stop token",
            SamplingParams {
                temperature: 0.0,
                repetition_penalty: 1.5,
                frequency_penalty: 0.25,
                n_logprobs: 3,
                ..SamplingParams::default()
            },
            vec![1_004],
        ),
        (
            "static allowed-token whitelist under stochastic sampling",
            SamplingParams {
                temperature: 0.8,
                seed: Some(29),
                allowed_token_ids: Some(vec![1_000, 1_001, 1_002, 1_003, 1_004, 1_005]),
                ..SamplingParams::default()
            },
            Vec::new(),
        ),
    ];

    for (label, sampling, stop_token_ids) in cases {
        let serial = relay_run(sampling.clone(), stop_token_ids.clone(), 24, 12, 1);
        let relayed = relay_run(sampling, stop_token_ids, 24, 12, 2);

        assert!(!serial.is_empty(), "[{label}] produced no tokens");
        assert_eq!(
            serial, relayed,
            "[{label}] device-relay tokens diverged from the depth-one serial oracle",
        );
    }
}

/// Decoding after a context image produces the same eight tokens and EOS at
/// queue depths 1 and 2.
#[test]
fn image_context_decode_is_depth_invariant() {
    // One command channel serves both runs: each run's scheduler drains the
    // submission made just before it is stepped.
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = uniserve_engine::EngineHandle::new(command_tx);

    let mut runs = Vec::new();
    for queue_depth in [1, 2] {
        let mut sim = SimEngine::new();
        sim.set_queue_depth(queue_depth);
        sim.set_text_len(8);
        let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
        let request = generation_request(
            RequestId(1),
            image_input(vec![1, 2], vec![3, 4], 0xD3C0DE, 4, 1),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            16,
        );
        let mut events = handle.submit(request).unwrap();
        let mut finish_reason = None;
        let mut tokens = Vec::new();
        for _ in 0..512 {
            scheduler.step(&commands);
            while let Ok(event) = events.try_recv() {
                match event {
                    EngineCoreOutput::TextToken { id, .. } => tokens.push(id),
                    EngineCoreOutput::Finished { reason, .. } => finish_reason = Some(reason),
                    _ => {}
                }
            }
            if finish_reason.is_some() {
                break;
            }
        }
        assert_eq!(finish_reason, Some(FinishReason::Eos));
        assert_eq!(tokens.len(), 8);
        runs.push(tokens);
    }

    assert_eq!(runs[0], runs[1]);
}

#[test]
fn stop_token_terminates_with_stop() {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut req = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        64,
    );
    // 1007 is request 1's first greedy token, so the request stops on its
    // first sample; `include_stop_token` is false, so nothing is emitted.
    req.stop_token_ids = vec![1007];
    let mut erx = handle.submit(req).unwrap();

    let mut reason = None;
    let mut text = 0;
    let deadline = Instant::now() + Duration::from_secs(10);
    while reason.is_none() && Instant::now() < deadline {
        if let Ok(ev) = erx.try_recv() {
            match ev {
                EngineCoreOutput::TextToken { .. } => text += 1,
                EngineCoreOutput::Finished { reason: r, .. } => reason = Some(r),
                _ => {}
            }
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    }
    handle.shutdown();
    let _ = jh.join();
    assert_eq!(reason, Some(FinishReason::Stop));
    assert_eq!(text, 0, "the stop token itself must not be emitted");
}

/// Issues `EngineHandle::abort` (when `abort`) or `EngineHandle::cancel` once
/// a text request has emitted its first token, and returns its finish reason.
///
/// A long simulated text keeps the request generating, so the command is
/// observed mid-flight.
fn run_until_control(abort: bool) -> FinishReason {
    let mut sim = SimEngine::new();
    sim.set_text_len(1024);
    sim.set_queue_depth(2);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        1024,
    );
    let mut erx = handle.submit(req).unwrap();

    // Wait until it is generating, then issue the control command.
    let mut saw_token = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !saw_token && Instant::now() < deadline {
        if let Ok(EngineCoreOutput::TextToken { .. }) = erx.try_recv() {
            saw_token = true;
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    }
    if abort {
        handle.abort(RequestId(1));
    } else {
        handle.cancel(RequestId(1));
    }

    let mut reason = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while reason.is_none() && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(EngineCoreOutput::Finished { reason: r, .. }) => reason = Some(r),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();
    reason.expect("request never finished")
}

#[test]
fn abort_and_cancel_are_distinct() {
    assert_eq!(run_until_control(false), FinishReason::Cancelled);
    assert_eq!(run_until_control(true), FinishReason::Aborted);
}

/// A stop-string cutoff reported through `EngineHandle::stop_at` finishes only
/// its own request with `Stop`; a concurrent request runs to its own end.
#[test]
fn stop_string_cutoff_is_request_local() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1024);
    sim.set_queue_depth(2);
    let scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));

    let mut unrelated_events = handle
        .submit(generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            6,
        ))
        .unwrap();

    let mut stopping = generation_request(
        RequestId(2),
        text_input(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        1024,
    );
    stopping.stop_strings = vec!["boundary".to_string()];
    let mut stopping_events = handle.submit(stopping).unwrap();

    // A request with stop strings is acknowledged by the frontend decoder, not
    // on receive (see `EngineHandle::submit`). The test plays that role for the
    // first token and, once a second token arrives, stops the output at that
    // one-token prefix.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut consumed_tokens = 0;
    while consumed_tokens < 2 && Instant::now() < deadline {
        match stopping_events.try_recv() {
            Ok(EngineCoreOutput::TextToken { .. }) => {
                consumed_tokens += 1;
                if consumed_tokens == 1 {
                    handle.acknowledge_at(RequestId(2), 1);
                }
            }
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert_eq!(consumed_tokens, 2);
    handle.stop_at(RequestId(2), 1);

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut stop_reason = None;
    let mut unrelated_reason = None;
    while Instant::now() < deadline {
        while let Ok(event) = stopping_events.try_recv() {
            if let EngineCoreOutput::Finished { reason, .. } = event {
                stop_reason = Some(reason);
            }
        }
        while let Ok(event) = unrelated_events.try_recv() {
            if let EngineCoreOutput::Finished { reason, .. } = event {
                unrelated_reason = Some(reason);
            }
        }
        if stop_reason.is_some() && unrelated_reason.is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = scheduler_thread.join();

    assert_eq!(stop_reason, Some(FinishReason::Stop));
    assert!(matches!(
        unrelated_reason,
        Some(FinishReason::Eos | FinishReason::MaxTokens)
    ));
}

/// Text requests complete on a worker whose unit pool holds a full-attention
/// group and a sliding-window group with different page shapes. The prompt
/// is several times the window, so the sliding group retires pages while the
/// request runs and its declared tables carry only in-window pages, and a
/// second request with the same prompt reuses the first one's prefix. Every
/// unit returns to the pool once both finish.
#[test]
fn hybrid_groups_share_one_unit_pool() {
    use std::sync::atomic::Ordering;
    use uniserve_core::{KvCacheGroup, KvGroupKind};

    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    sim.set_text_len(4);
    sim.set_num_units(64);
    let base = sim.mut_info_for_test().kv_cache.clone().unwrap().groups[0].clone();
    sim.set_groups(vec![
        KvCacheGroup {
            layer_ids: (0..4).collect(),
            ..base.clone()
        },
        KvCacheGroup {
            kind: KvGroupKind::SlidingWindow {
                window: 96,
                sink: 0,
            },
            page_tokens: 32,
            units_per_page: 3,
            layer_ids: (4..16).collect(),
            ..base
        },
    ]);
    let mut executor = SimExecutor::new(sim);
    let dispatched = executor.observe();
    let mut sched = Scheduler::new(Box::new(executor), ctrl(), 32).unwrap();
    // A 64-token chunk reads at most six sliding pages with the window.
    sched.set_long_prefill_threshold(64);
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    // 20 full pages of 64 tokens alone take 20 units; holding every sliding
    // page of 1280 tokens as well would take another 120, beyond the pool.
    let prompt: Vec<u32> = (0..1280u32).map(|i| (i % 61) + 3).collect();
    let run_one = |rid: u64| {
        let mut erx = handle
            .submit(generation_request(
                RequestId(rid),
                text_input(prompt.clone()),
                SamplingParams::default(),
                ImageParams::default(),
                GenerationConstraint::UndOnly,
                16,
            ))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match erx.try_recv() {
                Ok(EngineCoreOutput::Finished { reason, .. }) => return reason,
                Ok(EngineCoreOutput::Rejected { message, .. }) => {
                    panic!("request {rid} was rejected: {message}")
                }
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        panic!("request {rid} did not finish under hybrid groups");
    };

    assert_ne!(run_one(1), FinishReason::Error);
    let reused_before = stats.prefix.hit_tokens.load(Ordering::Relaxed);
    assert_ne!(run_one(2), FinishReason::Error);
    // The reused prefix is a multiple of the 64-token largest page whose
    // sliding window [M - 96, M) the first request published before retiring.
    let reused = stats.prefix.hit_tokens.load(Ordering::Relaxed) - reused_before;
    assert!(
        reused >= 1024 && reused.is_multiple_of(64),
        "reused {reused} tokens"
    );

    // Every unit returns to the pool once both requests retire. A request's
    // last pipelined step may still be in flight when its final event
    // arrives, so the pool is read once the engine has applied it.
    let deadline = Instant::now() + Duration::from_secs(10);
    while stats.kv_cache.free_units.load(Ordering::Relaxed)
        != stats.kv_cache.num_units.load(Ordering::Relaxed)
        && Instant::now() < deadline
    {
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = jh.join();
    assert_eq!(
        stats.kv_cache.free_units.load(Ordering::Relaxed),
        stats.kv_cache.num_units.load(Ordering::Relaxed)
    );

    // Every declared table covers whole pages of its group from its start
    // page; the sliding group's tables move their start and never hold more
    // than the pages a window and two pipelined chunks intersect.
    let mut retired = false;
    for event in dispatched.try_iter() {
        let BatchEvent::Submitted(batch) = event else {
            continue;
        };
        for (_, placement) in &batch.requests {
            for table in &placement.block_tables {
                let (page_tokens, units_per_page) = if table.group_id == 0 {
                    (64, 1)
                } else {
                    (32, 3)
                };
                assert_eq!(table.unit_ids.len() % units_per_page, 0);
                let pages = (table.unit_ids.len() / units_per_page) as u32;
                assert_eq!(
                    table.allocated_tokens,
                    (table.start_page + pages) * page_tokens
                );
                if table.group_id == 1 {
                    assert!(pages <= (96 + 2 * 64) / 32 + 1, "{table:?}");
                    retired |= table.start_page > 0;
                } else {
                    assert_eq!(table.start_page, 0);
                }
            }
        }
    }
    assert!(retired, "the sliding group never retired a page");
}

/// Two sequential requests with the same prompt: the first publishes its full
/// prompt blocks to the prefix cache, and the second reuses them.
#[test]
fn prefix_cache_reuses_shared_prompt() {
    use std::sync::atomic::Ordering;
    let mut sim = SimEngine::new();
    sim.set_text_len(4);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    // 600 tokens span 9 full 64-token pages plus a partial page; the
    // assertions below are lower bounds.
    let prompt: Vec<u32> = (0..600u32).map(|i| (i % 53) + 7).collect();

    let run_one = |rid: u64, handle: &EngineHandle| {
        let mut erx = handle
            .submit(generation_request(
                RequestId(rid),
                text_input(prompt.clone()),
                SamplingParams::default(),
                ImageParams::default(),
                GenerationConstraint::UndOnly,
                8,
            ))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut done = false;
        while !done && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(EngineCoreOutput::Finished { .. }) => done = true,
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        assert!(done, "req {rid} did not finish");
    };

    // Cold: req1 populates the prefix cache.
    run_one(1, &handle);
    assert!(
        stats.kv_cache.pages_stored.load(Ordering::Relaxed) >= 2,
        "req1 should cache >= 2 full prompt pages"
    );
    let hits_before = stats.prefix.hit_tokens.load(Ordering::Relaxed);

    // Warm: req2 (same prompt) reuses the cached prefix.
    run_one(2, &handle);
    let hits_after = stats.prefix.hit_tokens.load(Ordering::Relaxed);
    assert!(
        hits_after - hits_before >= 512,
        "req2 should reuse >= 512 cached prefix tokens (before={hits_before} after={hits_after})"
    );

    handle.shutdown();
    let _ = jh.join();
}

/// Per-request `CachePolicy` governs prefix reuse: isolation keys partition the
/// cache, `read: false` skips lookup but still publishes, and `write: false`
/// looks up but publishes nothing. Each stage uses a fresh isolation key so
/// earlier stages cannot supply hits.
#[test]
fn prefix_cache_enforces_read_write_and_isolation_policy() {
    use std::sync::atomic::Ordering;

    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));
    let prompt: Vec<u32> = (0..600_u32).map(|index| (index % 47) + 5).collect();

    let run = |id: u64, read: bool, write: bool, isolation_key: u64| {
        let mut request = generation_request(
            RequestId(id),
            text_input(prompt.clone()),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            8,
        );
        request.cache = uniserve_core::CachePolicy {
            read,
            write,
            isolation_key: Some(isolation_key),
        };
        let mut events = handle.submit(request).expect("submit cache request");
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match events.try_recv() {
                Ok(EngineCoreOutput::Finished { .. }) => return,
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        panic!("cache request {id} did not finish");
    };

    run(1, true, true, 11);
    let cold_hits = stats.prefix.hit_tokens.load(Ordering::Relaxed);
    run(2, true, true, 22);
    assert_eq!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed),
        cold_hits,
        "a different isolation key reused cached pages"
    );
    run(3, true, true, 11);
    assert!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed) >= cold_hits + 128,
        "the matching isolation key did not reuse its prefix"
    );

    let before_bypass = stats.prefix.hit_tokens.load(Ordering::Relaxed);
    run(4, false, true, 33);
    assert_eq!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed),
        before_bypass
    );
    run(5, true, true, 33);
    assert!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed) >= before_bypass + 128,
        "read bypass prevented a write-enabled request from publishing its prefix"
    );

    let before_no_store = stats.prefix.hit_tokens.load(Ordering::Relaxed);
    run(6, true, false, 44);
    run(7, true, true, 44);
    assert_eq!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed),
        before_no_store,
        "a write-disabled request published its prefix"
    );

    handle.shutdown();
    let _ = jh.join();
}

/// Under FCFS with a 256-token step budget and prefill chunks capped at 64
/// tokens, a 1000-token prompt is prefilled in many chunks; it and a
/// concurrent short request must both finish and emit text. The test observes
/// only completion, not how chunks and decodes share individual steps.
#[test]
fn chunked_prefill_progresses_with_decode() {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    let executor = Box::new(SimExecutor::new(sim));
    let mut sched = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs).unwrap();
    sched.set_long_prefill_threshold(64); // cap a prefill chunk at 64 tokens
    sched.set_token_budget(256); // leaves room for other decodes per step
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    // A long prompt plus a short concurrent request.
    let long_prompt: Vec<u32> = (0..1000u32).map(|i| (i % 91) + 7).collect();
    let mut erx1 = handle
        .submit(generation_request(
            RequestId(1),
            text_input(long_prompt),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            8,
        ))
        .unwrap();
    let mut erx2 = handle
        .submit(generation_request(
            RequestId(2),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            8,
        ))
        .unwrap();

    let collect = |erx: &mut uniserve_engine::EventRx| -> (usize, bool) {
        let mut text = 0;
        let mut done = false;
        let deadline = Instant::now() + Duration::from_secs(15);
        while !done && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(EngineCoreOutput::TextToken { .. }) => text += 1,
                Ok(EngineCoreOutput::Finished { .. }) => done = true,
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        (text, done)
    };
    let (t1, d1) = collect(&mut erx1);
    let (t2, d2) = collect(&mut erx2);
    handle.shutdown();
    let _ = jh.join();
    assert!(d1 && d2, "both requests must finish (d1={d1} d2={d2})");
    assert!(t1 > 0 && t2 > 0, "both must produce text (t1={t1} t2={t2})");
}

/// Runs one text request with the given sampling params and a sim `text_len`,
/// returning the emitted token ids, whether any token event carried a logprob,
/// and the finish reason (`None` when the request did not finish within the
/// 10-second deadline).
fn run_sampling(
    sampling: SamplingParams,
    text_len: usize,
    max_tokens: usize,
) -> (Vec<u32>, bool, Option<FinishReason>) {
    let mut sim = SimEngine::new();
    sim.set_text_len(text_len);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3]),
        sampling,
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        max_tokens,
    );
    let mut erx = handle.submit(req).unwrap();

    let mut toks = Vec::new();
    let mut any_logprob = false;
    let mut finished = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while finished.is_none() && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(EngineCoreOutput::TextToken { id, logprob, .. }) => {
                toks.push(id);
                if logprob.is_some() {
                    any_logprob = true;
                }
            }
            Ok(EngineCoreOutput::Finished { reason, .. }) => finished = Some(reason),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();
    (toks, any_logprob, finished)
}

#[test]
fn logprobs_flow_to_events() {
    let sp = SamplingParams {
        n_logprobs: 3,
        ..Default::default()
    };
    let (toks, any_logprob, finished) = run_sampling(sp, 8, 16);
    assert!(finished.is_some());
    assert!(!toks.is_empty());
    assert!(
        any_logprob,
        "logprobs requested but none populated on the event stream"
    );
}

#[test]
fn allowed_tokens_restricts_output() {
    let sp = SamplingParams {
        allowed_token_ids: Some(vec![1234]),
        ..Default::default()
    };
    let (toks, _lp, finished) = run_sampling(sp, 8, 6);
    assert!(finished.is_some());
    assert!(!toks.is_empty());
    assert!(
        toks.iter().all(|&t| t == 1234),
        "every token must be the single allowed id, got {toks:?}"
    );
}

/// The bad-word sequence [4321, 4321] bans 4321 only directly after a 4321:
/// the ban overrides the bias there, while the biased token still wins the
/// first position and the output stays within the allowlist.
#[test]
fn bad_word_suffix_overrides_bias_without_suppressing_its_prefix() {
    let sampling = SamplingParams {
        allowed_token_ids: Some(vec![1234, 4321]),
        logit_bias: vec![(4321, 1000.0)],
        bad_words_ids: vec![vec![4321, 4321]],
        ..Default::default()
    };
    let (tokens, _, finished) = run_sampling(sampling, 8, 6);
    assert!(finished.is_some());
    assert_eq!(tokens.len(), 6, "{tokens:?}, {finished:?}");
    assert_eq!(tokens[0], 4321);
    assert!(tokens.iter().all(|token| [1234, 4321].contains(token)));
    assert!(tokens.windows(2).all(|pair| pair != [4321, 4321]));
}

#[test]
fn logit_bias_forces_token() {
    // Strongly bias token 4321; it should dominate every step.
    let sp = SamplingParams {
        logit_bias: vec![(4321, 1000.0)],
        ..Default::default()
    };
    let (toks, _lp, finished) = run_sampling(sp, 8, 6);
    assert!(finished.is_some());
    assert!(
        toks.iter().all(|&t| t == 4321),
        "biased token must always win, got {toks:?}"
    );
}

#[test]
fn min_tokens_floor_overrides_early_eos() {
    // With text_len=1 the simulator prefers EOS from the second token on;
    // min_tokens=5 makes the scheduler suppress EOS until 5 tokens exist.
    let sp = SamplingParams {
        min_tokens: 5,
        ..Default::default()
    };
    let (toks, _lp, finished) = run_sampling(sp, 1, 50);
    assert!(finished.is_some());
    assert!(
        toks.len() >= 5,
        "min_tokens floor not honored, only {} tokens",
        toks.len()
    );
}

#[test]
fn default_sampling_is_unchanged() {
    // Default params are greedy (temperature 0), so the first token is the
    // simulator's natural token for request 1 at index 0: 1007.
    let (toks, _lp, finished) = run_sampling(SamplingParams::default(), 8, 16);
    assert!(finished.is_some());
    assert_eq!(toks[0], 1007);
}

#[test]
fn stochastic_sampling_reaches_synthetic_eos() {
    let sampling = SamplingParams {
        temperature: 1.0,
        seed: Some(7),
        ..Default::default()
    };
    let (_toks, _lp, finished) = run_sampling(sampling, 8, 16);
    assert_eq!(finished, Some(FinishReason::Eos));
}

/// The first request with a context image misses the encoder cache and leaves
/// its encoder output resident; a second request with the same image hash hits
/// that cached output.
#[test]
fn multimodal_encode_then_cache_hit() {
    use std::sync::atomic::Ordering;
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    let executor = SimExecutor::new(sim);
    let wake = executor.command_waker();
    let sched = Scheduler::new(Box::new(executor), ctrl(), 32).unwrap();
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::with_waker(tx, wake);
    let jh = thread::spawn(move || sched.run(rx));

    let run_img = |rid: u64, handle: &EngineHandle| -> (bool, Vec<String>) {
        let req = generation_request(
            RequestId(rid),
            image_input(vec![1, 2, 3], vec![9], 0xCAFE, 4, 1),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::GenOnly,
            8,
        );
        let mut erx = handle.submit(req).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut images = 0;
        let mut done = false;
        let mut seen = Vec::new();
        while !done && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(EngineCoreOutput::ImageDone { .. }) => {
                    seen.push("image_done".to_string());
                    images += 1;
                }
                Ok(EngineCoreOutput::Finished { reason, .. }) => {
                    seen.push(format!("finished:{reason:?}"));
                    done = true;
                }
                Ok(event) => seen.push(format!("{event:?}")),
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        (done && images > 0, seen)
    };

    let (ok, seen) = run_img(1, &handle);
    assert!(
        ok,
        "context-image request 1 must produce an image, saw {seen:?}"
    );
    assert_eq!(
        stats.encoder.cache_hits.load(Ordering::Relaxed),
        0,
        "first image is a cache miss"
    );
    // Encoder counters are snapshots that can trail the terminal event, so
    // poll them briefly instead of reading once.
    let cache_deadline = Instant::now() + Duration::from_secs(2);
    while stats.encoder.cached.load(Ordering::Relaxed) == 0 && Instant::now() < cache_deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert!(
        stats.encoder.cached.load(Ordering::Relaxed) >= 1,
        "encoder output should be cached"
    );

    let (ok, seen) = run_img(2, &handle);
    assert!(ok, "request 2 must produce an image, saw {seen:?}");
    let hit_deadline = Instant::now() + Duration::from_secs(2);
    while stats.encoder.cache_hits.load(Ordering::Relaxed) == 0 && Instant::now() < hit_deadline {
        thread::sleep(Duration::from_millis(1));
    }
    assert!(
        stats.encoder.cache_hits.load(Ordering::Relaxed) >= 1,
        "repeated image must hit the encoder cache"
    );

    handle.shutdown();
    let _ = jh.join();
}

/// Two requests with the same image hash are both submitted before the
/// scheduler first steps; once both finish, the encoder cache holds exactly
/// one entry for that image.
#[test]
fn concurrent_same_image_misses_converge_on_one_exact_cached_product() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = uniserve_engine::EngineHandle::new(command_tx);

    let mut sim = SimEngine::new();
    sim.set_text_len(6);
    sim.set_queue_depth(2);
    let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    let request = |request_id| {
        generation_request(
            RequestId(request_id),
            image_input(vec![1, 2, 3], vec![9], 0xCAFE, 4, 1),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            12,
        )
    };
    let mut first = handle.submit(request(1)).unwrap();
    let mut second = handle.submit(request(2)).unwrap();
    let mut reasons = [None, None];
    let mut text_tokens = [0_usize, 0_usize];
    let mut seen = [Vec::new(), Vec::new()];
    let deadline = Instant::now() + Duration::from_secs(5);

    while Instant::now() < deadline {
        scheduler.step(&commands);
        for (index, events) in [&mut first, &mut second].into_iter().enumerate() {
            while let Ok(event) = events.try_recv() {
                seen[index].push(format!("{event:?}"));
                match event {
                    EngineCoreOutput::TextToken { .. } => text_tokens[index] += 1,
                    EngineCoreOutput::Finished { reason, .. } => reasons[index] = Some(reason),
                    _ => {}
                }
            }
        }
        if reasons.iter().all(Option::is_some) {
            break;
        }
    }

    assert_eq!(
        reasons,
        [Some(FinishReason::Eos), Some(FinishReason::Eos)],
        "events={seen:?}"
    );
    assert!(text_tokens.iter().all(|count| *count > 0));
    assert_eq!(
        scheduler
            .stats_handle()
            .encoder
            .cached
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn und_only_image_context_encodes_then_produces_text_without_gen_output() {
    use std::sync::atomic::Ordering;

    let mut sim = SimEngine::new();
    sim.set_text_len(20);
    let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let request = generation_request(
        RequestId(81),
        image_input(vec![1, 2], vec![3, 4], 0x81, 4, 17),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        16,
    );
    let mut events = handle.submit(request).expect("submit request");
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut text_tokens = 0;
    let mut images = 0;
    let mut finished = false;
    while !finished && Instant::now() < deadline {
        match events.try_recv() {
            Ok(EngineCoreOutput::TextToken { .. }) => text_tokens += 1,
            Ok(EngineCoreOutput::ImageDone { .. }) => images += 1,
            Ok(EngineCoreOutput::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }

    handle.shutdown();
    let _ = jh.join();
    assert!(finished, "Und-only image-context request did not finish");
    assert!(text_tokens > 0, "Und-only request emitted no visible text");
    assert_eq!(images, 0, "Und-only request opened a Gen branch");
    assert!(
        stats.encoder.cached.load(Ordering::Relaxed) >= 1,
        "input image was not encoded and retained in the encoder cache"
    );
}

/// A request's worker admission carries the number of input images the
/// request holds, which worker image preprocessing reads when it divides a
/// pixel budget among them.
#[test]
fn admission_carries_the_request_input_image_count() {
    let mut sim = SimEngine::new();
    sim.set_text_len(4);
    let mut executor = SimExecutor::new(sim);
    let boundary = executor.observe();
    let sched = Scheduler::new(Box::new(executor), ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (prompt_token_ids, mut inputs) = image_input(vec![1, 2], vec![3, 4], 0x91, 4, 17);
    let second = ImageInput {
        hash: 0x92,
        ..inputs.images[0].clone()
    };
    inputs.images.push(second);
    let request = generation_request(
        RequestId(91),
        (prompt_token_ids, inputs),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        4,
    );
    let _events = handle.submit(request).expect("submit request");

    let deadline = Instant::now() + Duration::from_secs(10);
    let admission = loop {
        let event = boundary
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("the request is admitted");
        if let BatchEvent::Submitted(batch) = event
            && let Some(admission) = batch.commands.iter().find_map(|command| match command {
                uniserve_worker_ipc::BatchCommand::Start { request } => Some(request.clone()),
                _ => None,
            })
        {
            break admission;
        }
    };
    handle.shutdown();
    let _ = jh.join();

    assert_eq!(admission.input_images, 2);
}

/// A Default-constraint request triggered by its second greedy token (1008)
/// realizes all four budgeted images with three denoising steps each, resumes
/// text, and ends at its 200-token text bound. Its text and image sequence is
/// identical at queue depths 1 and 2.
#[test]
fn gen_branch_round_trip_preserves_publication_and_step_invariants() {
    let mut signatures = Vec::new();
    for queue_depth in [1, 2] {
        let mut sim = SimEngine::new();
        sim.set_text_len(1_000_000);
        sim.set_queue_depth(queue_depth);
        let executor = Box::new(SimExecutor::new(sim));
        let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let jh = thread::spawn(move || sched.run(rx));

        let req = with_trigger(
            generation_request(
                RequestId(1),
                text_input(vec![1, 2, 3]),
                SamplingParams::default(),
                ImageParams {
                    steps: 3,
                    max_images: 4,
                    ..Default::default()
                },
                GenerationConstraint::Default,
                200,
            ),
            ImageTrigger::Token { token_id: 1008 },
        );
        let mut erx = handle.submit(req).unwrap();

        let mut signature: Vec<(char, u32)> = Vec::new();
        let mut image_begins = 0;
        let mut image_steps = 0;
        let mut image_commits = 0;
        let mut finish_reason = None;
        let deadline = Instant::now() + Duration::from_secs(15);
        while finish_reason.is_none() && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(EngineCoreOutput::TextToken { id, .. }) => signature.push(('T', id)),
                Ok(EngineCoreOutput::ImageBegin { .. }) => image_begins += 1,
                Ok(EngineCoreOutput::ImageStep { .. }) => image_steps += 1,
                Ok(EngineCoreOutput::ImageCommit { .. }) => image_commits += 1,
                Ok(EngineCoreOutput::ImageDone { image_id, .. }) => signature.push(('I', image_id)),
                Ok(EngineCoreOutput::Finished { reason, .. }) => finish_reason = Some(reason),
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        handle.shutdown();
        let _ = jh.join();

        assert_eq!(
            finish_reason,
            Some(FinishReason::MaxTokens),
            "generated branch request must finish at its declared text bound"
        );
        assert_eq!(
            signature
                .iter()
                .filter(|(modality, _)| *modality == 'I')
                .count(),
            4,
            "request must realize its declared image cap"
        );
        assert!(
            signature.iter().any(|(modality, _)| *modality == 'T'),
            "request must resume visible Und output"
        );
        assert_eq!(image_begins, 4);
        assert_eq!(image_steps, 12);
        assert_eq!(image_commits, 4);
        signatures.push(signature);
    }

    assert_eq!(
        signatures[0], signatures[1],
        "device continuation must preserve the depth-one direct-trigger oracle"
    );
}

#[test]
fn interleave_c4_generated_images_complete() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = uniserve_engine::EngineHandle::new(command_tx);

    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    sim.set_text_len(1_000_000);
    let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    let mut events = HashMap::new();
    let mut results = HashMap::new();

    // Each trigger is the request's own greedy token at index `trigger_offset`,
    // so the four concurrent requests open their image branches at staggered
    // decode positions.
    for (raw_id, trigger_offset) in [(1_u64, 0_u32), (2, 1), (3, 2), (4, 3)] {
        let request_id = RequestId(raw_id);
        let trigger = 1_000 + raw_id as u32 * 7 + trigger_offset;
        let request = with_trigger(
            generation_request(
                request_id,
                text_input(vec![1, 2, 3]),
                SamplingParams::default(),
                ImageParams {
                    steps: 2,
                    max_images: 1,
                    ..Default::default()
                },
                GenerationConstraint::Default,
                24,
            ),
            ImageTrigger::Token { token_id: trigger },
        );
        events.insert(request_id, handle.submit(request).unwrap());
        results.insert(
            request_id,
            Collected {
                text: 0,
                images: 0,
                finished: false,
                reason: None,
            },
        );
    }

    let deadline = Instant::now() + Duration::from_secs(20);
    while results.values().any(|result| !result.finished) && Instant::now() < deadline {
        scheduler.step(&commands);
        for (id, event_rx) in &mut events {
            while let Ok(event) = event_rx.try_recv() {
                let result = results.get_mut(id).expect("request result exists");
                match event {
                    EngineCoreOutput::TextToken { .. } => result.text += 1,
                    EngineCoreOutput::ImageDone { .. } => result.images += 1,
                    EngineCoreOutput::Finished { reason, .. } if !result.finished => {
                        result.finished = true;
                        result.reason = Some(reason);
                    }
                    _ => {}
                }
            }
        }
        thread::sleep(Duration::from_millis(1));
    }

    assert!(
        results.values().all(|result| result.finished),
        "c4 interleave requests did not all finish"
    );
    for (id, result) in &results {
        assert_eq!(
            result.reason,
            Some(FinishReason::MaxTokens),
            "request {id:?} failed"
        );
        assert_eq!(result.images, 1, "request {id:?} missed its image branch");
    }
}

/// With artifact-sourced feedback and a declared one-token ViT encoder recipe,
/// the generated image is re-ingested before Und text resumes. The strong bias
/// on the trigger opens the branch on the first sample.
#[test]
fn generated_image_reingest_runs_declared_encoder_recipe_before_continuation() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut request = with_trigger(
        generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams {
                logit_bias: vec![(2222, 1000.0)],
                ..Default::default()
            },
            ImageParams {
                steps: 2,
                max_images: 1,
                retain_images: false,
                ..Default::default()
            },
            GenerationConstraint::Default,
            12,
        ),
        ImageTrigger::Token { token_id: 2222 },
    );
    request.image_generation.feedback_source = Some(FeedbackSource::ArtifactProduct);
    request.image_generation.feedback_next_token = FeedbackNextToken::Bos;
    request.image_generation.num_feedback_positions = 1;
    request.image_generation.feedback_encoders = vec![ImageEncoderInput {
        encoder: ImageIngestStep::VitEncode,
        num_kv_tokens: Some(1),
        max_kv_tokens: None,
    }];
    request.image_generation.sample_feedback_continuation = false;

    let mut events = handle.submit(request).unwrap();

    let mut sequence = Vec::new();
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    while !finished && Instant::now() < deadline {
        match events.try_recv() {
            Ok(EngineCoreOutput::TextToken { .. }) => sequence.push('T'),
            Ok(EngineCoreOutput::ImageDone { .. }) => sequence.push('I'),
            Ok(EngineCoreOutput::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(finished, "re-ingest request did not finish: {sequence:?}");
    assert_eq!(sequence.iter().filter(|&&event| event == 'I').count(), 1);
    let image = sequence.iter().position(|event| *event == 'I').unwrap();
    assert!(
        sequence[image + 1..].contains(&'T'),
        "Und continuation must begin only after feedback ingest: {sequence:?}"
    );
}

/// A `max_images` budget permits images but does not schedule them: image starts
/// must come from the model, not a scheduler-forced cadence. The simulator never
/// samples the trigger token 2222 here, so the request emits text only, and that
/// text is identical at queue depths 1 and 2.
#[test]
fn gen_branch_waits_for_model_image_starts() {
    let mut runs = Vec::new();
    for depth in [1, 2] {
        let mut sim = SimEngine::new();
        sim.set_text_len(1_000_000);
        sim.set_queue_depth(depth);
        let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let jh = thread::spawn(move || sched.run(rx));
        let request = with_trigger(
            generation_request(
                RequestId(1),
                text_input(vec![1, 2, 3]),
                SamplingParams::default(),
                ImageParams {
                    steps: 3,
                    max_images: 3,
                    ..Default::default()
                },
                GenerationConstraint::Default,
                40,
            ),
            ImageTrigger::Token { token_id: 2222 },
        );
        let mut events = handle.submit(request).unwrap();
        let mut tokens = Vec::new();
        let mut images = 0;
        let mut finish_reason = None;
        let deadline = Instant::now() + Duration::from_secs(15);
        while finish_reason.is_none() && Instant::now() < deadline {
            match events.try_recv() {
                Ok(EngineCoreOutput::TextToken { id, .. }) => tokens.push(id),
                Ok(EngineCoreOutput::ImageDone { .. }) => images += 1,
                Ok(EngineCoreOutput::Finished { reason, .. }) => finish_reason = Some(reason),
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        handle.shutdown();
        let _ = jh.join();
        assert_eq!(finish_reason, Some(FinishReason::MaxTokens));
        assert_eq!(images, 0, "a false transition predicate emitted an image");
        assert!(!tokens.is_empty(), "request emitted no text");
        runs.push(tokens);
    }
    assert_eq!(
        runs[0], runs[1],
        "false generation predicates changed the depth-one text result"
    );
}

/// A Gen-only request whose model requires text before an image decodes Und
/// tokens internally until it samples the trigger; the internally decoded
/// tokens are never emitted, and the discovered trigger opens exactly one
/// image.
#[test]
fn gen_only_can_discover_its_trigger_with_internal_und_decode() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut request = generation_request(
        RequestId(61),
        text_input(vec![1, 2, 3]),
        SamplingParams {
            logit_bias: vec![(2222, 1000.0)],
            ..SamplingParams::default()
        },
        ImageParams {
            steps: 2,
            max_images: 1,
            ..ImageParams::default()
        },
        GenerationConstraint::GenOnly,
        8,
    );
    request.image_generation.trigger = ImageTrigger::Token { token_id: 2222 };
    request.image_generation.requires_text_for_image = true;

    let mut events = handle.submit(request).expect("submit request");
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut visible_text = 0;
    let mut images = 0;
    let mut finished = false;
    while !finished && Instant::now() < deadline {
        match events.try_recv() {
            Ok(EngineCoreOutput::TextToken { .. }) => visible_text += 1,
            Ok(EngineCoreOutput::ImageDone { .. }) => images += 1,
            Ok(EngineCoreOutput::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(finished, "trigger-discovery request did not finish");
    assert_eq!(visible_text, 0, "Gen-only control tokens became visible");
    assert_eq!(images, 1, "the discovered trigger did not open Gen");
}

#[test]
fn und_only_round_close_trigger_cannot_open_gen() {
    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    let control = ctrl();
    let close_token_ids = control.eos.clone();
    let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), control, 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let request = with_trigger(
        generation_request(
            RequestId(62),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams {
                steps: 2,
                max_images: 1,
                ..ImageParams::default()
            },
            GenerationConstraint::UndOnly,
            16,
        ),
        ImageTrigger::RoundCloseThenSuffix {
            close_token_ids,
            trigger_token_ids: vec![1008],
        },
    );
    let mut events = handle.submit(request).expect("submit request");
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut text = 0;
    let mut images = 0;
    let mut finished = false;
    while !finished && Instant::now() < deadline {
        match events.try_recv() {
            Ok(EngineCoreOutput::TextToken { .. }) => text += 1,
            Ok(EngineCoreOutput::ImageDone { .. }) => images += 1,
            Ok(EngineCoreOutput::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(finished, "Und-only round-close request did not finish");
    assert!(text > 0, "Und-only request emitted no text");
    assert_eq!(images, 0, "Und-only round-close trigger opened Gen");
}

/// With the trigger token strongly biased, model-sampled image starts open
/// branches until the four-image budget is spent, and the first image precedes
/// any text.
#[test]
fn gen_branch_model_image_starts_spend_budget() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(sim));
    let trig = SpecialTokenIds {
        ..SpecialTokenIds::default()
    };
    let sched = Scheduler::new(executor, trig, 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams {
                logit_bias: vec![(2222, 1000.0)],
                ..Default::default()
            },
            ImageParams {
                steps: 3,
                max_images: 4,
                ..Default::default()
            },
            GenerationConstraint::Default,
            64,
        ),
        ImageTrigger::Token { token_id: 2222 },
    );
    let mut erx = handle.submit(req).unwrap();

    let mut seq = Vec::new();
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(EngineCoreOutput::TextToken { .. }) => seq.push('T'),
            Ok(EngineCoreOutput::ImageDone { .. }) => seq.push('I'),
            Ok(EngineCoreOutput::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(finished, "request did not finish (seq={seq:?})");
    let images = seq.iter().filter(|&&c| c == 'I').count();
    assert_eq!(
        images, 4,
        "biased image starts should spend the image budget"
    );
    let first_i = seq.iter().position(|&c| c == 'I').unwrap();
    let second_i = seq
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c == 'I')
        .nth(1)
        .unwrap()
        .0;
    assert_eq!(first_i, 0, "model image-start bias should draw before text");
    assert!(
        seq[first_i + 1..second_i].contains(&'T') || second_i == first_i + 1,
        "expected valid stream between model-triggered images (seq={seq:?})",
    );
}

/// Image-generating requests reserve their worst-case KV at admission. The
/// prompt plus the 32,768-token text budget alone exceeds this pool of 128
/// usable units of one 256-token page each, so the scheduler rejects the
/// request (a `Rejected` event and no `Finished`) instead of admitting it.
#[test]
fn gen_branch_rejects_oversized_worstcase_at_admission() {
    let mut sim = SimEngine::new();
    sim.set_text_len(8);
    sim.set_num_units(129);
    sim.set_page_tokens(256);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams {
            steps: 3,
            max_images: 1,
            retain_images: true,
            ..Default::default()
        },
        GenerationConstraint::Default,
        32_768,
    );
    let mut erx = handle.submit(req).unwrap();

    let mut rejected = false;
    let mut finished = None;
    let deadline = Instant::now() + Duration::from_secs(15);
    while finished.is_none() && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(EngineCoreOutput::Rejected { .. }) => rejected = true,
            Ok(EngineCoreOutput::Finished { reason, .. }) => finished = Some(reason),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(
        rejected,
        "oversized generated-branch request must be rejected"
    );
    assert_eq!(finished, None);
}

/// A Gen-only request finishes at its first generated-image commit instead of
/// spending the remaining image budget on hidden images or the understanding
/// budget on text filler. Internal Und decode discovers the biased trigger, Gen
/// opens, one image commits, and because Gen-only requests do not continue
/// after an image (`GenerationRequest::continues_after_image`), the scheduler
/// finishes the request with `ImageDone`.
#[test]
fn commit_eos_finishes_without_spending_remaining_budget() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut req = with_trigger(
        generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams {
                logit_bias: vec![(2222, 1000.0)],
                ..Default::default()
            },
            ImageParams {
                steps: 3,
                max_images: 3,
                ..Default::default()
            },
            GenerationConstraint::GenOnly,
            40,
        ),
        ImageTrigger::Token { token_id: 2222 },
    );
    req.image_generation.requires_text_for_image = true;

    let mut erx = handle.submit(req).unwrap();

    let mut text = 0usize;
    let mut images = 0usize;
    let mut finished = None;
    let deadline = Instant::now() + Duration::from_secs(15);
    while finished.is_none() && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(EngineCoreOutput::TextToken { .. }) => text += 1,
            Ok(EngineCoreOutput::ImageDone { .. }) => images += 1,
            Ok(EngineCoreOutput::Finished { reason, .. }) => finished = Some(reason),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert_eq!(finished, Some(FinishReason::ImageDone));
    assert_eq!(
        images, 1,
        "the gen commit should finish instead of spending the remaining image budget"
    );
    assert_eq!(
        text, 0,
        "the gen commit should not emit hidden Und text filler"
    );
}

#[test]
fn fcfs_policy_completes_text() {
    let out = batch_requests(
        2,
        SchedulingPolicy::Fcfs,
        &[(GenerationConstraint::UndOnly, 3)],
    );
    assert_eq!(out.len(), 3);
    for c in out.values() {
        assert!(c.finished);
        assert!(c.text > 0);
    }
}

#[test]
fn gen_branch_literal_trigger_starts_images() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000); // never EOS on its own
    sim.set_queue_depth(2);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    // Request 1 emits 1007, 1008, 1009, ... so the suffix [1008, 1009] matches
    // on its third token. Once the generated image is fed back into context,
    // the simulator restarts its sequence from 1007 and the suffix recurs.
    let req = with_trigger(
        generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams {
                steps: 3,
                max_images: 2,
                ..Default::default()
            },
            GenerationConstraint::Default,
            40,
        ),
        ImageTrigger::Suffix {
            token_ids: vec![1008, 1009],
        },
    );
    let mut erx = handle.submit(req).unwrap();

    let mut seq: Vec<char> = Vec::new();
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(EngineCoreOutput::TextToken { .. }) => seq.push('T'),
            Ok(EngineCoreOutput::ImageDone { .. }) => seq.push('I'),
            Ok(EngineCoreOutput::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(finished, "request did not finish (seq={seq:?})");
    let images = seq.iter().filter(|&&c| c == 'I').count();
    assert_eq!(
        images, 2,
        "literal trigger must start both images (seq={seq:?})"
    );
    // The suffix trigger fires mid-round, so at least the three tokens through
    // the suffix (1007, 1008, 1009) stream before the first image.
    let first_i = seq.iter().position(|&c| c == 'I').unwrap();
    assert!(
        seq[..first_i].iter().filter(|&&c| c == 'T').count() >= 3,
        "trigger text must stream before the image (seq={seq:?})"
    );
}

/// The image-likelihood knob: a logit bias on the image-start token steers
/// when generated-branch requests draw. A huge positive bias makes the very first
/// sampled token the image trigger (image before any text); a huge negative
/// bias keeps the pathway shut (the sim never EOSes here, so no image can
/// appear any other way).
#[test]
fn image_start_logit_bias_steers_gen_branch() {
    let run = |bias: f32| -> (usize, usize, bool) {
        let mut sim = SimEngine::new();
        sim.set_text_len(1_000_000); // never EOS on its own
        let executor = Box::new(SimExecutor::new(sim));
        let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let jh = thread::spawn(move || sched.run(rx));

        // 2222 is an image-start token inside the simulator's vocabulary that
        // its greedy sequence does not reach within this request's budget.
        let mut req = with_trigger(
            generation_request(
                RequestId(1),
                text_input(vec![1, 2, 3]),
                SamplingParams {
                    logit_bias: vec![(2222, bias)],
                    ..Default::default()
                },
                ImageParams {
                    steps: 3,
                    max_images: 1,
                    ..Default::default()
                },
                GenerationConstraint::Default,
                12,
            ),
            ImageTrigger::Token { token_id: 2222 },
        );
        req.sampling.seed = None;
        let mut erx = handle.submit(req).unwrap();

        let mut text_before_first_image = 0usize;
        let mut images = 0usize;
        let mut finished = false;
        let deadline = Instant::now() + Duration::from_secs(10);
        while !finished && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(EngineCoreOutput::TextToken { .. }) if images == 0 => {
                    text_before_first_image += 1
                }
                Ok(EngineCoreOutput::ImageDone { .. }) => images += 1,
                Ok(EngineCoreOutput::Finished { .. }) => finished = true,
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        handle.shutdown();
        let _ = jh.join();
        (text_before_first_image, images, finished)
    };

    // Eager: the trigger wins the first step — an image with no leading text.
    let (text_before, images, finished) = run(1000.0);
    assert!(finished);
    assert_eq!(images, 1, "positive bias must produce the image");
    assert_eq!(text_before, 0, "positive bias must draw before any text");

    // Suppressed: the trigger never wins; no image, terminates on max_tokens.
    let (_, images, finished) = run(-1000.0);
    assert!(finished);
    assert_eq!(images, 0, "negative bias must suppress the image pathway");
}

/// Native assistant prefixes can end at the image boundary. The generated branch must
/// honor that prefilled control token immediately after prefill instead of
/// waiting for the model to sample another image-start token.
#[test]
fn gen_branch_prefilled_image_start_begins_without_text() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            text_input(vec![10, 11, 2222]),
            SamplingParams::default(),
            ImageParams {
                steps: 3,
                max_images: 1,
                ..Default::default()
            },
            GenerationConstraint::Default,
            8,
        ),
        ImageTrigger::Token { token_id: 2222 },
    );
    let mut erx = handle.submit(req).unwrap();

    let mut text_before_first_image = 0usize;
    let mut images = 0usize;
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(EngineCoreOutput::TextToken { .. }) if images == 0 => text_before_first_image += 1,
            Ok(EngineCoreOutput::ImageDone { .. }) => images += 1,
            Ok(EngineCoreOutput::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(finished, "request did not finish");
    assert_eq!(images, 1, "prefilled image-start must produce one image");
    assert_eq!(
        text_before_first_image, 0,
        "prefilled image-start should jump straight to image generation",
    );
}

/// A request with a context image reasons in text and opens exactly one image
/// branch when its round closes on EOS right after the trigger suffix. With
/// `text_len` 2 the simulator's round for request 1 is 1007, 1008, EOS.
#[test]
fn context_image_request_commits_existing_image_context_at_round_close() {
    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    let executor = Box::new(SimExecutor::new(sim));
    let control = ctrl();
    let close_token_ids = control.eos.clone();
    let sched = Scheduler::new(executor, control, 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            image_input(Vec::new(), vec![1, 2, 3], 7, 1, 1),
            SamplingParams::default(),
            ImageParams {
                steps: 2,
                max_images: 1,
                ..Default::default()
            },
            GenerationConstraint::Default,
            40,
        ),
        ImageTrigger::RoundCloseThenSuffix {
            close_token_ids,
            trigger_token_ids: vec![1008],
        },
    );
    let mut erx = handle.submit(req).unwrap();

    let mut seq: Vec<char> = Vec::new();
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(EngineCoreOutput::TextToken { .. }) => seq.push('T'),
            Ok(EngineCoreOutput::ImageDone { .. }) => seq.push('I'),
            Ok(EngineCoreOutput::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(finished, "request did not finish (seq={seq:?})");
    assert!(
        seq.contains(&'T'),
        "expected text reasoning with existing image context (seq={seq:?})"
    );
    assert_eq!(
        seq.iter().filter(|&&c| c == 'I').count(),
        1,
        "IU image close should spend one image budget (seq={seq:?})"
    );
}

/// Image-budget enforcement under a strong image bias: once max_images is
/// spent, the scheduler adds the image-start token to each call's suppressed
/// tokens, so a bias that would otherwise force it forever (an invisible
/// un-actionable token stream) loses to the mask and generation returns to
/// ordinary text.
#[test]
fn image_budget_suppresses_biased_image_start() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000); // never EOS on its own
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams {
                logit_bias: vec![(2222, 1000.0)],
                ..Default::default()
            },
            ImageParams {
                steps: 3,
                max_images: 2,
                ..Default::default()
            },
            GenerationConstraint::Default,
            24,
        ),
        ImageTrigger::Token { token_id: 2222 },
    );
    let mut erx = handle.submit(req).unwrap();

    let mut images = 0usize;
    let mut post_budget_triggers = 0usize;
    let mut post_budget_text = 0usize;
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(EngineCoreOutput::TextToken { id, .. }) => {
                if images >= 2 {
                    if id == 2222 {
                        post_budget_triggers += 1;
                    } else {
                        post_budget_text += 1;
                    }
                }
            }
            Ok(EngineCoreOutput::ImageDone { .. }) => images += 1,
            Ok(EngineCoreOutput::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(finished, "request must terminate");
    assert_eq!(images, 2, "the bias draws the budgeted images");
    assert_eq!(
        post_budget_triggers, 0,
        "suppression must stop the biased trigger once the budget is spent"
    );
    assert!(
        post_budget_text > 0,
        "generation must return to ordinary text"
    );
}

/// Every logical KV reservation returns to the block pool after completion,
/// across text, image-only, and mixed requests, and no request remains running
/// or in flight.
#[test]
fn kv_resources_return_after_completion() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = uniserve_engine::EngineHandle::new(command_tx);

    let executor = Box::new(SimExecutor::new(SimEngine::new()));
    let mut sched = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs).unwrap();

    // Keep receivers alive — a dropped receiver is treated as a cancellation.
    let mut keep_alive = Vec::new();
    let cases = [
        GenerationConstraint::UndOnly,
        GenerationConstraint::GenOnly,
        GenerationConstraint::Default,
        GenerationConstraint::UndOnly,
    ];
    for (i, mode) in cases.iter().enumerate() {
        let req = generation_request(
            RequestId(i as u64 + 1),
            text_input(vec![1, 2, 3, 4, 5]),
            SamplingParams::default(),
            ImageParams {
                steps: 4,
                max_images: 1,
                ..Default::default()
            },
            *mode,
            16,
        );
        keep_alive.push(handle.submit(req).unwrap());
    }

    let mut idle_steps = 0;
    for _ in 0..5000 {
        let progressed = sched.step(&commands);
        // Converged when several consecutive steps make no progress.
        idle_steps = if progressed { 0 } else { idle_steps + 1 };
        if idle_steps >= 3 {
            break;
        }
    }

    assert_eq!(sched.stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(sched.stats.general.in_flight.load(Ordering::Relaxed), 0);
    assert_eq!(
        sched.stats.kv_cache.free_units.load(Ordering::Relaxed),
        sched.stats.kv_cache.num_units.load(Ordering::Relaxed)
    );
}

/// Dropping 128 event receivers at once cancels every request (dropping an
/// `EventRx` sends a cancel command), and all of them retire from the pending,
/// running, and in-flight accounting.
#[test]
fn cancellation_storm_retires_every_request() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = uniserve_engine::EngineHandle::new(command_tx);

    let executor = Box::new(SimExecutor::new(SimEngine::new()));
    let mut scheduler = Scheduler::new(executor, ctrl(), 32).unwrap();
    let receivers = (1..=128)
        .map(|request_id| {
            handle
                .submit(generation_request(
                    RequestId(request_id),
                    text_input(vec![1, 2, 3]),
                    SamplingParams::default(),
                    ImageParams::default(),
                    GenerationConstraint::UndOnly,
                    8,
                ))
                .unwrap()
        })
        .collect::<Vec<_>>();
    drop(receivers);

    for _ in 0..10_000 {
        scheduler.step(&commands);
        if scheduler.stats.general.running.load(Ordering::Relaxed) == 0
            && scheduler.stats.general.pending.load(Ordering::Relaxed) == 0
            && scheduler.stats.general.in_flight.load(Ordering::Relaxed) == 0
        {
            break;
        }
        thread::sleep(Duration::from_micros(50));
    }

    assert_eq!(scheduler.stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.stats.general.pending.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.stats.general.in_flight.load(Ordering::Relaxed), 0);
}

/// A client that never reads its events stalls its own request on output
/// capacity without holding execution: the concurrent request finishes, the
/// stalled request stays running with nothing in flight, and it retires once
/// its receiver is dropped.
#[test]
fn slow_client_releases_execution_slots_before_output_capacity_returns() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = uniserve_engine::EngineHandle::new(command_tx);

    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(sim));
    let mut scheduler =
        Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs).unwrap();
    let slow_events = handle
        .submit(generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            256,
        ))
        .unwrap();
    let mut fast_events = handle
        .submit(generation_request(
            RequestId(2),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            8,
        ))
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut fast_finished = false;
    while Instant::now() < deadline && !fast_finished {
        scheduler.step(&commands);
        while let Ok(event) = fast_events.try_recv() {
            fast_finished |= matches!(event, EngineCoreOutput::Finished { .. });
        }
    }
    assert!(
        fast_finished,
        "the consuming client must complete independently"
    );

    // Step until the loop stops making progress, which leaves the slow request
    // parked on output capacity.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && scheduler.step(&commands) {}
    assert_eq!(
        scheduler.stats.general.in_flight.load(Ordering::Relaxed),
        0,
        "an output-stalled request cannot retain an execution slot"
    );
    assert_eq!(scheduler.stats.general.running.load(Ordering::Relaxed), 1);

    drop(slow_events);
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline
        && (scheduler.stats.general.running.load(Ordering::Relaxed) > 0
            || scheduler.stats.general.in_flight.load(Ordering::Relaxed) > 0)
    {
        scheduler.step(&commands);
    }
    assert_eq!(scheduler.stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.stats.general.in_flight.load(Ordering::Relaxed), 0);
}

/// Executor whose worker process can be lost: once `lost` is set, every poll
/// reports an executor failure, as a worker exit does, and the scheduler
/// latches engine-fatal. Everything else is the simulator.
struct LosableExecutor {
    inner: SimExecutor,
    lost: Arc<AtomicBool>,
}

impl Executor for LosableExecutor {
    fn info(&self) -> &ExecutorInfo {
        self.inner.info()
    }

    fn is_ready(&self, worker: &WorkerId) -> bool {
        self.inner.is_ready(worker)
    }

    fn has_capacity(&self, worker: &WorkerId) -> bool {
        self.inner.has_capacity(worker)
    }

    fn command_has_capacity(&self, command: &uniserve_worker_ipc::BatchCommand) -> bool {
        self.inner.command_has_capacity(command)
    }

    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        self.inner.submit(batch)
    }

    fn poll(&mut self, timeout: Duration) -> Result<Option<BatchResult>, ExecutorError> {
        if self.lost.load(Ordering::SeqCst) {
            return Err(anyhow::anyhow!("worker process exited"));
        }
        self.inner.poll(timeout)
    }

    fn close(&mut self) -> Result<(), ExecutorError> {
        self.inner.close()
    }
}

/// A worker failure that stops the engine still delivers `Finished { Aborted }`
/// to a running request whose event channel is full, after every event it
/// published before the failure.
#[test]
fn engine_death_delivers_the_terminal_behind_a_full_event_channel() {
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(command_tx);

    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let lost = Arc::new(AtomicBool::new(false));
    let executor = Box::new(LosableExecutor {
        inner: SimExecutor::new(sim),
        lost: Arc::clone(&lost),
    });
    let mut scheduler =
        Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs).unwrap();
    let mut events = handle
        .submit(generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            256,
        ))
        .unwrap();

    // The client reads nothing, so the request fills its event channel and
    // parks on output capacity with nothing in flight.
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && scheduler.step(&commands) {}
    assert_eq!(scheduler.stats.general.in_flight.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.stats.general.running.load(Ordering::Relaxed), 1);

    lost.store(true, Ordering::SeqCst);
    assert!(scheduler.run(commands), "a lost worker stops the engine");

    let mut received = Vec::new();
    loop {
        match events.try_recv() {
            Ok(event) => received.push(event),
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => panic!("the stopped engine left the event stream open"),
        }
    }
    assert!(
        received.len() > EVENT_BUFFER_CAPACITY,
        "received {} events, ending with {:?}; the request published more than its channel holds",
        received.len(),
        received.last(),
    );
    let text_tokens = received
        .iter()
        .filter(|event| matches!(event, EngineCoreOutput::TextToken { .. }))
        .count();
    match received.last() {
        Some(EngineCoreOutput::Finished {
            reason,
            completion_tokens,
            ..
        }) => {
            assert_eq!(*reason, FinishReason::Aborted);
            assert_eq!(*completion_tokens, text_tokens);
        }
        other => panic!("the stream did not end with its terminal event: {other:?}"),
    }
}
