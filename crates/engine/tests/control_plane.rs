#![allow(clippy::unwrap_used, clippy::expect_used)]

//! End-to-end engine-loop lifecycle behavior over the GPU-free simulator.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};
use uniserve_worker_ipc::{ForwardMode, MediaCall};

use uniserve_core::{EngineCoreOutput, FinishReason};
use uniserve_core::{
    FeedbackNextToken, FeedbackSource, GenerationConstraint, GenerationLimits, GenerationRequest,
    ImageEncoderInput, ImageGenerationConfig, ImageIngestStep, ImageInput, ImageParams,
    ImageTrigger, MultimodalInputs, RequestId, SamplingParams,
};
use uniserve_engine::{
    EngineHandle, Scheduler, SchedulingPolicy, SimEngine, SimExecutor, SpecialTokenIds,
};

fn ctrl() -> SpecialTokenIds {
    SpecialTokenIds::default()
}

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
        let scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
        assert_eq!(
            scheduler.generation_limits().features,
            GenerationFeatures::UNDERSTANDING | expected,
        );
    }
}

#[test]
fn encoder_products_obey_worker_entry_capacity() {
    let mut sim = SimEngine::new();
    let info = sim.mut_info_for_test();
    info.buffer_pool_bytes = 8 << 30;
    info.encoder_entry_bytes = 128 << 20;
    info.encoder_cache_entries = 64;
    let scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
    let limits = scheduler.generation_limits();
    assert_eq!(limits.max_vision_feature_bytes, 128 << 20);
    assert_eq!(limits.max_latent_feature_bytes, 128 << 20);
    assert_eq!(limits.encoder_cache_entries, 64);
}

fn text_input(token_ids: Vec<u32>) -> (Vec<u32>, MultimodalInputs) {
    (token_ids, MultimodalInputs::default())
}

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
        ..ImageGenerationConfig::default()
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
fn batch_requests(
    depth: u32,
    policy: SchedulingPolicy,
    cases: &[(GenerationConstraint, usize)],
) -> HashMap<RequestId, Collected> {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(depth);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::with_policy(executor, ctrl(), 32, policy);
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
        if id.0 <= 3 {
            assert!(c.text > 0, "text req {id:?} produced no tokens");
        } else {
            assert_eq!(c.images, 1, "image req {id:?} produced {} images", c.images);
        }
    }
}

#[test]
fn cancellation_releases_latent_admission_for_a_waiting_image() {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(4);
    sim.mut_info_for_test().latent_page_units = 64;
    sim.mut_info_for_test().latent_pages = 65;
    sim.mut_info_for_test().buffer_pool_bytes = 16 << 20;
    sim.mut_info_for_test().max_batch_calls = 1024;
    let scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
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

/// Image generation exposes every declared denoise step in order before one committed image and terminal completion.
#[test]
fn image_events_cover_declared_denoise_steps() {
    const STEPS: u16 = 3;
    let scheduler = Scheduler::new(Box::new(SimExecutor::new(SimEngine::new())), ctrl(), 32);
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
    let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);

    assert_eq!(sched.config().max_batch, 3);
}

/// Single-worker correctness must be identical regardless of queue depth:
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

#[test]
fn call_window_metrics_record_the_full_lifecycle() {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    sim.set_text_len(6);
    let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
    let request = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3, 4, 5]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        16,
    );
    let mut events = scheduler.submit_for_test(request);
    let mut finished = false;
    for _ in 0..512 {
        scheduler.step();
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
    for _ in 0..32 {
        scheduler.step();
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
    let snapshot = reporter.snapshot(&scheduler.stats, 16);
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

fn relay_run(
    sampling: SamplingParams,
    stop_token_ids: Vec<u32>,
    max_und_tokens: usize,
    text_len: usize,
    depth: u32,
) -> Vec<u32> {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(depth);
    sim.set_text_len(text_len);
    let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
    let mut request = generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3, 4, 5]),
        sampling,
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        max_und_tokens,
    );
    request.stop_token_ids = stop_token_ids;
    let mut events = scheduler.submit_for_test(request);
    let mut tokens = Vec::new();
    let mut finished = false;
    for _ in 0..512 {
        scheduler.step();
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

#[test]
fn image_context_decode_is_depth_invariant() {
    let mut runs = Vec::new();
    for queue_depth in [1, 2] {
        let mut sim = SimEngine::new();
        sim.set_queue_depth(queue_depth);
        sim.set_text_len(8);
        let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
        let request = generation_request(
            RequestId(1),
            image_input(vec![1, 2], vec![3, 4], 0xD3C0DE, 4, 1),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            16,
        );
        let mut events = scheduler.submit_for_test(request);
        let mut finish_reason = None;
        let mut tokens = Vec::new();
        for _ in 0..512 {
            scheduler.step();
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
    let sched = Scheduler::new(executor, ctrl(), 32);
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

/// client cancel and server abort produce distinct finish reasons.
/// A "forever" sim (huge text_len) keeps the request running so the control
/// command is observed mid-flight.
fn run_until_control(abort: bool) -> FinishReason {
    let mut sim = SimEngine::new();
    sim.set_text_len(1024);
    sim.set_queue_depth(2);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32);
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

    // wait until it's actually generating, then issue the control command.
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

#[test]
fn stop_string_cutoff_is_request_local() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1024);
    sim.set_queue_depth(2);
    let scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
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

#[test]
fn hybrid_groups_handshake_runs() {
    use uniserve_core::{KvCacheGroup, KvGroupKind};
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    // Group 0 covers [0, 2048); group 1 covers [2048, 4096).
    sim.set_groups(vec![
        KvCacheGroup {
            num_blocks: 2048,
            kind: KvGroupKind::Full,
        },
        KvCacheGroup {
            num_blocks: 2048,
            kind: KvGroupKind::SlidingWindow {
                window: 4096,
                sink: 256,
            },
        },
    ]);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut erx = handle
        .submit(generation_request(
            RequestId(1),
            text_input(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            16,
        ))
        .unwrap();

    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(EngineCoreOutput::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();
    assert!(
        finished,
        "request did not complete under hybrid-group handshake"
    );
}

/// two requests sharing a (block-aligned) prompt prefix — the second
/// reuses the first's cached blocks and skips prefill over the shared prefix.
#[test]
fn prefix_cache_reuses_shared_prompt() {
    use std::sync::atomic::Ordering;
    let mut sim = SimEngine::new();
    sim.set_text_len(4);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32); // block_size 256
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    // 600 tokens => 2 full 256-token blocks + a partial; the 2 full blocks are
    // the cacheable shared prefix.
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

    // cold: req1 populates the prefix cache.
    run_one(1, &handle);
    assert!(
        stats.kv_cache.blocks_stored.load(Ordering::Relaxed) >= 2,
        "req1 should cache >= 2 full prompt blocks"
    );
    let hits_before = stats.prefix.hits.load(Ordering::Relaxed);

    // warm: req2 (same prompt) reuses the cached prefix.
    run_one(2, &handle);
    let hits_after = stats.prefix.hits.load(Ordering::Relaxed);
    assert!(
        hits_after - hits_before >= 2,
        "req2 should reuse >= 2 cached prefix blocks (before={hits_before} after={hits_after})"
    );
    assert!(stats.prefix.hit_tokens.load(Ordering::Relaxed) >= 512);

    handle.shutdown();
    let _ = jh.join();
}

#[test]
fn prefix_cache_enforces_read_write_and_isolation_policy() {
    use std::sync::atomic::Ordering;

    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
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
    let cold_hits = stats.prefix.hits.load(Ordering::Relaxed);
    run(2, true, true, 22);
    assert_eq!(
        stats.prefix.hits.load(Ordering::Relaxed),
        cold_hits,
        "a different isolation key reused cached blocks"
    );
    run(3, true, true, 11);
    assert!(
        stats.prefix.hits.load(Ordering::Relaxed) >= cold_hits + 2,
        "the matching isolation key did not reuse its prefix"
    );

    let before_bypass = stats.prefix.hits.load(Ordering::Relaxed);
    run(4, false, true, 33);
    assert_eq!(stats.prefix.hits.load(Ordering::Relaxed), before_bypass);
    run(5, true, true, 33);
    assert!(
        stats.prefix.hits.load(Ordering::Relaxed) >= before_bypass + 2,
        "read bypass prevented a write-enabled request from publishing its prefix"
    );

    let before_no_store = stats.prefix.hits.load(Ordering::Relaxed);
    run(6, true, false, 44);
    run(7, true, true, 44);
    assert_eq!(
        stats.prefix.hits.load(Ordering::Relaxed),
        before_no_store,
        "a write-disabled request published its prefix"
    );

    handle.shutdown();
    let _ = jh.join();
}

/// under Fcfs with a small per-step token budget and a low chunk cap, a
/// long prompt is prefilled in budget-sized chunks while a concurrent request's
/// decodes proceed in the same steps — both complete correctly.
#[test]
fn chunked_prefill_progresses_with_decode() {
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    let executor = Box::new(SimExecutor::new(sim));
    let mut sched = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs);
    sched.set_long_prefill_threshold(64); // cap a prefill chunk at 64 tokens
    sched.set_token_budget(256); // leaves room for other decodes per step
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    // long prompt (≈ 4 full 256-blocks) + a short concurrent request.
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
/// returning (emitted token ids, whether any logprob was populated, finished).
fn run_sampling(
    sampling: SamplingParams,
    text_len: usize,
    max_tokens: usize,
) -> (Vec<u32>, bool, Option<FinishReason>) {
    let mut sim = SimEngine::new();
    sim.set_text_len(text_len);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32);
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
    // strongly bias token 4321; it should dominate every step.
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
    // text_len=1 makes the sim want to stop almost immediately; min_tokens=5
    // forces at least 5 generated tokens before EOS is permitted.
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
    // No params set => greedy argmax of the synthetic distribution = the natural
    // token (1000 + (1*7+n)%5000); first token is 1007.
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

/// An image-in-prompt request encodes the image before prefill; a second request
/// attaches the resident encoder output to its own KV without rerunning the vision tower.
#[test]
fn multimodal_encode_then_cache_hit() {
    use std::sync::atomic::Ordering;
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    let executor = SimExecutor::new(sim);
    let wake = executor.command_waker();
    let sched = Scheduler::new(Box::new(executor), ctrl(), 32);
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

#[test]
fn concurrent_same_image_misses_converge_on_one_exact_cached_product() {
    let mut sim = SimEngine::new();
    sim.set_text_len(6);
    sim.set_queue_depth(2);
    let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
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
    let mut first = scheduler.submit_for_test(request(1));
    let mut second = scheduler.submit_for_test(request(2));
    let mut reasons = [None, None];
    let mut text_tokens = [0_usize, 0_usize];
    let mut seen = [Vec::new(), Vec::new()];
    let deadline = Instant::now() + Duration::from_secs(5);

    while Instant::now() < deadline {
        scheduler.step();
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
    let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
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

#[test]
fn gen_branch_round_trip_preserves_publication_and_step_invariants() {
    let mut signatures = Vec::new();
    for queue_depth in [1, 2] {
        let mut sim = SimEngine::new();
        sim.set_text_len(1_000_000);
        sim.set_queue_depth(queue_depth);
        let executor = Box::new(SimExecutor::new(sim));
        let sched = Scheduler::new(executor, ctrl(), 32);
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
    let mut sim = SimEngine::new();
    sim.set_queue_depth(2);
    sim.set_text_len(1_000_000);
    let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
    let mut events = HashMap::new();
    let mut results = HashMap::new();

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
        events.insert(request_id, scheduler.submit_for_test(request));
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
        scheduler.step();
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

#[test]
fn generated_image_reingest_runs_declared_encoder_recipe_before_continuation() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32);
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

/// Native multi-image passages can be requested intentionally through max_images;
/// image starts must come from the model, not a scheduler-forced cadence.
#[test]
fn gen_branch_waits_for_model_image_starts() {
    let mut runs = Vec::new();
    for depth in [1, 2] {
        let mut sim = SimEngine::new();
        sim.set_text_len(1_000_000);
        sim.set_queue_depth(depth);
        let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
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

#[test]
fn gen_only_can_discover_its_trigger_with_internal_und_decode() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), ctrl(), 32);
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
    let sched = Scheduler::new(Box::new(SimExecutor::new(sim)), control, 32);
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

#[test]
fn gen_branch_model_image_starts_spend_budget() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(sim));
    let trig = SpecialTokenIds {
        ..SpecialTokenIds::default()
    };
    let sched = Scheduler::new(executor, trig, 32);
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

#[test]
fn gen_branch_rejects_oversized_worstcase_at_admission() {
    let mut sim = SimEngine::new();
    sim.set_text_len(8);
    sim.set_num_blocks(128);
    sim.set_block_size(256);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32);
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

/// A Gen-only request whose behavior finishes after its first generated-image
/// commit stops there: the commit terminates the lineage rather than spending
/// the remaining image budget on hidden images or the understanding budget on
/// text filler. Internal Und decode discovers the biased trigger, Gen opens,
/// one image commits, and `finish_after_gen_commit` closes the request.
#[test]
fn commit_eos_finishes_without_spending_remaining_budget() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32);
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
    // Sim emits 1000 + ((id*7 + n) % 5000) for request id=1: 1007, 1008, 1009…
    // After an image commits, the sim resets and the round repeats from 1007.
    let sched = Scheduler::new(executor, ctrl(), 32);
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
    // The trigger fires mid-round: each image is preceded by the trigger text.
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
        // an image-start token inside the sim's vocab
        let sched = Scheduler::new(executor, ctrl(), 32);
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let jh = thread::spawn(move || sched.run(rx));

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

/// Native assistant prefixes can end at the image boundary. Generated branch must
/// honor that prefilled control token immediately after prefill instead of
/// waiting for the model to sample another image-start token.
#[test]
fn gen_branch_prefilled_image_start_begins_without_text() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32);
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

#[test]
fn context_image_request_commits_existing_image_context_at_round_close() {
    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    let executor = Box::new(SimExecutor::new(sim));
    let control = ctrl();
    let close_token_ids = control.eos.clone();
    let sched = Scheduler::new(executor, control, 32);
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
/// spent, the image-start token is suppressed host-side, so a bias that would
/// otherwise force it forever (an invisible un-actionable token stream) loses
/// to the mask and generation returns to ordinary text.
#[test]
fn image_budget_suppresses_biased_image_start() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000); // never EOS on its own
    let executor = Box::new(SimExecutor::new(sim));
    let sched = Scheduler::new(executor, ctrl(), 32);
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

/// Every logical KV reservation returns to the block manager after completion.
#[test]
fn kv_resources_return_after_completion() {
    let executor = Box::new(SimExecutor::new(SimEngine::new()));
    let mut sched = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs);

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
        keep_alive.push(sched.submit_for_test(req));
    }

    let mut idle_steps = 0;
    for _ in 0..5000 {
        let progressed = sched.step();
        // converged when several consecutive steps make no progress.
        idle_steps = if progressed { 0 } else { idle_steps + 1 };
        if idle_steps >= 3 {
            break;
        }
    }

    assert_eq!(sched.stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(sched.stats.general.in_flight.load(Ordering::Relaxed), 0);
    assert_eq!(
        sched.stats.kv_cache.free_blocks.load(Ordering::Relaxed),
        sched.stats.kv_cache.num_blocks.load(Ordering::Relaxed)
    );
}

#[test]
fn cancellation_storm_retires_every_request() {
    let executor = Box::new(SimExecutor::new(SimEngine::new()));
    let mut scheduler = Scheduler::new(executor, ctrl(), 32);
    let receivers = (1..=128)
        .map(|request_id| {
            scheduler.submit_for_test(generation_request(
                RequestId(request_id),
                text_input(vec![1, 2, 3]),
                SamplingParams::default(),
                ImageParams::default(),
                GenerationConstraint::UndOnly,
                8,
            ))
        })
        .collect::<Vec<_>>();
    drop(receivers);

    for _ in 0..10_000 {
        scheduler.step();
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

#[test]
fn slow_client_releases_execution_slots_before_output_capacity_returns() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(sim));
    let mut scheduler = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs);
    let slow_events = scheduler.submit_for_test(generation_request(
        RequestId(1),
        text_input(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        256,
    ));
    let mut fast_events = scheduler.submit_for_test(generation_request(
        RequestId(2),
        text_input(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        8,
    ));

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut fast_finished = false;
    while Instant::now() < deadline && !fast_finished {
        scheduler.step();
        while let Ok(event) = fast_events.try_recv() {
            fast_finished |= matches!(event, EngineCoreOutput::Finished { .. });
        }
    }
    assert!(
        fast_finished,
        "the consuming client must complete independently"
    );

    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && scheduler.step() {}
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
        scheduler.step();
    }
    assert_eq!(scheduler.stats.general.running.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.stats.general.in_flight.load(Ordering::Relaxed), 0);
}
