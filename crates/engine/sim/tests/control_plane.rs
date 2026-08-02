#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Full-stack GPU-free control-plane integration tests: drive the
//! real `Scheduler` over a `LocalExecutor`+`SimEngine` and assert the lifecycle/event
//! contract. This is the regression harness every workstream relies on.

use std::collections::{HashMap, HashSet};
use std::thread;
use std::time::{Duration, Instant};

use uniserve_core::{
    ContextSegment, FeedbackNextToken, FeedbackSource, GeneratedImageFeedbackRecipe,
    GenerationBehaviorDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
    GenerationRequest, GenerationResourceBounds, GenerationRuntimeCapabilities, ImageIngestRecipe,
    ImageKvEffect, ImageParams, ImageSegment, RequestId, SamplingParams, SegmentPlacement,
    TriggerPolicyDescriptor, UndVisibility,
};
use uniserve_engine_api::{EngineHandle, FinishReason, GenEvent, PublicModality};
use uniserve_executor::{ControlAck, ControlOp, Executor};
use uniserve_scheduler::{ControlTokens, Scheduler, SchedulerConfig, SchedulingPolicy};
use uniserve_sim::SimEngine;
use uniserve_sim::SimExecutor;
use uniserve_worker_ipc::MultiprocExecutor;

fn ctrl() -> ControlTokens {
    ControlTokens::default()
}

fn text_context(token_ids: Vec<u32>) -> Vec<ContextSegment> {
    vec![ContextSegment::UndTokens {
        token_ids,
        visibility: UndVisibility::Internal,
    }]
}

fn context_with_image(
    before: Vec<u32>,
    after: Vec<u32>,
    hash: u64,
    logical_positions: u32,
    physical_tokens: u32,
) -> Vec<ContextSegment> {
    let position = before.len() as u32;
    vec![
        ContextSegment::UndTokens {
            token_ids: before,
            visibility: UndVisibility::Internal,
        },
        ContextSegment::Image {
            image: ImageSegment {
                hash,
                b64: "aW1hZ2U=".to_string(),
                placement: SegmentPlacement::AtToken { position },
            },
            ingest: ImageIngestRecipe::vit_only(
                logical_positions,
                ImageKvEffect::Exact {
                    tokens: physical_tokens,
                },
            ),
        },
        ContextSegment::UndTokens {
            token_ids: after,
            visibility: UndVisibility::Internal,
        },
    ]
}

fn generation_request(
    request_id: RequestId,
    context: Vec<ContextSegment>,
    sampling: SamplingParams,
    image: ImageParams,
    constraint: GenerationConstraint,
    max_und_tokens: usize,
) -> GenerationRequest {
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
    let behavior = GenerationBehaviorDescriptor::resolve(constraint, &policy);
    let cache = Default::default();
    let capabilities = GenerationRuntimeCapabilities {
        supports_understanding: true,
        supports_vision_encode: true,
        supports_latent_encode: false,
        supports_image_generation: true,
        max_latent_units: 1_024,
        latent_downsample: 16,
        max_vae_grid_tokens: 1_024,
        max_vit_grid_tokens: 64,
        max_latent_feature_bytes: 1 << 20,
        max_vision_feature_bytes: 1 << 20,
        commit_marker_tokens: 2,
        max_cfg_branches: 3,
        scratch_capacity_tokens: 1 << 20,
        scratch_block_size: 64,
        encoder_cache_entries: 256,
    };
    let resources = GenerationResourceBounds::conservative(
        &context,
        &[],
        &behavior,
        &policy,
        &image,
        max_und_tokens,
        &cache,
        &capabilities,
    )
    .expect("bounded simulation request");
    GenerationRequest {
        request_id,
        context,
        negative_context: Vec::new(),
        constraint,
        behavior,
        sampling,
        image,
        max_und_tokens,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        lora_id: None,
        grammar: None,
        cache,
        policy,
        resources,
    }
}

fn with_trigger(
    mut request: GenerationRequest,
    trigger: TriggerPolicyDescriptor,
) -> GenerationRequest {
    request.policy.trigger = trigger;
    request.behavior = GenerationBehaviorDescriptor::resolve(request.constraint, &request.policy);
    request.resources.generated_feedback_makes_non_replayable =
        request.behavior.generated_image_feedback;
    request
}

struct Collected {
    text: usize,
    images: usize,
    finished: bool,
    reason: Option<FinishReason>,
}

/// Run the given (constraint, count) requests through a fresh scheduler at `depth`,
/// returning per-request collected counts.
fn run_requests(
    depth: u32,
    policy: SchedulingPolicy,
    specs: &[(GenerationConstraint, usize)],
) -> HashMap<RequestId, Collected> {
    let mut sim = SimEngine::new();
    sim.set_pipeline_depth(depth);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::with_policy(executor, ctrl(), 32, policy);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut rxs: HashMap<RequestId, uniserve_engine_api::EventRx> = HashMap::new();
    let mut id = 1u64;
    for (constraint, n) in specs {
        for _ in 0..*n {
            let req = generation_request(
                RequestId(id),
                text_context(vec![1, 2, 3, 4, 5]),
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
                    GenEvent::TextToken { .. } => c.text += 1,
                    GenEvent::ImageDone { .. } => c.images += 1,
                    GenEvent::Finished { reason, .. } if !c.finished => {
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
    let out = run_requests(
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
fn scheduler_submits_mixed_op_kind_batches() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{
        Batch, CompletionReport, EngineCaps, ExecutionCapability, WorkVariant,
    };

    type PartitionLog = Arc<Mutex<Vec<Vec<(ExecutionCapability, u32, Vec<WorkVariant>)>>>>;

    struct Recording {
        inner: SimExecutor,
        batches: PartitionLog,
    }

    impl Executor for Recording {
        fn caps(&self) -> EngineCaps {
            self.inner.caps()
        }
        fn pipeline_depth(&self) -> usize {
            self.inner.pipeline_depth()
        }
        fn in_flight(&self) -> usize {
            self.inner.in_flight()
        }
        fn can_submit(&self) -> bool {
            self.inner.can_submit()
        }
        fn submit(&mut self, b: Batch) -> anyhow::Result<()> {
            self.batches.lock().unwrap().push(
                b.partitions
                    .iter()
                    .map(|partition| {
                        (
                            partition.execution,
                            partition.submission_group,
                            partition
                                .operations
                                .iter()
                                .map(|operation| operation.work.variant())
                                .collect(),
                        )
                    })
                    .collect(),
            );
            self.inner.submit(b)
        }
        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.inner.next_result()
        }
        fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
            self.inner.control(op)
        }
        fn control_wait(
            &mut self,
            op: ControlOp,
            targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            self.inner.control_wait(op, targets)
        }
        fn shutdown(&mut self) {
            self.inner.shutdown();
        }
    }

    let mut sim = SimEngine::new();
    sim.set_text_len(32);
    sim.set_pipeline_depth(2);
    let batches = Arc::new(Mutex::new(Vec::new()));
    let exec = Recording {
        inner: SimExecutor::new(Box::new(sim)),
        batches: batches.clone(),
    };
    let sched = Scheduler::new(Box::new(exec), ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut text_rx = handle
        .submit(generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            64,
        ))
        .unwrap();
    let mut image_rx = handle
        .submit(generation_request(
            RequestId(2),
            text_context(vec![4, 5, 6]),
            SamplingParams::default(),
            ImageParams {
                steps: 1,
                ..Default::default()
            },
            GenerationConstraint::GenOnly,
            0,
        ))
        .unwrap();

    let mut text_done = false;
    let mut image_done = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    while (!text_done || !image_done) && Instant::now() < deadline {
        while let Ok(ev) = text_rx.try_recv() {
            if matches!(ev, GenEvent::Finished { .. }) {
                text_done = true;
            }
        }
        while let Ok(ev) = image_rx.try_recv() {
            if matches!(ev, GenEvent::Finished { .. }) {
                image_done = true;
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(text_done, "text request did not finish");
    assert!(image_done, "image request did not finish");
    let batches = batches.lock().unwrap();
    assert!(
        batches.iter().any(|partitions| {
            let token = partitions
                .iter()
                .find(|(_, _, kinds)| kinds.contains(&WorkVariant::TokenDecode));
            let flow = partitions
                .iter()
                .find(|(_, _, kinds)| kinds.contains(&WorkVariant::GenFlow));
            matches!(
                (token, flow),
                (
                    Some((ExecutionCapability::TensorizedMixed, token_group, _)),
                    Some((ExecutionCapability::TensorizedMixed, flow_group, _)),
                ) if token_group == flow_group
            )
        }),
        "expected one batch mixing decode_und and denoise_gen, got {batches:?}",
    );
    assert!(
        batches.iter().any(|partitions| {
            let token = partitions
                .iter()
                .find(|(_, _, kinds)| kinds.contains(&WorkVariant::TokenDecode));
            let materialize = partitions
                .iter()
                .find(|(_, _, kinds)| kinds.contains(&WorkVariant::Materialize));
            matches!(
                (token, materialize),
                (
                    Some((ExecutionCapability::DomainHomogeneous, token_group, _)),
                    Some((ExecutionCapability::DomainHomogeneous, materialize_group, _)),
                ) if token_group != materialize_group
            )
        }),
        "expected decode_und and commit_gen to retain independent physical submissions, got {batches:?}",
    );
}

#[test]
fn scheduler_services_a_ready_prompt_at_the_next_available_slot() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{Batch, CompletionReport, EngineCaps, WorkVariant};

    type BatchLog = Arc<Mutex<Vec<Vec<(RequestId, WorkVariant)>>>>;

    struct Recording {
        inner: SimExecutor,
        batches: BatchLog,
    }

    impl Executor for Recording {
        fn caps(&self) -> EngineCaps {
            self.inner.caps()
        }
        fn pipeline_depth(&self) -> usize {
            self.inner.pipeline_depth()
        }
        fn in_flight(&self) -> usize {
            self.inner.in_flight()
        }
        fn can_submit(&self) -> bool {
            self.inner.can_submit()
        }
        fn submit(&mut self, b: Batch) -> anyhow::Result<()> {
            self.batches.lock().unwrap().push(
                b.operations()
                    .map(|operation| (operation.request_key.session_id, operation.work.variant()))
                    .collect(),
            );
            self.inner.submit(b)
        }
        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.inner.next_result()
        }
        fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
            self.inner.control(op)
        }
        fn control_wait(
            &mut self,
            op: ControlOp,
            targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            self.inner.control_wait(op, targets)
        }
        fn shutdown(&mut self) {
            self.inner.shutdown();
        }
    }

    fn run() {
        let mut sim = SimEngine::new();
        sim.set_pipeline_depth(2);
        sim.set_text_len(64);
        sim.mut_caps_for_test()
            .execution_constraints
            .max_batch_operations = 1024;
        let batches = Arc::new(Mutex::new(Vec::new()));
        let exec = Recording {
            inner: SimExecutor::new(Box::new(sim)),
            batches: batches.clone(),
        };
        let mut sched = Scheduler::with_config(
            Box::new(exec),
            ctrl(),
            SchedulerConfig {
                max_batch: 2,
                max_num_batched_tokens: 8192,
                max_num_seqs: 16,
                ..Default::default()
            },
        );

        let greedy_lookahead = SamplingParams {
            temperature: 0.0,
            ignore_eos: true,
            ..SamplingParams::default()
        };
        let _rx1 = sched.submit_for_test(generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3]),
            greedy_lookahead.clone(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            64,
        ));
        let _rx2 = sched.submit_for_test(generation_request(
            RequestId(2),
            text_context(vec![4, 5, 6]),
            greedy_lookahead.clone(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            64,
        ));

        let mut saw_decode_pressure = false;
        for _ in 0..40 {
            sched.step();
            let log = batches.lock().unwrap();
            saw_decode_pressure = log.iter().any(|batch| {
                batch.len() == 2
                    && batch
                        .iter()
                        .all(|(_, kind)| *kind == WorkVariant::TokenDecode)
            });
            if saw_decode_pressure {
                break;
            }
        }
        assert!(
            saw_decode_pressure,
            "setup must reach a full decode batch before testing admission"
        );

        let before = batches.lock().unwrap().len();
        let _rx3 = sched.submit_for_test(generation_request(
            RequestId(3),
            text_context(vec![7, 8, 9]),
            greedy_lookahead.clone(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            64,
        ));

        for _ in 0..40 {
            sched.step();
            if batches.lock().unwrap().len() > before {
                break;
            }
        }
        let log = batches.lock().unwrap();
        let prefill_batch = log.get(before).unwrap_or_else(|| {
            panic!("expected the ready prompt at the next available slot: {log:?}")
        });
        assert!(
            prefill_batch
                .iter()
                .any(|(id, kind)| *id == RequestId(3) && *kind == WorkVariant::TokenExtend),
            "the ready prompt should receive prefill service, got {prefill_batch:?}"
        );
    }

    run();
}

#[test]
fn scheduler_respects_worker_image_latent_capacity_for_denoise_batches() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{Batch, CompletionReport, EngineCaps, ResourceClass, WorkVariant};

    struct Recording {
        inner: SimExecutor,
        batches: Arc<Mutex<Vec<Vec<WorkVariant>>>>,
    }

    impl Executor for Recording {
        fn caps(&self) -> EngineCaps {
            self.inner.caps()
        }
        fn pipeline_depth(&self) -> usize {
            self.inner.pipeline_depth()
        }
        fn in_flight(&self) -> usize {
            self.inner.in_flight()
        }
        fn can_submit(&self) -> bool {
            self.inner.can_submit()
        }
        fn submit(&mut self, b: Batch) -> anyhow::Result<()> {
            self.batches.lock().unwrap().push(
                b.operations()
                    .map(|operation| operation.work.variant())
                    .collect(),
            );
            self.inner.submit(b)
        }
        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.inner.next_result()
        }
        fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
            self.inner.control(op)
        }
        fn control_wait(
            &mut self,
            op: ControlOp,
            targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            self.inner.control_wait(op, targets)
        }
        fn shutdown(&mut self) {
            self.inner.shutdown();
        }
    }

    let mut sim = SimEngine::new();
    sim.mut_caps_for_test().resource_classes = vec![ResourceClass::ImageLatent];
    sim.mut_caps_for_test().max_latent_size = 1024;
    sim.mut_caps_for_test().latent_downsample = 16;
    sim.mut_caps_for_test()
        .execution_constraints
        .max_batch_operations = 1024;
    let batches = Arc::new(Mutex::new(Vec::new()));
    let exec = Recording {
        inner: SimExecutor::new(Box::new(sim)),
        batches: batches.clone(),
    };
    let sched = Scheduler::new(Box::new(exec), ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut receivers = Vec::new();
    for id in 1..=3 {
        let image_rx = handle
            .submit(generation_request(
                RequestId(id),
                text_context(vec![4, 5, 6]),
                SamplingParams::default(),
                ImageParams {
                    steps: 1,
                    ..Default::default()
                },
                GenerationConstraint::GenOnly,
                0,
            ))
            .unwrap();
        receivers.push(image_rx);
    }

    let mut done = vec![false; receivers.len()];
    let deadline = Instant::now() + Duration::from_secs(15);
    while done.iter().any(|finished| !*finished) && Instant::now() < deadline {
        for (idx, rx) in receivers.iter_mut().enumerate() {
            while let Ok(ev) = rx.try_recv() {
                if matches!(ev, GenEvent::Finished { .. }) {
                    done[idx] = true;
                }
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(
        done.iter().all(|finished| *finished),
        "image requests did not finish"
    );
    let batches = batches.lock().unwrap();
    assert!(
        batches.iter().all(|kinds| {
            [WorkVariant::GenTransition, WorkVariant::GenFlow]
                .into_iter()
                .all(|target| kinds.iter().filter(|kind| **kind == target).count() <= 1)
        }),
        "expected every latent-producing batch to respect one-image capacity, got {batches:?}",
    );
}

/// The flow phase runs exactly `image.steps` denoise quanta and then commits —
/// never `image.steps + 1`. The worker signals no flow-completion, so
/// termination is host-driven off the committed step count; this guards the
/// off-by-one that planned a spurious extra denoise operation.
#[test]
fn flow_phase_plans_exactly_image_steps_then_commits() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{Batch, CompletionReport, Control, EngineCaps, WorkVariant};

    struct Recording {
        inner: SimExecutor,
        ops: Arc<Mutex<Vec<WorkVariant>>>,
        public_event_limits: Arc<Mutex<Vec<u64>>>,
    }
    impl Executor for Recording {
        fn caps(&self) -> EngineCaps {
            self.inner.caps()
        }
        fn pipeline_depth(&self) -> usize {
            self.inner.pipeline_depth()
        }
        fn in_flight(&self) -> usize {
            self.inner.in_flight()
        }
        fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
            self.ops
                .lock()
                .unwrap()
                .extend(batch.operations().map(|op| op.work.variant()));
            self.public_event_limits
                .lock()
                .unwrap()
                .extend(batch.controls.iter().filter_map(|control| match control {
                    Control::Commit {
                        public_event_limit, ..
                    } => Some(*public_event_limit),
                    _ => None,
                }));
            self.inner.submit(batch)
        }
        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.inner.next_result()
        }
        fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
            self.inner.control(op)
        }
        fn control_wait(
            &mut self,
            op: ControlOp,
            targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            self.inner.control_wait(op, targets)
        }
        fn shutdown(&mut self) {
            self.inner.shutdown();
        }
    }

    const STEPS: u16 = 3;
    let ops = Arc::new(Mutex::new(Vec::new()));
    let public_event_limits = Arc::new(Mutex::new(Vec::new()));
    let exec = Recording {
        inner: SimExecutor::new(Box::new(SimEngine::new())),
        ops: ops.clone(),
        public_event_limits: public_event_limits.clone(),
    };
    let sched = Scheduler::new(Box::new(exec), ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut erx = handle
        .submit(generation_request(
            RequestId(1),
            text_context(vec![4, 5, 6]),
            SamplingParams::default(),
            ImageParams {
                steps: STEPS,
                ..Default::default()
            },
            GenerationConstraint::GenOnly,
            0,
        ))
        .unwrap();

    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    while !finished && Instant::now() < deadline {
        while let Ok(event) = erx.try_recv() {
            if matches!(event, GenEvent::Finished { .. }) {
                finished = true;
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(
        finished,
        "gen-only image request never finished — the flow phase did not terminate"
    );
    let ops = ops.lock().unwrap();
    let flow_count = ops
        .iter()
        .filter(|kind| **kind == WorkVariant::GenFlow)
        .count();
    assert_eq!(
        flow_count, STEPS as usize,
        "flow must plan exactly image.steps denoise quanta, got {ops:?}"
    );
    let last_flow = ops.iter().rposition(|kind| *kind == WorkVariant::GenFlow);
    let commit = ops
        .iter()
        .position(|kind| *kind == WorkVariant::Materialize);
    assert!(
        commit.is_some(),
        "the flow must be followed by a commit (materialize), got {ops:?}"
    );
    assert!(
        last_flow < commit,
        "the commit must follow the final flow quantum, got {ops:?}"
    );
    let public_event_limits = public_event_limits.lock().unwrap();
    assert!(public_event_limits.len() > usize::from(STEPS));
    assert!(
        public_event_limits
            .windows(2)
            .all(|pair| pair[0] <= pair[1]),
        "semantic commits must carry a monotonic public-event bound: {public_event_limits:?}"
    );
}

#[test]
fn scheduler_clamps_max_batch_to_worker_caps() {
    let mut sim = SimEngine::new();
    sim.mut_caps_for_test()
        .execution_constraints
        .max_batch_operations = 3;
    let sched = Scheduler::new(Box::new(SimExecutor::new(Box::new(sim))), ctrl(), 32);

    assert_eq!(sched.config().max_batch, 3);
}

/// Single-worker correctness must be identical regardless of pipeline depth:
/// the same prompts produce the same per-request token counts at depth 1 and 2.
#[test]
fn pipeline_depth_is_token_identical() {
    let d1 = run_requests(
        1,
        SchedulingPolicy::Fcfs,
        &[(GenerationConstraint::UndOnly, 4)],
    );
    let d2 = run_requests(
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
fn image_context_decode_pipeline_preserves_natural_eos_with_ordered_successors() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{
        Batch, CompletionReport, Control, EngineCaps, OpId, Point, WorkVariant,
    };

    type OperationLog = Arc<Mutex<Vec<(OpId, WorkVariant, Option<(OpId, u64)>, usize, bool)>>>;

    struct Recording {
        inner: SimExecutor,
        operations: OperationLog,
    }

    impl Executor for Recording {
        fn caps(&self) -> EngineCaps {
            self.inner.caps()
        }

        fn pipeline_depth(&self) -> usize {
            self.inner.pipeline_depth()
        }

        fn in_flight(&self) -> usize {
            self.inner.in_flight()
        }

        fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
            let in_flight = self.inner.in_flight();
            for envelope in batch.operations() {
                let parent = match &envelope.parent.point {
                    Point::Fixed { point_index, .. } => {
                        Some((envelope.parent.producer_op_id, u64::from(*point_index)))
                    }
                    Point::Device { .. } => None,
                };
                let releases_parent = envelope.parent.producer_op_id.0 == 0
                    || batch.controls.iter().any(|control| {
                        matches!(
                            control,
                            Control::Release {
                                request_key,
                                op_id,
                            } if *request_key == envelope.request_key
                                && *op_id == envelope.parent.producer_op_id
                        )
                    });
                self.operations.lock().unwrap().push((
                    envelope.op_id,
                    envelope.work.variant(),
                    parent,
                    in_flight,
                    releases_parent,
                ));
            }
            self.inner.submit(batch)
        }

        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            self.inner.poll()
        }

        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.inner.next_result()
        }

        fn control(&mut self, operation: ControlOp) -> anyhow::Result<u64> {
            self.inner.control(operation)
        }

        fn control_wait(
            &mut self,
            operation: ControlOp,
            targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            self.inner.control_wait(operation, targets)
        }

        fn shutdown(&mut self) {
            self.inner.shutdown();
        }
    }

    let mut runs = Vec::new();
    for pipeline_depth in [1, 2] {
        let mut sim = SimEngine::new();
        sim.set_pipeline_depth(pipeline_depth);
        sim.set_text_len(8);
        let operations = Arc::new(Mutex::new(Vec::new()));
        let executor = Recording {
            inner: SimExecutor::new(Box::new(sim)),
            operations: operations.clone(),
        };
        let mut scheduler = Scheduler::new(Box::new(executor), ctrl(), 32);
        let mut request = generation_request(
            RequestId(1),
            context_with_image(vec![1, 2], vec![3, 4], 0xD3C0DE, 4, 1),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            16,
        );
        if pipeline_depth == 1 {
            request.stop_token_ids = vec![u32::MAX];
        }
        let mut events = scheduler.submit_for_test(request);

        let mut finish_reason = None;
        let mut text_tokens = 0;
        for _ in 0..256 {
            scheduler.step();
            while let Ok(event) = events.try_recv() {
                match event {
                    GenEvent::TextToken { .. } => text_tokens += 1,
                    GenEvent::Finished { reason, .. } => finish_reason = Some(reason),
                    _ => {}
                }
            }
            if finish_reason.is_some() {
                break;
            }
        }
        drop(scheduler);
        runs.push((
            pipeline_depth,
            finish_reason,
            text_tokens,
            Arc::try_unwrap(operations).unwrap().into_inner().unwrap(),
        ));
    }

    for (_, finish_reason, text_tokens, operations) in &runs {
        assert_eq!(*finish_reason, Some(FinishReason::Eos));
        assert_eq!(*text_tokens, 8);
        assert!(operations.len() >= 2);
        assert!(
            operations
                .iter()
                .all(|(_, _, _, _, releases_parent)| *releases_parent),
            "every successor retires its parent after submission"
        );
        // A fixed producer exports operation-local point one when it advances
        // state and point zero when it preserves state.
        for (_, _, parent, _, _) in operations {
            if let Some((producer_op_id, point)) = parent {
                let expected = if producer_op_id.0 == 0 {
                    0
                } else {
                    u64::from(
                        operations
                            .iter()
                            .find(|(op_id, _, _, _, _)| op_id == producer_op_id)
                            .expect("fixed parent producer must precede its consumer")
                            .1
                            .advances_state(),
                    )
                };
                assert_eq!(*point, expected);
            }
        }
    }

    let serial = &runs[0].3;
    assert!(
        serial.iter().any(|(_, kind, parent, in_flight, _)| {
            *kind == WorkVariant::TokenDecode && parent.is_none() && *in_flight == 0
        }),
        "a resolved selected-point product remains the next decode parent"
    );
    let pipelined = &runs[1].3;
    assert!(
        pipelined.iter().any(|(_, kind, parent, in_flight, _)| {
            *kind == WorkVariant::TokenDecode && parent.is_none() && *in_flight > 0
        }),
        "an in-flight selected-point product feeds the next decode"
    );
}

#[test]
fn staged_product_reachability_uses_a_fixed_handoff_then_same_pool_device_lineage() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{Batch, CompletionReport, EngineCaps, Point, WorkVariant};

    struct StagedReachability {
        inner: SimExecutor,
        operations: Arc<Mutex<Vec<(WorkVariant, Point)>>>,
    }

    impl Executor for StagedReachability {
        fn caps(&self) -> EngineCaps {
            self.inner.caps()
        }

        fn pipeline_depth(&self) -> usize {
            self.inner.pipeline_depth()
        }

        fn in_flight(&self) -> usize {
            self.inner.in_flight()
        }

        fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
            self.operations.lock().unwrap().extend(
                batch
                    .operations()
                    .map(|operation| (operation.work.variant(), operation.parent.point.clone())),
            );
            self.inner.submit(batch)
        }

        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            self.inner.poll()
        }

        fn device_products_reachable(&self, producer: WorkVariant, consumer: WorkVariant) -> bool {
            producer == WorkVariant::TokenDecode && consumer == WorkVariant::TokenDecode
        }

        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.inner.next_result()
        }

        fn control(&mut self, operation: ControlOp) -> anyhow::Result<u64> {
            self.inner.control(operation)
        }

        fn control_wait(
            &mut self,
            operation: ControlOp,
            targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            self.inner.control_wait(operation, targets)
        }

        fn shutdown(&mut self) {
            self.inner.shutdown();
        }
    }

    let mut sim = SimEngine::new();
    sim.set_pipeline_depth(2);
    sim.set_text_len(6);
    let operations = Arc::new(Mutex::new(Vec::new()));
    let executor = StagedReachability {
        inner: SimExecutor::new(Box::new(sim)),
        operations: operations.clone(),
    };
    let mut scheduler = Scheduler::new(Box::new(executor), ctrl(), 32);
    let mut events = scheduler.submit_for_test(generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        12,
    ));

    let mut finished = false;
    for _ in 0..256 {
        scheduler.step();
        while let Ok(event) = events.try_recv() {
            if matches!(event, GenEvent::Finished { .. }) {
                finished = true;
            }
        }
        if finished {
            break;
        }
    }
    assert!(finished);
    let operations = operations.lock().unwrap();
    let decode_parents = operations
        .iter()
        .filter_map(|(variant, point)| (*variant == WorkVariant::TokenDecode).then_some(point))
        .collect::<Vec<_>>();
    assert!(matches!(decode_parents.first(), Some(Point::Fixed { .. })));
    assert!(
        decode_parents
            .iter()
            .skip(1)
            .any(|point| matches!(point, Point::Device { .. }))
    );
}

/// an explicit stop token id terminates the request with FinishReason::Stop.
/// The SimEngine's first sampled token for request id=1 is deterministic:
/// `1000 + ((1*7 + 0) % 5000) = 1007`.
#[test]
fn stop_token_terminates_with_stop() {
    let mut sim = SimEngine::new();
    sim.set_pipeline_depth(2);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut req = generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
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
                GenEvent::TextToken { .. } => text += 1,
                GenEvent::Finished { reason: r, .. } => reason = Some(r),
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
    sim.set_pipeline_depth(2);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
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
        if let Ok(GenEvent::TextToken { .. }) = erx.try_recv() {
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
            Ok(GenEvent::Finished { reason: r, .. }) => reason = Some(r),
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
fn exact_prefix_controls_close_the_selected_semantic_versions() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{Batch, CompletionReport, Control, EngineCaps, Point, VersionRef};

    struct Recording {
        inner: SimExecutor,
        controls: Arc<Mutex<Vec<Control>>>,
    }

    impl Executor for Recording {
        fn caps(&self) -> EngineCaps {
            self.inner.caps()
        }
        fn pipeline_depth(&self) -> usize {
            self.inner.pipeline_depth()
        }
        fn in_flight(&self) -> usize {
            self.inner.in_flight()
        }
        fn can_submit(&self) -> bool {
            self.inner.can_submit()
        }
        fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
            self.controls
                .lock()
                .unwrap()
                .extend(batch.controls.iter().cloned());
            self.inner.submit(batch)
        }
        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.inner.next_result()
        }
        fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
            self.inner.control(op)
        }
        fn control_wait(
            &mut self,
            op: ControlOp,
            targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            self.inner.control_wait(op, targets)
        }
        fn shutdown(&mut self) {
            self.inner.shutdown();
        }
    }

    let mut sim = SimEngine::new();
    sim.set_text_len(1024);
    sim.set_pipeline_depth(4);
    let controls = Arc::new(Mutex::new(Vec::new()));
    let executor = Recording {
        inner: SimExecutor::new(Box::new(sim)),
        controls: Arc::clone(&controls),
    };
    let scheduler = Scheduler::new(Box::new(executor), ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));

    let request = generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        1024,
    );
    let mut events = handle.submit(request).unwrap();
    let mut consumed_tokens = 0;
    let deadline = Instant::now() + Duration::from_secs(10);
    while consumed_tokens < 2 && Instant::now() < deadline {
        match events.try_recv() {
            Ok(GenEvent::TextToken { .. }) => {
                consumed_tokens += 1;
            }
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert_eq!(consumed_tokens, 2);

    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let committed = controls
            .lock()
            .unwrap()
            .iter()
            .filter(|control| matches!(control, Control::Commit { .. }))
            .count();
        if committed > consumed_tokens {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.cancel_at(RequestId(1), consumed_tokens);

    let deadline = Instant::now() + Duration::from_secs(10);
    let close_cutoff = loop {
        let cutoff = controls.lock().unwrap().iter().find_map(|control| {
            if let Control::Close { cutoff, .. } = control {
                Some(cutoff.clone())
            } else {
                None
            }
        });
        if cutoff.is_some() || Instant::now() >= deadline {
            break cutoff;
        }
        thread::sleep(Duration::from_millis(1));
    };
    handle.shutdown();
    let _ = scheduler_thread.join();

    let committed: Vec<(u64, VersionRef)> = controls
        .lock()
        .unwrap()
        .iter()
        .filter_map(|control| {
            if let Control::Commit {
                control_seq,
                selected,
                ..
            } = control
            {
                Some((*control_seq, selected.clone()))
            } else {
                None
            }
        })
        .collect();
    assert!(committed.len() > consumed_tokens);
    let close_cutoff = close_cutoff.expect("cancelled lineage must close");
    assert_eq!(close_cutoff, committed[consumed_tokens - 1].1);
    let close_seq = controls
        .lock()
        .unwrap()
        .iter()
        .find_map(|control| {
            if let Control::Close { control_seq, .. } = control {
                Some(*control_seq)
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(close_seq, committed.last().unwrap().0 + 1);

    let mut sim = SimEngine::new();
    sim.set_text_len(1024);
    sim.set_pipeline_depth(4);
    let stop_controls = Arc::new(Mutex::new(Vec::new()));
    let executor = Recording {
        inner: SimExecutor::new(Box::new(sim)),
        controls: Arc::clone(&stop_controls),
    };
    let scheduler = Scheduler::new(Box::new(executor), ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));
    let mut request = generation_request(
        RequestId(2),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        1024,
    );
    request.stop_strings = vec!["boundary".to_string()];
    let mut events = handle.submit(request).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut consumed_tokens = 0;
    while consumed_tokens < 2 && Instant::now() < deadline {
        match events.try_recv() {
            Ok(GenEvent::TextToken { .. }) => {
                consumed_tokens += 1;
                if consumed_tokens == 1 {
                    handle.acknowledge_at(RequestId(2), consumed_tokens);
                    handle.acknowledge_at(RequestId(2), consumed_tokens);
                }
            }
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert_eq!(consumed_tokens, 2);
    handle.stop_at(RequestId(2), consumed_tokens);

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut finish_reason = None;
    while Instant::now() < deadline {
        while let Ok(event) = events.try_recv() {
            if let GenEvent::Finished { reason, .. } = event {
                finish_reason = Some(reason);
            }
        }
        let closed = stop_controls
            .lock()
            .unwrap()
            .iter()
            .any(|control| matches!(control, Control::Close { .. }));
        if closed && finish_reason.is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = scheduler_thread.join();
    assert_eq!(finish_reason, Some(FinishReason::Stop));

    let stop_controls = stop_controls.lock().unwrap();
    let committed: Vec<(u64, VersionRef)> = stop_controls
        .iter()
        .filter_map(|control| {
            if let Control::Commit {
                control_seq,
                selected,
                ..
            } = control
            {
                Some((*control_seq, selected.clone()))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(committed.len(), consumed_tokens - 1);
    let (close_seq, close_cutoff, close_reason) = stop_controls
        .iter()
        .find_map(|control| {
            if let Control::Close {
                control_seq,
                cutoff,
                reason,
                ..
            } = control
            {
                Some((*control_seq, cutoff, *reason))
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(close_reason, uniserve_worker_wire::CloseReason::Completed);
    assert_eq!(close_seq, committed[0].0 + 1);
    assert_eq!(
        close_cutoff.producer_op_id.0,
        committed[0].1.producer_op_id.0 + 1
    );
    let (
        Point::Fixed {
            point_index: committed_point,
            ..
        },
        Point::Fixed {
            point_index: close_point,
            ..
        },
    ) = (&committed[0].1.point, &close_cutoff.point)
    else {
        panic!("semantic controls must use fixed versions");
    };
    assert_eq!(*committed_point, 1);
    assert_eq!(*close_point, 1);
}

/// the EngineCaps hybrid-group handshake builds a multi-group block
/// manager and the scheduler still drives requests to completion.
#[test]
fn hybrid_groups_handshake_runs() {
    use uniserve_core::{KvCacheGroupSpec, KvGroupKind};
    let mut sim = SimEngine::new();
    sim.set_pipeline_depth(2);
    // block 0 padding; group 0 full [1,2048), group 1 sliding-window [2048,4096).
    sim.set_groups(vec![
        KvCacheGroupSpec {
            group_id: 0,
            block_offset: 1,
            num_blocks: 2047,
            kind: KvGroupKind::Full,
        },
        KvCacheGroupSpec {
            group_id: 1,
            block_offset: 2048,
            num_blocks: 2048,
            kind: KvGroupKind::SlidingWindow {
                window: 4096,
                sink: 256,
            },
        },
    ]);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut erx = handle
        .submit(generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3]),
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
            Ok(GenEvent::Finished { .. }) => finished = true,
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
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
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
                text_context(prompt.clone()),
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
                Ok(GenEvent::Finished { .. }) => done = true,
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

    // resetting the prefix cache clears it.
    assert!(handle.reset_prefix_cache(false).unwrap());

    handle.shutdown();
    let _ = jh.join();
}

#[test]
fn prefix_cache_enforces_read_write_and_isolation_policy() {
    use std::sync::atomic::Ordering;

    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    let sched = Scheduler::new(Box::new(SimExecutor::new(Box::new(sim))), ctrl(), 32);
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));
    let prompt: Vec<u32> = (0..600_u32).map(|index| (index % 47) + 5).collect();

    let run = |id: u64, read: bool, write: bool, isolation_key: u64| {
        let mut request = generation_request(
            RequestId(id),
            text_context(prompt.clone()),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            8,
        );
        request.cache = uniserve_core::GenerationCachePolicyDescriptor {
            read,
            write,
            isolation_key: Some(isolation_key),
        };
        let mut events = handle.submit(request).expect("submit cache request");
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match events.try_recv() {
                Ok(GenEvent::Finished { .. }) => return,
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
    sim.set_pipeline_depth(2);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
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
            text_context(long_prompt),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            8,
        ))
        .unwrap();
    let mut erx2 = handle
        .submit(generation_request(
            RequestId(2),
            text_context(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            8,
        ))
        .unwrap();

    let collect = |erx: &mut uniserve_engine_api::EventRx| -> (usize, bool) {
        let mut text = 0;
        let mut done = false;
        let deadline = Instant::now() + Duration::from_secs(15);
        while !done && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(GenEvent::TextToken { .. }) => text += 1,
                Ok(GenEvent::Finished { .. }) => done = true,
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

/// Run one text request with the given sampling params and a sim `text_len`,
/// returning (emitted token ids, whether any logprob was populated, finished).
fn run_sampling(
    sampling: SamplingParams,
    text_len: usize,
    max_tokens: usize,
) -> (Vec<u32>, bool, Option<FinishReason>) {
    let mut sim = SimEngine::new();
    sim.set_text_len(text_len);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
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
            Ok(GenEvent::TextToken { id, logprob, .. }) => {
                toks.push(id);
                if logprob.is_some() {
                    any_logprob = true;
                }
            }
            Ok(GenEvent::Finished { reason, .. }) => finished = Some(reason),
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
    sim.set_pipeline_depth(2);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let run_img = |rid: u64, handle: &EngineHandle| -> (bool, Vec<String>) {
        let req = generation_request(
            RequestId(rid),
            context_with_image(vec![1, 2, 3], vec![9], 0xCAFE, 4, 1),
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
                Ok(GenEvent::ImageDone { .. }) => {
                    seen.push("image_done".to_string());
                    images += 1;
                }
                Ok(GenEvent::Finished { reason, .. }) => {
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

    handle.reset_encoder_cache();
    thread::sleep(Duration::from_millis(20));
    handle.shutdown();
    let _ = jh.join();
}

#[test]
fn concurrent_same_image_misses_converge_on_one_exact_cached_product() {
    let mut sim = SimEngine::new();
    sim.set_text_len(6);
    sim.set_pipeline_depth(2);
    let mut scheduler = Scheduler::new(Box::new(SimExecutor::new(Box::new(sim))), ctrl(), 32);
    let request = |request_id| {
        generation_request(
            RequestId(request_id),
            context_with_image(vec![1, 2, 3], vec![9], 0xCAFE, 4, 1),
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
                    GenEvent::TextToken { .. } => text_tokens[index] += 1,
                    GenEvent::Finished { reason, .. } => reasons[index] = Some(reason),
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
        "events={seen:?}, health={:?}",
        scheduler.health_snapshot()
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
    let sched = Scheduler::new(Box::new(SimExecutor::new(Box::new(sim))), ctrl(), 32);
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let request = generation_request(
        RequestId(81),
        context_with_image(vec![1, 2], vec![3, 4], 0x81, 4, 17),
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
            Ok(GenEvent::TextToken { .. }) => text_tokens += 1,
            Ok(GenEvent::ImageDone { .. }) => images += 1,
            Ok(GenEvent::Finished { .. }) => finished = true,
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
    for pipeline_depth in [1, 2] {
        let mut sim = SimEngine::new();
        sim.set_text_len(1_000_000);
        sim.set_pipeline_depth(pipeline_depth);
        let caps = sim.mut_caps_for_test();
        caps.execution_constraints.route_capabilities[0]
            .credits
            .per_request
            .device_products = 6;
        caps.route_capability_digest = caps.compute_route_capability_digest();
        let executor = Box::new(SimExecutor::new(Box::new(sim)));
        let sched = Scheduler::new(executor, ctrl(), 32);
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let jh = thread::spawn(move || sched.run(rx));

        let req = with_trigger(
            generation_request(
                RequestId(1),
                text_context(vec![1, 2, 3]),
                SamplingParams::default(),
                ImageParams {
                    steps: 3,
                    max_images: 4,
                    ..Default::default()
                },
                GenerationConstraint::Default,
                200,
            ),
            TriggerPolicyDescriptor::Token { token_id: 1008 },
        );
        let mut erx = handle.submit(req).unwrap();

        let mut signature: Vec<(char, u32)> = Vec::new();
        let mut publications = Vec::new();
        let mut image_begins = 0;
        let mut image_steps = 0;
        let mut image_commits = 0;
        let mut finish_reason = None;
        let deadline = Instant::now() + Duration::from_secs(15);
        while finish_reason.is_none() && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(GenEvent::TextToken {
                    id,
                    public_commit: Some(commit),
                    ..
                }) => {
                    assert_eq!(commit.modality, PublicModality::Text);
                    signature.push(('T', id));
                    publications.push(commit);
                }
                Ok(GenEvent::ImageBegin { .. }) => image_begins += 1,
                Ok(GenEvent::ImageStep { .. }) => image_steps += 1,
                Ok(GenEvent::ImageCommit { .. }) => image_commits += 1,
                Ok(GenEvent::ImageDone {
                    image_id,
                    public_commit: Some(commit),
                    ..
                }) => {
                    assert_eq!(commit.modality, PublicModality::Image);
                    signature.push(('I', image_id));
                    publications.push(commit);
                }
                Ok(GenEvent::TextToken {
                    public_commit: None,
                    ..
                })
                | Ok(GenEvent::ImageDone {
                    public_commit: None,
                    ..
                }) => panic!("visible event omitted its exact publication identity"),
                Ok(GenEvent::Finished { reason, .. }) => finish_reason = Some(reason),
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
        assert!(publications.windows(2).all(|pair| {
            pair[0].event_seq < pair[1].event_seq && pair[0].committed_at <= pair[1].committed_at
        }));
        assert!(publications.iter().all(|commit| {
            commit.semantic_root.semantic_digest.len() == 64
                && commit
                    .semantic_root
                    .semantic_digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
        }));
        signatures.push(signature);
    }

    assert_eq!(
        signatures[0], signatures[1],
        "device continuation must preserve the depth-one direct-trigger oracle"
    );
}

#[test]
fn generated_image_reingest_runs_declared_encoder_recipe_before_continuation() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut request = with_trigger(
        generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3]),
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
        TriggerPolicyDescriptor::Token { token_id: 2222 },
    );
    request.policy.feedback = Some(GeneratedImageFeedbackRecipe {
        source: FeedbackSource::ArtifactProduct,
        next_und_token: FeedbackNextToken::Bos,
        ingest: ImageIngestRecipe::vit_only(1, ImageKvEffect::Exact { tokens: 1 }),
        sample_continuation: false,
    });
    request.behavior = GenerationBehaviorDescriptor::resolve(request.constraint, &request.policy);
    let mut events = handle.submit(request).unwrap();

    let mut sequence = Vec::new();
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    while !finished && Instant::now() < deadline {
        match events.try_recv() {
            Ok(GenEvent::TextToken { .. }) => sequence.push('T'),
            Ok(GenEvent::ImageDone { .. }) => sequence.push('I'),
            Ok(GenEvent::Finished { .. }) => finished = true,
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
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams {
                steps: 3,
                max_images: 3,
                ..Default::default()
            },
            GenerationConstraint::Default,
            40,
        ),
        TriggerPolicyDescriptor::Token { token_id: 2222 },
    );
    let mut erx = handle.submit(req).unwrap();

    let mut seq = Vec::new();
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(GenEvent::TextToken { .. }) => seq.push('T'),
            Ok(GenEvent::ImageDone { .. }) => seq.push('I'),
            Ok(GenEvent::Finished { .. }) => finished = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert!(finished, "request did not finish (seq={seq:?})");
    let images = seq.iter().filter(|&&c| c == 'I').count();
    assert_eq!(
        images, 0,
        "image starts must be model-triggered (seq={seq:?})"
    );
    assert!(
        seq.iter().all(|&c| c == 'T'),
        "expected text-only stream (seq={seq:?})"
    );
}

#[test]
fn gen_only_can_discover_its_trigger_with_internal_und_decode() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let sched = Scheduler::new(Box::new(SimExecutor::new(Box::new(sim))), ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut request = generation_request(
        RequestId(61),
        text_context(vec![1, 2, 3]),
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
    request.policy.trigger = TriggerPolicyDescriptor::Token { token_id: 2222 };
    request.policy.gen_only_start = uniserve_core::GenOnlyStartPolicyDescriptor::DiscoverTrigger;
    request.behavior = GenerationBehaviorDescriptor::resolve(request.constraint, &request.policy);
    assert!(request.behavior.und_decode);
    assert!(!request.behavior.start_gen_after_context);

    let mut events = handle.submit(request).expect("submit request");
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut visible_text = 0;
    let mut images = 0;
    let mut finished = false;
    while !finished && Instant::now() < deadline {
        match events.try_recv() {
            Ok(GenEvent::TextToken { .. }) => visible_text += 1,
            Ok(GenEvent::ImageDone { .. }) => images += 1,
            Ok(GenEvent::Finished { .. }) => finished = true,
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
    let sched = Scheduler::new(Box::new(SimExecutor::new(Box::new(sim))), control, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let request = with_trigger(
        generation_request(
            RequestId(62),
            text_context(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams {
                steps: 2,
                max_images: 1,
                ..ImageParams::default()
            },
            GenerationConstraint::UndOnly,
            16,
        ),
        TriggerPolicyDescriptor::RoundCloseThenSuffix {
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
            Ok(GenEvent::TextToken { .. }) => text += 1,
            Ok(GenEvent::ImageDone { .. }) => images += 1,
            Ok(GenEvent::Finished { .. }) => finished = true,
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
    let caps = sim.mut_caps_for_test();
    caps.execution_constraints.route_capabilities[0]
        .credits
        .per_request
        .device_products = 10;
    caps.route_capability_digest = caps.compute_route_capability_digest();
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let trig = ControlTokens {
        ..ControlTokens::default()
    };
    let sched = Scheduler::new(executor, trig, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3]),
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
        TriggerPolicyDescriptor::Token { token_id: 2222 },
    );
    let mut erx = handle.submit(req).unwrap();

    let mut seq = Vec::new();
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(GenEvent::TextToken { .. }) => seq.push('T'),
            Ok(GenEvent::ImageDone { .. }) => seq.push('I'),
            Ok(GenEvent::Finished { .. }) => finished = true,
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
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
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
            Ok(GenEvent::Rejected { .. }) => rejected = true,
            Ok(GenEvent::Finished { reason, .. }) => finished = Some(reason),
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
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut req = with_trigger(
        generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3]),
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
        TriggerPolicyDescriptor::Token { token_id: 2222 },
    );
    req.policy.gen_only_start = uniserve_core::GenOnlyStartPolicyDescriptor::DiscoverTrigger;
    req.behavior = GenerationBehaviorDescriptor::resolve(req.constraint, &req.policy);
    assert!(req.behavior.finish_after_gen_commit);
    let mut erx = handle.submit(req).unwrap();

    let mut text = 0usize;
    let mut images = 0usize;
    let mut finished = None;
    let deadline = Instant::now() + Duration::from_secs(15);
    while finished.is_none() && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(GenEvent::TextToken { .. }) => text += 1,
            Ok(GenEvent::ImageDone { .. }) => images += 1,
            Ok(GenEvent::Finished { reason, .. }) => finished = Some(reason),
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

/// the async Executor seam fronts a MultiWorkerExecutor (2 ranks) with no
/// scheduler change — a batch fans out to both ranks, results join, and requests
/// complete identically to the single-worker path.
#[test]
fn multiworker_executor_drives_scheduler_unchanged() {
    let mk = |rank| {
        let mut sim = SimEngine::new();
        sim.set_pipeline_depth(2);
        sim.mut_caps_for_test().rank.tp_rank = rank;
        sim.mut_caps_for_test().rank.tp_size = 2;
        Box::new(SimExecutor::new(Box::new(sim))) as Box<dyn Executor>
    };
    let executor = Box::new(MultiprocExecutor::new(vec![mk(0), mk(1)]).unwrap());
    // rank-aware caps reflect the topology at the handshake.
    assert_eq!(executor.caps().rank.tp_size, 2);
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut rxs = std::collections::HashMap::new();
    for id in 1..=3u64 {
        let erx = handle
            .submit(generation_request(
                RequestId(id),
                text_context(vec![1, 2, 3]),
                SamplingParams::default(),
                ImageParams::default(),
                GenerationConstraint::UndOnly,
                16,
            ))
            .unwrap();
        rxs.insert(RequestId(id), erx);
    }
    let mut done = 0;
    let deadline = Instant::now() + Duration::from_secs(15);
    let total = rxs.len();
    while done < total && Instant::now() < deadline {
        for erx in rxs.values_mut() {
            while let Ok(ev) = erx.try_recv() {
                if matches!(ev, GenEvent::Finished { .. }) {
                    done += 1;
                }
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    handle.shutdown();
    let _ = jh.join();
    assert_eq!(
        done, total,
        "all requests must finish under the multi-worker executor"
    );
}

#[test]
fn fcfs_policy_completes_text() {
    let out = run_requests(
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

/// Sequence admission carries the prefix-cache reuse boundary: zero for a cold
/// session and cached blocks times block size for a reused prompt prefix.
#[test]
fn sequence_admission_carries_the_prefix_reuse_boundary() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{Batch, CompletionReport, EngineCaps};

    #[derive(Default)]
    struct Log {
        registrations: Vec<(RequestId, u32)>,
    }

    struct Recording {
        inner: SimExecutor,
        log: Arc<Mutex<Log>>,
    }
    impl Executor for Recording {
        fn caps(&self) -> EngineCaps {
            self.inner.caps()
        }
        fn pipeline_depth(&self) -> usize {
            self.inner.pipeline_depth()
        }
        fn in_flight(&self) -> usize {
            self.inner.in_flight()
        }
        fn submit(&mut self, b: Batch) -> anyhow::Result<()> {
            let mut log = self.log.lock().unwrap();
            for admission in &b.admissions {
                if let Some(und) = &admission.und {
                    log.registrations
                        .push((admission.request_key.session_id, und.kv.prefix_len));
                }
            }
            drop(log);
            self.inner.submit(b)
        }
        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.inner.next_result()
        }
        fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
            self.inner.control(op)
        }
        fn control_wait(
            &mut self,
            op: ControlOp,
            targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            self.inner.control_wait(op, targets)
        }
        fn shutdown(&mut self) {
            self.inner.shutdown();
        }
    }

    let mut sim = SimEngine::new();
    sim.set_text_len(4);
    let block_size = sim.mut_caps_for_test().block_size;
    assert_eq!(block_size, 64, "test geometry assumes the sim block size");
    let log = Arc::new(Mutex::new(Log::default()));
    let exec = Recording {
        inner: SimExecutor::new(Box::new(sim)),
        log: log.clone(),
    };
    let mut sched = Scheduler::new(Box::new(exec), ctrl(), 32);

    // 600 tokens => 9 full 64-token blocks + a partial; the 9 full blocks are
    // the cacheable shared prefix.
    let prompt: Vec<u32> = (0..600u32).map(|i| (i % 53) + 7).collect();
    let run_one = |rid: u64, sched: &mut Scheduler| {
        let _erx = sched.submit_for_test(generation_request(
            RequestId(rid),
            text_context(prompt.clone()),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            8,
        ));
        for _ in 0..10_000 {
            if !sched.step() {
                break;
            }
        }
    };

    // cold: req1 populates the prefix cache and registers with no reuse.
    run_one(1, &mut sched);
    // warm: req2 (same prompt) registers with the reused prefix boundary.
    run_one(2, &mut sched);

    let log = log.lock().unwrap();
    let prefix_of = |rid: u64| -> Vec<u32> {
        log.registrations
            .iter()
            .filter(|(r, _)| r.0 == rid)
            .map(|(_, p)| *p)
            .collect()
    };
    assert_eq!(
        prefix_of(1),
        vec![0],
        "a cold admission must register prefix_len == 0"
    );
    assert_eq!(
        prefix_of(2),
        vec![9 * block_size],
        "a prefix-cache-hit admission must register prefix_len == cached blocks x block size"
    );
}

/// Structured outputs end to end under the sim: a guided choice gates on grammar
/// compilation (skipped_waiting), masks every decode step to the choice trie,
/// and terminates after one alternative completes.
#[test]
fn guided_choice_constrains_output() {
    use uniserve_engine_api::GrammarSpec;
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000); // the grammar, not the sim EOS, must terminate it
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let mut req = generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        64,
    );
    req.grammar = Some(GrammarSpec::Choice {
        token_sequences: vec![vec![2000, 2001, 2002], vec![3000]],
    });
    let mut erx = handle.submit(req).unwrap();

    let mut toks = Vec::new();
    let mut reason = None;
    let deadline = Instant::now() + Duration::from_secs(10);
    while reason.is_none() && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(GenEvent::TextToken { id, .. }) => toks.push(id),
            Ok(GenEvent::Finished { reason: r, .. }) => reason = Some(r),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();

    assert_eq!(
        reason,
        Some(FinishReason::Eos),
        "the completed grammar must force EOS"
    );
    assert!(
        toks == vec![2000, 2001, 2002] || toks == vec![3000],
        "output must be exactly one allowed choice, got {toks:?}"
    );
}

/// The literal image trigger (ThinkMorph's textual visual-thinking
/// signal): when the generated round text completes `image_start_ids`, the
/// next image begins immediately — without waiting for EOS or BAGEL's
/// <|vision_start|> token. The sim never emits EOS here (huge text_len), so
/// images can only come from the literal trigger.
#[test]
fn gen_branch_literal_trigger_starts_images() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000); // never EOS on its own
    sim.set_pipeline_depth(2);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    // Sim emits 1000 + ((id*7 + n) % 5000) for request id=1: 1007, 1008, 1009…
    // After an image commits, the sim resets and the round repeats from 1007.
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams {
                steps: 3,
                max_images: 2,
                ..Default::default()
            },
            GenerationConstraint::Default,
            40,
        ),
        TriggerPolicyDescriptor::Suffix {
            token_ids: vec![1008, 1009],
        },
    );
    let mut erx = handle.submit(req).unwrap();

    let mut seq: Vec<char> = Vec::new();
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(GenEvent::TextToken { .. }) => seq.push('T'),
            Ok(GenEvent::ImageDone { .. }) => seq.push('I'),
            Ok(GenEvent::Finished { .. }) => finished = true,
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
        let executor = Box::new(SimExecutor::new(Box::new(sim)));
        // an image-start token inside the sim's vocab
        let sched = Scheduler::new(executor, ctrl(), 32);
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let jh = thread::spawn(move || sched.run(rx));

        let mut req = with_trigger(
            generation_request(
                RequestId(1),
                text_context(vec![1, 2, 3]),
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
            TriggerPolicyDescriptor::Token { token_id: 2222 },
        );
        req.sampling.seed = None;
        let mut erx = handle.submit(req).unwrap();

        let mut text_before_first_image = 0usize;
        let mut images = 0usize;
        let mut finished = false;
        let deadline = Instant::now() + Duration::from_secs(10);
        while !finished && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(GenEvent::TextToken { .. }) if images == 0 => text_before_first_image += 1,
                Ok(GenEvent::ImageDone { .. }) => images += 1,
                Ok(GenEvent::Finished { .. }) => finished = true,
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
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            text_context(vec![10, 11, 2222]),
            SamplingParams::default(),
            ImageParams {
                steps: 3,
                max_images: 1,
                ..Default::default()
            },
            GenerationConstraint::Default,
            8,
        ),
        TriggerPolicyDescriptor::Token { token_id: 2222 },
    );
    let mut erx = handle.submit(req).unwrap();

    let mut text_before_first_image = 0usize;
    let mut images = 0usize;
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(GenEvent::TextToken { .. }) if images == 0 => text_before_first_image += 1,
            Ok(GenEvent::ImageDone { .. }) => images += 1,
            Ok(GenEvent::Finished { .. }) => finished = true,
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
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let control = ctrl();
    let close_token_ids = control.eos.clone();
    let sched = Scheduler::new(executor, control, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            context_with_image(Vec::new(), vec![1, 2, 3], 7, 1, 1),
            SamplingParams::default(),
            ImageParams {
                steps: 2,
                max_images: 1,
                ..Default::default()
            },
            GenerationConstraint::Default,
            40,
        ),
        TriggerPolicyDescriptor::RoundCloseThenSuffix {
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
            Ok(GenEvent::TextToken { .. }) => seq.push('T'),
            Ok(GenEvent::ImageDone { .. }) => seq.push('I'),
            Ok(GenEvent::Finished { .. }) => finished = true,
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
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3]),
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
        TriggerPolicyDescriptor::Token { token_id: 2222 },
    );
    let mut erx = handle.submit(req).unwrap();

    let mut images = 0usize;
    let mut post_budget_triggers = 0usize;
    let mut post_budget_text = 0usize;
    let mut finished = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !finished && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(GenEvent::TextToken { id, .. }) => {
                if images >= 2 {
                    if id == 2222 {
                        post_budget_triggers += 1;
                    } else {
                        post_budget_text += 1;
                    }
                }
            }
            Ok(GenEvent::ImageDone { .. }) => images += 1,
            Ok(GenEvent::Finished { .. }) => finished = true,
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

///: the resource ledger proves consistency — every
/// per-request lease (KV residency, denoise latents, scratch) is released after
/// the request completes, so the ledger drains to zero when the engine is idle.
#[test]
fn resource_leases_drain_to_zero_after_completion() {
    use std::sync::atomic::Ordering;
    let executor = Box::new(SimExecutor::new(Box::new(SimEngine::new())));
    let mut sched = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs);

    // Keep receivers alive — a dropped receiver is treated as a cancellation.
    let mut keep_alive = Vec::new();
    let specs = [
        GenerationConstraint::UndOnly,
        GenerationConstraint::GenOnly,
        GenerationConstraint::Default,
        GenerationConstraint::UndOnly,
    ];
    for (i, mode) in specs.iter().enumerate() {
        let req = generation_request(
            RequestId(i as u64 + 1),
            text_context(vec![1, 2, 3, 4, 5]),
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

    let mut max_active = 0usize;
    let mut idle_steps = 0;
    for _ in 0..5000 {
        let progressed = sched.step();
        max_active = max_active.max(sched.stats.resources.active.load(Ordering::Relaxed));
        // converged when several consecutive steps make no progress.
        idle_steps = if progressed { 0 } else { idle_steps + 1 };
        if idle_steps >= 3 {
            break;
        }
    }

    assert!(
        max_active > 0,
        "leases must actually be issued during the run"
    );
    assert_eq!(
        sched.stats.resources.active.load(Ordering::Relaxed),
        0,
        "every per-request lease must be released after completion (no leak)",
    );
    assert_eq!(
        sched.stats.general.running.load(Ordering::Relaxed),
        0,
        "running request gauge must be refreshed after the final request finishes",
    );
    assert_eq!(
        sched
            .stats
            .resources
            .invariant_violations
            .load(Ordering::Relaxed),
        0,
        "no resource-invariant violations",
    );
}

/// Host KV pages remain bound to their worker session until the exact close
/// acknowledgement orders worker retirement ahead of cross-request reuse.
#[test]
fn kv_page_ownership_turns_over_after_close_acknowledgement() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{Batch, CompletionReport, EngineCaps};

    struct OwnershipChecking {
        inner: SimExecutor,
        owners: Arc<Mutex<HashMap<u32, RequestId>>>,
    }

    impl Executor for OwnershipChecking {
        fn caps(&self) -> EngineCaps {
            self.inner.caps()
        }

        fn pipeline_depth(&self) -> usize {
            self.inner.pipeline_depth()
        }

        fn in_flight(&self) -> usize {
            self.inner.in_flight()
        }

        fn can_submit(&self) -> bool {
            self.inner.can_submit()
        }

        fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
            {
                let mut owners = self.owners.lock().unwrap();
                for operation in batch.operations() {
                    let session_id = operation.request_key.session_id;
                    for block in &operation.new_kv_blocks {
                        if let Some(owner) = owners.get(&block.0) {
                            anyhow::ensure!(
                                *owner == session_id,
                                "KV page {} remains owned by session {}",
                                block.0,
                                owner.0
                            );
                        }
                        owners.insert(block.0, session_id);
                    }
                }
            }
            self.inner.submit(batch)
        }

        fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
            self.inner.poll()
        }

        fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
            self.inner.next_result()
        }

        fn control(&mut self, operation: ControlOp) -> anyhow::Result<u64> {
            if let ControlOp::DropSession(id) = &operation {
                self.owners.lock().unwrap().retain(|_, owner| owner != id);
            }
            self.inner.control(operation)
        }

        fn control_wait(
            &mut self,
            operation: ControlOp,
            targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            if let ControlOp::DropSession(id) = &operation {
                self.owners.lock().unwrap().retain(|_, owner| owner != id);
            }
            self.inner.control_wait(operation, targets)
        }

        fn shutdown(&mut self) {
            self.inner.shutdown();
        }
    }

    let mut sim = SimEngine::new();
    sim.set_num_blocks(2);
    sim.set_pipeline_depth(2);
    sim.set_text_len(1);
    let owners = Arc::new(Mutex::new(HashMap::new()));
    let executor = OwnershipChecking {
        inner: SimExecutor::new(Box::new(sim)),
        owners: Arc::clone(&owners),
    };
    let mut scheduler =
        Scheduler::with_policy(Box::new(executor), ctrl(), 32, SchedulingPolicy::Fcfs);
    let mut receivers = (1..=2)
        .map(|request_id| {
            scheduler.submit_for_test(generation_request(
                RequestId(request_id),
                text_context(vec![1]),
                SamplingParams::default(),
                ImageParams::default(),
                GenerationConstraint::UndOnly,
                1,
            ))
        })
        .collect::<Vec<_>>();
    let mut finished = HashSet::new();
    for _ in 0..256 {
        scheduler.step();
        for (index, receiver) in receivers.iter_mut().enumerate() {
            while let Ok(event) = receiver.try_recv() {
                if matches!(event, GenEvent::Finished { .. }) {
                    finished.insert(index);
                }
            }
        }
        if finished.len() == receivers.len()
            && scheduler.health_snapshot().in_flight == 0
            && scheduler.health_snapshot().active_credit_requests == 0
        {
            break;
        }
    }

    assert_eq!(finished.len(), receivers.len());
    assert_eq!(scheduler.health_snapshot().active_credit_requests, 0);
    assert_eq!(scheduler.health_snapshot().free_blocks, 1);
    assert!(owners.lock().unwrap().is_empty());
}

/// The scheduler exposes structured facts, explainable decisions, and latency history.
#[test]
fn policy_facts_and_decisions_are_recorded() {
    let executor = Box::new(SimExecutor::new(Box::new(SimEngine::new())));
    let mut sched = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs);

    let mut keep_alive = Vec::new();
    for i in 0..3u64 {
        keep_alive.push(sched.submit_for_test(generation_request(
            RequestId(i + 1),
            text_context(vec![1, 2, 3, 4, 5]),
            SamplingParams::default(),
            ImageParams {
                steps: 4,
                ..Default::default()
            },
            GenerationConstraint::UndOnly,
            16,
        )));
    }
    let mut idle = 0;
    for _ in 0..5000 {
        if sched.step() {
            idle = 0;
        } else {
            idle += 1;
        }
        if idle >= 3 {
            break;
        }
    }

    // Structured facts: idle after completion, blocks returned.
    let snap = sched.policy_snapshot();
    assert_eq!(snap.running, 0);
    assert_eq!(snap.in_flight, 0);
    assert!(snap.free_blocks > 0 && snap.total_blocks > 0);

    // Latency history populated for the op kinds that ran.
    assert!(
        sched.op_latency_us("token_decode").is_some()
            || sched.op_latency_us("token_extend").is_some(),
        "per-op latency history must be observed"
    );

    // Explainable decisions: every request was admitted.
    let decisions = sched.take_policy_decisions();
    let admitted = decisions
        .iter()
        .filter(|d| d.reason == uniserve_scheduler::PolicyReason::Admitted)
        .count();
    assert_eq!(
        admitted, 3,
        "all three requests should record an Admitted decision"
    );
    // draining empties the ring
    assert!(sched.take_policy_decisions().is_empty());
}

///: a request's lifecycle is reconstructable from its trace
/// (admitted → op submitted [op_id + kind] → op resolved → finished), and the
/// engine exposes a self-describing, leak-free health snapshot.
#[test]
fn lifecycle_trace_and_health_snapshot() {
    let executor = Box::new(SimExecutor::new(Box::new(SimEngine::new())));
    let mut sched = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs);
    let _keep = sched.submit_for_test(generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3, 4, 5]),
        SamplingParams::default(),
        ImageParams {
            steps: 4,
            ..Default::default()
        },
        GenerationConstraint::UndOnly,
        8,
    ));
    let mut idle = 0;
    for _ in 0..5000 {
        if sched.step() {
            idle = 0;
        } else {
            idle += 1;
        }
        if idle >= 3 {
            break;
        }
    }

    // Health snapshot: idle, leak-free, alive (read before draining traces).
    let h = sched.health_snapshot();
    assert_eq!(h.running, 0);
    assert_eq!(h.active_credit_requests, 0);
    assert_eq!(h.resource_invariant_violations, 0);
    assert!(!h.fatal);
    assert!(h.completed_traces >= 1);
    assert!(!h.supported_work.is_empty());

    let traces = sched.take_completed_traces();
    assert_eq!(traces.len(), 1);
    let t = &traces[0];
    assert!(
        t.was_admitted() && t.is_finished(),
        "trace must span admit→finish"
    );
    assert!(t.resolved_ops() > 0, "ops must resolve");
    assert_eq!(t.request_id, RequestId(1), "trace is tied to the request");
    assert_eq!(t.trace_id.0, 1, "trace id is assigned");
    let submitted: Vec<_> = t
        .events
        .iter()
        .filter(|e| e.kind == uniserve_scheduler::TraceEventKind::OpSubmitted)
        .collect();
    assert!(!submitted.is_empty());
    assert!(
        submitted
            .iter()
            .all(|e| e.op_id.is_some() && e.op_kind.is_some()),
        "every submitted op carries an op_id + kind for correlation"
    );
}

#[test]
fn slow_cpu_continuation_suspends_only_its_request_lineage() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use uniserve_scheduler::{LogitsProcessor, MaskContribution, ProcCtx, ProcessorDeclaration};

    struct ControlledProcessor {
        release: Arc<AtomicBool>,
    }

    impl LogitsProcessor for ControlledProcessor {
        fn name(&self) -> &'static str {
            "controlled"
        }

        fn declaration(&self) -> ProcessorDeclaration {
            ProcessorDeclaration {
                snapshotable: true,
                deterministic: true,
                max_output_tokens: 1,
                max_outstanding_tasks: 1,
            }
        }

        fn is_argmax_invariant(&self) -> bool {
            true
        }

        fn contribute(&self, context: &ProcCtx<'_>) -> MaskContribution {
            if context.sampling.seed == Some(1)
                && context.n_generated == 0
                && !self.release.load(Ordering::Acquire)
            {
                while !self.release.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            MaskContribution::default()
        }
    }

    let release = Arc::new(AtomicBool::new(false));
    let executor = Box::new(SimExecutor::new(Box::new(SimEngine::new())));
    let mut scheduler = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs)
        .with_logits_processor(Box::new(ControlledProcessor {
            release: Arc::clone(&release),
        }));
    let mut slow_request = generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        4,
    );
    slow_request.sampling.seed = Some(1);
    let mut fast_request = generation_request(
        RequestId(2),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        4,
    );
    fast_request.sampling.seed = Some(2);
    let mut slow_events = scheduler.submit_for_test(slow_request);
    let mut fast_events = scheduler.submit_for_test(fast_request);

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut fast_finished = false;
    while Instant::now() < deadline && !fast_finished {
        scheduler.step();
        while let Ok(event) = fast_events.try_recv() {
            fast_finished |= matches!(event, GenEvent::Finished { .. });
        }
        thread::sleep(Duration::from_millis(1));
    }
    assert!(
        fast_finished,
        "an unrelated request must complete while one CPU continuation is suspended"
    );

    release.store(true, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut slow_finished = false;
    while Instant::now() < deadline && !slow_finished {
        scheduler.step();
        while let Ok(event) = slow_events.try_recv() {
            slow_finished |= matches!(event, GenEvent::Finished { .. });
        }
        thread::sleep(Duration::from_millis(1));
    }
    assert!(slow_finished);
}

#[test]
fn cpu_continuation_timeout_closes_only_its_request_lineage() {
    use uniserve_scheduler::{LogitsProcessor, MaskContribution, ProcCtx, ProcessorDeclaration};

    struct DelayedProcessor;

    impl LogitsProcessor for DelayedProcessor {
        fn name(&self) -> &'static str {
            "delayed"
        }

        fn declaration(&self) -> ProcessorDeclaration {
            ProcessorDeclaration {
                snapshotable: true,
                deterministic: true,
                max_output_tokens: 1,
                max_outstanding_tasks: 1,
            }
        }

        fn is_argmax_invariant(&self) -> bool {
            true
        }

        fn contribute(&self, context: &ProcCtx<'_>) -> MaskContribution {
            if context.sampling.seed == Some(11) && context.n_generated == 0 {
                thread::sleep(Duration::from_millis(200));
            }
            MaskContribution::default()
        }
    }

    let executor = Box::new(SimExecutor::new(Box::new(SimEngine::new())));
    let mut scheduler = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs)
        .with_logits_processor(Box::new(DelayedProcessor))
        .with_cpu_task_timeout(Duration::from_millis(25));
    let mut slow_request = generation_request(
        RequestId(11),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        4,
    );
    slow_request.sampling.seed = Some(11);
    let mut fast_request = generation_request(
        RequestId(12),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        4,
    );
    fast_request.sampling.seed = Some(12);
    let mut slow_events = scheduler.submit_for_test(slow_request);
    let mut fast_events = scheduler.submit_for_test(fast_request);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut slow_reason = None;
    let mut fast_reason = None;
    while Instant::now() < deadline && (slow_reason.is_none() || fast_reason.is_none()) {
        scheduler.step();
        while let Ok(event) = slow_events.try_recv() {
            if let GenEvent::Finished { reason, .. } = event {
                slow_reason = Some(reason);
            }
        }
        while let Ok(event) = fast_events.try_recv() {
            if let GenEvent::Finished { reason, .. } = event {
                fast_reason = Some(reason);
            }
        }
        thread::sleep(Duration::from_millis(1));
    }

    assert_eq!(slow_reason, Some(FinishReason::Error));
    assert!(fast_reason.is_some_and(|reason| reason != FinishReason::Error));
}

#[test]
fn cancellation_storm_reclaims_every_request_credit() {
    let executor = Box::new(SimExecutor::new(Box::new(SimEngine::new())));
    let mut scheduler = Scheduler::new(executor, ctrl(), 32);
    let receivers = (1..=128)
        .map(|request_id| {
            scheduler.submit_for_test(generation_request(
                RequestId(request_id),
                text_context(vec![1, 2, 3]),
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
        let health = scheduler.health_snapshot();
        if health.running == 0 && health.pending == 0 && health.in_flight == 0 {
            break;
        }
        thread::sleep(Duration::from_micros(50));
    }

    let health = scheduler.health_snapshot();
    assert_eq!(health.running, 0);
    assert_eq!(health.pending, 0);
    assert_eq!(health.in_flight, 0);
    assert_eq!(health.active_credit_requests, 0);
    assert_eq!(health.resource_invariant_violations, 0);
}

#[test]
fn cpu_failure_storm_is_request_local_and_reclaims_every_credit() {
    use uniserve_scheduler::{LogitsProcessor, MaskContribution, ProcCtx, ProcessorDeclaration};

    struct BoundViolatingProcessor;

    impl LogitsProcessor for BoundViolatingProcessor {
        fn name(&self) -> &'static str {
            "bound_violating"
        }

        fn declaration(&self) -> ProcessorDeclaration {
            ProcessorDeclaration {
                snapshotable: true,
                deterministic: true,
                max_output_tokens: 1,
                max_outstanding_tasks: 1,
            }
        }

        fn is_argmax_invariant(&self) -> bool {
            true
        }

        fn contribute(&self, _context: &ProcCtx<'_>) -> MaskContribution {
            MaskContribution {
                allowed: None,
                suppress: vec![1, 2],
            }
        }
    }

    let executor = Box::new(SimExecutor::new(Box::new(SimEngine::new())));
    let mut scheduler = Scheduler::new(executor, ctrl(), 32)
        .with_logits_processor(Box::new(BoundViolatingProcessor));
    let mut receivers = (1..=128)
        .map(|request_id| {
            scheduler.submit_for_test(generation_request(
                RequestId(request_id),
                text_context(vec![1, 2, 3]),
                SamplingParams::default(),
                ImageParams::default(),
                GenerationConstraint::UndOnly,
                8,
            ))
        })
        .collect::<Vec<_>>();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut failed = HashSet::new();
    while Instant::now() < deadline && failed.len() < receivers.len() {
        scheduler.step();
        for (index, receiver) in receivers.iter_mut().enumerate() {
            while let Ok(event) = receiver.try_recv() {
                if matches!(
                    event,
                    GenEvent::Finished {
                        reason: FinishReason::Error,
                        ..
                    }
                ) {
                    failed.insert(index);
                }
            }
        }
        thread::sleep(Duration::from_micros(50));
    }

    assert_eq!(failed.len(), receivers.len());
    let health = scheduler.health_snapshot();
    assert_eq!(health.running, 0);
    assert_eq!(health.pending, 0);
    assert_eq!(health.in_flight, 0);
    assert_eq!(health.active_credit_requests, 0);
    assert_eq!(health.resource_invariant_violations, 0);
}

#[test]
fn slow_client_releases_execution_credits_before_output_capacity_returns() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let mut scheduler = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Fcfs);
    let slow_events = scheduler.submit_for_test(generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        256,
    ));
    let mut fast_events = scheduler.submit_for_test(generation_request(
        RequestId(2),
        text_context(vec![1, 2, 3]),
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
            fast_finished |= matches!(event, GenEvent::Finished { .. });
        }
    }
    assert!(
        fast_finished,
        "the consuming client must complete independently"
    );

    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && scheduler.step() {}
    let health = scheduler.health_snapshot();
    assert_eq!(
        health.in_flight, 0,
        "an output-credit-stalled request cannot retain an execution slot"
    );
    assert_eq!(health.running, 1);

    drop(slow_events);
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline
        && (scheduler.health_snapshot().running > 0
            || scheduler.health_snapshot().active_credit_requests > 0)
    {
        scheduler.step();
    }
    let health = scheduler.health_snapshot();
    assert_eq!(health.running, 0);
    assert_eq!(health.active_credit_requests, 0);
}
