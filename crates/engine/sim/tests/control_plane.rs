#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Full-stack GPU-free control-plane integration tests: drive the
//! real `Scheduler` over a `LocalExecutor`+`SimEngine` and assert the lifecycle/event
//! contract. This is the regression harness every workstream relies on.

use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};

use uniserve_core::{
    CommitRecipe, ContextSegment, FeedbackNextToken, FeedbackWriteback,
    GeneratedImageFeedbackRecipe, GenerationBehaviorDescriptor, GenerationConstraint,
    GenerationPolicyDescriptor, GenerationRequest, GenerationResourceBounds,
    GenerationRuntimeCapabilities, ImageIngestRecipe, ImageKvEffect, ImageParams, ImageSegment,
    OpKind, RequestId, SamplingParams, SegmentPlacement, TriggerPolicyDescriptor, UndVisibility,
};
use uniserve_engine_api::{EngineHandle, FinishReason, GenEvent};
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
            commit: CommitRecipe::CommitGenThenWriteback,
            writeback: FeedbackWriteback::DirectKv,
            next_und_token: FeedbackNextToken::EndOfImage,
            logical_positions: 2,
            physical_kv_tokens: ImageKvEffect::WorkerDefined,
        }),
        ..GenerationPolicyDescriptor::default()
    };
    let behavior = GenerationBehaviorDescriptor::resolve(constraint, &policy);
    let cache = Default::default();
    let capabilities = GenerationRuntimeCapabilities {
        supported_ops: vec![
            OpKind::PrefillUnd,
            OpKind::DecodeUnd,
            OpKind::VitEncode,
            OpKind::DenoiseGen,
            OpKind::CommitGen,
            OpKind::CommitWriteback,
        ],
        max_latent_units: 1_024,
        latent_downsample: 16,
        max_vae_grid_tokens: 1_024,
        max_vit_grid_tokens: 64,
        commit_marker_tokens: 2,
        max_cfg_branches: 3,
        scratch_capacity_tokens: 1 << 20,
        scratch_block_size: 64,
        encoder_cache_entries: 256,
        generated_image_commit: uniserve_core::GeneratedImageCommitCapabilities {
            inline: true,
            separate_writeback: true,
        },
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

    let mut rxs: HashMap<RequestId, tokio::sync::mpsc::UnboundedReceiver<GenEvent>> =
        HashMap::new();
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
    use uniserve_worker_wire::{EngineCaps, ForwardBatch, ForwardResult, OpKind};

    struct Recording {
        inner: SimExecutor,
        batches: Arc<Mutex<Vec<Vec<OpKind>>>>,
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
        fn submit(&mut self, b: ForwardBatch) -> anyhow::Result<()> {
            self.batches
                .lock()
                .unwrap()
                .push(b.ops.iter().map(|op| op.kind).collect());
            self.inner.submit(b)
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
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
    sim.set_text_len(8);
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
        batches.iter().any(|kinds| {
            kinds.contains(&OpKind::DecodeUnd) && kinds.contains(&OpKind::DenoiseGen)
        }),
        "expected one batch mixing decode_und and denoise_gen, got {batches:?}",
    );
    assert!(
        batches.iter().any(|kinds| {
            kinds.contains(&OpKind::DecodeUnd) && kinds.contains(&OpKind::CommitGen)
        }),
        "expected one batch mixing decode_und and commit_gen, got {batches:?}",
    );
}

#[test]
fn scheduler_uses_idle_pipeline_slot_for_prefill_without_decode_lookahead_pressure() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{
        EngineCaps, ExecutionConstraints, ForwardBatch, ForwardResult, OpKind,
    };

    type BatchLog = Arc<Mutex<Vec<Vec<(RequestId, OpKind)>>>>;

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
        fn submit(&mut self, b: ForwardBatch) -> anyhow::Result<()> {
            self.batches
                .lock()
                .unwrap()
                .push(b.ops.iter().map(|op| (op.req_id, op.kind)).collect());
            self.inner.submit(b)
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
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
        sim.mut_caps_for_test().execution_constraints = ExecutionConstraints {
            max_batch_ops: 1024,
        };
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

        let _rx1 = sched.submit_for_test(generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3]),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            64,
        ));
        let _rx2 = sched.submit_for_test(generation_request(
            RequestId(2),
            text_context(vec![4, 5, 6]),
            SamplingParams::default(),
            ImageParams::default(),
            GenerationConstraint::UndOnly,
            64,
        ));

        let mut saw_decode_pressure = false;
        for _ in 0..40 {
            sched.step();
            let log = batches.lock().unwrap();
            saw_decode_pressure = log.iter().any(|batch| {
                batch.len() == 2 && batch.iter().all(|(_, kind)| *kind == OpKind::DecodeUnd)
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
            SamplingParams::default(),
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
            panic!("expected a prefill batch after admitting request 3: {log:?}")
        });
        assert!(
            prefill_batch
                .iter()
                .any(|(id, kind)| *id == RequestId(3) && *kind == OpKind::PrefillUnd),
            "new request prefill should use an idle pipeline slot, got {prefill_batch:?}"
        );
        assert!(
            prefill_batch
                .iter()
                .all(|(_, kind)| *kind != OpKind::DecodeUnd),
            "idle-slot text prefill must not mix with decode, got {prefill_batch:?}"
        );
    }

    run();
}

#[test]
fn scheduler_respects_worker_image_latent_capacity_for_denoise_batches() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{
        EngineCaps, ExecutionConstraints, ForwardBatch, ForwardResult, OpKind, ResourceClass,
    };

    struct Recording {
        inner: SimExecutor,
        batches: Arc<Mutex<Vec<Vec<OpKind>>>>,
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
        fn submit(&mut self, b: ForwardBatch) -> anyhow::Result<()> {
            self.batches
                .lock()
                .unwrap()
                .push(b.ops.iter().map(|op| op.kind).collect());
            self.inner.submit(b)
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
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
    sim.mut_caps_for_test().execution_constraints = ExecutionConstraints {
        max_batch_ops: 1024,
    };
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
        batches.iter().all(|kinds| kinds
            .iter()
            .filter(|kind| **kind == OpKind::DenoiseGen)
            .count()
            <= 1),
        "expected at most one denoise_gen per batch under one-image latent cap, got {batches:?}",
    );
}

#[test]
fn scheduler_clamps_max_batch_to_worker_caps() {
    use uniserve_worker_wire::ExecutionConstraints;

    let mut sim = SimEngine::new();
    sim.mut_caps_for_test().execution_constraints = ExecutionConstraints { max_batch_ops: 3 };
    let sched = Scheduler::new(Box::new(SimExecutor::new(Box::new(sim))), ctrl(), 32);

    assert_eq!(sched.config().max_batch, 3);
}

#[test]
fn decode_lookahead_uses_last_sampled_token_source_for_safe_text() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{EngineCaps, ForwardBatch, ForwardResult, OpKind, TokenSource};

    type OperationLog = Arc<Mutex<Vec<(OpKind, TokenSource, (u32, u32))>>>;

    struct Recording {
        inner: SimExecutor,
        ops: OperationLog,
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
        fn submit(&mut self, b: ForwardBatch) -> anyhow::Result<()> {
            self.ops.lock().unwrap().extend(
                b.ops
                    .iter()
                    .map(|op| (op.kind, op.token_source, op.pos_range)),
            );
            self.inner.submit(b)
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
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
    sim.set_text_len(128);
    sim.set_pipeline_depth(2);
    let ops = Arc::new(Mutex::new(Vec::new()));
    let exec = Recording {
        inner: SimExecutor::new(Box::new(sim)),
        ops: ops.clone(),
    };
    let mut sched = Scheduler::new(Box::new(exec), ctrl(), 32);
    let sampling = SamplingParams {
        ignore_eos: true,
        ..Default::default()
    };
    let _keep = sched.submit_for_test(generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
        sampling,
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        4,
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

    let ops = ops.lock().unwrap();
    assert!(
        ops.iter().any(|(kind, source, _)| {
            *kind == OpKind::DecodeUnd && *source == TokenSource::LastSampled
        }),
        "safe greedy text decode should submit at least one lookahead op, got {ops:?}"
    );
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
    sim.set_text_len(1_000_000); // effectively never EOS on its own
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
        1_000_000,
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

    let collect = |erx: &mut tokio::sync::mpsc::UnboundedReceiver<GenEvent>| -> (usize, bool) {
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

/// a high-priority arrival preempts a lower-priority running request; the
/// preempted request later resumes (recomputing) and both finish. Driven by
/// manual stepping for determinism.
#[test]
fn priority_preemption_and_recompute() {
    let mut sim = SimEngine::new();
    sim.set_num_blocks(2); // block 0 padding => exactly 1 usable block
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let mut sched = Scheduler::with_policy(executor, ctrl(), 32, SchedulingPolicy::Priority);
    let stats = sched.stats_handle();

    // low-priority A, then (later) high-priority B; 1 usable block forces a choice.
    let mut a = generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        6,
    );
    a.priority = 10;
    let mut arx = sched.submit_for_test(a);

    // step until A is running and decoding (holds the only block).
    for _ in 0..4 {
        sched.step();
    }

    let mut b = generation_request(
        RequestId(2),
        text_context(vec![4, 5, 6]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        6,
    );
    b.priority = 0; // higher priority (lower value)
    let mut brx = sched.submit_for_test(b);

    // drive to completion.
    for _ in 0..2000 {
        if !sched.step() {
            break;
        }
    }

    use std::sync::atomic::Ordering;
    assert!(
        stats.general.preemptions.load(Ordering::Relaxed) >= 1,
        "expected at least one preemption"
    );

    let drain = |erx: &mut tokio::sync::mpsc::UnboundedReceiver<GenEvent>| -> (usize, bool) {
        let mut text = 0;
        let mut done = false;
        while let Ok(ev) = erx.try_recv() {
            match ev {
                GenEvent::TextToken { .. } => text += 1,
                GenEvent::Finished { .. } => done = true,
                _ => {}
            }
        }
        (text, done)
    };
    let (ta, da) = drain(&mut arx);
    let (tb, db) = drain(&mut brx);
    assert!(db, "high-priority B must finish (text={tb})");
    assert!(da, "preempted A must resume and finish (text={ta})");
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
            Ok(GenEvent::TextToken { id, logprob }) => {
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
        context_with_image(vec![1, 2], vec![3, 4], 0x81, 4, 1),
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

/// a single generated branch request emits at least two images separated by
/// text in one stream, and finishes only on a terminal condition.
#[test]
fn gen_branch_round_trip_text_image_text_image() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    sim.set_pipeline_depth(2);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let control = ControlTokens { ..ctrl() };
    let sched = Scheduler::new(executor, control, 32);
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
            200,
        ),
        TriggerPolicyDescriptor::Token { token_id: 2222 },
    );
    let mut erx = handle.submit(req).unwrap();

    // record the modality sequence: 'T' for a text token, 'I' for an image.
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

    assert!(finished, "generated branch request did not finish");
    let images = seq.iter().filter(|&&c| c == 'I').count();
    assert!(
        images >= 2,
        "expected >= 2 images, got {images} (seq={:?})",
        seq
    );
    assert!(
        seq.contains(&'T'),
        "expected native text tokens around images (seq={:?})",
        seq
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
        commit: CommitRecipe::CommitGen,
        writeback: FeedbackWriteback::Reingest {
            ingest: Box::new(ImageIngestRecipe::vit_only(
                1,
                ImageKvEffect::Exact { tokens: 1 },
            )),
        },
        next_und_token: FeedbackNextToken::Bos,
        logical_positions: 1,
        physical_kv_tokens: ImageKvEffect::Exact { tokens: 1 },
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
        images, 3,
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

#[test]
fn commit_eos_finishes_without_spending_remaining_budget() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let control = ControlTokens { ..ctrl() };
    sim.set_commit_token(Some(control.eos[0]));
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, control, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let req = with_trigger(
        generation_request(
            RequestId(1),
            text_context(vec![1, 2, 3, 2222]),
            SamplingParams {
                logit_bias: vec![(2222, 1000.0)],
                ..Default::default()
            },
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

    assert_eq!(finished, Some(FinishReason::Eos));
    assert_eq!(
        images, 1,
        "commit-side EOS should finish instead of starting hidden images"
    );
    assert_eq!(text, 0, "commit-side EOS should not emit text filler");
}

/// the async Executor seam fronts a MultiWorkerExecutor (2 ranks) with no
/// scheduler change — a batch fans out to both ranks, results join, and requests
/// complete identically to the single-worker path.
#[test]
fn multiworker_executor_drives_scheduler_unchanged() {
    let mk = || {
        let mut sim = SimEngine::new();
        sim.set_pipeline_depth(2);
        Box::new(SimExecutor::new(Box::new(sim))) as Box<dyn Executor>
    };
    let executor = Box::new(MultiprocExecutor::new(vec![mk(), mk()]));
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

/// Stateful-diff contract: a recording executor asserts that a request's static
/// state crosses exactly once (`new_reqs`), per-step ops carry only block deltas,
/// and preemption resets the registration (the request re-registers after
/// `drop_request`).
#[test]
fn stateful_diff_contract_registers_once_and_resends_after_preemption() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{EngineCaps, ForwardBatch, ForwardResult};

    #[derive(Default, Clone)]
    struct Log {
        new_reqs: Vec<RequestId>,
        drops: Vec<RequestId>,
        blocks_per_op: Vec<(RequestId, usize)>,
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
        fn submit(&mut self, b: ForwardBatch) -> anyhow::Result<()> {
            let mut log = self.log.lock().unwrap();
            for nr in &b.new_reqs {
                log.new_reqs.push(nr.req_id);
            }
            for op in &b.ops {
                log.blocks_per_op.push((op.req_id, op.new_block_ids.len()));
            }
            drop(log);
            self.inner.submit(b)
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
            self.inner.next_result()
        }
        fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
            if let ControlOp::DropRequest(id) = &op {
                self.log.lock().unwrap().drops.push(*id);
            }
            self.inner.control(op)
        }
        fn control_wait(
            &mut self,
            op: ControlOp,
            targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<ControlAck>> {
            if let ControlOp::DropRequest(id) = &op {
                self.log.lock().unwrap().drops.push(*id);
            }
            self.inner.control_wait(op, targets)
        }
        fn shutdown(&mut self) {
            self.inner.shutdown();
        }
    }

    let mut sim = SimEngine::new();
    sim.set_num_blocks(2); // 1 usable block forces preemption pressure
    let log = Arc::new(Mutex::new(Log::default()));
    let exec = Recording {
        inner: SimExecutor::new(Box::new(sim)),
        log: log.clone(),
    };
    let mut sched = Scheduler::with_policy(Box::new(exec), ctrl(), 32, SchedulingPolicy::Priority);

    let mut a = generation_request(
        RequestId(1),
        text_context(vec![1, 2, 3]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        6,
    );
    a.priority = 10;
    let _arx = sched.submit_for_test(a);
    for _ in 0..4 {
        sched.step();
    }

    let mut b = generation_request(
        RequestId(2),
        text_context(vec![4, 5, 6]),
        SamplingParams::default(),
        ImageParams::default(),
        GenerationConstraint::UndOnly,
        6,
    );
    b.priority = 0;
    let _brx = sched.submit_for_test(b);
    for _ in 0..2000 {
        if !sched.step() {
            break;
        }
    }

    let log = log.lock().unwrap();
    // Request 1 was preempted (drop_request) and re-registered on resumption:
    // it appears in new_reqs once per registration, i.e. exactly twice.
    let reg_1 = log.new_reqs.iter().filter(|r| r.0 == 1).count();
    let reg_2 = log.new_reqs.iter().filter(|r| r.0 == 2).count();
    assert_eq!(
        reg_2, 1,
        "request 2 must register exactly once: {:?}",
        log.new_reqs
    );
    assert_eq!(
        reg_1, 2,
        "preempted request 1 must re-register: {:?}",
        log.new_reqs
    );
    assert!(
        log.drops.iter().any(|r| r.0 == 1),
        "preemption must drop the worker record"
    );

    // Per-step decode ops carry no block ids while the request stays within its
    // allocation — the first op after (re-)registration carries the initial
    // blocks in NewRequestData, so per-op deltas are empty until growth.
    let deltas_1: Vec<usize> = log
        .blocks_per_op
        .iter()
        .filter(|(r, _)| r.0 == 1)
        .map(|(_, n)| *n)
        .collect();
    assert!(
        deltas_1.iter().all(|&n| n == 0),
        "per-op block deltas must be empty within the initial allocation: {deltas_1:?}"
    );
}

/// `NewRequestData.prefix_len` carries the scheduler's prefix-cache reuse
/// boundary to the worker as a typed field: 0 on a cold admission, and
/// cached-blocks x block-size when the admission reuses a cached prompt prefix.
#[test]
fn new_request_data_carries_the_prefix_reuse_boundary() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{EngineCaps, ForwardBatch, ForwardResult};

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
        fn submit(&mut self, b: ForwardBatch) -> anyhow::Result<()> {
            let mut log = self.log.lock().unwrap();
            for nr in &b.new_reqs {
                log.registrations.push((nr.req_id, nr.prefix_len));
            }
            drop(log);
            self.inner.submit(b)
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            self.inner.poll()
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
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

///: the scheduler exposes structured facts (PolicySnapshot), explainable
/// decisions (admit/reject), and per-op-kind latency history — all observed
/// from the existing inline policy without changing scheduling behavior.
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
        sched.op_latency_us("decode_und").is_some() || sched.op_latency_us("prefill_und").is_some(),
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
    assert_eq!(h.active_leases, 0);
    assert_eq!(h.resource_invariant_violations, 0);
    assert!(!h.fatal);
    assert!(h.completed_traces >= 1);
    assert!(!h.supported_ops.is_empty());

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
