#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Full-stack GPU-free control-plane integration tests: drive the
//! real `Scheduler` over a `LocalExecutor`+`SimEngine` and assert the FSM/event
//! contract. This is the regression harness every workstream relies on.

use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};

use uniserve_core::{GenMode, ImageParams, RequestId, SamplingParams};
use uniserve_engine_api::{EngineHandle, FinishReason, GenEvent, GenerateRequest, MmItem};
use uniserve_executor::{ControlAck, ControlOp, Executor};
use uniserve_scheduler::{ControlTokens, Scheduler, SchedulerConfig, SchedulingPolicy};
use uniserve_sim::SimEngine;
use uniserve_sim::SimExecutor;
use uniserve_worker_ipc::MultiprocExecutor;

fn ctrl() -> ControlTokens {
    ControlTokens::default()
}

struct Collected {
    text: usize,
    images: usize,
    finished: bool,
    reason: Option<FinishReason>,
}

/// Run the given (mode, count) requests through a fresh scheduler at `depth`,
/// returning per-request collected counts.
fn run_requests(
    depth: u32,
    policy: SchedulingPolicy,
    specs: &[(GenMode, usize)],
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
    for (mode, n) in specs {
        for _ in 0..*n {
            let (etx, erx) = tokio::sync::mpsc::unbounded_channel();
            let req = GenerateRequest::new(
                RequestId(id),
                vec![1, 2, 3, 4, 5],
                SamplingParams::default(),
                ImageParams {
                    steps: 4,
                    ..Default::default()
                },
                *mode,
                16,
                etx,
            );
            handle.submit(req).unwrap();
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
        &[(GenMode::Text, 3), (GenMode::Image, 2)],
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

    let (text_tx, mut text_rx) = tokio::sync::mpsc::unbounded_channel();
    handle
        .submit(GenerateRequest::new(
            RequestId(1),
            vec![1, 2, 3],
            SamplingParams::default(),
            ImageParams::default(),
            GenMode::Text,
            64,
            text_tx,
        ))
        .unwrap();
    let (image_tx, mut image_rx) = tokio::sync::mpsc::unbounded_channel();
    handle
        .submit(GenerateRequest::new(
            RequestId(2),
            vec![4, 5, 6],
            SamplingParams::default(),
            ImageParams {
                steps: 1,
                ..Default::default()
            },
            GenMode::Image,
            0,
            image_tx,
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
fn scheduler_prioritizes_new_prefill_over_decode_batch_slots() {
    use std::sync::{Arc, Mutex};
    use uniserve_worker_wire::{
        EngineCaps, ExecutionConstraints, ForwardBatch, ForwardResult, OpKind,
    };

    struct Recording {
        inner: SimExecutor,
        batches: Arc<Mutex<Vec<Vec<(RequestId, OpKind)>>>>,
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
        sim.set_pipeline_depth(1);
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

        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        sched.submit_for_test(GenerateRequest::new(
            RequestId(1),
            vec![1, 2, 3],
            SamplingParams::default(),
            ImageParams::default(),
            GenMode::Text,
            64,
            tx1,
        ));
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched.submit_for_test(GenerateRequest::new(
            RequestId(2),
            vec![4, 5, 6],
            SamplingParams::default(),
            ImageParams::default(),
            GenMode::Text,
            64,
            tx2,
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
        let (tx3, _rx3) = tokio::sync::mpsc::unbounded_channel();
        sched.submit_for_test(GenerateRequest::new(
            RequestId(3),
            vec![7, 8, 9],
            SamplingParams::default(),
            ImageParams::default(),
            GenMode::Text,
            64,
            tx3,
        ));

        for _ in 0..40 {
            sched.step();
            if batches.lock().unwrap().len() > before {
                break;
            }
        }
        let log = batches.lock().unwrap();
        let next_batch = log.get(before).unwrap_or_else(|| {
            panic!("expected a submitted batch after admitting request 3: {log:?}")
        });
        assert!(
            next_batch
                .iter()
                .any(|(id, kind)| *id == RequestId(3) && *kind == OpKind::PrefillUnd),
            "new request prefill should take a batch slot ahead of older decodes, got {next_batch:?}"
        );
    }

    // und/gen mixing is unconditional; the new-prefill-over-decode admission
    // priority holds regardless, so this runs once.
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
        let (image_tx, image_rx) = tokio::sync::mpsc::unbounded_channel();
        handle
            .submit(GenerateRequest::new(
                RequestId(id),
                vec![4, 5, 6],
                SamplingParams::default(),
                ImageParams {
                    steps: 1,
                    ..Default::default()
                },
                GenMode::Image,
                0,
                image_tx,
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

    struct Recording {
        inner: SimExecutor,
        ops: Arc<Mutex<Vec<(OpKind, TokenSource, (u32, u32))>>>,
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
    let (etx, erx) = tokio::sync::mpsc::unbounded_channel();
    let _keep = erx;
    let mut sampling = SamplingParams::default();
    sampling.ignore_eos = true;
    sched.submit_for_test(GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        sampling,
        ImageParams::default(),
        GenMode::Text,
        4,
        etx,
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
    let d1 = run_requests(1, SchedulingPolicy::Fcfs, &[(GenMode::Text, 4)]);
    let d2 = run_requests(2, SchedulingPolicy::Fcfs, &[(GenMode::Text, 4)]);
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

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let mut req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams::default(),
        ImageParams::default(),
        GenMode::Text,
        64,
        etx,
    );
    req.stop_token_ids = vec![1007];
    handle.submit(req).unwrap();

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

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams::default(),
        ImageParams::default(),
        GenMode::Text,
        1_000_000,
        etx,
    );
    handle.submit(req).unwrap();

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

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    handle
        .submit(GenerateRequest::new(
            RequestId(1),
            vec![1, 2, 3],
            SamplingParams::default(),
            ImageParams::default(),
            GenMode::Text,
            16,
            etx,
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
        let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
        handle
            .submit(GenerateRequest::new(
                RequestId(rid),
                prompt.clone(),
                SamplingParams::default(),
                ImageParams::default(),
                GenMode::Text,
                8,
                etx,
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
    handle.reset_prefix_cache();
    thread::sleep(Duration::from_millis(20));

    handle.shutdown();
    let _ = jh.join();
}

/// under Fcfs with a small per-step token budget and a low chunk cap, a
/// long prompt is prefilled in budget-sized chunks while a concurrent request's
/// decodes proceed in the same steps — both complete correctly.
#[test]
fn chunked_prefill_interleaves_with_decode() {
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
    let (etx1, mut erx1) = tokio::sync::mpsc::unbounded_channel();
    handle
        .submit(GenerateRequest::new(
            RequestId(1),
            long_prompt,
            SamplingParams::default(),
            ImageParams::default(),
            GenMode::Text,
            8,
            etx1,
        ))
        .unwrap();
    let (etx2, mut erx2) = tokio::sync::mpsc::unbounded_channel();
    handle
        .submit(GenerateRequest::new(
            RequestId(2),
            vec![1, 2, 3],
            SamplingParams::default(),
            ImageParams::default(),
            GenMode::Text,
            8,
            etx2,
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
    let (atx, mut arx) = tokio::sync::mpsc::unbounded_channel();
    let mut a = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams::default(),
        ImageParams::default(),
        GenMode::Text,
        6,
        atx,
    );
    a.priority = 10;
    sched.submit_for_test(a);

    // step until A is running and decoding (holds the only block).
    for _ in 0..4 {
        sched.step();
    }

    let (btx, mut brx) = tokio::sync::mpsc::unbounded_channel();
    let mut b = GenerateRequest::new(
        RequestId(2),
        vec![4, 5, 6],
        SamplingParams::default(),
        ImageParams::default(),
        GenMode::Text,
        6,
        btx,
    );
    b.priority = 0; // higher priority (lower value)
    sched.submit_for_test(b);

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
) -> (Vec<u32>, bool, bool) {
    let mut sim = SimEngine::new();
    sim.set_text_len(text_len);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        sampling,
        ImageParams::default(),
        GenMode::Text,
        max_tokens,
        etx,
    );
    handle.submit(req).unwrap();

    let mut toks = Vec::new();
    let mut any_logprob = false;
    let mut done = false;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done && Instant::now() < deadline {
        match erx.try_recv() {
            Ok(GenEvent::TextToken { id, logprob }) => {
                toks.push(id);
                if logprob.is_some() {
                    any_logprob = true;
                }
            }
            Ok(GenEvent::Finished { .. }) => done = true,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    handle.shutdown();
    let _ = jh.join();
    (toks, any_logprob, done)
}

#[test]
fn logprobs_flow_to_events() {
    let sp = SamplingParams {
        n_logprobs: 3,
        ..Default::default()
    };
    let (toks, any_logprob, done) = run_sampling(sp, 8, 16);
    assert!(done);
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
    let (toks, _lp, done) = run_sampling(sp, 8, 6);
    assert!(done);
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
    let (toks, _lp, done) = run_sampling(sp, 8, 6);
    assert!(done);
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
    let (toks, _lp, done) = run_sampling(sp, 1, 50);
    assert!(done);
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
    let (toks, _lp, done) = run_sampling(SamplingParams::default(), 8, 16);
    assert!(done);
    assert_eq!(toks[0], 1007);
}

/// an image-in-prompt request encodes the image (VitEncode) before
/// prefill and then produces text; a second request with the same image hits the
/// encoder cache and skips re-encode (no embedding ever crosses the wire).
#[test]
fn multimodal_encode_then_cache_hit() {
    use std::sync::atomic::Ordering;
    use uniserve_engine_api::MmItem;
    let mut sim = SimEngine::new();
    sim.set_pipeline_depth(2);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let stats = sched.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let run_img = |rid: u64, handle: &EngineHandle| -> bool {
        let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
        // prompt_ids include placeholder positions for the image span [3, 7).
        let mut req = GenerateRequest::new(
            RequestId(rid),
            vec![1, 2, 3, 0, 0, 0, 0, 9],
            SamplingParams::default(),
            ImageParams::default(),
            GenMode::Text,
            8,
            etx,
        );
        req.mm_items = vec![MmItem {
            hash: 0xCAFE,
            position: 3,
            num_tokens: 4,
            b64: String::new(),
        }];
        handle.submit(req).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut text = 0;
        let mut done = false;
        while !done && Instant::now() < deadline {
            match erx.try_recv() {
                Ok(GenEvent::TextToken { .. }) => text += 1,
                Ok(GenEvent::Finished { .. }) => done = true,
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        done && text > 0
    };

    assert!(
        run_img(1, &handle),
        "image-in-prompt request 1 must produce text"
    );
    assert_eq!(
        stats.encoder.cache_hits.load(Ordering::Relaxed),
        0,
        "first image is a cache miss"
    );
    assert!(
        stats.encoder.cached.load(Ordering::Relaxed) >= 1,
        "encoder output should be cached"
    );

    assert!(run_img(2, &handle), "request 2 must produce text");
    assert!(
        stats.encoder.cache_hits.load(Ordering::Relaxed) >= 1,
        "repeated image must hit the encoder cache"
    );

    handle.reset_encoder_cache();
    thread::sleep(Duration::from_millis(20));
    handle.shutdown();
    let _ = jh.join();
}

/// a single AutoInterleave request emits at least two images separated by
/// text in one stream, and finishes only on a terminal condition.
#[test]
fn interleave_round_trip_text_image_text_image() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    sim.set_pipeline_depth(2);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let control = ControlTokens {
        start_of_image: 2222,
        ..ctrl()
    };
    let sched = Scheduler::new(executor, control, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams {
            logit_bias: vec![(2222, 1000.0)],
            ..Default::default()
        },
        ImageParams {
            steps: 3,
            max_images: 2,
            ..Default::default()
        },
        GenMode::AutoInterleave,
        200,
        etx,
    );
    handle.submit(req).unwrap();

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

    assert!(finished, "interleave request did not finish");
    let images = seq.iter().filter(|&&c| c == 'I').count();
    assert!(
        images >= 2,
        "expected >= 2 images, got {images} (seq={:?})",
        seq
    );
    assert!(
        seq.iter().any(|&c| c == 'T'),
        "expected native text tokens around images (seq={:?})",
        seq
    );
}

/// Native multi-image passages can be requested intentionally through max_images;
/// image starts must come from the model, not a scheduler-forced cadence.
#[test]
fn auto_interleave_waits_for_model_image_starts() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams::default(),
        ImageParams {
            steps: 3,
            max_images: 3,
            ..Default::default()
        },
        GenMode::AutoInterleave,
        40,
        etx,
    );
    handle.submit(req).unwrap();

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
fn auto_interleave_model_image_starts_spend_budget() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let trig = ControlTokens {
        start_of_image: 2222,
        ..ControlTokens::default()
    };
    let sched = Scheduler::new(executor, trig, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams {
            logit_bias: vec![(2222, 1000.0)],
            ..Default::default()
        },
        ImageParams {
            steps: 3,
            max_images: 3,
            ..Default::default()
        },
        GenMode::AutoInterleave,
        40,
        etx,
    );
    handle.submit(req).unwrap();

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
fn auto_interleave_long_token_budget_is_not_rejected_up_front() {
    let mut sim = SimEngine::new();
    sim.set_text_len(8);
    sim.set_num_blocks(128);
    sim.set_block_size(256);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, ctrl(), 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams::default(),
        ImageParams {
            steps: 3,
            max_images: 1,
            retain_images: true,
            ..Default::default()
        },
        GenMode::AutoInterleave,
        32_768,
        etx,
    );
    handle.submit(req).unwrap();

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
        !rejected,
        "auto-interleave should not reserve 32768 tokens up front"
    );
    assert_eq!(finished, Some(FinishReason::Eos));
}

#[test]
fn commit_eos_finishes_without_spending_remaining_budget() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let control = ControlTokens {
        start_of_image: 2222,
        ..ctrl()
    };
    sim.set_commit_token(Some(control.eos[0]));
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let sched = Scheduler::new(executor, control, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3, 2222],
        SamplingParams {
            logit_bias: vec![(2222, 1000.0)],
            ..Default::default()
        },
        ImageParams {
            steps: 3,
            max_images: 3,
            ..Default::default()
        },
        GenMode::AutoInterleave,
        40,
        etx,
    );
    handle.submit(req).unwrap();

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
        let (etx, erx) = tokio::sync::mpsc::unbounded_channel();
        handle
            .submit(GenerateRequest::new(
                RequestId(id),
                vec![1, 2, 3],
                SamplingParams::default(),
                ImageParams::default(),
                GenMode::Text,
                16,
                etx,
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
    let out = run_requests(2, SchedulingPolicy::Fcfs, &[(GenMode::Text, 3)]);
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

    let (atx, _arx) = tokio::sync::mpsc::unbounded_channel();
    let mut a = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams::default(),
        ImageParams::default(),
        GenMode::Text,
        6,
        atx,
    );
    a.priority = 10;
    sched.submit_for_test(a);
    for _ in 0..4 {
        sched.step();
    }

    let (btx, _brx) = tokio::sync::mpsc::unbounded_channel();
    let mut b = GenerateRequest::new(
        RequestId(2),
        vec![4, 5, 6],
        SamplingParams::default(),
        ImageParams::default(),
        GenMode::Text,
        6,
        btx,
    );
    b.priority = 0;
    sched.submit_for_test(b);
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

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let mut req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams::default(),
        ImageParams::default(),
        GenMode::Text,
        64,
        etx,
    );
    req.grammar = Some(GrammarSpec::Choice(vec![
        vec![2000, 2001, 2002],
        vec![3000],
    ]));
    handle.submit(req).unwrap();

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

/// The auto-interleave literal trigger (ThinkMorph's textual visual-thinking
/// signal): when the generated round text completes `image_start_ids`, the
/// next image begins immediately — without waiting for EOS or BAGEL's
/// <|vision_start|> token. The sim never emits EOS here (huge text_len), so
/// images can only come from the literal trigger.
#[test]
fn auto_interleave_literal_trigger_starts_images() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000); // never EOS on its own
    sim.set_pipeline_depth(2);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    // Sim emits 1000 + ((id*7 + n) % 5000) for request id=1: 1007, 1008, 1009…
    // After an image commits, the sim resets and the round repeats from 1007.
    let trig = ControlTokens {
        image_start_ids: vec![1008, 1009],
        ..ControlTokens::default()
    };
    let sched = Scheduler::new(executor, trig, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams::default(),
        ImageParams {
            steps: 3,
            max_images: 2,
            ..Default::default()
        },
        GenMode::AutoInterleave,
        40,
        etx,
    );
    handle.submit(req).unwrap();

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
/// when interleave requests draw. A huge positive bias makes the very first
/// sampled token the image trigger (image before any text); a huge negative
/// bias keeps the pathway shut (the sim never EOSes here, so no image can
/// appear any other way).
#[test]
fn image_start_logit_bias_steers_interleave() {
    let run = |bias: f32| -> (usize, usize, bool) {
        let mut sim = SimEngine::new();
        sim.set_text_len(1_000_000); // never EOS on its own
        let executor = Box::new(SimExecutor::new(Box::new(sim)));
        // an image-start token inside the sim's vocab
        let trig = ControlTokens {
            start_of_image: 2222,
            ..ControlTokens::default()
        };
        let sched = Scheduler::new(executor, trig, 32);
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let jh = thread::spawn(move || sched.run(rx));

        let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
        let mut req = GenerateRequest::new(
            RequestId(1),
            vec![1, 2, 3],
            SamplingParams {
                logit_bias: vec![(2222, bias)],
                ..Default::default()
            },
            ImageParams {
                steps: 3,
                max_images: 1,
                ..Default::default()
            },
            GenMode::AutoInterleave,
            12,
            etx,
        );
        req.sampling.seed = None;
        handle.submit(req).unwrap();

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

/// Native assistant prefixes can end at the image boundary. AutoInterleave must
/// honor that prefilled control token immediately after prefill instead of
/// waiting for the model to sample another image-start token.
#[test]
fn auto_interleave_prefilled_image_start_begins_without_text() {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let trig = ControlTokens {
        start_of_image: 2222,
        ..ControlTokens::default()
    };
    let sched = Scheduler::new(executor, trig, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenerateRequest::new(
        RequestId(1),
        vec![10, 11, 2222],
        SamplingParams::default(),
        ImageParams {
            steps: 3,
            max_images: 1,
            ..Default::default()
        },
        GenMode::AutoInterleave,
        8,
        etx,
    );
    handle.submit(req).unwrap();

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
fn understanding_interleave_commits_existing_image_context_at_round_close() {
    let mut sim = SimEngine::new();
    sim.set_text_len(2);
    let executor = Box::new(SimExecutor::new(Box::new(sim)));
    let control = ControlTokens {
        image_start_ids: vec![1007],
        ..ctrl()
    };
    let sched = Scheduler::new(executor, control, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let mut req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams::default(),
        ImageParams {
            steps: 2,
            max_images: 1,
            ..Default::default()
        },
        GenMode::InterleaveUnd,
        40,
        etx,
    );
    req.mm_items = vec![MmItem {
        hash: 7,
        position: 0,
        num_tokens: 1,
        b64: "image".into(),
    }];
    handle.submit(req).unwrap();

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
    let trig = ControlTokens {
        start_of_image: 2222,
        ..ControlTokens::default()
    };
    let sched = Scheduler::new(executor, trig, 32);
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let jh = thread::spawn(move || sched.run(rx));

    let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3],
        SamplingParams {
            logit_bias: vec![(2222, 1000.0)],
            ..Default::default()
        },
        ImageParams {
            steps: 3,
            max_images: 2,
            ..Default::default()
        },
        GenMode::AutoInterleave,
        24,
        etx,
    );
    handle.submit(req).unwrap();

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
        GenMode::Text,
        GenMode::Image,
        GenMode::AutoInterleave,
        GenMode::Text,
    ];
    for (i, mode) in specs.iter().enumerate() {
        let (etx, erx) = tokio::sync::mpsc::unbounded_channel();
        keep_alive.push(erx);
        let req = GenerateRequest::new(
            RequestId(i as u64 + 1),
            vec![1, 2, 3, 4, 5],
            SamplingParams::default(),
            ImageParams {
                steps: 4,
                max_images: 1,
                ..Default::default()
            },
            *mode,
            16,
            etx,
        );
        sched.submit_for_test(req);
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
        let (etx, erx) = tokio::sync::mpsc::unbounded_channel();
        keep_alive.push(erx);
        sched.submit_for_test(GenerateRequest::new(
            RequestId(i + 1),
            vec![1, 2, 3, 4, 5],
            SamplingParams::default(),
            ImageParams {
                steps: 4,
                ..Default::default()
            },
            GenMode::Text,
            16,
            etx,
        ));
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
    let (etx, erx) = tokio::sync::mpsc::unbounded_channel();
    let _keep = erx; // keep the receiver alive (a dropped one cancels)
    sched.submit_for_test(GenerateRequest::new(
        RequestId(1),
        vec![1, 2, 3, 4, 5],
        SamplingParams::default(),
        ImageParams {
            steps: 4,
            ..Default::default()
        },
        GenMode::Text,
        8,
        etx,
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
    assert!(t.program_id.0 >= 1, "program id assigned");
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
