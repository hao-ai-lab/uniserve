//! Event-driven `run` smoke test against the stub Python worker.
//!
//! Unlike `worker_smoke` (which drives `step` directly), this exercises the
//! production owner-thread reactor `EngineLoop::run` over the iceoryx2
//! event-driven boundary: the worker parks on its request listener, the
//! scheduler parks on {result, command, death}, and the command ingress wakes
//! the park via the executor's command waker. It also exercises the idle path
//! (submit after the engine has gone idle and parked) and graceful shutdown.
//!
//! Run with PYTHONPATH set to the repo root.
use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};

use uniserve_core::Event;
use uniserve_core::{
    ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
    GenerationRequest, GenerationResourceBounds, ImageParams, RequestId, SamplingParams,
    UndVisibility,
};
use uniserve_engine::{
    Command, ControlTokens, EngineHandle, EngineLoop, UniprocExecutor, WorkerProcessArgs,
};

type Rxs = HashMap<RequestId, uniserve_engine::EventRx>;

fn submit_text(handle: &EngineHandle, rxs: &mut Rxs, id: u64) -> anyhow::Result<()> {
    let constraint = GenerationConstraint::UndOnly;
    let policy = GenerationPolicyDescriptor::default();
    let request = GenerationRequest {
        request_id: RequestId(id),
        context: vec![ContextSegment::UndTokens {
            token_ids: vec![1, 2, 3],
            visibility: UndVisibility::Internal,
        }],
        negative_context: Vec::new(),
        constraint,
        behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_und_tokens: 16,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        policy,
        resources: GenerationResourceBounds {
            context_tokens: 3,
            max_kv_tokens: 19,
            ..GenerationResourceBounds::default()
        },
    };
    let rx = handle.submit(request).map_err(|e| anyhow::anyhow!(e))?;
    rxs.insert(RequestId(id), rx);
    Ok(())
}

/// Drain events until every request in `ids` has emitted `Finished`, or until
/// `deadline`. Returns the set that finished.
fn await_finished(rxs: &mut Rxs, ids: &[RequestId], deadline: Instant) -> Vec<RequestId> {
    let mut finished = Vec::new();
    while finished.len() < ids.len() && Instant::now() < deadline {
        let mut saw_any = false;
        for id in ids {
            if finished.contains(id) {
                continue;
            }
            if let Some(rx) = rxs.get_mut(id) {
                while let Ok(ev) = rx.try_recv() {
                    saw_any = true;
                    if matches!(ev, Event::Finished { .. }) {
                        finished.push(*id);
                        break;
                    }
                }
            }
        }
        if !saw_any {
            thread::sleep(Duration::from_millis(2));
        }
    }
    finished
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let worker_config = WorkerProcessArgs {
        stub: true,
        cuda_graph: false,
        ..WorkerProcessArgs::default()
    };
    let engine = UniprocExecutor::spawn(uniserve_engine::WorkerProcessArgs {
        python: "python3".into(),
        model: String::new(),
        device: "cpu".into(),
        world_size: 1,
        pipeline_depth: 2,
        req_slot_cap: 1 << 20,
        resp_slot_cap: 8 << 20,
        kv_token_capacity: None,
        block_size: 256,
        max_batch_operations: 32,
        max_batch_tokens: 8192,
        attention_backend: uniserve_engine::AttentionBackend::Auto,
        supported_ops: uniserve_worker_ipc::OpKind::ALL.to_vec(),
        transfer_backend: uniserve_engine::TransferBackend::Inproc,
        ..worker_config
    })?;
    let waker = engine.command_waker();
    let sched = EngineLoop::new(Box::new(engine), ControlTokens::default(), 32);

    let (tx, rx) = crossbeam_channel::unbounded::<Command>();
    let handle = EngineHandle::with_waker(tx, waker);
    let jh = thread::spawn(move || sched.run(rx));

    let mut rxs: Rxs = HashMap::new();
    let mut ok = true;

    // Round 1: three concurrent requests.
    let r1 = [RequestId(1), RequestId(2), RequestId(3)];
    for id in &r1 {
        submit_text(&handle, &mut rxs, id.0)?;
    }
    let done = await_finished(&mut rxs, &r1, Instant::now() + Duration::from_secs(30));
    println!("round 1 finished: {}/{}", done.len(), r1.len());
    ok &= done.len() == r1.len();

    // Let the engine go fully idle and park, then submit again: this only
    // completes promptly if the command waker interrupts the idle park.
    thread::sleep(Duration::from_millis(300));
    let r2 = [RequestId(4)];
    let t_submit = Instant::now();
    submit_text(&handle, &mut rxs, 4)?;
    let done2 = await_finished(&mut rxs, &r2, Instant::now() + Duration::from_secs(30));
    let wake_latency = t_submit.elapsed();
    println!(
        "round 2 (idle-wake) finished: {}/{} in {:?}",
        done2.len(),
        r2.len(),
        wake_latency
    );
    ok &= done2.len() == r2.len();

    handle.shutdown();
    let fatal = jh
        .join()
        .map_err(|_| anyhow::anyhow!("scheduler thread panicked"))?;
    ok &= !fatal;

    println!(
        "\nEVENT-DRIVEN RUN SMOKE: {}",
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
