//! Event-driven `run` smoke test against the stub Python worker.
//!
//! Unlike `worker_smoke` (which drives `step` directly), this exercises the
//! production owner-thread reactor `Scheduler::run` over the iceoryx2
//! event-driven boundary: the worker parks on its request listener, the
//! scheduler parks on {result, command, death}, and the command ingress wakes
//! the park via the executor's command waker. It also exercises the idle path
//! (submit after the engine has gone idle and parked) and graceful shutdown.
//!
//! Run with PYTHONPATH set to the repo root.
use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};

use uniserve_core::{GenMode, ImageParams, RequestId, SamplingParams};
use uniserve_engine_api::{Command, EngineHandle, GenEvent, GenerateRequest};
use uniserve_executor::Executor;
use uniserve_scheduler::{ControlTokens, Scheduler};
use uniserve_worker_ipc::{UniprocExecutor, WorkerLaunchConfig};

type Rxs = HashMap<RequestId, tokio::sync::mpsc::UnboundedReceiver<GenEvent>>;

fn submit_text(handle: &EngineHandle, rxs: &mut Rxs, id: u64) -> anyhow::Result<()> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    rxs.insert(RequestId(id), rx);
    let req = GenerateRequest::new(
        RequestId(id),
        vec![1, 2, 3],
        SamplingParams::default(),
        ImageParams::default(),
        GenMode::Text,
        16,
        tx,
    );
    handle.submit(req).map_err(|e| anyhow::anyhow!(e))
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
                    if matches!(ev, GenEvent::Finished { .. }) {
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

    let worker_config = WorkerLaunchConfig {
        stub: true,
        ..WorkerLaunchConfig::default()
    };
    let engine = UniprocExecutor::spawn_with_config(
        "python3",
        "",
        "cpu",
        2,
        1 << 20,
        8 << 20,
        None,
        256,
        "auto",
        &worker_config,
    )?;
    let event_driven = engine.event_driven();
    println!("event_driven = {event_driven}");
    let waker = engine.command_waker();
    let sched = Scheduler::new(Box::new(engine), ControlTokens::default(), 32);

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
