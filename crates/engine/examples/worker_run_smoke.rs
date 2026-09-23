//! Exercises the event-driven engine loop against a Python worker process.
//!
//! The engine parks on result, command, and worker-death events while the worker
//! parks on request notifications. The flow includes idle wakeup and graceful
//! shutdown.
//!
//! Set `PYTHONPATH` to the repository root before running the example.
use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};

use uniserve_core::EngineCoreOutput;
use uniserve_core::{
    GenerationConstraint, GenerationRequest, ImageGenerationConfig, ImageParams, RequestId,
    SamplingParams,
};
use uniserve_engine::{
    Command, EngineHandle, Scheduler, SpecialTokenIds, WorkerExecutor, WorkerGroup, WorkerId,
    WorkerProcessArgs,
};

type Rxs = HashMap<RequestId, uniserve_engine::EventRx>;

fn submit_text(handle: &EngineHandle, rxs: &mut Rxs, id: u64) -> anyhow::Result<()> {
    let constraint = GenerationConstraint::UndOnly;
    let policy = ImageGenerationConfig::default();
    let request = GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: vec![1, 2, 3],
        multimodal_inputs: Default::default(),
        negative_prompt_token_ids: Vec::new(),
        constraint,

        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_und_tokens: 16,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        image_generation: policy,
    };
    let rx = handle.submit(request).map_err(|e| anyhow::anyhow!(e))?;
    rxs.insert(RequestId(id), rx);
    Ok(())
}

/// Drains events until every request in `ids` has emitted `Finished`, or until
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
                    if matches!(ev, EngineCoreOutput::Finished { .. }) {
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
        worker_id: "local".into(),
        stub: true,
        ..WorkerProcessArgs::default()
    };
    let engine = WorkerGroup::spawn(uniserve_engine::WorkerProcessArgs {
        worker_id: "local".into(),
        python: "python3".into(),
        model: String::new(),
        ranks: uniserve_engine::WorkerConfig::placed(
            &["localhost".to_owned()],
            "cpu",
            1,
            2,
            uniserve_engine::WorkerConfig::single_component(uniserve_engine::DEFAULT_COMPONENT, 1),
        )
        .ranks,
        queue_depth: 2,
        req_slot_cap: 1 << 20,
        resp_slot_cap: 8 << 20,
        kv_token_capacity: None,
        block_size: 256,
        max_batch_calls: 32,
        max_batch_tokens: 8192,
        attention_backend: uniserve_engine::AttentionBackend::Auto,
        transfer: Default::default(),
        ..worker_config
    })?;
    let engine =
        WorkerExecutor::try_new(vec![(WorkerId("local".into()), engine)], Default::default())?;
    let waker = engine.command_waker();
    let sched = Scheduler::new(Box::new(engine), SpecialTokenIds::default(), 32)?;

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
