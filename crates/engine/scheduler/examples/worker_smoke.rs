//! Two-process IPC smoke test: spawn the (stub) Python worker, handshake caps,
//! and drive a few requests through the scheduler over the shared-memory ring.
use std::collections::HashMap;

use uniserve_core::{GenMode, ImageParams, RequestId, SamplingParams};
use uniserve_engine_api::{GenEvent, GenerateRequest};
use uniserve_executor::Executor;
use uniserve_scheduler::{ControlTokens, Scheduler};
use uniserve_worker_ipc::{UniprocExecutor, WorkerLaunchConfig};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    // pipeline_depth=2 exercises the descriptor ring with batches in flight.
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
    println!("caps from worker: {:?}", engine.caps());

    let ctrl = ControlTokens {
        image_start_ids: vec![],
        ..ControlTokens::default()
    };
    let mut sched = Scheduler::new(Box::new(engine), ctrl, 32);

    let mut rxs: HashMap<RequestId, (&str, tokio::sync::mpsc::UnboundedReceiver<GenEvent>)> =
        HashMap::new();
    // we drive step directly here instead of the run thread
    let mut reqs = Vec::new();
    let mk = |id: u64,
              mode: GenMode,
              kind: &'static str,
              rxs: &mut HashMap<RequestId, (&'static str, _)>| {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        rxs.insert(RequestId(id), (kind, rx));
        GenerateRequest::new(
            RequestId(id),
            vec![1, 2, 3],
            SamplingParams::default(),
            ImageParams {
                steps: 4,
                height: 128,
                width: 128,
                ..Default::default()
            },
            mode,
            20,
            tx,
        )
    };
    reqs.push(mk(1, GenMode::Text, "text", &mut rxs));
    reqs.push(mk(2, GenMode::Text, "text", &mut rxs));
    reqs.push(mk(3, GenMode::Image, "image", &mut rxs));
    for r in reqs {
        sched.submit_for_test(r);
    }
    for _ in 0..400 {
        if !sched.step() {
            break;
        }
    }

    let mut counts: HashMap<RequestId, (usize, usize, bool)> = HashMap::new();
    for (id, (_k, rx)) in rxs.iter_mut() {
        while let Ok(ev) = rx.try_recv() {
            let e = counts.entry(*id).or_insert((0, 0, false));
            match ev {
                GenEvent::TextToken { .. } => e.0 += 1,
                GenEvent::ImageDone { .. } => e.1 += 1,
                GenEvent::Finished { .. } => e.2 = true,
                _ => {}
            }
        }
    }
    let mut ok = true;
    for (id, (k, _)) in &rxs {
        let (toks, imgs, fin) = counts.get(id).copied().unwrap_or((0, 0, false));
        println!(
            "req {:?} [{}]: text={} imgs={} finished={}",
            id, k, toks, imgs, fin
        );
        if !fin {
            ok = false;
        }
    }
    println!("peak ops in a batch: {}", sched.peak_ops_in_batch);
    println!(
        "\nTWO-PROCESS IPC SMOKE: {}",
        if ok { "PASS" } else { "FAIL" }
    );
    Ok(())
}
