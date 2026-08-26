//! Two-process IPC smoke test: spawn the (stub) Python worker, handshake caps,
//! and drive a few requests through the scheduler over the shared-memory ring.
use std::collections::HashMap;

use uniserve_core::GenEvent;
use uniserve_core::{
    ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
    GenerationRequest, GenerationResourceBounds, ImageParams, RequestId, SamplingParams,
    TriggerPolicyDescriptor, UndVisibility,
};
use uniserve_engine::executor::Executor;
use uniserve_engine::scheduler::{ControlTokens, Scheduler};
use uniserve_engine::worker::{UniprocExecutor, WorkerLaunchConfig};

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
        32,
        8192,
        "auto",
        &worker_config,
    )?;
    println!("caps from worker: {:?}", engine.caps());

    let ctrl = ControlTokens {
        ..ControlTokens::default()
    };
    let mut sched = Scheduler::new(Box::new(engine), ctrl, 32);

    let mut rxs: HashMap<RequestId, (&str, uniserve_engine::EventRx)> = HashMap::new();
    // we drive step directly here instead of the run thread
    let mut reqs = Vec::new();
    let mk = |id: u64, constraint: GenerationConstraint| generation_request(id, constraint);
    reqs.push((mk(1, GenerationConstraint::UndOnly), "text"));
    reqs.push((mk(2, GenerationConstraint::UndOnly), "text"));
    reqs.push((mk(3, GenerationConstraint::GenOnly), "image"));
    for (request, kind) in reqs {
        let id = request.request_id;
        let rx = sched.submit_for_test(request);
        rxs.insert(id, (kind, rx));
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

fn generation_request(id: u64, constraint: GenerationConstraint) -> GenerationRequest {
    let policy = GenerationPolicyDescriptor {
        trigger: TriggerPolicyDescriptor::Token { token_id: 1 },
        gen_only_start: uniserve_core::GenOnlyStartPolicyDescriptor::Immediate,
        ..GenerationPolicyDescriptor::default()
    };
    GenerationRequest {
        request_id: RequestId(id),
        context: vec![ContextSegment::UndTokens {
            token_ids: vec![1, 2, 3],
            visibility: UndVisibility::Internal,
        }],
        negative_context: Vec::new(),
        constraint,
        behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
        sampling: SamplingParams::default(),
        image: ImageParams {
            steps: 4,
            height: 128,
            width: 128,
            ..Default::default()
        },
        max_und_tokens: 20,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        policy,
        resources: GenerationResourceBounds {
            context_tokens: 3,
            max_kv_tokens: 23,
            ..GenerationResourceBounds::default()
        },
    }
}
