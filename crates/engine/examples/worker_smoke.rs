//! Demonstrates worker startup, capability negotiation, and shared-memory execution.
use std::collections::HashMap;

use uniserve_core::Event;
use uniserve_core::{
    ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
    GenerationRequest, GenerationResourceBounds, ImageParams, RequestId, SamplingParams,
    TriggerPolicyDescriptor, UndVisibility,
};
use uniserve_engine::{
    AttentionBackend, ControlTokens, EngineLoop, Executor, Worker, WorkerExecutor, WorkerId,
    WorkerProcessArgs,
};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    // pipeline_depth=2 exercises the descriptor ring with batches in flight.
    let worker_config = WorkerProcessArgs {
        worker_id: "local".into(),
        stub: true,
        ..WorkerProcessArgs::default()
    };
    let engine = Worker::spawn(WorkerProcessArgs {
        worker_id: "local".into(),
        python: "python3".into(),
        model: String::new(),
        ranks: uniserve_engine::WorkerConfig::model("cpu", 1, 2).ranks,
        pipeline_depth: 2,
        req_slot_cap: 1 << 20,
        resp_slot_cap: 8 << 20,
        kv_token_capacity: None,
        block_size: 256,
        max_batch_operations: 32,
        max_batch_tokens: 8192,
        attention_backend: AttentionBackend::Auto,
        supported_ops: uniserve_worker_ipc::OpCode::ALL.to_vec(),
        transfer: Default::default(),
        ..worker_config
    })?;
    let engine =
        WorkerExecutor::try_new(vec![(WorkerId("local".into()), engine)], Default::default())?;
    println!("info from worker: {:?}", engine.info());

    let ctrl = ControlTokens {
        ..ControlTokens::default()
    };
    let mut sched = EngineLoop::new(Box::new(engine), ctrl, 32);

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
                Event::TextToken { .. } => e.0 += 1,
                Event::ImageDone { .. } => e.1 += 1,
                Event::Finished { .. } => e.2 = true,
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
