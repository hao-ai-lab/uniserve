//! Demonstrates worker startup, capability negotiation, and execution over the
//! worker IPC transport.
//!
//! One CPU rank is launched with `stub: true`, so the Python worker runs its
//! deterministic test model without weights. The example prints the reported
//! worker info, drives `Scheduler::step` on the main thread instead of a
//! scheduler thread, and prints PASS when every request finished. Unlike
//! `worker_run_smoke`, it exits successfully whether or not they finished.
//!
//! Set `PYTHONPATH` to the repository root before running the example.

use std::collections::HashMap;

use uniserve_core::EngineCoreOutput;
use uniserve_core::{
    GenerationConstraint, GenerationRequest, ImageGenerationConfig, ImageParams, ImageTrigger,
    RequestId, SamplingParams,
};
use uniserve_engine::{
    AttentionBackend, Executor, Scheduler, SpecialTokenIds, WorkerExecutor, WorkerGroup, WorkerId,
    WorkerProcessArgs,
};

fn main() -> anyhow::Result<()> {
    // The handle keeps the sender alive, so `Scheduler::step` never sees a
    // disconnected channel while the loop below runs.
    let (command_tx, commands) = crossbeam_channel::unbounded();
    let handle = uniserve_engine::EngineHandle::new(command_tx);

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let worker_config = WorkerProcessArgs {
        worker_id: "local".into(),
        stub: true,
        ..WorkerProcessArgs::default()
    };
    let engine = WorkerGroup::spawn(WorkerProcessArgs {
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
        // Allows two physical runs in flight on the rank at once.
        queue_depth: 2,
        req_slot_cap: 1 << 20,
        resp_slot_cap: 8 << 20,
        kv_token_capacity: None,
        block_size: Some(256),
        max_batch_calls: 32,
        max_batch_tokens: 8192,
        attention_backend: AttentionBackend::Auto,
        transfer: Default::default(),
        ..worker_config
    })?;
    let engine =
        WorkerExecutor::try_new(vec![(WorkerId("local".into()), engine)], Default::default())?;
    println!("info from worker: {:?}", engine.info());

    let ctrl = SpecialTokenIds {
        ..SpecialTokenIds::default()
    };
    let mut sched = Scheduler::new(Box::new(engine), ctrl, 32)?;

    let mut rxs: HashMap<RequestId, (&str, uniserve_engine::EventRx)> = HashMap::new();
    let mut reqs = Vec::new();
    let mk = |id: u64, constraint: GenerationConstraint| generation_request(id, constraint);
    reqs.push((mk(1, GenerationConstraint::UndOnly), "text"));
    reqs.push((mk(2, GenerationConstraint::UndOnly), "text"));
    reqs.push((mk(3, GenerationConstraint::GenOnly), "image"));
    for (request, kind) in reqs {
        let id = request.request_id;
        let rx = handle.submit(request)?;
        rxs.insert(id, (kind, rx));
    }

    // Drive the scheduler on this thread instead of `Scheduler::run`. A step
    // returns `false` when it makes no progress with no batch in flight, as
    // after the requests finish, or when its bounded wait for an in-flight
    // result times out; either ends the loop. The iteration bound caps a run
    // that keeps progressing.
    for _ in 0..400 {
        if !sched.step(&commands) {
            break;
        }
    }

    // Per request: text tokens, images, and whether `Finished` arrived.
    let mut counts: HashMap<RequestId, (usize, usize, bool)> = HashMap::new();
    for (id, (_k, rx)) in rxs.iter_mut() {
        while let Ok(ev) = rx.try_recv() {
            let e = counts.entry(*id).or_insert((0, 0, false));
            match ev {
                EngineCoreOutput::TextToken { .. } => e.0 += 1,
                EngineCoreOutput::ImageDone { .. } => e.1 += 1,
                EngineCoreOutput::Finished { .. } => e.2 = true,
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
    println!(
        "\nTWO-PROCESS IPC SMOKE: {}",
        if ok { "PASS" } else { "FAIL" }
    );
    Ok(())
}

fn generation_request(id: u64, constraint: GenerationConstraint) -> GenerationRequest {
    let policy = ImageGenerationConfig {
        trigger: ImageTrigger::Token { token_id: 1 },
        requires_text_for_image: false,
        ..ImageGenerationConfig::default()
    };
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: vec![1, 2, 3],
        multimodal_inputs: Default::default(),
        negative_prompt_token_ids: Vec::new(),
        constraint,
        sampling: SamplingParams::default(),
        image: ImageParams {
            steps: 4,
            height: 128,
            width: 128,
            ..Default::default()
        },
        max_und_tokens: 20,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        image_generation: policy,
    }
}
