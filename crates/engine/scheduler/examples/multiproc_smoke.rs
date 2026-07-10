//! Multi-rank IPC smoke test: spawn stub Python worker processes — one
//! descriptor ring per rank — behind a `MultiprocExecutor`, drive requests
//! through the scheduler, and exercise correlated control fan-out.
use std::collections::HashMap;

use uniserve_core::{
    ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
    GenerationRequest, GenerationResourceBounds, ImageParams, RequestId, SamplingParams,
    TriggerPolicyDescriptor, UndVisibility,
};
use uniserve_engine_api::GenEvent;
use uniserve_executor::{ControlOp, Executor};
use uniserve_scheduler::{ControlTokens, Scheduler};
use uniserve_worker_ipc::MultiprocExecutor;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let world = 2usize;
    let mut executor = MultiprocExecutor::spawn(
        "python3",
        "",
        "cpu",
        world,
        2,
        1 << 20,
        8 << 20,
        None,
        256,
        "auto",
    )?;
    println!(
        "caps from {} ranks: tp_size={}",
        world,
        executor.caps().rank.tp_size
    );
    assert_eq!(executor.caps().rank.tp_size as usize, world);

    // The collective_rpc shape: per-rank correlated acks over the real rings.
    let acks = executor.control_wait(ControlOp::ResetPrefixCache, None)?;
    println!("control acks: {acks:?}");
    assert_eq!(acks.len(), world);
    assert!(acks.iter().all(|a| a.ok));

    let mut sched = Scheduler::new(Box::new(executor), ControlTokens::default(), 32);

    let mut rxs: HashMap<RequestId, tokio::sync::mpsc::UnboundedReceiver<GenEvent>> =
        HashMap::new();
    for id in 1..=3u64 {
        let constraint = if id == 3 {
            GenerationConstraint::GenOnly
        } else {
            GenerationConstraint::UndOnly
        };
        let request = generation_request(id, constraint);
        rxs.insert(RequestId(id), sched.submit_for_test(request));
    }
    for _ in 0..400 {
        if !sched.step() {
            break;
        }
    }

    let mut ok = true;
    for (id, rx) in rxs.iter_mut() {
        let (mut toks, mut imgs, mut fin) = (0, 0, false);
        while let Ok(ev) = rx.try_recv() {
            match ev {
                GenEvent::TextToken { .. } => toks += 1,
                GenEvent::ImageDone { .. } => imgs += 1,
                GenEvent::Finished { .. } => fin = true,
                _ => {}
            }
        }
        println!("req {id:?}: text={toks} imgs={imgs} finished={fin}");
        if !fin {
            ok = false;
        }
    }
    println!(
        "\nMULTIPROC IPC SMOKE: {}",
        if ok { "PASS" } else { "FAIL" }
    );
    assert!(ok);
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
            ..ImageParams::default()
        },
        max_und_tokens: 20,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        lora_id: None,
        grammar: None,
        cache: Default::default(),
        policy,
        resources: GenerationResourceBounds {
            context_tokens: 3,
            max_kv_tokens: 23,
            ..GenerationResourceBounds::default()
        },
    }
}
