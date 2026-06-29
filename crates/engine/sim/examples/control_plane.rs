//! GPU-free control-plane test in Rust: drive the Scheduler with
//! SimEngine over concurrent text + image requests; check the FSM/events.
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use uniserve_core::{GenMode, ImageParams, RequestId, SamplingParams};
use uniserve_engine_api::{EngineHandle, GenEvent, GenerateRequest, Prompt};
use uniserve_scheduler::{ControlTokens, Scheduler};
use uniserve_sim::SimEngine;
use uniserve_sim::SimExecutor;

fn main() {
    let _ = Prompt::default();
    let ctrl = ControlTokens::default();
    let executor = Box::new(SimExecutor::new(Box::new(SimEngine::new())));
    let sched = Scheduler::new(executor, ctrl, 32);
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(cmd_tx);
    let jh = thread::spawn(move || sched.run(cmd_rx));

    let mut rxs: HashMap<RequestId, (String, tokio::sync::mpsc::UnboundedReceiver<GenEvent>)> =
        HashMap::new();
    let mut next_id = 1u64;
    let mut mk = |mode: GenMode, kind: &str, rxs: &mut HashMap<_, _>| {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let id = RequestId(next_id);
        next_id += 1;
        let req = GenerateRequest::new(
            id,
            vec![1, 2, 3],
            SamplingParams::default(),
            ImageParams {
                steps: 6,
                ..Default::default()
            },
            mode,
            20,
            tx,
        );
        rxs.insert(id, (kind.to_string(), rx));
        req
    };

    for _ in 0..3 {
        handle.submit(mk(GenMode::Text, "text", &mut rxs)).unwrap();
    }
    for _ in 0..2 {
        handle
            .submit(mk(GenMode::Image, "image", &mut rxs))
            .unwrap();
    }

 // collect until every request is Finished
    let mut done = 0usize;
    let total = rxs.len();
    let mut counts: HashMap<RequestId, (String, usize, usize, bool)> = HashMap::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while done < total && std::time::Instant::now() < deadline {
        for (id, (kind, rx)) in rxs.iter_mut() {
            while let Ok(ev) = rx.try_recv() {
                let e = counts.entry(*id).or_insert((kind.clone(), 0, 0, false));
                match ev {
                    GenEvent::TextToken { .. } => e.1 += 1,
                    GenEvent::ImageDone { .. } => e.2 += 1,
                    GenEvent::Finished { .. } if !e.3 => {
                        e.3 = true;
                        done += 1;
                    }
                    _ => {}
                }
            }
        }
        thread::sleep(Duration::from_millis(2));
    }

    let mut ok = done == total;
    for (id, (kind, toks, imgs, fin)) in &counts {
        println!(
            "req {:?} [{}]: text_tokens={} images={} finished={}",
            id, kind, toks, imgs, fin
        );
        if kind == "text" && *toks == 0 {
            ok = false;
        }
        if kind == "image" && *imgs == 0 {
            ok = false;
        }
    }
    handle.shutdown();
    let _ = jh.join();
    println!(
        "\nRUST SIM CONTROL-PLANE: {}",
        if ok { "PASS" } else { "FAIL" }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
