#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Block-diffusion text generation on the GPU-free simulator.
//!
//! A canvas-generating request prefills its prompt without sampling, then
//! generates text one block at a time: token-denoising steps run on the
//! worker-resident canvas of the block until a step stops it and reports the
//! block's tokens, which are published and, unless they end the request,
//! committed to the context by a causal prefill before the next block
//! starts. The simulator stops block `b` of request `r` at step
//! `sim_canvas_stop_step(r, b, max_steps)` and fills it with the synthetic
//! text `sim_text_token(r, i)` followed by EOS, so the expected output is
//! known exactly. Tests observe calls at the executor boundary, the protocol
//! the worker consumes.

use std::thread;
use std::time::{Duration, Instant};

use uniserve_core::{
    CachePolicy, CanvasSampling, EngineCoreOutput, FinishReason, GenerationConstraint,
    GenerationRequest, ImageGenerationConfig, ImageParams, MultimodalInputs, RejectionKind,
    Request, RequestId, SamplingParams,
};
use uniserve_engine::{
    BatchEvent, EngineHandle, EventRx, ExecutionBatch, Scheduler, SchedulerConfig, SimEngine,
    SimExecutor, SpecialTokenIds, sim_canvas_stop_step, sim_text_token,
};
use uniserve_worker_ipc::{Call, CallKind, ForwardBatch, ForwardMode};

/// Tokens of one block.
const CANVAS: u32 = 16;
/// Denoising steps a block runs at most.
const MAX_STEPS: u32 = 4;

/// A block-diffusion worker with 16-token KV pages whose synthetic text
/// holds `text_len` tokens before EOS.
fn worker(text_len: usize) -> SimEngine {
    let mut sim = SimEngine::new();
    sim.mut_info_for_test().supported_calls = vec![
        CallKind::Forward(ForwardMode::Prefill),
        CallKind::Forward(ForwardMode::TokenDenoising),
    ];
    sim.set_page_tokens(16);
    sim.set_text_len(text_len);
    sim
}

fn canvas_request(request_id: u64, prompt_len: u32, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(request_id),
        prompt_token_ids: (0..prompt_len).map(|index| 1_000 + index).collect(),
        negative_prompt_token_ids: Vec::new(),
        multimodal_inputs: MultimodalInputs::default(),
        constraint: GenerationConstraint::Default,
        sampling: SamplingParams {
            seed: Some(7),
            ..SamplingParams::default()
        },
        image: ImageParams::default(),
        max_und_tokens: max_tokens,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy::default(),
        include_stop_token: false,
        image_generation: ImageGenerationConfig::default(),
        readout: Vec::new(),
        canvas: Some(CanvasSampling {
            canvas_length: CANVAS,
            max_steps: MAX_STEPS,
            entropy_bound: 0.1,
            t_min: 0.4,
            t_max: 0.8,
            confidence_threshold: 0.005,
            stability_threshold: 1,
        }),
    }
}

/// A block-diffusion scheduler running on its own thread, with the batches
/// its executor accepts.
struct Running {
    handle: EngineHandle,
    batches: crossbeam_channel::Receiver<BatchEvent>,
    thread: thread::JoinHandle<bool>,
}

impl Running {
    fn start(sim: SimEngine, config: SchedulerConfig) -> Self {
        let mut executor = SimExecutor::new(sim);
        let batches = executor.observe();
        let scheduler =
            Scheduler::with_config(Box::new(executor), SpecialTokenIds::default(), config).unwrap();
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let thread = thread::spawn(move || scheduler.run(rx));
        Self {
            handle,
            batches,
            thread,
        }
    }

    fn submit(&self, request: GenerationRequest) -> EventRx {
        self.handle
            .submit(Request::BlockDiffusion(request))
            .unwrap()
    }

    /// Stops the scheduler and returns every batch it submitted, in order.
    fn stop(self) -> Vec<ExecutionBatch> {
        self.handle.shutdown();
        assert!(!self.thread.join().unwrap(), "the scheduler failed");
        self.batches
            .try_iter()
            .filter_map(|event| match event {
                BatchEvent::Submitted(batch) => Some(batch),
                BatchEvent::Resolved { .. } => None,
            })
            .collect()
    }
}

/// Collects one request's events through its terminal event.
fn events(mut receiver: EventRx) -> Vec<EngineCoreOutput> {
    let mut events = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        match receiver.try_recv() {
            Ok(event) => {
                let terminal = matches!(
                    event,
                    EngineCoreOutput::Finished { .. }
                        | EngineCoreOutput::Rejected { .. }
                        | EngineCoreOutput::Error { .. }
                );
                events.push(event);
                if terminal {
                    return events;
                }
            }
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    panic!("request did not finish: {events:?}");
}

/// The token blocks a request published, in order.
fn published_blocks(events: &[EngineCoreOutput]) -> Vec<Vec<u32>> {
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, EngineCoreOutput::TextToken { .. })),
        "canvas text is published by block: {events:?}"
    );
    events
        .iter()
        .filter_map(|event| match event {
            EngineCoreOutput::TextTokens { ids } => Some(ids.clone()),
            _ => None,
        })
        .collect()
}

fn finish_of(events: &[EngineCoreOutput]) -> (FinishReason, usize) {
    match events.last() {
        Some(EngineCoreOutput::Finished {
            reason,
            completion_tokens,
            ..
        }) => (reason.clone(), *completion_tokens),
        other => panic!("request ended with {other:?}"),
    }
}

/// The calls of one request in submission order.
fn calls_of(batches: &[ExecutionBatch], request: u64) -> Vec<(Call, ForwardBatch)> {
    batches
        .iter()
        .flat_map(|batch| &batch.requests)
        .filter(|(call, _)| call.request_key.request_id == RequestId(request))
        .map(|(call, placement)| (call.clone(), placement.forward.clone()))
        .collect()
}

fn text(request: u64, range: std::ops::Range<usize>) -> Vec<u32> {
    range
        .map(|index| sim_text_token(RequestId(request), index))
        .collect()
}

/// Each block is denoised by one step per call on its resident canvas over
/// the whole context, published whole once a step stops it, and committed as
/// a causal prefill that the next block reads; the block holding EOS ends
/// the request there, uncommitted.
#[test]
fn blocks_are_denoised_published_and_committed_until_eos() {
    let running = Running::start(worker(40), SchedulerConfig::default());
    let answer = events(running.submit(canvas_request(1, 24, 1_000)));
    let batches = running.stop();

    assert_eq!(
        published_blocks(&answer),
        vec![text(1, 0..16), text(1, 16..32), text(1, 32..40)]
    );
    // EOS itself is not output, but it is the 41st generated token.
    assert_eq!(finish_of(&answer), (FinishReason::Eos, 41));

    let calls = calls_of(&batches, 1);
    let prompt: Vec<_> = calls
        .iter()
        .take_while(|(call, _)| call.code == CallKind::Forward(ForwardMode::Prefill))
        .collect();
    assert!(
        prompt.iter().all(|(call, _)| call.token_output.is_none()),
        "a canvas prompt samples no token"
    );
    assert_eq!(
        prompt
            .iter()
            .flat_map(|(call, _)| call.input_token_ids.clone())
            .collect::<Vec<_>>(),
        (0..24).map(|index| 1_000 + index).collect::<Vec<_>>()
    );

    // After the prompt, each block runs its steps from zero until the step
    // that stops it, then commits its tokens, except the last block.
    let mut expected = Vec::new();
    for block in 0..3u32 {
        let context = 24 + block * CANVAS;
        for step in 0..=sim_canvas_stop_step(RequestId(1), block, MAX_STEPS) {
            expected.push((
                CallKind::Forward(ForwardMode::TokenDenoising),
                Some((block, step)),
                Vec::new(),
                context,
                vec![CANVAS],
                vec![context + CANVAS],
                vec![false],
            ));
        }
        if block < 2 {
            let tokens = text(
                1,
                (block * CANVAS) as usize..((block + 1) * CANVAS) as usize,
            );
            expected.push((
                CallKind::Forward(ForwardMode::Prefill),
                None,
                tokens,
                context,
                vec![CANVAS],
                vec![context + CANVAS],
                vec![true],
            ));
        }
    }
    let generation: Vec<_> = calls[prompt.len()..]
        .iter()
        .map(|(call, forward)| {
            (
                call.code,
                call.canvas.map(|step| (step.block, step.step)),
                call.input_token_ids.clone(),
                call.coordinates.kv_visible_len,
                forward.query_lens.clone(),
                forward.seq_lens.clone(),
                forward.write_kv.clone(),
            )
        })
        .collect();
    assert_eq!(generation, expected);
    assert!(
        calls[prompt.len()..]
            .iter()
            .all(|(call, _)| call.token_output.is_none()),
        "block-diffusion generation samples no token by decode"
    );
}

/// The completion limit truncates the block that reaches it, and no block
/// follows.
#[test]
fn the_completion_limit_truncates_inside_a_block() {
    let running = Running::start(worker(1_000), SchedulerConfig::default());
    let answer = events(running.submit(canvas_request(2, 8, 20)));
    let batches = running.stop();

    assert_eq!(
        published_blocks(&answer),
        vec![text(2, 0..16), text(2, 16..20)]
    );
    assert_eq!(finish_of(&answer), (FinishReason::MaxTokens, 20));
    let commits = calls_of(&batches, 2)
        .into_iter()
        .filter(|(call, _)| {
            call.code == CallKind::Forward(ForwardMode::Prefill)
                && call.coordinates.kv_visible_len > 0
        })
        .count();
    assert_eq!(commits, 1, "only the first block is committed");
}

/// A stop token inside a block ends the request there; it is published only
/// when the request includes stop tokens in its output.
#[test]
fn a_stop_token_ends_the_request_inside_its_block() {
    let running = Running::start(worker(1_000), SchedulerConfig::default());
    let mut excluded = canvas_request(3, 8, 1_000);
    excluded.stop_token_ids = vec![sim_text_token(RequestId(3), 21)];
    let mut included = canvas_request(4, 8, 1_000);
    included.stop_token_ids = vec![sim_text_token(RequestId(4), 5)];
    included.include_stop_token = true;
    let excluded_answer = events(running.submit(excluded));
    let included_answer = events(running.submit(included));
    running.stop();

    assert_eq!(
        published_blocks(&excluded_answer),
        vec![text(3, 0..16), text(3, 16..21)]
    );
    assert_eq!(finish_of(&excluded_answer).0, FinishReason::Stop);
    assert_eq!(published_blocks(&included_answer), vec![text(4, 0..6)]);
    assert_eq!(finish_of(&included_answer).0, FinishReason::Stop);
}

/// A canvas step and a block commit each need a whole canvas of one step's
/// token budget, so a canvas longer than the budget is refused at
/// submission, as is a canvas-generating request without a seed.
#[test]
fn unschedulable_or_unseeded_canvases_are_refused() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 8,
        long_prefill_threshold: 8,
        ..SchedulerConfig::default()
    };
    let running = Running::start(worker(40), config);
    let oversized = events(running.submit(canvas_request(5, 8, 64)));
    let mut unseeded_request = canvas_request(6, 8, 64);
    unseeded_request.sampling.seed = None;
    let unseeded = events(running.submit(unseeded_request));
    running.stop();

    for answer in [oversized, unseeded] {
        assert!(
            matches!(
                answer.last(),
                Some(EngineCoreOutput::Rejected {
                    kind: RejectionKind::Invalid,
                    ..
                })
            ),
            "{answer:?}"
        );
    }
}
