#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Readout requests on a block-diffusion runtime over the GPU-free simulator.
//!
//! A readout prefills its prompt without sampling, denoises its canvas rows
//! in token-denoising passes that read the prompt's KV without writing it,
//! and answers once with every candidate's log-probability before finishing.
//! The simulator reports `sim_candidate_logprob(request, token)` for each
//! candidate, so the expected answer is known exactly. Tests observe calls at
//! the executor boundary, the protocol the worker consumes.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use uniserve_core::{
    CachePolicy, EngineCoreOutput, FinishReason, GenerationConstraint, GenerationRequest,
    ImageEncoderInput, ImageGenerationConfig, ImageIngestStep, ImageInput, ImageParams,
    MultimodalInputs, ReadoutRow, ReadoutSlot, RejectionKind, Request, RequestId, SamplingParams,
};
use uniserve_engine::{
    BatchEvent, EngineHandle, EventRx, ExecutionBatch, Scheduler, SchedulerConfig, SchedulerStats,
    SimEngine, SimExecutor, SpecialTokenIds, sim_candidate_logprob,
};
use uniserve_worker_ipc::{Call, CallKind, ForwardMode, MediaCall};

/// Canvas token that marks an answer slot.
const MASK: u32 = 4;

/// A block-diffusion worker: prompt prefill, canvas denoising and vision
/// encoding, no token decode, with 16-token KV pages.
fn readout_worker() -> SimEngine {
    let mut sim = SimEngine::new();
    sim.mut_info_for_test().supported_calls = vec![
        CallKind::Forward(ForwardMode::Prefill),
        CallKind::Forward(ForwardMode::TokenDenoising),
        CallKind::Media(MediaCall::VisionEncoding),
    ];
    sim.set_page_tokens(16);
    sim
}

/// One canvas row of `length` tokens whose masked slots sit at the listed
/// positions, each reading the candidates listed with it.
fn row(length: usize, slots: &[(u32, &[u32])]) -> ReadoutRow {
    let mut token_ids: Vec<u32> = (0..length as u32).map(|index| 100 + index).collect();
    for (position, _) in slots {
        token_ids[*position as usize] = MASK;
    }
    ReadoutRow {
        token_ids,
        slots: slots
            .iter()
            .map(|(position, candidates)| ReadoutSlot {
                position: *position,
                candidates: candidates.to_vec(),
            })
            .collect(),
    }
}

fn prompt(length: u32, first: u32) -> Vec<u32> {
    (0..length).map(|index| first + index).collect()
}

fn readout_request(
    request_id: u64,
    prompt_token_ids: Vec<u32>,
    readout: Vec<ReadoutRow>,
) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(request_id),
        prompt_token_ids,
        negative_prompt_token_ids: Vec::new(),
        multimodal_inputs: MultimodalInputs::default(),
        constraint: GenerationConstraint::Default,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_und_tokens: 0,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy::default(),
        include_stop_token: false,
        image_generation: ImageGenerationConfig::default(),
        readout,
        canvas: None,
    }
}

/// The answer the simulator gives: every candidate in row, slot, and
/// candidate order.
fn expected_logprobs(request: &GenerationRequest) -> Vec<f32> {
    request
        .readout
        .iter()
        .flat_map(|row| &row.slots)
        .flat_map(|slot| &slot.candidates)
        .map(|&token| sim_candidate_logprob(request.request_id, token))
        .collect()
}

/// A block-diffusion scheduler running on its own thread, with the batches
/// its executor accepts.
struct Running {
    handle: EngineHandle,
    batches: crossbeam_channel::Receiver<BatchEvent>,
    stats: Arc<SchedulerStats>,
    thread: thread::JoinHandle<bool>,
}

impl Running {
    fn start(sim: SimEngine, config: SchedulerConfig) -> Self {
        let mut executor = SimExecutor::new(sim);
        let batches = executor.observe();
        let scheduler =
            Scheduler::with_config(Box::new(executor), SpecialTokenIds::default(), config).unwrap();
        let stats = scheduler.stats_handle();
        let (tx, rx) = crossbeam_channel::unbounded();
        let handle = EngineHandle::new(tx);
        let thread = thread::spawn(move || scheduler.run(rx));
        Self {
            handle,
            batches,
            stats,
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

/// The calls of one request in submission order.
fn calls_of(
    batches: &[ExecutionBatch],
    request: u64,
) -> Vec<(Call, uniserve_worker_ipc::ForwardBatch)> {
    batches
        .iter()
        .flat_map(|batch| &batch.requests)
        .filter(|(call, _)| call.request_key.request_id == RequestId(request))
        .map(|(call, placement)| (call.clone(), placement.forward.clone()))
        .collect()
}

/// Asserts the answer events of a completed readout: one `Readout` with
/// every expected log-probability, then `Completed` with nothing generated.
fn assert_answered(events: &[EngineCoreOutput], request: &GenerationRequest) {
    let answers: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            EngineCoreOutput::Readout { candidate_logprobs } => Some(candidate_logprobs.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(answers, vec![expected_logprobs(request)]);
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, EngineCoreOutput::TextToken { .. })),
        "a readout generated text: {events:?}"
    );
    match events.last() {
        Some(EngineCoreOutput::Finished {
            reason,
            completion_tokens,
            ..
        }) => {
            assert_eq!(*reason, FinishReason::Completed);
            assert_eq!(*completion_tokens, 0);
        }
        other => panic!("readout ended with {other:?}"),
    }
}

/// The prompt is prefilled without sampling, and each canvas row becomes one
/// read-only row of a token-denoising pass over the whole prompt, whose
/// slots name the masked canvas tokens.
#[test]
fn a_readout_prefills_without_sampling_and_answers_every_candidate_once() {
    let running = Running::start(readout_worker(), SchedulerConfig::default());
    let request = readout_request(
        1,
        prompt(40, 1000),
        vec![
            row(32, &[(3, &[7, 8]), (9, &[9])]),
            row(32, &[(4, &[10, 11, 12])]),
        ],
    );
    let answer = events(running.submit(request.clone()));
    let batches = running.stop();
    assert_answered(&answer, &request);

    let calls = calls_of(&batches, 1);
    let (prefills, canvases): (Vec<_>, Vec<_>) = calls
        .iter()
        .partition(|(call, _)| call.code == CallKind::Forward(ForwardMode::Prefill));
    assert!(
        prefills.iter().all(|(call, _)| call.token_output.is_none()),
        "a readout prompt samples no token"
    );
    assert_eq!(
        prefills
            .iter()
            .flat_map(|(call, _)| call.input_token_ids.clone())
            .collect::<Vec<_>>(),
        request.prompt_token_ids
    );

    // Both rows fit one pass: they read the 40 prompt tokens and write
    // nothing, and the slots index the masked tokens of the concatenation.
    assert_eq!(canvases.len(), 1);
    let (canvas, forward) = &canvases[0];
    assert_eq!(canvas.code, CallKind::Forward(ForwardMode::TokenDenoising));
    assert_eq!(canvas.coordinates.kv_visible_len, 40);
    assert_eq!(canvas.coordinates.logical_position, 40);
    assert_eq!(
        canvas.input_token_ids,
        [
            request.readout[0].token_ids.clone(),
            request.readout[1].token_ids.clone()
        ]
        .concat()
    );
    assert_eq!(forward.query_lens, vec![32, 32]);
    assert_eq!(forward.seq_lens, vec![72, 72]);
    assert_eq!(forward.write_kv, vec![false, false]);
    let readout = canvas.readout.as_ref().unwrap();
    assert_eq!(readout.slot_tokens, vec![3, 9, 36]);
    assert!(
        readout
            .slot_tokens
            .iter()
            .all(|&token| canvas.input_token_ids[token as usize] == MASK)
    );
    assert_eq!(readout.candidate_ids, vec![7, 8, 9, 10, 11, 12]);
}

/// Rows join a pass whole while they fit the step's token budget, so a
/// readout whose rows exceed one budget answers after several passes, still
/// in report order.
#[test]
fn canvas_rows_split_into_passes_within_the_step_budget() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 64,
        long_prefill_threshold: 64,
        ..SchedulerConfig::default()
    };
    let running = Running::start(readout_worker(), config);
    let rows = (0..5)
        .map(|index| row(24, &[(index + 1, &[20 + index, 30 + index])]))
        .collect();
    let request = readout_request(2, prompt(40, 1000), rows);
    let answer = events(running.submit(request.clone()));
    let batches = running.stop();
    assert_answered(&answer, &request);

    let passes: Vec<_> = calls_of(&batches, 2)
        .into_iter()
        .filter(|(call, _)| call.readout.is_some())
        .collect();
    assert_eq!(
        passes
            .iter()
            .map(|(_, forward)| forward.query_lens.clone())
            .collect::<Vec<_>>(),
        vec![vec![24, 24], vec![24, 24], vec![24]]
    );
    assert_eq!(
        passes
            .iter()
            .flat_map(|(call, _)| call.input_token_ids.clone())
            .collect::<Vec<_>>(),
        request
            .readout
            .iter()
            .flat_map(|row| row.token_ids.clone())
            .collect::<Vec<_>>()
    );
}

/// A canvas row is never split, so a row longer than the step budget can
/// never run and is refused at submission.
#[test]
fn a_canvas_row_beyond_the_step_budget_is_refused() {
    let config = SchedulerConfig {
        max_num_batched_tokens: 64,
        long_prefill_threshold: 64,
        ..SchedulerConfig::default()
    };
    let running = Running::start(readout_worker(), config);
    let request = readout_request(3, prompt(8, 1000), vec![row(80, &[(1, &[5])])]);
    let answer = events(running.submit(request));
    running.stop();
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

/// A later readout of the same state reuses the prompt pages an earlier one
/// cached and prefills only its own suffix.
#[test]
fn readouts_of_one_state_reuse_the_shared_prompt_pages() {
    let running = Running::start(readout_worker(), SchedulerConfig::default());
    let shared = prompt(48, 1000);
    let first = readout_request(
        4,
        [shared.clone(), vec![5, 6, 7, 8, 9]].concat(),
        vec![row(16, &[(2, &[40, 41])])],
    );
    assert_answered(&events(running.submit(first.clone())), &first);
    let hits_before = running.stats.prefix.hit_tokens.load(Ordering::Relaxed);

    let second = readout_request(
        5,
        [shared, vec![15, 16, 17, 18, 19, 20, 21]].concat(),
        vec![row(16, &[(2, &[40, 41])])],
    );
    assert_answered(&events(running.submit(second.clone())), &second);
    let hits = running.stats.prefix.hit_tokens.load(Ordering::Relaxed) - hits_before;
    let batches = running.stop();

    assert_eq!(hits, 48);
    let prefills: Vec<_> = calls_of(&batches, 5)
        .into_iter()
        .filter(|(call, _)| call.code == CallKind::Forward(ForwardMode::Prefill))
        .collect();
    assert_eq!(prefills.len(), 1);
    assert_eq!(prefills[0].0.coordinates.kv_visible_len, 48);
    assert_eq!(
        prefills[0].0.input_token_ids,
        vec![15, 16, 17, 18, 19, 20, 21]
    );
}

/// An image in the prompt is encoded and written into KV before the canvas
/// reads the prompt, whose positions then include the image's.
#[test]
fn an_image_readout_reads_the_encoded_image_with_its_prompt() {
    let running = Running::start(readout_worker(), SchedulerConfig::default());
    let mut request = readout_request(6, prompt(16, 1000), vec![row(16, &[(1, &[50, 51, 52])])]);
    request.multimodal_inputs.images.push(ImageInput {
        hash: 99,
        b64: "aW1hZ2U=".into(),
        position: 10,
        num_positions: 9,
        encoders: vec![ImageEncoderInput {
            encoder: ImageIngestStep::VitEncode,
            num_kv_tokens: Some(9),
            max_kv_tokens: None,
        }],
    });
    let answer = events(running.submit(request.clone()));
    let batches = running.stop();
    assert_answered(&answer, &request);

    let calls = calls_of(&batches, 6);
    assert!(
        calls
            .iter()
            .any(|(call, _)| call.code == CallKind::Media(MediaCall::VisionEncoding))
    );
    let (canvas, forward) = calls
        .iter()
        .find(|(call, _)| call.readout.is_some())
        .unwrap();
    assert_eq!(canvas.coordinates.logical_position, 25);
    assert_eq!(canvas.coordinates.kv_visible_len, 25);
    assert_eq!(forward.seq_lens, vec![41]);
}

/// A ready canvas pass takes the step before another request's remaining
/// prefill: completing a readout releases its KV.
#[test]
fn a_ready_canvas_pass_runs_before_waiting_prefill() {
    let mut sim = readout_worker();
    sim.set_queue_depth(1);
    sim.set_results_on_wait(true);
    let config = SchedulerConfig {
        max_num_batched_tokens: 64,
        long_prefill_threshold: 64,
        ..SchedulerConfig::default()
    };
    // Both requests are queued before the scheduler starts, so the first
    // step prefills all of the short prompt and the start of the long one.
    let mut executor = SimExecutor::new(sim);
    let batches = executor.observe();
    let scheduler =
        Scheduler::with_config(Box::new(executor), SpecialTokenIds::default(), config).unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let short = readout_request(7, prompt(16, 1000), vec![row(32, &[(1, &[60])])]);
    let long = readout_request(8, prompt(200, 3000), vec![row(32, &[(1, &[61])])]);
    let short_events = handle
        .submit(Request::BlockDiffusion(short.clone()))
        .unwrap();
    let long_events = handle
        .submit(Request::BlockDiffusion(long.clone()))
        .unwrap();
    let thread = thread::spawn(move || scheduler.run(rx));
    assert_answered(&events(short_events), &short);
    assert_answered(&events(long_events), &long);
    handle.shutdown();
    assert!(!thread.join().unwrap());

    let submitted: Vec<Vec<(u64, CallKind)>> = batches
        .try_iter()
        .filter_map(|event| match event {
            BatchEvent::Submitted(batch) => Some(
                batch
                    .requests
                    .iter()
                    .map(|(call, _)| (call.request_key.request_id.0, call.code))
                    .collect(),
            ),
            BatchEvent::Resolved { .. } => None,
        })
        .collect();
    let prefill = CallKind::Forward(ForwardMode::Prefill);
    assert_eq!(submitted[0], vec![(7, prefill), (8, prefill)]);
    assert_eq!(
        submitted[1],
        vec![(7, CallKind::Forward(ForwardMode::TokenDenoising))]
    );
}

/// A block-diffusion runtime refuses requests of another family, and a
/// generating request it cannot decode.
#[test]
fn a_block_diffusion_runtime_serves_only_readouts() {
    let running = Running::start(readout_worker(), SchedulerConfig::default());
    let autoregressive = readout_request(9, prompt(8, 1000), Vec::new());
    let refused = events(
        running
            .handle
            .submit(Request::Ar(autoregressive.clone()))
            .unwrap(),
    );
    let mut generating = readout_request(10, prompt(8, 1000), Vec::new());
    generating.max_und_tokens = 4;
    let undecodable = events(running.submit(generating));
    running.stop();

    for answer in [refused, undecodable] {
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
