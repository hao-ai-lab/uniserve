#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Readout requests on a block-diffusion runtime over the GPU-free simulator.
//!
//! A readout prefills its prompt without sampling, denoises its canvas rows
//! in token-denoising passes that read the prompt's KV without writing it,
//! and answers once with every candidate's log-probability before finishing.
//! The simulator reports `sim_candidate_logprob(request, token)` for each
//! candidate, so the expected answer is known exactly. Tests observe calls at
//! the executor boundary, the protocol the worker consumes.

use std::collections::{BTreeMap, HashMap};
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
    BatchEvent, ComponentConfig, EngineHandle, EventRx, ExecutionBatch, Scheduler, SchedulerConfig,
    SchedulerStats, SimEngine, SimExecutor, SpecialTokenIds, sim_candidate_logprob,
};
use uniserve_worker_ipc::{Call, CallKind, ComponentInfo, ForwardMode, MediaCall};

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
        self.stop_with_resolutions()
            .into_iter()
            .filter_map(|event| match event {
                BatchEvent::Submitted(batch) => Some(batch),
                BatchEvent::Resolved { .. } => None,
            })
            .collect()
    }

    /// Stops the scheduler and returns every submission and resolution the
    /// executor saw, in order.
    fn stop_with_resolutions(self) -> Vec<BatchEvent> {
        self.handle.shutdown();
        assert!(!self.thread.join().unwrap(), "the scheduler failed");
        self.batches.try_iter().collect()
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

/// A worker reporting a non-finite candidate log-probability fails the
/// readout without an answer, and the request's terminal error names the
/// field, the readout row and slot the value belongs to, and why it was
/// rejected.
#[test]
fn a_non_finite_readout_logprob_fails_the_request_with_its_cause() {
    let mut worker = readout_worker();
    worker.set_non_finite_candidate(11);
    let running = Running::start(worker, SchedulerConfig::default());
    let request = readout_request(
        1,
        prompt(40, 1000),
        vec![
            row(32, &[(3, &[7, 8]), (9, &[9])]),
            row(32, &[(4, &[10, 11, 12])]),
        ],
    );
    let answer = events(running.submit(request));
    running.stop();

    assert!(
        !answer
            .iter()
            .any(|event| matches!(event, EngineCoreOutput::Readout { .. })),
        "a rejected readout gives no answer: {answer:?}"
    );
    let Some(EngineCoreOutput::Error { message }) = answer.last() else {
        panic!("the readout did not fail with an error: {answer:?}");
    };
    // Candidate 11 is the second of the three read by the only slot of
    // the second row, the fifth of the six candidates the pass reports.
    for part in [
        "candidate_logprobs[4]",
        "not finite",
        "readout row 1, slot 0",
        "1 of 6",
    ] {
        assert!(message.contains(part), "{part:?} missing from {message:?}");
    }
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

/// A vision input image of `tokens` KV tokens and as many positions at
/// prompt position `position`.
fn vision_image(hash: u64, position: u32, tokens: u32) -> ImageInput {
    ImageInput {
        hash,
        b64: "aW1hZ2U=".into(),
        position,
        num_positions: tokens,
        encoders: vec![ImageEncoderInput {
            encoder: ImageIngestStep::VitEncode,
            num_kv_tokens: Some(tokens),
            max_kv_tokens: None,
        }],
    }
}

/// The context prefills of one request, with their forward rows, in
/// submission order.
fn context_prefills(
    batches: &[ExecutionBatch],
    request: u64,
) -> Vec<(Call, uniserve_worker_ipc::ForwardBatch)> {
    calls_of(batches, request)
        .into_iter()
        .filter(|(call, _)| call.code == CallKind::Forward(ForwardMode::Prefill))
        .collect()
}

/// A prompt with an image is written into KV by one context prefill, after
/// the image is encoded: the prefill carries the prompt tokens and the image
/// block at its prompt position, as one forward row per segment, and the
/// canvas then reads the whole context.
#[test]
fn an_image_prompt_is_written_by_one_context_prefill() {
    let running = Running::start(readout_worker(), SchedulerConfig::default());
    let mut request = readout_request(9, prompt(16, 1000), vec![row(16, &[(1, &[50, 51])])]);
    request
        .multimodal_inputs
        .images
        .push(vision_image(99, 10, 9));
    let answer = events(running.submit(request.clone()));
    let batches = running.stop();
    assert_answered(&answer, &request);

    let calls = calls_of(&batches, 9);
    let kinds: Vec<_> = calls.iter().map(|(call, _)| call.code).collect();
    assert_eq!(
        kinds,
        vec![
            CallKind::Media(MediaCall::VisionEncoding),
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::TokenDenoising),
        ]
    );
    let feature = calls[0].0.encoder_output.clone().unwrap();
    let (prefill, forward) = &calls[1];
    assert_eq!(prefill.input_token_ids, prompt(16, 1000));
    assert_eq!(prefill.vision_inputs.len(), 1);
    assert_eq!(prefill.vision_inputs[0].offset, 10);
    assert_eq!(
        prefill.vision_inputs[0].feature.buffer_id(),
        feature.buffer_id()
    );
    assert_eq!(prefill.coordinates.kv_visible_len, 0);
    // Text before the image, the image block, text after it.
    assert_eq!(forward.query_lens, vec![10, 9, 6]);
    assert_eq!(forward.seq_lens, vec![10, 19, 25]);
    assert_eq!(forward.write_kv, vec![true, true, true]);
    assert_eq!(calls[2].0.coordinates.kv_visible_len, 25);
}

/// Every image a context prefill reaches is encoded before it, and the
/// prefill writes all of their blocks between its prompt tokens.
#[test]
fn a_context_prefill_writes_every_image_it_reaches() {
    let running = Running::start(readout_worker(), SchedulerConfig::default());
    let mut request = readout_request(10, prompt(20, 1000), vec![row(16, &[(1, &[50, 51])])]);
    request
        .multimodal_inputs
        .images
        .push(vision_image(101, 4, 5));
    request
        .multimodal_inputs
        .images
        .push(vision_image(102, 12, 7));
    let answer = events(running.submit(request.clone()));
    let batches = running.stop();
    assert_answered(&answer, &request);

    let calls = calls_of(&batches, 10);
    let encodes: Vec<_> = calls
        .iter()
        .filter(|(call, _)| call.code == CallKind::Media(MediaCall::VisionEncoding))
        .map(|(call, _)| call.encoder_output.clone().unwrap().buffer_id())
        .collect();
    assert_eq!(encodes.len(), 2);
    let prefills = context_prefills(&batches, 10);
    assert_eq!(prefills.len(), 1);
    let (prefill, forward) = &prefills[0];
    assert_eq!(
        prefill
            .vision_inputs
            .iter()
            .map(|input| (input.offset, input.feature.buffer_id()))
            .collect::<Vec<_>>(),
        vec![(4, encodes[0]), (12, encodes[1])]
    );
    assert_eq!(forward.query_lens, vec![4, 5, 8, 7, 8]);
    assert_eq!(forward.seq_lens, vec![4, 9, 17, 24, 32]);
}

/// The worker's prefill graph limit bounds numerical segments, even when one
/// request's context has more segments than the graph can hold.
#[test]
fn image_contexts_respect_the_workers_prefill_row_capacity() {
    let mut sim = readout_worker();
    sim.mut_info_for_test().max_prefill_calls = 3;
    let running = Running::start(sim, SchedulerConfig::default());
    let requests: Vec<_> = (0..8u64)
        .map(|index| {
            let mut request = readout_request(
                100 + index,
                prompt(20, 1000 + 100 * index as u32),
                vec![row(16, &[(1, &[50, 51])])],
            );
            request.multimodal_inputs.images.extend([
                vision_image(1000 + index * 2, 4, 5),
                vision_image(1001 + index * 2, 12, 7),
            ]);
            request
        })
        .collect();
    let streams: Vec<_> = requests
        .iter()
        .map(|request| running.submit(request.clone()))
        .collect();
    for (stream, request) in streams.into_iter().zip(&requests) {
        assert_answered(&events(stream), request);
    }
    let batches = running.stop();
    for batch in &batches {
        let rows: usize = batch
            .requests
            .iter()
            .filter(|(call, _)| call.code == CallKind::Forward(ForwardMode::Prefill))
            .map(|(_, placement)| placement.forward.query_lens.len())
            .sum();
        assert!(rows <= 3, "prefill batch has {rows} numerical rows");
    }
    for request in &requests {
        let prefills = context_prefills(&batches, request.request_id.0);
        let tokens: Vec<_> = prefills
            .iter()
            .flat_map(|(call, _)| call.input_token_ids.iter().copied())
            .collect();
        assert_eq!(tokens, request.prompt_token_ids);
        // Chunking preserves both complete image blocks and every text token.
        assert_eq!(
            prefills
                .iter()
                .map(|(_, forward)| forward.query_lens.iter().sum::<u32>())
                .sum::<u32>(),
            32
        );
        assert_eq!(
            prefills
                .iter()
                .map(|(call, _)| call.vision_inputs.len())
                .sum::<usize>(),
            2
        );
    }
}

/// An image block is never split across context prefills: a chunk that
/// cannot hold the whole block ends before it, and the next chunk starts
/// with the block.
#[test]
fn a_context_chunk_ends_before_an_image_block_it_cannot_hold() {
    let config = SchedulerConfig {
        long_prefill_threshold: 16,
        ..SchedulerConfig::default()
    };
    let running = Running::start(readout_worker(), config);
    let mut request = readout_request(11, prompt(30, 1000), vec![row(16, &[(1, &[50, 51])])]);
    request
        .multimodal_inputs
        .images
        .push(vision_image(103, 12, 9));
    let answer = events(running.submit(request.clone()));
    let batches = running.stop();
    assert_answered(&answer, &request);

    let chunks: Vec<_> = context_prefills(&batches, 11)
        .into_iter()
        .map(|(call, forward)| {
            (
                call.input_token_ids.len(),
                call.vision_inputs
                    .iter()
                    .map(|input| input.offset)
                    .collect::<Vec<_>>(),
                forward.query_lens,
            )
        })
        .collect();
    assert_eq!(
        chunks,
        vec![
            (12, vec![], vec![12]),
            (7, vec![0], vec![9, 7]),
            (11, vec![], vec![11]),
        ]
    );
}

/// The worker prepares each inline input image as one task on its rank's
/// host lane, so an image encode waits in the scheduler while the lane is
/// full instead of reaching the worker: with one host task per rank, image
/// encodes of different requests are never in flight together, and every
/// request is still answered.
#[test]
fn an_image_encode_waits_for_a_free_host_lane_task() {
    let mut sim = readout_worker();
    // A deep queue and results delivered only when the scheduler waits keep
    // every ready encode schedulable at once, leaving the lane to bound them.
    sim.set_queue_depth(16);
    sim.set_results_on_wait(true);
    let info = sim.mut_info_for_test();
    info.host_lane_capacity = 1;
    info.components = vec![ComponentInfo {
        name: "model".to_owned(),
        config: ComponentConfig::parallel(vec![0], Default::default()),
        outputs: Vec::new(),
    }];
    info.media_components = BTreeMap::from([(MediaCall::VisionEncoding, "model".to_owned())]);
    let running = Running::start(sim, SchedulerConfig::default());

    // Each prompt opens with its image, so every encode is ready at once.
    let requests: Vec<_> = (0..3u64)
        .map(|index| {
            let mut request = readout_request(
                20 + index,
                prompt(16, 1000),
                vec![row(16, &[(1, &[50, 51])])],
            );
            request.multimodal_inputs.images.push(ImageInput {
                hash: 200 + index,
                b64: "aW1hZ2U=".into(),
                position: 0,
                num_positions: 9,
                encoders: vec![ImageEncoderInput {
                    encoder: ImageIngestStep::VitEncode,
                    num_kv_tokens: Some(9),
                    max_kv_tokens: None,
                }],
            });
            request
        })
        .collect();
    let streams: Vec<_> = requests
        .iter()
        .map(|request| running.submit(request.clone()))
        .collect();
    for (stream, request) in streams.into_iter().zip(&requests) {
        assert_answered(&events(stream), request);
    }
    let timeline = running.stop_with_resolutions();

    // Encodes in flight after each submission: a batch's encodes stay in
    // flight until its result is resolved.
    let mut in_flight: HashMap<u64, usize> = HashMap::new();
    let mut encodes = 0;
    for event in timeline {
        match event {
            BatchEvent::Submitted(batch) => {
                let count = batch
                    .requests
                    .iter()
                    .filter(|(call, _)| call.code == CallKind::Media(MediaCall::VisionEncoding))
                    .count();
                encodes += count;
                in_flight.insert(batch.id, count);
                assert!(
                    in_flight.values().sum::<usize>() <= 1,
                    "image encodes exceeded the host lane: {in_flight:?}"
                );
            }
            BatchEvent::Resolved { batch_id } => {
                in_flight.remove(&batch_id);
            }
        }
    }
    assert_eq!(encodes, requests.len());
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

/// With room for two batches in flight, a readout's canvas pass is queued
/// behind the prefill that completes its prompt, before that prefill
/// resolves, and the answer is unchanged.
#[test]
fn a_readout_pass_is_queued_behind_its_prompt_prefill() {
    let mut sim = readout_worker();
    sim.set_queue_depth(2);
    sim.set_results_on_wait(true);
    let running = Running::start(sim, SchedulerConfig::default());
    let request = readout_request(11, prompt(40, 1000), vec![row(32, &[(3, &[7, 8])])]);
    let answer = events(running.submit(request.clone()));
    running.handle.shutdown();
    assert!(!running.thread.join().unwrap(), "the scheduler failed");
    assert_answered(&answer, &request);

    // The submission of the first batch holding a call of `code`.
    let history: Vec<BatchEvent> = running.batches.try_iter().collect();
    let submitted = |code: CallKind| {
        history
            .iter()
            .position(|event| match event {
                BatchEvent::Submitted(batch) => {
                    batch.requests.iter().any(|(call, _)| call.code == code)
                }
                BatchEvent::Resolved { .. } => false,
            })
            .unwrap()
    };
    let prefill = submitted(CallKind::Forward(ForwardMode::Prefill));
    let pass = submitted(CallKind::Forward(ForwardMode::TokenDenoising));
    let BatchEvent::Submitted(prefill_batch) = &history[prefill] else {
        unreachable!("the position names a submission");
    };
    let prefill_resolved = history
        .iter()
        .position(|event| {
            matches!(event, BatchEvent::Resolved { batch_id } if *batch_id == prefill_batch.id)
        })
        .unwrap();
    assert!(
        pass < prefill_resolved,
        "the readout pass waited for its prefill to resolve"
    );
}

/// A readout pass is queued behind the context prefill that writes the last
/// image of its prompt, before that prefill resolves, as behind a text
/// prompt's last prefill.
#[test]
fn a_readout_pass_is_queued_behind_its_image_context_prefill() {
    let mut sim = readout_worker();
    sim.set_queue_depth(2);
    sim.set_results_on_wait(true);
    let running = Running::start(sim, SchedulerConfig::default());
    let mut request = readout_request(12, prompt(24, 1000), vec![row(16, &[(3, &[7, 8])])]);
    request
        .multimodal_inputs
        .images
        .push(vision_image(104, 20, 9));
    let answer = events(running.submit(request.clone()));
    running.handle.shutdown();
    assert!(!running.thread.join().unwrap(), "the scheduler failed");
    assert_answered(&answer, &request);

    let history: Vec<BatchEvent> = running.batches.try_iter().collect();
    let submitted = |code: CallKind| {
        history
            .iter()
            .position(|event| match event {
                BatchEvent::Submitted(batch) => {
                    batch.requests.iter().any(|(call, _)| call.code == code)
                }
                BatchEvent::Resolved { .. } => false,
            })
            .unwrap()
    };
    let prefill = submitted(CallKind::Forward(ForwardMode::Prefill));
    let pass = submitted(CallKind::Forward(ForwardMode::TokenDenoising));
    let BatchEvent::Submitted(prefill_batch) = &history[prefill] else {
        unreachable!("the position names a submission");
    };
    assert_eq!(prefill_batch.requests[0].0.vision_inputs.len(), 1);
    let prefill_resolved = history
        .iter()
        .position(|event| {
            matches!(event, BatchEvent::Resolved { batch_id } if *batch_id == prefill_batch.id)
        })
        .unwrap();
    assert!(
        pass < prefill_resolved,
        "the readout pass waited for its image context prefill to resolve"
    );
}

/// An image whose encoder product the encoder cache holds joins its context
/// prefill without an encode, reading the cached product.
#[test]
fn a_cached_image_joins_its_context_prefill_without_an_encode() {
    let running = Running::start(readout_worker(), SchedulerConfig::default());
    let mut first = readout_request(13, prompt(16, 1000), vec![row(16, &[(1, &[50, 51])])]);
    first.multimodal_inputs.images.push(vision_image(105, 8, 9));
    assert_answered(&events(running.submit(first.clone())), &first);
    let mut second = readout_request(14, prompt(12, 2000), vec![row(16, &[(1, &[50, 51])])]);
    second
        .multimodal_inputs
        .images
        .push(vision_image(105, 4, 9));
    assert_answered(&events(running.submit(second.clone())), &second);
    let batches = running.stop();

    let encoded = calls_of(&batches, 13)
        .into_iter()
        .find(|(call, _)| call.code == CallKind::Media(MediaCall::VisionEncoding))
        .and_then(|(call, _)| call.encoder_output)
        .unwrap();
    let calls = calls_of(&batches, 14);
    assert!(
        calls
            .iter()
            .all(|(call, _)| call.code != CallKind::Media(MediaCall::VisionEncoding))
    );
    let prefills = context_prefills(&batches, 14);
    assert_eq!(prefills.len(), 1);
    assert_eq!(prefills[0].0.vision_inputs[0].offset, 4);
    assert_eq!(
        prefills[0].0.vision_inputs[0].feature.buffer_id(),
        encoded.buffer_id()
    );
}
