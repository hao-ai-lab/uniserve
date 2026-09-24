#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Lane-occupancy scheduling of video requests over the GPU-free simulator.
//!
//! The observation point is the executor boundary: the batches the simulator
//! accepts and the moment it hands each result back. The simulator delivers a
//! result only when the scheduler waits for one, so every batch stays in
//! flight until the scheduler blocks for a completion, and the record of
//! submissions and resolutions is a function of scheduling decisions alone.

use std::collections::{BTreeMap, HashMap};
use std::thread;
use std::time::{Duration, Instant};

use uniserve_core::{
    DiffusionRequest, DiffusionSamplingParams, EngineCoreOutput, FinishReason, RejectionKind,
    Request, RequestId,
};
use uniserve_engine::{
    BatchEvent, ComponentConfig, ComponentDistribution, EngineHandle, ExecutionBatch, Scheduler,
    SimEngine, SimExecutor, SpecialTokenIds,
};
use uniserve_worker_ipc::{
    CallKind, ComponentInfo, DType, DimBound, MediaCall, OutputInfo, ShapeBound,
};

/// Denoising steps the simulated model advertises and every request follows.
const STEPS: u32 = 3;

/// Every request carries the same short prompt.
const PROMPT: [u32; 3] = [11, 12, 13];

/// Declares one loaded component with the outputs it produces.
///
/// A distributed component reconstructs one media unit per rank per round
/// (`units_per_rank` of one) and keeps the default parallel config, whose
/// world size of one is what a temporal-unit distribution requires.
fn component(
    name: &str,
    ranks: Vec<usize>,
    distributed: bool,
    outputs: Vec<OutputInfo>,
) -> ComponentInfo {
    let mut config = ComponentConfig::parallel(ranks, Default::default());
    if distributed {
        config.distribution = Some(ComponentDistribution::TemporalUnits);
    }

    ComponentInfo {
        name: name.to_owned(),
        config,
        outputs,
    }
}

/// A small tensor result, indexed by media unit along a device-actual leading
/// axis when `units` bounds it.
fn output(name: &str, units: Option<u32>) -> OutputInfo {
    let mut dims = vec![DimBound::Static(4)];
    if let Some(max) = units {
        dims.insert(0, DimBound::Device { max });
    }

    OutputInfo {
        name: name.to_owned(),
        dtype: DType::BF16,
        shape_bound: ShapeBound { dims },
    }
}

/// A decoded media unit result as a video decoder declares it: units along a
/// device-actual leading axis, then frames, height, width and RGB channels.
fn media_units(name: &str, units: u32) -> OutputInfo {
    OutputInfo {
        name: name.to_owned(),
        dtype: DType::U8,
        shape_bound: ShapeBound {
            dims: vec![
                DimBound::Device { max: units },
                DimBound::Static(4),
                DimBound::Static(16),
                DimBound::Static(24),
                DimBound::Static(3),
            ],
        },
    }
}

/// A video worker whose text encoder, denoiser, audio decoder and muxer live on
/// rank 0 and whose video decoder and video encoder are distributed over
/// `decoder_ranks` ranks, each reconstructing and encoding one media unit per
/// round.
///
/// Calls route to components the way a loaded model reports them: the video
/// encoder component encodes the media units the decoder produced, and the
/// muxer encodes audio and assembles the artifact.
fn video_worker(decoder_ranks: usize, host_lane_capacity: u32) -> SimEngine {
    let mut sim = SimEngine::new();
    // A deep executor queue keeps queue capacity from bounding which calls
    // are in flight together, leaving that to the lane rules under test.
    sim.set_queue_depth(64);
    sim.set_results_on_wait(true);

    let info = sim.mut_info_for_test();
    info.supported_calls = MediaCall::VIDEO
        .iter()
        .map(|call| CallKind::Media(*call))
        .collect();
    info.num_inference_steps = STEPS;
    // The denoiser's latent pool holds the reserved sentinel page plus two
    // pages of samples per request slot.
    info.latent_page_units = 256;
    info.latent_pages = 2 * info.request_slots + 1;
    info.world_size = decoder_ranks as u32;
    info.host_lane_capacity = host_lane_capacity;
    info.components = vec![
        component(
            "text_encoder",
            vec![0],
            false,
            vec![output("conditioning", Some(64))],
        ),
        component(
            "denoiser",
            vec![0],
            false,
            vec![output("video_latent", None), output("audio_latent", None)],
        ),
        component(
            "video_decoder",
            (0..decoder_ranks).collect(),
            true,
            vec![media_units("units", 32)],
        ),
        component(
            "video_encoder",
            (0..decoder_ranks).collect(),
            true,
            vec![output("encoded", Some(32))],
        ),
        component("audio_decoder", vec![0], true, vec![output("audio", None)]),
        component("muxer", vec![0], false, Vec::new()),
    ];
    info.media_components = BTreeMap::from([
        (MediaCall::TextEncoding, "text_encoder".to_owned()),
        (MediaCall::LatentPreparation, "denoiser".to_owned()),
        (MediaCall::Denoising, "denoiser".to_owned()),
        (MediaCall::VideoDecoding, "video_decoder".to_owned()),
        (MediaCall::VideoEncoding, "video_encoder".to_owned()),
        (MediaCall::AudioDecoding, "audio_decoder".to_owned()),
        (MediaCall::AudioEncoding, "muxer".to_owned()),
        (MediaCall::Muxing, "muxer".to_owned()),
    ]);

    sim
}

/// A video request decoded in `video_units` media units.
fn video_request(id: u64, video_units: u32) -> Request {
    Request::Diffusion(DiffusionRequest {
        request_id: RequestId(id),
        prompt_token_ids: PROMPT.to_vec(),
        priority: 0,
        sampling: DiffusionSamplingParams {
            num_frames: 16,
            video_units,
            num_inference_steps: STEPS,
            seed: id,
        },
    })
}

/// One media call as the executor received it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ObservedCall {
    request: RequestId,
    call: MediaCall,
    /// Media units the call covers: its decode range's `max_units`, or one
    /// when the call carries no decode range.
    units: u32,
}

impl ObservedCall {
    fn is(&self, request: RequestId, call: MediaCall) -> bool {
        self.request == request && self.call == call
    }

    /// Whether the call runs on a rank's bounded host executor.
    fn is_host_task(&self) -> bool {
        matches!(
            self.call,
            MediaCall::VideoEncoding | MediaCall::AudioEncoding | MediaCall::Muxing
        )
    }

    /// Whether the call decodes or encodes media after denoising.
    fn is_media_output(&self) -> bool {
        self.is_host_task()
            || matches!(
                self.call,
                MediaCall::VideoDecoding | MediaCall::AudioDecoding
            )
    }
}

/// One submission as the executor accepted it, with the calls of earlier
/// submissions that were still in flight at that moment.
#[derive(Debug, Clone)]
struct Submission {
    calls: Vec<ObservedCall>,
    in_flight: Vec<ObservedCall>,
}

impl Submission {
    /// Calls that ran at the same time as this submission's calls: those it
    /// carried and those still in flight when it was accepted.
    fn concurrent(&self) -> impl Iterator<Item = &ObservedCall> {
        self.calls.iter().chain(self.in_flight.iter())
    }
}

/// The record of one served workload: what the executor saw and what each
/// request's stream delivered.
struct Served {
    /// Media submissions in the order the executor accepted them.
    submissions: Vec<Submission>,
    /// Every event each request's stream delivered.
    outcomes: HashMap<RequestId, Vec<EngineCoreOutput>>,
}

impl Served {
    /// Submissions carrying a call of this request and media call, with their index.
    fn submissions_of(
        &self,
        request: RequestId,
        media_call: MediaCall,
    ) -> Vec<(usize, &Submission)> {
        self.submissions
            .iter()
            .enumerate()
            .filter(|(_, submission)| {
                submission
                    .calls
                    .iter()
                    .any(|call| call.is(request, media_call))
            })
            .collect()
    }

    /// Asserts that a request delivered a video artifact and then finished.
    fn assert_completed(&self, request: RequestId) {
        let events = &self.outcomes[&request];
        let artifact = events.iter().find_map(|event| match event {
            EngineCoreOutput::Artifact(artifact) => Some(artifact),
            _ => None,
        });
        let artifact = artifact
            .unwrap_or_else(|| panic!("request {request:?} delivered no artifact: {events:?}"));
        assert_eq!(artifact.content_type, "video/mp4");
        assert!(!artifact.media.as_bytes().is_empty());
        assert!(
            matches!(
                events.last(),
                Some(EngineCoreOutput::Finished {
                    reason: FinishReason::Completed,
                    ..
                })
            ),
            "request {request:?} did not finish after its artifact: {events:?}"
        );
    }
}

/// A consumer that reads nothing until the request has already finished still
/// receives the artifact followed by the completed terminal event. Events that
/// do not fit the stream's bounded channel wait in the scheduler's per-request
/// output journal until the consumer drains the channel.
#[test]
fn slow_consumer_receives_the_completed_video_before_the_terminal_event() {
    let mut executor = SimExecutor::new(video_worker(1, 1));
    let boundary = executor.observe();
    let scheduler = Scheduler::new(Box::new(executor), SpecialTokenIds::default(), 32).unwrap();
    let (tx, commands) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);
    let mut stream = handle
        .submit(Request::Diffusion(DiffusionRequest {
            request_id: RequestId(90),
            prompt_token_ids: PROMPT.to_vec(),
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: 128,
                video_units: 32,
                num_inference_steps: STEPS,
                seed: 1,
            },
        }))
        .unwrap();
    let engine = thread::spawn(move || scheduler.run(commands));

    // Let the entire workload finish while the caller retains an unread stream.
    // `Scheduler::finish_media` enqueues the terminal event on the request's
    // output before it queues the request's `Finish` lifecycle command, so a
    // submitted `Finish` at the executor boundary shows the request has ended.
    // A stalled lane instead exhausts the deadline and fails `recv_timeout`.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let event = boundary
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .unwrap();
        if let BatchEvent::Submitted(batch) = event
            && batch
                .commands
                .iter()
                .any(|command| matches!(command, uniserve_worker_ipc::BatchCommand::Finish { .. }))
        {
            break;
        }
    }

    // Only now drain the stream, until its terminal event. The engine keeps
    // running meanwhile, so it can flush journaled events into the channel as
    // reads free room.
    let mut events = Vec::new();
    while Instant::now() < deadline {
        if let Ok(event) = stream.try_recv() {
            let finished = matches!(event, EngineCoreOutput::Finished { .. });
            events.push(event);
            if finished {
                break;
            }
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    }
    handle.shutdown();
    // `Scheduler::run` returns `true` only when the engine died rather than
    // shut down.
    assert!(!engine.join().unwrap());

    Served {
        submissions: Vec::new(),
        outcomes: HashMap::from([(RequestId(90), events)]),
    }
    .assert_completed(RequestId(90));
}

/// Media calls of one batch; a batch carrying only lifecycle commands has none.
fn media_calls(batch: &ExecutionBatch) -> Vec<ObservedCall> {
    batch
        .requests
        .iter()
        .filter_map(|(call, placement)| {
            let CallKind::Media(media_call) = call.code else {
                return None;
            };
            Some(ObservedCall {
                request: call.request_key.request_id,
                call: media_call,
                units: placement.decode.as_ref().map_or(1, |range| range.max_units),
            })
        })
        .collect()
}

/// Serves the requests, queued together in order, until each stream reaches
/// a terminal event.
fn serve(sim: SimEngine, requests: Vec<Request>) -> Served {
    serve_bounded(sim, requests, None)
}

/// Serves the requests with an optional bound on waiting request state.
fn serve_bounded(sim: SimEngine, requests: Vec<Request>, max_num_waiting: Option<usize>) -> Served {
    let mut executor = SimExecutor::new(sim);
    let boundary = executor.observe();
    let mut scheduler = Scheduler::new(Box::new(executor), SpecialTokenIds::default(), 32).unwrap();
    if let Some(bound) = max_num_waiting {
        scheduler.set_max_num_waiting(bound);
    }
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = EngineHandle::new(tx);

    // Every request is queued before the loop starts, so one admission pass
    // sees them together, resident in submission order.
    let mut streams = requests
        .into_iter()
        .map(|request| {
            let id = request.request_id();
            (id, handle.submit(request).expect("submit media request"))
        })
        .collect::<Vec<_>>();
    let engine_loop = thread::spawn(move || scheduler.run(rx));

    let mut outcomes: HashMap<RequestId, Vec<EngineCoreOutput>> = HashMap::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut terminal = 0;
    while terminal < streams.len() && Instant::now() < deadline {
        let mut idle = true;
        for (id, stream) in streams.iter_mut() {
            while let Ok(event) = stream.try_recv() {
                idle = false;
                if matches!(
                    event,
                    EngineCoreOutput::Finished { .. }
                        | EngineCoreOutput::Rejected { .. }
                        | EngineCoreOutput::Error { .. }
                ) {
                    terminal += 1;
                }
                outcomes.entry(*id).or_default().push(event);
            }
        }
        if idle {
            thread::sleep(Duration::from_millis(1));
        }
    }
    handle.shutdown();
    let died = engine_loop.join().expect("engine loop thread");
    assert!(!died, "the engine loop died");
    assert_eq!(
        terminal,
        streams.len(),
        "every request reaches a terminal event: {outcomes:?}"
    );

    // The executor is gone with the scheduler, so the record is complete.
    // Replaying it attaches to each media submission the calls of earlier
    // media submissions not yet resolved when it was accepted; batches without
    // media calls are left out of the record.
    let mut in_flight: Vec<(u64, Vec<ObservedCall>)> = Vec::new();
    let mut submissions = Vec::new();
    for event in boundary.iter() {
        match event {
            BatchEvent::Submitted(batch) => {
                let calls = media_calls(&batch);
                if calls.is_empty() {
                    continue;
                }
                submissions.push(Submission {
                    calls: calls.clone(),
                    in_flight: in_flight
                        .iter()
                        .flat_map(|(_, calls)| calls.iter().copied())
                        .collect(),
                });
                in_flight.push((batch.id, calls));
            }
            BatchEvent::Resolved { batch_id } => in_flight.retain(|(id, _)| *id != batch_id),
        }
    }

    Served {
        submissions,
        outcomes,
    }
}

#[test]
fn a_younger_request_denoises_only_after_the_older_request_completes_its_steps() {
    let older = RequestId(1);
    let younger = RequestId(2);
    let served = serve(
        video_worker(2, 1),
        vec![video_request(older.0, 2), video_request(younger.0, 2)],
    );
    served.assert_completed(older);
    served.assert_completed(younger);

    let younger_steps = served.submissions_of(younger, MediaCall::Denoising);
    assert_eq!(younger_steps.len(), STEPS as usize);
    for (index, submission) in &younger_steps {
        // The denoiser lane is exclusive: no step of the older request is in
        // flight, or dispatched alongside, while the younger request steps.
        assert!(
            !submission
                .concurrent()
                .any(|call| call.is(older, MediaCall::Denoising)),
            "the younger request denoised while the older request held the denoiser lane: {submission:?}"
        );
        // The lane passes to the younger request only once the older one has
        // dispatched every step, so its first step follows the older's last.
        let older_steps_dispatched = served.submissions[..*index]
            .iter()
            .flat_map(|earlier| earlier.calls.iter())
            .filter(|call| call.is(older, MediaCall::Denoising))
            .count();
        assert_eq!(
            older_steps_dispatched, STEPS as usize,
            "the younger request denoised before the older request dispatched every step"
        );
    }

    // Lane sharing: the older request decodes and encodes its media while the
    // younger request denoises.
    assert!(
        younger_steps.iter().any(|(_, submission)| {
            submission
                .in_flight
                .iter()
                .any(|call| call.request == older && call.is_media_output())
        }),
        "the older request's media output never overlapped the younger request's denoising: {:?}",
        served.submissions
    );
}

#[test]
fn a_distributed_decoder_admits_calls_up_to_its_rank_width_in_media_units() {
    // Each decoder rank reconstructs one media unit per round, so the lane
    // holds as many units as the decoder has ranks.
    const DECODER_RANKS: u32 = 2;

    let older = RequestId(1);
    let younger = RequestId(2);
    let served = serve(
        video_worker(DECODER_RANKS as usize, 1),
        vec![
            video_request(older.0, DECODER_RANKS + 1),
            video_request(younger.0, 1),
        ],
    );
    served.assert_completed(older);
    served.assert_completed(younger);

    // A request with more units than the lane holds decodes in successive
    // rounds: a full round, then the remainder.
    let older_rounds = served
        .submissions_of(older, MediaCall::VideoDecoding)
        .iter()
        .flat_map(|(_, submission)| submission.calls.iter())
        .filter(|call| call.is(older, MediaCall::VideoDecoding))
        .map(|call| call.units)
        .collect::<Vec<_>>();
    assert_eq!(older_rounds, vec![DECODER_RANKS, 1]);

    // The units in flight on the decoder lane never exceed what it holds.
    for submission in &served.submissions {
        let units = |calls: &[ObservedCall]| {
            calls
                .iter()
                .filter(|call| call.call == MediaCall::VideoDecoding)
                .map(|call| call.units)
                .sum::<u32>()
        };
        let submitted = units(&submission.calls);
        if submitted == 0 {
            continue;
        }
        assert!(
            units(&submission.in_flight) + submitted <= DECODER_RANKS,
            "decode calls exceeded the lane's media units: {submission:?}"
        );
    }

    // The younger request's decode is admitted while the older request's is
    // in flight, once the older request's remainder round left units free.
    let younger_decodes = served.submissions_of(younger, MediaCall::VideoDecoding);
    assert_eq!(younger_decodes.len(), 1);
    assert!(
        younger_decodes[0]
            .1
            .concurrent()
            .any(|call| call.is(older, MediaCall::VideoDecoding)),
        "the younger request's decode waited for the older request to finish decoding: {:?}",
        served.submissions
    );
}

#[test]
fn a_host_lane_admits_no_more_tasks_than_its_rank_advertises() {
    for capacity in [1_usize, 2] {
        let request = RequestId(1);
        let served = serve(
            video_worker(1, capacity as u32),
            vec![video_request(request.0, 1)],
        );
        served.assert_completed(request);

        // Every host task of this worker lands on rank 0's host lane, whose
        // occupancy never exceeds the advertised capacity. With one rank and
        // one media unit, each host call places exactly one task, so counting
        // calls counts the lane's tasks.
        let mut peak = 0;
        for submission in &served.submissions {
            let submitted = submission
                .calls
                .iter()
                .filter(|call| call.is_host_task())
                .count();
            if submitted == 0 {
                continue;
            }
            let occupied = submission
                .in_flight
                .iter()
                .filter(|call| call.is_host_task())
                .count();
            assert!(
                occupied + submitted <= capacity,
                "host tasks exceeded the lane capacity {capacity}: {submission:?}"
            );
            peak = peak.max(occupied + submitted);
        }

        // The video and audio encodes of one request are the tasks that can
        // overlap: a lane of two holds both, a lane of one serializes them.
        assert_eq!(
            peak,
            capacity.min(2),
            "host lane capacity {capacity} was not used up to its bound: {:?}",
            served.submissions
        );
    }
}

#[test]
fn an_admitted_video_request_reports_scheduling_before_its_progress() {
    let request = RequestId(1);
    let served = serve(video_worker(2, 1), vec![video_request(request.0, 2)]);
    served.assert_completed(request);

    // A job's public phases start at admission: the scheduling event carries
    // the queue interval and precedes every progress report.
    let events = &served.outcomes[&request];
    let scheduled = events
        .iter()
        .position(|event| matches!(event, EngineCoreOutput::Scheduled { .. }))
        .unwrap_or_else(|| panic!("the request was never reported as scheduled: {events:?}"));
    let first_progress = events
        .iter()
        .position(|event| matches!(event, EngineCoreOutput::MediaProgress { .. }))
        .expect("the request reported progress");
    assert!(scheduled < first_progress, "{events:?}");
    let EngineCoreOutput::Scheduled {
        queued_at,
        scheduled_at,
    } = events[scheduled]
    else {
        unreachable!("position matched a scheduling event");
    };
    assert!(queued_at <= scheduled_at);
}

#[test]
fn a_full_waiting_queue_rejects_a_video_request_as_overloaded() {
    let admitted = RequestId(1);
    let refused = RequestId(2);
    let served = serve_bounded(
        video_worker(2, 1),
        vec![video_request(admitted.0, 2), video_request(refused.0, 2)],
        Some(1),
    );
    served.assert_completed(admitted);

    // The second request arrives while the first still waits, so the bound
    // refuses it as retryable overload rather than as an invalid request.
    let events = &served.outcomes[&refused];
    assert!(
        matches!(
            events.as_slice(),
            [EngineCoreOutput::Rejected {
                kind: RejectionKind::Overloaded,
                ..
            }]
        ),
        "{events:?}"
    );
}
