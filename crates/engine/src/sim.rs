//! GPU-free executor for scheduler and frontend behavior.
//!
//! The simulator consumes the same typed admissions and calls as worker
//! executors. It enforces request identity, lifecycle, and dependency invariants
//! and reports accepted tokens, progress, and synthetic media.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::thread::JoinHandle;
use std::time::Duration;
use uniserve_worker_ipc::{ForwardMode, PipelineStage, TransferMode};

use crate::executor::{
    BatchResult, ExecutionBatch, Executor, ExecutorInfo, ExecutorSubmitError, WorkerId,
    logical_result,
};
use base64::Engine as _;
use crossbeam_channel::{Receiver, Sender};
use uniserve_core::philox;
use uniserve_core::{
    CommandWaker, ImageParams, RequestId, SampleOutput, SamplingParams, TokenLogprob,
    try_apply_sampling_counts,
};
use uniserve_worker_ipc::{
    BatchOutput, CallKind, DrawLayout, ErrorCode, FinishFlags, NewRequest, CallStatus,
    RegistrationAck, RequestOutput, SamplingState, Call, TensorRef, TimingCounters,
    WorkerInfo,
};

const DEFAULT_TEXT_LEN: usize = 8;
const FAKE_EOS_TOKEN: u32 = 151_645;
const SYNTH_VOCAB_SIZE: usize = FAKE_EOS_TOKEN as usize + 1;
const DEFAULT_DENOISE_STEPS: u16 = 50;
const DEFAULT_IMAGE_HW: (u32, u32) = (512, 512);

enum Job {
    Batch(ExecutionBatch),
    Shutdown,
}

/// Runs the deterministic model simulator on a bounded asynchronous executor seam.
pub struct SimExecutor {
    executor_info: ExecutorInfo,
    depth: usize,
    to_worker: Sender<Job>,
    from_worker: Receiver<anyhow::Result<(BatchResult, Vec<uniserve_worker_ipc::BatchCommand>)>>,
    progress_tx: Sender<()>,
    progress_rx: Receiver<()>,
    in_flight: usize,
    handle: Option<JoinHandle<()>>,
    admissions: HashSet<uniserve_worker_ipc::RequestKey>,
    products: HashSet<TensorRef>,
}

impl SimExecutor {
    /// Starts an asynchronous simulator with its advertised queue depth.
    pub fn new(engine: SimEngine) -> Self {
        let depth = (engine.info().queue_depth as usize).max(1);
        Self::with_depth(engine, depth)
    }

    /// Starts an asynchronous simulator with an explicit in-flight batch limit.
    ///
    /// # Panics
    ///
    /// Panics if the simulator thread cannot be spawned.
    pub fn with_depth(mut engine: SimEngine, depth: usize) -> Self {
        let info = engine.info().clone();
        let depth = depth.max(1);
        let (to_worker, jobs) = crossbeam_channel::unbounded();
        let (results_tx, from_worker) = crossbeam_channel::unbounded();
        let (progress_tx, progress_rx) = crossbeam_channel::bounded(1);
        let handle = std::thread::Builder::new()
            .name("uniserve-sim-executor".into())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    match job {
                        Job::Batch(batch) => {
                            let commands = batch.commands.clone();
                            let result = engine.execute(batch).and_then(|report| {
                                report.validate()?;
                                Ok((
                                    logical_result(
                                        crate::executor::WorkerResult::receive(report),
                                        true,
                                        &commands,
                                    ),
                                    commands,
                                ))
                            });
                            if results_tx.send(result).is_err() {
                                break;
                            }
                        }
                        Job::Shutdown => break,
                    }
                }
            })
            .expect("spawn sim executor thread");
        Self {
            executor_info: ExecutorInfo::single(WorkerId("sim".to_owned()), info.clone()),
            depth,
            to_worker,
            from_worker,
            progress_tx,
            progress_rx,
            in_flight: 0,
            handle: Some(handle),
            admissions: HashSet::new(),
            products: HashSet::new(),
        }
    }

    /// Returns a waker that interrupts executor polling after command enqueue.
    pub fn command_waker(&self) -> CommandWaker {
        let progress = self.progress_tx.clone();
        CommandWaker::new(move || {
            let _ = progress.try_send(());
        })
    }
}

impl SimExecutor {
    /// Returns the result of a submitted batch.
    fn poll_batch(
        &mut self,
        timeout: Duration,
    ) -> anyhow::Result<Option<(BatchResult, Vec<uniserve_worker_ipc::BatchCommand>)>> {
        if self.handle.is_none() {
            let _ = self.progress_rx.recv_timeout(timeout);
            return Ok(None);
        }
        let result = crossbeam_channel::select! {
            recv(self.from_worker) -> result => match result {
                Ok(result) => Ok(Some(result?)),
                Err(_) => Err(anyhow::anyhow!("sim executor thread disconnected")),
            },
            recv(self.progress_rx) -> wake => match wake {
                Ok(()) => Ok(None),
                Err(_) => Err(anyhow::anyhow!("sim executor progress channel disconnected")),
            },
            default(timeout) => Ok(None),
        }?;
        if result.is_some() {
            self.in_flight = self.in_flight.saturating_sub(1);
        }
        Ok(result)
    }

    /// Closes the simulated physical worker.
    fn shutdown(&mut self) -> anyhow::Result<()> {
        let _ = self.to_worker.send(Job::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        Ok(())
    }
}

impl Executor for SimExecutor {
    fn has_capacity(&self, worker: &WorkerId) -> bool {
        self.is_ready(worker) && self.in_flight < self.depth
    }

    fn command_has_capacity(&self, command: &uniserve_worker_ipc::BatchCommand) -> bool {
        use uniserve_worker_ipc::BatchCommand;
        let has_owner = match command {
            BatchCommand::Free { buffer } => self
                .products
                .iter()
                .any(|product| product.buffer_id() == *buffer),
            _ => self.admissions.contains(&command.request_key()),
        };
        !has_owner || (self.handle.is_some() && self.in_flight < self.depth)
    }
    fn is_ready(&self, worker: &WorkerId) -> bool {
        self.handle.is_some()
            && self
                .executor_info
                .workers
                .iter()
                .any(|(id, _)| id == worker)
    }

    /// Returns the worker metadata.
    fn info(&self) -> &ExecutorInfo {
        &self.executor_info
    }

    /// Lowers and submits a logical batch while preserving executor backpressure semantics.
    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError> {
        if self.handle.is_none() {
            return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                "Executor is closed"
            )));
        }
        for (_, placement) in &batch.requests {
            if !self.is_ready(&placement.worker) {
                return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                    "Worker cannot accept this request"
                )));
            }
        }
        if self.in_flight >= self.depth {
            return Err(ExecutorSubmitError::WouldBlock(batch));
        }
        batch.validate().map_err(ExecutorSubmitError::Failed)?;
        self.admissions
            .extend(batch.admissions().map(|request| request.request_key));
        self.products.extend(
            batch
                .requests
                .iter()
                .flat_map(|(call, _)| call.tensor_outputs().cloned()),
        );
        self.to_worker.send(Job::Batch(batch)).map_err(|_| {
            ExecutorSubmitError::Failed(anyhow::anyhow!("sim executor thread gone"))
        })?;
        self.in_flight += 1;
        Ok(())
    }

    /// Polls for the next completed worker call.
    fn poll(&mut self, timeout: Duration) -> anyhow::Result<Option<BatchResult>> {
        let Some((result, commands)) = self.poll_batch(timeout)? else {
            return Ok(None);
        };
        for receipt in &result.command_results {
            if receipt.outcome == crate::executor::CommandOutcome::Failed {
                continue;
            }
            let command = commands
                .iter()
                .filter(|command| {
                    !matches!(command, uniserve_worker_ipc::BatchCommand::Start { .. })
                })
                .nth(receipt.command_index as usize)
                .expect("logical command has its registered payload");
            use uniserve_worker_ipc::BatchCommand;
            match command {
                BatchCommand::Free { buffer } => self
                    .products
                    .retain(|product| product.buffer_id() != *buffer),
                BatchCommand::Finish {
                    request_key,
                    retained_buffers,
                    ..
                } => {
                    self.admissions.remove(request_key);
                    self.products.retain(|product| {
                        product.request_key != *request_key
                            || retained_buffers.contains(&product.buffer_id())
                    });
                }
                _ => {}
            }
        }
        Ok(Some(result))
    }

    /// Closes the component and releases its resources.
    fn close(&mut self) -> anyhow::Result<()> {
        self.shutdown()
    }
}

impl Drop for SimExecutor {
    /// Releases resources owned by this value.
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// Current synthetic execution state and products retained for their consumers.
#[derive(Clone)]
struct SimRequestState {
    admission: NewRequest,
    logical_position: u32,
    /// Whether the model, rather than this simulator, owns the request's
    /// logical positions. An image contributes positions through its rope
    /// advance, which the simulator does not implement, so once a request has
    /// ingested one it takes the position each call states.
    positions_from_model: bool,
    kv_visible_len: u32,
    emitted: usize,
    flow_step: u16,
    predicate_values: HashMap<TensorRef, bool>,
    /// Committed penalty counts in ascending token order.
    ///
    /// Each generated token folds in during execution, allowing a successor to
    /// observe it before the predecessor is host-visible. Device executors encode
    /// the same state as a resident count tensor plus per-call deltas.
    penalty_counts: BTreeMap<u32, u32>,
}

impl SimRequestState {
    /// Creates request state from an admitted request.
    fn new(admission: NewRequest) -> Self {
        let prefix_len = admission
            .ar
            .as_ref()
            .map_or(0, |branch| branch.initial_position);
        Self {
            admission,
            logical_position: prefix_len,
            positions_from_model: false,
            kv_visible_len: prefix_len,
            emitted: 0,
            flow_step: 0,
            predicate_values: HashMap::new(),
            penalty_counts: BTreeMap::new(),
        }
    }

    /// Returns the committed penalty histogram in ascending token order.
    fn recent_counts(&self) -> Vec<(u32, u32)> {
        self.penalty_counts
            .iter()
            .map(|(token, count)| (*token, *count))
            .collect()
    }

    /// Folds one generated token into the committed penalty base.
    fn fold_penalty_token(&mut self, token: u32) {
        self.penalty_counts
            .entry(token)
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);
    }

    /// Returns shared access to the simulated sampling configuration.
    fn sampling(&self) -> Option<&SamplingParams> {
        self.admission.ar.as_ref().map(|und| &und.sampling)
    }

    /// Returns shared access to the simulated image configuration.
    fn image(&self) -> Option<&ImageParams> {
        self.admission.umm.as_ref().map(|branch| &branch.image)
    }
}

/// Deterministic local model engine that implements the worker lifecycle protocol.
pub struct SimEngine {
    info: WorkerInfo,
    text_len: usize,
    fake_eos: u32,
    vocab: usize,
    requests: HashMap<RequestId, SimRequestState>,
}

impl SimEngine {
    /// Constructs a simulator with deterministic text and image capabilities.
    pub fn new() -> Self {
        let info = WorkerInfo {
            supported_ops: CallKind::ALL.to_vec(),
            latent_page_units: 64,
            latent_pages: 1_025,
            buffer_pool_bytes: 257_u64 * (256 << 20),
            encoder_cache_entries: 256,
            encoder_entry_bytes: 256 << 20,
            max_batch_ops: 1024,
            max_unresolved_ops: 2,
            model_name: "sim".to_owned(),
            ..WorkerInfo::default()
        };
        Self {
            info,
            text_len: DEFAULT_TEXT_LEN,
            fake_eos: FAKE_EOS_TOKEN,
            vocab: SYNTH_VOCAB_SIZE,
            requests: HashMap::new(),
        }
    }

    /// Produces deterministic logits for one simulated autoregressive position.
    fn synth_logits(
        vocab: usize,
        text_len: usize,
        fake_eos: u32,
        request_id: RequestId,
        index: usize,
    ) -> Vec<f32> {
        let mut logits = vec![0.0; vocab];
        let natural = if index >= text_len {
            fake_eos
        } else {
            1_000 + ((request_id.0 as u32 * 7 + index as u32) % 5_000)
        };
        logits[natural as usize] = 10.0;
        let alternate_one = 1_000 + ((request_id.0 as u32 * 13 + index as u32 + 1) % 5_000);
        let alternate_two = 1_000 + ((request_id.0 as u32 * 29 + index as u32 + 2) % 5_000);
        if alternate_one != natural {
            logits[alternate_one as usize] = 8.0;
        }
        if alternate_two != natural {
            logits[alternate_two as usize] = 6.0;
        }
        logits[fake_eos as usize] = if index >= text_len { 100.0 } else { 1.0 };
        logits
    }

    /// Applies request sampling controls to deterministic simulated logits.
    fn sample(
        vocab: usize,
        text_len: usize,
        fake_eos: u32,
        call: &Call,
        request: &SimRequestState,
        index: usize,
        state: Option<&SamplingState>,
    ) -> anyhow::Result<Option<SampleOutput>> {
        let request_id = call.request_key.request_id;
        let mut logits = Self::synth_logits(vocab, text_len, fake_eos, request_id, index);
        match request.sampling() {
            Some(sampling) => {
                let draw = if sampling.temperature > 0.0 {
                    let rng = call.rng.ok_or_else(|| {
                        anyhow::anyhow!("stochastic sampling call has no RNG coordinates")
                    })?;
                    anyhow::ensure!(
                        rng.draw_layout == DrawLayout::TargetSampling,
                        "stochastic sampling call uses the wrong RNG layout"
                    );
                    anyhow::ensure!(
                        rng.seed == sampling.seed.unwrap_or(0),
                        "call RNG seed disagrees with admitted sampling"
                    );
                    let key = philox::sampling_key(
                        rng.seed,
                        call.request_key.engine_id,
                        request_id.0,
                        call.request_key.request_epoch,
                        philox::DRAW_LAYOUT_TARGET,
                    );
                    philox::sampling_uniform(key, rng.semantic_index_base, 0, 0)
                } else {
                    0.0
                };
                // Penalty counts are device-resident, not carried in the staged
                // state: the successor reads the committed base folded from
                // ancestral tokens before any of them is host-observed.
                let recent_counts = request.recent_counts();
                // Per-step masks omit an unchanged request allowlist. A dynamic
                // suppression delta must still preserve that admitted constraint.
                let allowed = state
                    .and_then(|value| value.allowed_token_ids.as_deref())
                    .or(sampling.allowed_token_ids.as_deref());
                let suppress = state
                    .map(|value| value.suppressed_token_ids.as_slice())
                    .filter(|tokens| !tokens.is_empty());
                // Processor step 2 forced-token constraint: a decode call
                // samples a single span point, so its forced token is the first
                // entry of the schedule and overrides any allowed-token mask.
                let forced = sampling
                    .forced_token_ids
                    .first()
                    .copied()
                    .map(|token| [token]);
                let allowed = forced.as_ref().map(|slot| slot.as_slice()).or(allowed);
                Ok(try_apply_sampling_counts(
                    &mut logits,
                    sampling,
                    &recent_counts,
                    allowed,
                    suppress,
                    sampling.n_logprobs as usize,
                    draw,
                ))
            }
            None => Ok(Some(SampleOutput {
                token: if index >= text_len {
                    fake_eos
                } else {
                    1_000 + ((request_id.0 as u32 * 7 + index as u32) % 5_000)
                },
                logprob: 0.0,
                top: Vec::new(),
            })),
        }
    }

    /// Executes one call against its request, producing the terminal
    /// [`RequestOutput`] and any resolved output-product values.
    fn execute_call(
        vocab: usize,
        text_len: usize,
        fake_eos: u32,
        call: &Call,
        request: &mut SimRequestState,
    ) -> anyhow::Result<RequestOutput> {
        // A call states the coordinates it executes at. The simulator holds the
        // same request state a rank does, so a disagreement is a scheduling
        // failure and not a condition it can execute through. A call with no
        // request predecessor carries no coordinates.
        if call.predecessor.is_some() {
            let stated = call.coordinates;
            if request.positions_from_model {
                request.logical_position = stated.logical_position;
            }
            let held = uniserve_worker_ipc::CallCoordinates {
                logical_position: request.logical_position,
                kv_visible_len: request.kv_visible_len,
                kv_computed_len: request.kv_visible_len,
                flow_step: u32::from(request.flow_step),
            };
            anyhow::ensure!(
                stated == held,
                "call {:?} states coordinates {stated:?} that disagree with the request's own progress {held:?}",
                call.call_id
            );
        }
        let mut record = RequestOutput {
            sampled_logprob: None,
            top_logprobs: Vec::new(),
            prompt_logprobs: Vec::new(),
            request_key: call.request_key,
            call_id: call.call_id,
            status: CallStatus::Ok,
            product_generations: call
                .tensor_outputs()
                .map(|out| out.generation)
                .collect(),
            error_code: None,
            timing_counters: TimingCounters::default(),
            code: call.code,
            position: 0,
            kv_visible_len: 0,
            kv_computed_len: 0,
            num_completed_steps: 0,
            committed_tokens: Vec::new(),
            finish_flags: FinishFlags::default(),
            media_output: None,
            kv_output: None,
        };
        record.position = request.logical_position;
        set_kv_lengths(&mut record, request.kv_visible_len);

        match call.code {
            work @ (CallKind::Forward(ForwardMode::Prefill)
            | CallKind::Forward(ForwardMode::Decode)
            | CallKind::Forward(ForwardMode::Verify)) => {
                let visual_state =
                    call.vision_input.is_some() || call.latent_feature_input.is_some();
                let samples_token = call.token_output.is_some();
                if visual_state {
                    request.positions_from_model = true;
                    request.kv_visible_len = request
                        .kv_visible_len
                        .saturating_add(call.bounds.max_tokens);
                    set_kv_lengths(&mut record, request.kv_visible_len);
                    if call.completion_output.is_some() {
                        request.emitted = 0;
                    }
                }
                if !visual_state && !samples_token {
                    request.kv_visible_len = request
                        .kv_visible_len
                        .saturating_add(call.bounds.max_tokens);
                    record.position = request.logical_position;
                    set_kv_lengths(&mut record, request.kv_visible_len);
                } else if samples_token {
                    let index = request.emitted;
                    let sampling_state = call.sampling_state.as_ref();
                    let Some(output) = Self::sample(
                        vocab,
                        text_len,
                        fake_eos,
                        call,
                        request,
                        index,
                        sampling_state,
                    )?
                    else {
                        record.status = CallStatus::Error;
                        record.product_generations.clear();
                        record.error_code = Some(ErrorCode::InvalidCall);
                        return Ok(record);
                    };
                    record.finish_flags.eos = output.token == fake_eos;
                    if let Some(state) = sampling_state {
                        record.finish_flags.stop =
                            state.finish_token_ids.binary_search(&output.token).is_ok()
                                && !record.finish_flags.eos;
                        record.finish_flags.length = state.force_finish;
                    }
                    let admitted_stops = request
                        .admission
                        .ar
                        .as_ref()
                        .map_or(&[][..], |und| und.finish_token_ids.as_slice());
                    let continuation = admitted_stops.binary_search(&output.token).is_err()
                        && sampling_state.is_none_or(|state| {
                            !state.force_finish
                                && state.finish_token_ids.binary_search(&output.token).is_err()
                        });
                    if let Some(token_product) = call.token_output.as_ref() {
                        request
                            .predicate_values
                            .insert(token_product.clone(), continuation);
                    }
                    let transition = sampling_state.is_some_and(|state| {
                        state
                            .transition_token_ids
                            .binary_search(&output.token)
                            .is_ok()
                    });
                    for completion in call.transition_output.iter() {
                        request
                            .predicate_values
                            .insert(completion.clone(), transition);
                    }
                    record.position = 1;
                    if !visual_state {
                        let query_tokens = match work {
                            CallKind::Forward(ForwardMode::Prefill) => {
                                call.bounds.max_tokens
                            }
                            CallKind::Forward(ForwardMode::Decode)
                            | CallKind::Forward(ForwardMode::Verify) => 1,
                            _ => unreachable!(),
                        };
                        request.logical_position =
                            request.logical_position.saturating_add(query_tokens);
                        request.kv_visible_len =
                            request.kv_visible_len.saturating_add(query_tokens);
                        record.position = request.logical_position;
                        set_kv_lengths(&mut record, request.kv_visible_len);
                    }
                    match work {
                        CallKind::Forward(ForwardMode::Prefill) => {
                            request.emitted = request.emitted.max(1)
                        }
                        CallKind::Forward(ForwardMode::Decode)
                        | CallKind::Forward(ForwardMode::Verify) => {
                            request.emitted = request.emitted.saturating_add(1)
                        }
                        _ => unreachable!(),
                    }
                    record.committed_tokens = vec![output.token];
                    // Fold the generated token into the device-resident penalty
                    // base so the next call's penalties see it before this
                    // one is host-observed. A false-predicate no-op never reaches
                    // this branch, so a retracted point is never folded.
                    request.fold_penalty_token(output.token);
                    record.sampled_logprob = request
                        .sampling()
                        .is_some_and(SamplingParams::generated_logprobs_requested)
                        .then_some(output.logprob);
                    record.top_logprobs = output
                        .top
                        .iter()
                        .map(|(token_id, logprob, rank)| TokenLogprob {
                            token_id: *token_id,
                            logprob: *logprob,
                            rank: *rank,
                        })
                        .collect();
                }
            }
            CallKind::Pipeline(PipelineStage::TextEncoding)
            | CallKind::Pipeline(PipelineStage::VisionEncoding)
            | CallKind::Pipeline(PipelineStage::LatentEncoding) => {}
            CallKind::Transfer(TransferMode::Tensor)
            | CallKind::Transfer(TransferMode::KvPublish)
            | CallKind::Transfer(TransferMode::KvInstall) => {
                set_kv_lengths(&mut record, request.kv_visible_len);
            }
            CallKind::Pipeline(PipelineStage::LatentPreparation) => {}
            CallKind::Pipeline(PipelineStage::Denoising) => {
                let steps = call.bounds.max_tokens.max(1) as u16;
                request.flow_step = request.flow_step.saturating_add(steps);
                let total = request
                    .image()
                    .map(|image| image.steps)
                    .unwrap_or(DEFAULT_DENOISE_STEPS);
                record.num_completed_steps = u32::from(request.flow_step);
                // Denoise completion surfaces as a length finish once the flow has
                // advanced through every scheduled step.
                record.finish_flags.length = request.flow_step >= total;
            }
            CallKind::Pipeline(
                PipelineStage::VideoDecoding
                | PipelineStage::AudioDecoding
                | PipelineStage::VideoEncoding
                | PipelineStage::AudioEncoding
                | PipelineStage::Muxing,
            ) => {}
            CallKind::Pipeline(PipelineStage::ImageDecoding) => {
                request.flow_step = 0;
                if let Some(image) = request.image().cloned() {
                    let (height, width) = if image.height > 0 && image.width > 0 {
                        (image.height, image.width)
                    } else {
                        DEFAULT_IMAGE_HW
                    };
                    let png_base64 = synthetic_png_b64(width, height)?;
                    use std::io::Write;
                    let mut storage = tempfile::Builder::new()
                        .prefix("uniserve-image-")
                        .tempfile_in("/dev/shm")?;
                    storage.write_all(png_base64.as_bytes())?;
                    let (_file, path) = storage.keep()?;
                    record.media_output = Some(uniserve_worker_ipc::MediaOutput {
                        handle: uniserve_worker_ipc::ArtifactHandle::PosixShm {
                            name: path.file_name().unwrap().to_str().unwrap().to_owned(),
                        },
                        bytes: png_base64.len() as u64,
                    });
                }
            }
        }

        for completion in call
            .completion_output
            .iter()
            .chain(call.transition_output.iter())
        {
            request
                .predicate_values
                .entry(completion.clone())
                .or_insert(true);
        }

        Ok(record)
    }

    /// Builds a completion for an call resolved entirely by its execution predicate.
    fn predicated_completion(
        call: &Call,
        request: &SimRequestState,
    ) -> RequestOutput {
        RequestOutput {
            sampled_logprob: None,
            top_logprobs: Vec::new(),
            prompt_logprobs: Vec::new(),
            request_key: call.request_key,
            call_id: call.call_id,
            status: CallStatus::Predicated,
            product_generations: Vec::new(),
            error_code: None,
            timing_counters: TimingCounters::default(),
            code: call.code,
            position: request.logical_position,
            kv_visible_len: request.kv_visible_len,
            kv_computed_len: request.kv_visible_len,
            num_completed_steps: u32::from(request.flow_step),
            committed_tokens: Vec::new(),
            finish_flags: FinishFlags::default(),
            media_output: None,
            kv_output: None,
        }
    }

    /// Sets the advertised unresolved-batch capacity, clamped to at least one.
    pub fn set_queue_depth(&mut self, depth: u32) {
        self.info.queue_depth = depth.max(1);
    }

    /// Sets the number of synthetic tokens emitted before the configured EOS.
    pub fn set_text_len(&mut self, length: usize) {
        self.text_len = length;
    }

    /// Configures EOS and expands the synthetic vocabulary for control tokens.
    pub fn configure_control_tokens(&mut self, eos: u32, control_tokens: &[u32]) {
        self.fake_eos = eos;
        let max_token = control_tokens
            .iter()
            .copied()
            .chain(std::iter::once(eos))
            .max()
            .unwrap_or(eos);
        self.vocab = self.vocab.max(
            usize::try_from(max_token)
                .unwrap_or(usize::MAX)
                .saturating_add(1),
        );
    }

    /// Replaces the simulator's KV cache groups.
    ///
    /// # Panics
    ///
    /// Panics when the simulated worker does not expose a KV cache.
    pub fn set_groups(&mut self, groups: Vec<uniserve_core::KvCacheGroup>) {
        self.info
            .kv_cache
            .as_mut()
            .expect("sim AR worker has a KV cache")
            .groups = groups;
    }

    /// Sets the total KV block capacity and its sole group capacity when applicable.
    ///
    /// # Panics
    ///
    /// Panics when the simulated worker does not expose a KV cache.
    pub fn set_num_blocks(&mut self, count: u32) {
        let kv_cache = self
            .info
            .kv_cache
            .as_mut()
            .expect("sim AR worker has a KV cache");
        kv_cache.num_blocks = count;
        if kv_cache.groups.len() == 1 {
            kv_cache.groups[0].num_blocks = count;
        }
    }

    /// Sets the simulator's KV block size in tokens.
    ///
    /// # Panics
    ///
    /// Panics when the simulated worker does not expose a KV cache.
    pub fn set_block_size(&mut self, size: u32) {
        self.info
            .kv_cache
            .as_mut()
            .expect("sim AR worker has a KV cache")
            .block_size = size;
    }

    /// Returns mutable access to the simulator's advertised capabilities.
    pub fn mut_info_for_test(&mut self) -> &mut WorkerInfo {
        &mut self.info
    }
}

/// Sets every logical KV frontier to the visible token position.
fn set_kv_lengths(lengths: &mut RequestOutput, visible: u32) {
    lengths.kv_visible_len = visible;
    lengths.kv_computed_len = visible;
}

/// Encodes a deterministic RGB gradient as a base64 PNG artifact.
fn synthetic_png_b64(width: u32, height: u32) -> anyhow::Result<String> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        let row_bytes = width as usize * 3;
        let mut image = vec![0; row_bytes * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let index = y * row_bytes + x * 3;
                image[index] = ((x * 255) / width.max(1) as usize) as u8;
                image[index + 1] = ((y * 255) / height.max(1) as usize) as u8;
                image[index + 2] = 128;
            }
        }
        writer.write_image_data(&image)?;
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

impl Default for SimEngine {
    /// Returns the default value.
    fn default() -> Self {
        Self::new()
    }
}

impl SimEngine {
    /// Returns the worker metadata.
    fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Executes selected call kinds in order against deterministic model state.
    fn execute(&mut self, batch: ExecutionBatch) -> anyhow::Result<BatchOutput> {
        batch.validate()?;
        let batch_id = batch.id;
        let closed = batch
            .commands
            .iter()
            .filter_map(|control| match control {
                uniserve_worker_ipc::BatchCommand::Finish { request_key, .. } => Some(*request_key),
                _ => None,
            })
            .collect::<Vec<_>>();
        let admissions = batch.admissions().cloned().collect::<Vec<_>>();
        for admission in admissions {
            match self.requests.get(&admission.request_key.request_id) {
                Some(request) => anyhow::ensure!(
                    request.admission == admission,
                    "request {} was readmitted with a different descriptor",
                    admission.request_key.request_id.0
                ),
                None => {
                    self.requests.insert(
                        admission.request_key.request_id,
                        SimRequestState::new(admission),
                    );
                }
            }
        }
        for command in &batch.commands {
            if let uniserve_worker_ipc::BatchCommand::Free { buffer } = command {
                if let Some(request) = self.requests.get_mut(&buffer.owner.request_id) {
                    request
                        .predicate_values
                        .retain(|product, _| product.buffer_id() != *buffer);
                }
            }
        }
        let vocab = self.vocab;
        let text_len = self.text_len;
        let fake_eos = self.fake_eos;
        let mut completions = Vec::with_capacity(batch.requests.len());
        for (call, _) in batch.requests {
            let request = self
                .requests
                .get_mut(&call.request_key.request_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "request {} has no admission",
                        call.request_key.request_id.0
                    )
                })?;
            anyhow::ensure!(
                call.request_key == request.admission.request_key,
                "call identity {:?} does not match its admitted lineage",
                call.request_key
            );
            let predicate_value = call
                .predicate
                .as_ref()
                .as_ref()
                .map(|predicate| {
                    request
                        .predicate_values
                        .get(predicate)
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "computation {:?} predicate names an unresolved product",
                                call.call_id
                            )
                        })
                })
                .transpose()?;
            // Binding the input captures its value. The producer relay then has
            // no future acquisition owner, just as in the physical Worker.
            if let Some(predecessor) = call.predecessor {
                request
                    .predicate_values
                    .retain(|product, _| product.producer_call_id != predecessor);
            }
            if let Some(predicate) = call.predicate.as_ref() {
                request.predicate_values.remove(predicate);
            }
            if let Some(predicate_value) = predicate_value {
                if !predicate_value {
                    for output in call.tensor_outputs() {
                        request.predicate_values.insert(output.clone(), false);
                    }
                    let completion = Self::predicated_completion(&call, request);
                    completions.push(completion);
                    continue;
                }
            }

            let completion =
                Self::execute_call(vocab, text_len, fake_eos, &call, request)?;
            completions.push(completion);
        }
        let report = BatchOutput {
            batch_id,
            completions,
            products: Vec::new(),
            registration: RegistrationAck { visible: true },
            worker_exec_us: None,
            forward_stats: None,
        };
        for request_key in closed {
            if self
                .requests
                .get(&request_key.request_id)
                .is_some_and(|request| request.admission.request_key == request_key)
            {
                self.drop_request(request_key.request_id)?;
            }
        }
        Ok(report)
    }

    /// Removes all simulated state for a request.
    fn drop_request(&mut self, request_id: RequestId) -> anyhow::Result<()> {
        self.requests.remove(&request_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_worker_ipc::{
        ArRequestParams, Bounds, CallId, DType, ForwardBatch, RequestKey, ShapeBound,
        TensorRef,
    };

    fn request_key() -> RequestKey {
        RequestKey::new(0, RequestId(9), 1)
    }

    fn admission() -> NewRequest {
        NewRequest::new(
            request_key(),
            1,
            Some(ArRequestParams {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                initial_position: 0,
            }),
            None,
        )
        .expect("admission")
    }

    fn token_output(call_id: CallId) -> TensorRef {
        TensorRef {
            request_key: request_key(),
            producer_call_id: call_id,
            output_index: 0,
            generation: 1,
            dtype: DType::I64,
            shape_bound: ShapeBound::default(),
        }
    }

    fn batch(batch_id: u64, request_index: u32) -> ExecutionBatch {
        let request_key = request_key();
        let admission = admission();
        let parent = CallId::new(0, 0);
        let call = Call {
            coordinates: uniserve_worker_ipc::CallCoordinates::default(),
            token_input: None,

            token_output: Some(token_output(CallId::new(batch_id, request_index))),
            vision_input: None,
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,

            input_image: None,
            kv_input: None,
            kv_output: None,
            input_token_ids: Vec::new(),
            sampling_state: None,
            request_key,
            call_id: CallId::new(batch_id, request_index),
            predecessor: Some(parent),
            entry: "model".into(),
            code: CallKind::Forward(ForwardMode::Prefill),
            bounds: Bounds {
                max_tokens: 2,
                ..Bounds::default()
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            predicate: None,
            rng: None,
        };
        ExecutionBatch::new(
            batch_id,
            vec![(
                call,
                crate::executor::RequestPlacement {
                    worker: WorkerId("sim".into()),
                    block_tables: Vec::new(),
                    new_cache_pages: Vec::new(),
                    forward: ForwardBatch::default(),
                    latent: None,
                    decode: None,
                    buffers: Vec::new(),
                },
            )],
            vec![uniserve_worker_ipc::BatchCommand::Start { request: admission }],
            Vec::new(),
        )
    }

    #[test]
    fn same_request_computations_preserve_coordinates_and_sequential_tokens() {
        let mut executor = SimExecutor::new(SimEngine::new());
        let mut selected = batch(1, 3);
        let mut successor = batch(1, 4).requests.remove(0);
        successor.0.predecessor = Some(CallId::new(1, 3));
        // The selected prompt covers two positions, so its successor in the
        // same batch enters where it left off.
        successor.0.coordinates = uniserve_worker_ipc::CallCoordinates {
            logical_position: 2,
            kv_visible_len: 2,
            kv_computed_len: 2,
            flow_step: 0,
        };
        selected.requests.push(successor);
        executor.submit(selected).expect("submit logical batch");
        let report = executor
            .poll(Duration::from_secs(5))
            .expect("poll")
            .expect("result");
        assert!(report.done);
        let outputs = report
            .results
            .iter()
            .map(|result| &result.output)
            .collect::<Vec<_>>();
        assert_eq!(
            outputs
                .iter()
                .map(|output| output.call_id)
                .collect::<Vec<_>>(),
            vec![CallId::new(1, 3), CallId::new(1, 4)]
        );
        // Greedy synthetic tokens are 1000 + request_id * 7 + accepted index.
        assert_eq!(outputs[0].committed_tokens, vec![1063]);
        assert_eq!(outputs[1].committed_tokens, vec![1064]);
        assert_eq!(outputs[1].kv_visible_len, 4);
        executor.close().expect("close simulator");
    }

    #[test]
    fn default_limits_admit_public_image_geometry() {
        let engine = SimEngine::new();
        let info = engine.info();
        let latent_units = (2_048_u64 / 16) * (1_152_u64 / 16);
        assert!(latent_units <= info.latent_capacity_units());
    }
}
