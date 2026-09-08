//! GPU-free executor for scheduler and frontend behavior.
//!
//! The simulator consumes the same typed admissions and operations as worker
//! executors. It enforces lifecycle, version, and replay invariants and returns
//! resolved products with stable request, operation, and point identities.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::executor::{
    Batch, BatchResult, Executor, ExecutorInfo, ExecutorSubmitError, LogicalResultTracker,
    WorkerId, lower_batch,
};
use crate::worker::RunSubmitError;
use base64::Engine as _;
use crossbeam_channel::{Receiver, Sender};
use uniserve_core::philox;
use uniserve_core::product_blob::{LogprobBlob, RankedToken};
use uniserve_core::{
    CommandWaker, ImageParams, RequestId, SampleOutput, SamplingParams, try_apply_sampling_counts,
};
use uniserve_worker_ipc::{
    CheckpointPoint, DrawLayout, ErrorCode, InlineValue, LogicalLengths, ModelOutput, NewRequest,
    OpCode, OpStatus, Operation, ProductKind, ProductPayload, ProductRef, RegistrationAck,
    ResultData, ResultPayload, Run as PhysicalRun, RunResult, SamplingState, TimingCounters,
    TokenSpan, WorkerInfo, decode_sampling_state_bytes,
};

const DEFAULT_TEXT_LEN: usize = 8;
const FAKE_EOS_TOKEN: u32 = 151_645;
const SYNTH_VOCAB_SIZE: usize = FAKE_EOS_TOKEN as usize + 1;
const DEFAULT_DENOISE_STEPS: u16 = 50;
const DEFAULT_IMAGE_HW: (u32, u32) = (512, 512);

enum Job {
    Batch(PhysicalRun),
    Shutdown,
}

/// Runs the deterministic model simulator on a bounded asynchronous executor seam.
pub struct SimExecutor {
    executor_info: ExecutorInfo,
    depth: usize,
    to_worker: Sender<Job>,
    from_worker: Receiver<anyhow::Result<RunResult>>,
    progress_tx: Sender<()>,
    progress_rx: Receiver<()>,
    in_flight: usize,
    handle: Option<JoinHandle<()>>,
    next_collective_seq: u64,
    logical_results: LogicalResultTracker,
    admissions: HashSet<uniserve_worker_ipc::RequestKey>,
    products: HashSet<ProductRef>,
}

impl SimExecutor {
    /// Starts an asynchronous simulator with its advertised queue depth.
    pub fn new(engine: SimEngine) -> Self {
        let depth = (engine.info().queue_depth as usize).max(1);
        Self::with_depth(engine, depth)
    }

    /// Starts an asynchronous simulator with an explicit in-flight run limit.
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
                            if results_tx.send(engine.execute(batch)).is_err() {
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
            next_collective_seq: 1,
            logical_results: LogicalResultTracker::default(),
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
    /// Submits the run.
    fn submit_run(&mut self, batch: PhysicalRun) -> Result<(), RunSubmitError> {
        if self.in_flight >= self.depth {
            return Err(RunSubmitError::WouldBlock(batch));
        }
        batch
            .validate()
            .map_err(anyhow::Error::from)
            .map_err(RunSubmitError::Failed)?;
        self.to_worker
            .send(Job::Batch(batch))
            .map_err(|_| RunSubmitError::Failed(anyhow::anyhow!("sim executor thread gone")))?;
        self.in_flight += 1;
        Ok(())
    }

    /// Returns the result of a submitted physical run.
    fn poll_run(&mut self, timeout: Duration) -> anyhow::Result<Option<RunResult>> {
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
    fn submit(&mut self, batch: Batch) -> Result<(), ExecutorSubmitError> {
        if self.handle.is_none() {
            return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                "Executor is closed"
            )));
        }
        for op in &batch.ops {
            if !self.is_ready(&op.target.0) {
                return Err(ExecutorSubmitError::Failed(anyhow::anyhow!(
                    "Worker cannot accept this request"
                )));
            }
        }
        if self.in_flight >= self.depth {
            return Err(ExecutorSubmitError::WouldBlock(batch));
        }
        let run = lower_batch(&batch, &mut self.next_collective_seq)
            .map_err(ExecutorSubmitError::Failed)?;
        self.logical_results
            .register(&batch)
            .map_err(ExecutorSubmitError::Failed)?;
        match self.submit_run(run) {
            Ok(()) => {
                self.admissions
                    .extend(batch.admissions().map(|request| request.request_key));
                self.products.extend(
                    batch
                        .ops
                        .iter()
                        .flat_map(|op| op.payload.outputs.iter().cloned()),
                );
                Ok(())
            }
            Err(RunSubmitError::WouldBlock(_)) => {
                self.logical_results.unregister(batch.id);
                Err(ExecutorSubmitError::WouldBlock(batch))
            }
            Err(RunSubmitError::Failed(error)) => {
                self.logical_results.unregister(batch.id);
                Err(ExecutorSubmitError::Failed(error))
            }
        }
    }

    /// Polls for the next completed worker operation.
    fn poll(&mut self, timeout: Duration) -> anyhow::Result<Option<BatchResult>> {
        let Some(report) = self.poll_run(timeout)? else {
            return Ok(None);
        };
        let commands = self.logical_results.commands(report.batch_id).to_vec();
        let result = self.logical_results.apply(report)?;
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
                }
                | BatchCommand::Retire {
                    request_key,
                    retained_buffers,
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

/// The completion and its resolved product values recorded for a committed
/// operation, replayed verbatim when the same operation is resubmitted.
#[derive(Clone)]
struct RecordedCompletion {
    operation: Operation,
    completion: ModelOutput,
    products: Vec<ProductPayload>,
}

/// One lineage's authoritative state: the last committed fixed point, the
/// synthetic token cursor, denoise progress, and the terminal record of every
/// committed operation for replay.
#[derive(Clone)]
struct SimRequestState {
    admission: NewRequest,
    point_index: u32,
    logical_position: u32,
    kv_visible_len: u32,
    kv_published_len: u32,
    emitted: usize,
    flow_step: u16,
    predicate_values: HashMap<ProductRef, bool>,
    terminal: BTreeMap<u64, RecordedCompletion>,
    /// Committed penalty counts in ascending token order.
    ///
    /// Each generated token folds in during execution, allowing a successor to
    /// observe it before the predecessor is host-visible. Device executors encode
    /// the same state as a resident count tensor plus per-operation deltas.
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
            point_index: 0,
            logical_position: prefix_len,
            kv_visible_len: prefix_len,
            kv_published_len: 0,
            emitted: 0,
            flow_step: 0,
            predicate_values: HashMap::new(),
            terminal: BTreeMap::new(),
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
            supported_ops: OpCode::ALL.to_vec(),
            latent_page_units: 64,
            latent_pages: 1_025,
            buffer_pool_bytes: 257_u64 * (256 << 20),
            max_batch_ops: 1024,
            max_unresolved_ops: 2,
            model_name: "sim".to_owned(),
            weight_version: 0,
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
        operation: &Operation,
        request: &SimRequestState,
        index: usize,
        state: Option<&SamplingState>,
    ) -> anyhow::Result<Option<SampleOutput>> {
        let request_id = operation.request_key.request_id;
        let mut logits = Self::synth_logits(vocab, text_len, fake_eos, request_id, index);
        match request.sampling() {
            Some(sampling) => {
                let draw = if sampling.temperature > 0.0 {
                    let rng = operation.rng().ok_or_else(|| {
                        anyhow::anyhow!("stochastic sampling operation has no RNG coordinates")
                    })?;
                    anyhow::ensure!(
                        rng.draw_layout == DrawLayout::TargetSampling,
                        "stochastic sampling operation uses the wrong RNG layout"
                    );
                    anyhow::ensure!(
                        rng.seed == sampling.seed.unwrap_or(0),
                        "operation RNG seed disagrees with admitted sampling"
                    );
                    let key = philox::sampling_key(
                        rng.seed,
                        operation.request_key.authority_id,
                        request_id.0,
                        operation.request_key.epoch,
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
                let allowed = state
                    .and_then(|value| value.allowed_token_ids.as_deref())
                    .or_else(|| {
                        state
                            .is_none()
                            .then_some(sampling.allowed_token_ids.as_deref())
                            .flatten()
                    });
                let suppress = state
                    .map(|value| value.suppressed_token_ids.as_slice())
                    .filter(|tokens| !tokens.is_empty());
                // Processor step 2 forced-token constraint: a decode operation
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

    /// Executes one operation against its request, producing the terminal
    /// [`ModelOutput`] and any resolved output-product values.
    fn execute_operation(
        vocab: usize,
        text_len: usize,
        fake_eos: u32,
        operation: &Operation,
        request: &mut SimRequestState,
        input_products: &[ProductPayload],
    ) -> anyhow::Result<(ModelOutput, Vec<ProductPayload>)> {
        let point_index = request.point_index;
        let selected_point = u32::from(operation.advances_state());
        let mut record = ModelOutput {
            request_key: operation.request_key,
            op_id: operation.op_id,
            completion_slot_generation: ((operation.op_id.0 - 1) % u64::from(u32::MAX) + 1) as u32,
            status: OpStatus::Ok,
            selected_point,
            product_generations: operation
                .outputs()
                .iter()
                .map(|out| out.generation)
                .collect(),
            error_code: None,
            timing_counters: TimingCounters::default(),
            payload: ResultPayload::for_kind(operation.kind(), ResultData::default()),
        };
        record.logical_lengths_mut().token_len = request.logical_position;
        set_kv_lengths(
            record.logical_lengths_mut(),
            request.kv_visible_len,
            request.kv_visible_len,
            request.kv_published_len,
        );
        let mut products = Vec::new();

        match operation.kind() {
            work @ (OpCode::ArExtend | OpCode::ArDecode | OpCode::ArVerify) => {
                let visual_state = operation.inputs().iter().any(|input| {
                    matches!(
                        input.kind,
                        ProductKind::VisionFeature | ProductKind::LatentFeature
                    )
                });
                let samples_token = operation
                    .outputs()
                    .iter()
                    .any(|output| output.kind == ProductKind::Token);
                if visual_state {
                    request.kv_visible_len = request
                        .kv_visible_len
                        .saturating_add(operation.bounds().max_tokens);
                    set_kv_lengths(
                        record.logical_lengths_mut(),
                        request.kv_visible_len,
                        request.kv_visible_len,
                        request.kv_published_len,
                    );
                    if operation
                        .outputs()
                        .iter()
                        .any(|output| output.kind == ProductKind::Completion)
                    {
                        request.emitted = 0;
                    }
                }
                if !visual_state && !samples_token {
                    request.kv_visible_len = request
                        .kv_visible_len
                        .saturating_add(operation.bounds().max_tokens);
                    record.logical_lengths_mut().token_len = request.logical_position;
                    set_kv_lengths(
                        record.logical_lengths_mut(),
                        request.kv_visible_len,
                        request.kv_visible_len,
                        request.kv_published_len,
                    );
                } else if samples_token {
                    let index = request.emitted;
                    let sampling_state = operation_sampling_state(operation, input_products)?;
                    let Some(output) = Self::sample(
                        vocab,
                        text_len,
                        fake_eos,
                        operation,
                        request,
                        index,
                        sampling_state.as_ref(),
                    )?
                    else {
                        record.status = OpStatus::Error;
                        record.selected_point = point_index;
                        record.product_generations.clear();
                        record.error_code = Some(ErrorCode::InvalidOperation);
                        return Ok((record, Vec::new()));
                    };
                    record.finish_flags_mut().eos = output.token == fake_eos;
                    if let Some(state) = sampling_state.as_ref() {
                        record.finish_flags_mut().stop =
                            state.finish_token_ids.binary_search(&output.token).is_ok()
                                && !record.finish_flags().eos;
                        record.finish_flags_mut().length = state.force_finish;
                    }
                    let admitted_stops = request
                        .admission
                        .ar
                        .as_ref()
                        .map_or(&[][..], |und| und.finish_token_ids.as_slice());
                    let continuation = admitted_stops.binary_search(&output.token).is_err()
                        && sampling_state.as_ref().is_none_or(|state| {
                            !state.force_finish
                                && state.finish_token_ids.binary_search(&output.token).is_err()
                        });
                    if let Some(token_product) = operation
                        .outputs()
                        .iter()
                        .find(|output| output.kind == ProductKind::Token)
                    {
                        request
                            .predicate_values
                            .insert(token_product.clone(), continuation);
                    }
                    let transition = sampling_state.as_ref().is_some_and(|state| {
                        state
                            .transition_token_ids
                            .binary_search(&output.token)
                            .is_ok()
                    });
                    for completion in operation.outputs().iter().filter(|candidate| {
                        candidate.kind == ProductKind::Completion
                            && matches!(candidate.output_index, 4 | 6)
                    }) {
                        request
                            .predicate_values
                            .insert(completion.clone(), transition);
                    }
                    *record.token_span_mut() = TokenSpan {
                        base: index as u32,
                        len: 1,
                    };
                    record.logical_lengths_mut().token_len = 1;
                    if !visual_state {
                        let query_tokens = match work {
                            OpCode::ArExtend => operation.bounds().max_tokens,
                            OpCode::ArDecode | OpCode::ArVerify => 1,
                            _ => unreachable!(),
                        };
                        request.logical_position =
                            request.logical_position.saturating_add(query_tokens);
                        request.kv_visible_len =
                            request.kv_visible_len.saturating_add(query_tokens);
                        record.logical_lengths_mut().token_len = request.logical_position;
                        set_kv_lengths(
                            record.logical_lengths_mut(),
                            request.kv_visible_len,
                            request.kv_visible_len,
                            request.kv_published_len,
                        );
                    }
                    match work {
                        OpCode::ArExtend => request.emitted = request.emitted.max(1),
                        OpCode::ArDecode | OpCode::ArVerify => {
                            request.emitted = request.emitted.saturating_add(1)
                        }
                        _ => unreachable!(),
                    }
                    *record.committed_tokens_mut() = vec![output.token];
                    // Fold the generated token into the device-resident penalty
                    // base so the next operation's penalties see it before this
                    // one is host-observed. A false-predicate no-op never reaches
                    // this branch, so a retracted point is never folded.
                    request.fold_penalty_token(output.token);
                    let blob = LogprobBlob {
                        sampled_logprob: request
                            .sampling()
                            .is_some_and(SamplingParams::generated_logprobs_requested)
                            .then_some(output.logprob),
                        top_logprobs: output
                            .top
                            .iter()
                            .map(|(token_id, logprob, rank)| RankedToken {
                                token_id: *token_id,
                                logprob: *logprob,
                                rank: *rank,
                            })
                            .collect(),
                        prompt_logprobs: Vec::new(),
                    };
                    if !blob.is_empty() {
                        products.push(ProductPayload {
                            product: output_ref(operation, ProductKind::Logprob)?,
                            value: InlineValue::Bytes(blob.encode()),
                        });
                    }
                }
            }
            OpCode::EncoderText | OpCode::EncoderVision | OpCode::EncoderLatent => {}
            work @ (OpCode::TransferProduct
            | OpCode::TransferKvPublish
            | OpCode::TransferKvInstall) => {
                if work == OpCode::TransferKvPublish {
                    request.kv_published_len = request.kv_visible_len;
                }
                set_kv_lengths(
                    record.logical_lengths_mut(),
                    request.kv_visible_len,
                    request.kv_visible_len,
                    request.kv_published_len,
                );
            }
            OpCode::DiffusionPrepare => {}
            OpCode::DiffusionStep => {
                let steps = operation.bounds().max_tokens.max(1) as u16;
                request.flow_step = request.flow_step.saturating_add(steps);
                let total = request
                    .image()
                    .map(|image| image.steps)
                    .unwrap_or(DEFAULT_DENOISE_STEPS);
                record.logical_lengths_mut().latent_len = u32::from(request.flow_step);
                // Denoise completion surfaces as a length finish once the flow has
                // advanced through every scheduled step.
                record.finish_flags_mut().length = request.flow_step >= total;
            }
            OpCode::DiffusionDecode | OpCode::MediaAppend => {}
            OpCode::DiffusionFinalize => {
                request.flow_step = 0;
                if let Some(image) = request.image().cloned() {
                    let (height, width) = if image.height > 0 && image.width > 0 {
                        (image.height, image.width)
                    } else {
                        DEFAULT_IMAGE_HW
                    };
                    let png_base64 = synthetic_png_b64(width, height)?;
                    products.push(ProductPayload {
                        product: output_ref(operation, ProductKind::Artifact)?,
                        value: InlineValue::Bytes(png_base64.into_bytes()),
                    });
                }
            }
        }

        for completion in operation
            .outputs()
            .iter()
            .filter(|output| output.kind == ProductKind::Completion)
        {
            request
                .predicate_values
                .entry(completion.clone())
                .or_insert(true);
        }

        // Every declared output surfaces as a resolvable product for the host.
        // The Artifact/Logprob payloads above carry real bytes; the remaining
        // declared KV, token, latent, and feature references surface as
        // presence-only entries addressed by identity, since the host does not
        // consume their values.
        for output in operation.outputs() {
            if products
                .iter()
                .any(|payload| payload.product.output_index == output.output_index)
            {
                continue;
            }
            products.push(ProductPayload {
                product: output.clone(),
                value: InlineValue::Bytes(Vec::new()),
            });
        }

        Ok((record, products))
    }

    /// Builds a completion for an operation resolved entirely by its execution predicate.
    fn predicated_completion(operation: &Operation, request: &SimRequestState) -> ModelOutput {
        ModelOutput {
            request_key: operation.request_key,
            op_id: operation.op_id,
            completion_slot_generation: ((operation.op_id.0 - 1) % u64::from(u32::MAX) + 1) as u32,
            status: OpStatus::Predicated,
            selected_point: request.point_index,
            product_generations: Vec::new(),
            error_code: None,
            timing_counters: TimingCounters::default(),
            payload: ResultPayload::for_kind(
                operation.kind(),
                ResultData {
                    logical_lengths: LogicalLengths {
                        token_len: request.logical_position,
                        kv_visible_len: request.kv_visible_len,
                        kv_computed_len: request.kv_visible_len,
                        ..LogicalLengths::default()
                    },
                    token_span: TokenSpan {
                        base: request.emitted.min(u32::MAX as usize) as u32,
                        len: 0,
                    },
                    ..ResultData::default()
                },
            ),
        }
    }

    /// Sets the advertised unresolved-run capacity, clamped to at least one.
    pub fn set_pipeline_depth(&mut self, depth: u32) {
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

/// Decodes the branch-local state declared for one sampling operation.
fn operation_sampling_state(
    operation: &Operation,
    input_products: &[ProductPayload],
) -> anyhow::Result<Option<SamplingState>> {
    let mut references = operation
        .inputs()
        .iter()
        .filter(|input| input.kind == ProductKind::SamplingState);
    let Some(reference) = references.next() else {
        return Ok(None);
    };
    anyhow::ensure!(
        references.next().is_none(),
        "operation has multiple sampling-state inputs"
    );
    let payload = input_products
        .iter()
        .find(|payload| payload.product == *reference)
        .ok_or_else(|| anyhow::anyhow!("operation sampling-state input has no payload"))?;
    let bytes = payload
        .value
        .bytes()
        .ok_or_else(|| anyhow::anyhow!("sampling state cannot be a transfer handle"))?;
    Ok(Some(decode_sampling_state_bytes(bytes)?))
}

/// Sets every logical KV frontier to the visible token position.
fn set_kv_lengths(lengths: &mut LogicalLengths, visible: u32, _committed: u32, _published: u32) {
    lengths.kv_visible_len = visible;
    lengths.kv_computed_len = visible;
}

/// Returns the declared output-product reference for a completion value.
fn output_ref(operation: &Operation, kind: ProductKind) -> anyhow::Result<ProductRef> {
    operation
        .outputs()
        .iter()
        .find(|output| output.kind == kind)
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "operation {:?} has no declared {kind:?} output",
                operation.op_id
            )
        })
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

    /// Executes one physical run against deterministic in-memory model state.
    fn execute(&mut self, batch: PhysicalRun) -> anyhow::Result<RunResult> {
        batch.validate()?;
        let batch_id = batch.batch_id;
        let run_id = batch.run_id;
        let closed = batch
            .commands
            .iter()
            .filter_map(|control| match control {
                uniserve_worker_ipc::BatchCommand::Finish { request_key, .. }
                | uniserve_worker_ipc::BatchCommand::Retire { request_key, .. } => {
                    Some(request_key.request_id)
                }
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
        let input_products = batch.input_products;
        let vocab = self.vocab;
        let text_len = self.text_len;
        let fake_eos = self.fake_eos;
        let mut completions = Vec::with_capacity(batch.operations.len());
        let mut products = Vec::new();
        for operation in batch.operations {
            let request = self
                .requests
                .get_mut(&operation.request_key.request_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "request {} has no admission",
                        operation.request_key.request_id.0
                    )
                })?;
            if let Some(recorded) = request.terminal.get(&operation.op_id.0) {
                anyhow::ensure!(
                    recorded.operation == operation,
                    "operation {} conflicts with its terminal record",
                    operation.op_id.0
                );
                completions.push(recorded.completion.clone());
                products.extend(recorded.products.clone());
                continue;
            }
            anyhow::ensure!(
                operation.request_key == request.admission.request_key,
                "operation identity {:?} does not match its admitted lineage",
                operation.request_key
            );
            if let Some(parent) = &operation.parent {
                match &parent.point {
                    CheckpointPoint::Fixed(point) => {
                        anyhow::ensure!(
                            *point == request.point_index,
                            "operation {} ({}) parent point {} does not match request point {}",
                            operation.op_id.0,
                            operation.kind().as_str(),
                            point,
                            request.point_index
                        );
                    }
                    CheckpointPoint::DeviceSelected => {
                        // A device-relay successor roots on its predecessor's
                        // selected point before host observation. By the time it
                        // runs, the predecessor has committed and advanced this
                        // request, so its point is the current request point and its
                        // terminal record names the referenced producer. The base
                        // point is the request's tracked point, not the absent
                        // device point index.
                        let producer_op_id = parent.op_id.0;
                        let recorded = request.terminal.get(&producer_op_id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "device parent names unknown predecessor op {producer_op_id}"
                            )
                        })?;
                        anyhow::ensure!(
                            recorded.completion.selected_point == request.point_index,
                            "device parent predecessor is not the request's committed point"
                        );
                    }
                }
            }

            if let Some(predicate) = operation.predicate().as_ref() {
                let predicate_value = request
                    .predicate_values
                    .get(predicate)
                    .copied()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "operation {} predicate names an unresolved product",
                            operation.op_id.0
                        )
                    })?;
                if !predicate_value {
                    for output in operation.outputs() {
                        request.predicate_values.insert(output.clone(), false);
                    }
                    let completion = Self::predicated_completion(&operation, request);
                    request.terminal.insert(
                        operation.op_id.0,
                        RecordedCompletion {
                            operation: operation.clone(),
                            completion: completion.clone(),
                            products: Vec::new(),
                        },
                    );
                    completions.push(completion);
                    continue;
                }
            }

            let (completion, op_products) = Self::execute_operation(
                vocab,
                text_len,
                fake_eos,
                &operation,
                request,
                &input_products,
            )?;
            if operation.advances_state() && completion.status == OpStatus::Ok {
                request.point_index = completion.selected_point;
            }
            request.terminal.insert(
                operation.op_id.0,
                RecordedCompletion {
                    operation,
                    completion: completion.clone(),
                    products: op_products.clone(),
                },
            );
            completions.push(completion);
            products.extend(op_products);
        }
        let report = RunResult {
            batch_id,
            run_id,
            completions,
            products,
            registration: RegistrationAck { visible: true },
            worker_exec_us: None,
            forward_stats: None,
            done: true,
        };
        for request_id in closed {
            self.drop_request(request_id)?;
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
        ArRequestParams, Bounds, Checkpoint, DType, OpId, OpPayload, PointRange, ProductRef,
        RequestKey, ShapeBound, StorageClass,
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

    fn token_outputs(op_id: OpId) -> Vec<ProductRef> {
        [
            (ProductKind::Token, DType::U32, Vec::new()),
            (ProductKind::SelectedPoint, DType::U32, Vec::new()),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (kind, dtype, dims))| ProductRef {
            request_key: request_key(),
            producer_op_id: op_id,
            output_index: index as u16,
            generation: index as u32 + 1,
            kind,
            storage_class: StorageClass::RequestRelay,
            dtype,
            shape_bound: ShapeBound { dims },
            point_range: PointRange {
                base_point: 0,
                max_points: 1,
            },
        })
        .collect()
    }

    fn batch(run_id: u64, op_id: u64) -> PhysicalRun {
        let request_key = request_key();
        let admission = admission();
        let parent = Checkpoint::admission_root(OpId(1));
        let operation = Operation {
            request_key,
            op_id: OpId(op_id),
            parent: Some(parent),
            entry: "model".into(),
            payload: OpPayload::new(
                OpCode::ArExtend,
                Bounds {
                    max_points: 1,
                    max_tokens: 2,
                    ..Bounds::default()
                },
                Vec::new(),
                token_outputs(OpId(op_id)),
                None,
                None,
                0,
            ),
        }
        .sealed();
        PhysicalRun::new(run_id, vec![admission], vec![operation])
    }

    #[test]
    fn duplicate_operation_replays_one_committed_record() {
        let mut engine = SimEngine::new();
        let first = engine.execute(batch(4, 17)).expect("first execution");
        let replay = engine.execute(batch(5, 17)).expect("replay execution");
        assert_eq!(first.completions, replay.completions);
    }

    #[test]
    fn extend_commits_the_greedy_synthetic_token() {
        // The synthetic distribution's natural token for request 9 at index 0 is
        // `1000 + (9*7 + 0) % 5000 = 1063`; greedy default sampling commits it.
        let mut engine = SimEngine::new();
        let report = engine.execute(batch(1, 3)).expect("execution");
        let completion = report.completions().next().unwrap();
        assert_eq!(report.completions().count(), 1);
        assert_eq!(completion.committed_tokens(), vec![1063]);
        assert_eq!(completion.selected_point, 1);
        assert!(report.registration.visible);
    }

    #[test]
    fn default_limits_admit_public_image_geometry() {
        let engine = SimEngine::new();
        let info = engine.info();
        let latent_units = (2_048_u64 / 16) * (1_152_u64 / 16);
        assert!(latent_units <= info.latent_capacity_units());
    }
}
