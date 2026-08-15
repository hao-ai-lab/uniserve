//! GPU-free execution engine for scheduler and frontend conformance tests.
//!
//! The simulator is a strict peer of the production execution protocol: it
//! consumes typed admissions and operations, enforces lifecycle, version, and
//! replay invariants, and returns one [`CompletionRecord`] per operation with the
//! resolved output-product values a host consumes. Every state point is named by
//! its point index and semantic digest, exactly as a real worker names it.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{BTreeMap, HashMap};
use std::thread::JoinHandle;
use std::time::Duration;

use base64::Engine as _;
use crossbeam_channel::{Receiver, Sender};
use uniserve_core::philox;
use uniserve_core::product_blob::{LogprobBlob, RankedToken};
use uniserve_core::{
    CommandWaker, ImageParams, RequestId, SampleOutput, SamplingParams, try_apply_sampling_counts,
};
use uniserve_executor::{ControlAck, ControlOp, Executor, ModelEngine};
use uniserve_worker_wire::{
    Admission, Batch, CompletionRecord, CompletionReport, Digest, DrawLayout, ErrorCode,
    FinishFlags, GenMode, LogicalLengths, MixedExecutionCapability, OpStatus, Operation,
    PartitionCompletion, Point, ProductKind, ProductPayload, ProductRef, RegistrationAck,
    RequestKind, ResourceClass, SamplingState, TimingCounters, TokenMode, TokenSpan, TransferMode,
    Work, WorkVariant, WorkerCapabilities, decode_sampling_state_bytes,
};

const DEFAULT_TEXT_LEN: usize = 8;
const FAKE_EOS_TOKEN: u32 = 151_645;
const SYNTH_VOCAB_SIZE: usize = FAKE_EOS_TOKEN as usize + 1;
const DEFAULT_DENOISE_STEPS: u16 = 50;
const DEFAULT_IMAGE_HW: (u32, u32) = (512, 512);

enum Job {
    Batch(Batch),
    Drop(RequestId),
    Shutdown,
}

/// Runs a synchronous [`ModelEngine`] on a bounded asynchronous executor seam.
pub struct SimExecutor {
    caps: WorkerCapabilities,
    depth: usize,
    to_worker: Sender<Job>,
    from_worker: Receiver<anyhow::Result<CompletionReport>>,
    progress_tx: Sender<()>,
    progress_rx: Receiver<()>,
    in_flight: usize,
    next_call_id: u64,
    handle: Option<JoinHandle<()>>,
}

impl SimExecutor {
    pub fn new(engine: Box<dyn ModelEngine>) -> Self {
        let depth = (engine.caps().pipeline_depth as usize).max(1);
        Self::with_depth(engine, depth)
    }

    pub fn with_depth(mut engine: Box<dyn ModelEngine>, depth: usize) -> Self {
        let caps = engine.caps();
        let depth = depth.max(1);
        let (to_worker, jobs) = crossbeam_channel::unbounded();
        let (results_tx, from_worker) = crossbeam_channel::unbounded();
        let (progress_tx, progress_rx) = crossbeam_channel::bounded(1);
        let worker_progress = progress_tx.clone();
        let handle = std::thread::Builder::new()
            .name("uniserve-sim-executor".into())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    match job {
                        Job::Batch(batch) => {
                            if results_tx.send(engine.execute(batch)).is_err() {
                                break;
                            }
                            let _ = worker_progress.try_send(());
                        }
                        Job::Drop(session_id) => {
                            let _ = engine.drop_session(session_id);
                        }
                        Job::Shutdown => break,
                    }
                }
            })
            .expect("spawn sim executor thread");
        Self {
            caps,
            depth,
            to_worker,
            from_worker,
            progress_tx,
            progress_rx,
            in_flight: 0,
            next_call_id: 1,
            handle: Some(handle),
        }
    }

    fn apply_control(&mut self, operation: &ControlOp) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.caps
                .supported_controls
                .contains(&operation.request_kind()),
            "sim executor does not support control {}",
            operation.method()
        );
        match operation {
            ControlOp::DropSession(session_id) => {
                self.to_worker
                    .send(Job::Drop(*session_id))
                    .map_err(|_| anyhow::anyhow!("sim executor thread gone"))?;
            }
            ControlOp::CopyKv(_) | ControlOp::ReleaseProducts(_) => {}
            ControlOp::SnapshotSession(_) | ControlOp::RestoreSession { .. } => {
                unreachable!("unsupported controls are rejected before dispatch")
            }
        }
        Ok(())
    }
}

impl Executor for SimExecutor {
    fn caps(&self) -> WorkerCapabilities {
        self.caps.clone()
    }

    fn pipeline_depth(&self) -> usize {
        self.depth
    }

    fn in_flight(&self) -> usize {
        self.in_flight
    }

    fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
        batch.validate()?;
        self.to_worker
            .send(Job::Batch(batch))
            .map_err(|_| anyhow::anyhow!("sim executor thread gone"))?;
        self.in_flight += 1;
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
        match self.from_worker.try_recv() {
            Ok(result) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                Ok(Some(result?))
            }
            Err(crossbeam_channel::TryRecvError::Empty) => Ok(None),
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                Err(anyhow::anyhow!("sim executor thread disconnected"))
            }
        }
    }

    fn command_waker(&self) -> CommandWaker {
        let progress = self.progress_tx.clone();
        CommandWaker::new(move || {
            let _ = progress.try_send(());
        })
    }

    fn park_for_event(&mut self, timeout: Duration) -> anyhow::Result<()> {
        match self.progress_rx.recv_timeout(timeout) {
            Ok(()) | Err(crossbeam_channel::RecvTimeoutError::Timeout) => Ok(()),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => Err(anyhow::anyhow!(
                "sim executor progress channel disconnected"
            )),
        }
    }

    fn wait_result_timeout(
        &mut self,
        timeout: Duration,
    ) -> anyhow::Result<Option<CompletionReport>> {
        match self.from_worker.recv_timeout(timeout) {
            Ok(result) => {
                self.in_flight = self.in_flight.saturating_sub(1);
                Ok(Some(result?))
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => Ok(None),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                Err(anyhow::anyhow!("sim executor thread disconnected"))
            }
        }
    }

    fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
        let result = self
            .from_worker
            .recv()
            .map_err(|_| anyhow::anyhow!("sim executor thread disconnected"))?;
        self.in_flight = self.in_flight.saturating_sub(1);
        result
    }

    fn control(&mut self, operation: ControlOp) -> anyhow::Result<u64> {
        let call_id = self.next_call_id;
        self.next_call_id = self.next_call_id.saturating_add(1);
        self.apply_control(&operation)?;
        Ok(call_id)
    }

    fn control_wait(
        &mut self,
        operation: ControlOp,
        _targets: Option<&[u32]>,
    ) -> anyhow::Result<Vec<ControlAck>> {
        self.apply_control(&operation)?;
        Ok(vec![ControlAck {
            rank: 0,
            ok: true,
            message: None,
            snapshot: None,
        }])
    }

    fn shutdown(&mut self) {
        let _ = self.to_worker.send(Job::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SimExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The completion and its resolved product values recorded for a committed
/// operation, replayed verbatim when the same operation is resubmitted.
#[derive(Clone)]
struct RecordedCompletion {
    plan_digest: Digest,
    completion: CompletionRecord,
    products: Vec<ProductPayload>,
}

/// One lineage's authoritative state: the last committed fixed point, the
/// synthetic token cursor, denoise progress, and the terminal record of every
/// committed operation for replay.
#[derive(Clone)]
struct SimSession {
    admission: Admission,
    point_index: u32,
    committed_semantic: Digest,
    logical_position: u32,
    kv_visible_len: u32,
    kv_published_len: u32,
    emitted: usize,
    flow_step: u16,
    predicate_values: HashMap<ProductRef, bool>,
    terminal: BTreeMap<u64, RecordedCompletion>,
    /// Device-resident committed penalty count base. Each generated token folds
    /// in as its operation executes; a successor reads it before its
    /// predecessor is host-observed, so penalties are device-continuous. The
    /// GPU-free oracle keeps a host histogram; the production worker keeps the
    /// equivalent device count tensor plus per-operation deltas.
    penalty_counts: BTreeMap<u32, u32>,
}

impl SimSession {
    fn new(admission: Admission) -> Self {
        let committed_semantic = admission.digest.clone();
        let prefix_len = admission
            .und
            .as_ref()
            .map_or(0, |branch| branch.kv.prefix_len);
        Self {
            admission,
            point_index: 0,
            committed_semantic,
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

    /// The committed recent-output histogram used for penalties, in canonical
    /// ascending token order.
    fn recent_counts(&self) -> Vec<(u32, u32)> {
        self.penalty_counts
            .iter()
            .map(|(token, count)| (*token, *count))
            .collect()
    }

    /// Fold one generated token into the committed penalty base.
    fn fold_penalty_token(&mut self, token: u32) {
        self.penalty_counts
            .entry(token)
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);
    }

    fn sampling(&self) -> Option<&SamplingParams> {
        self.admission.und.as_ref().map(|und| &und.sampling)
    }

    fn image(&self) -> Option<&ImageParams> {
        self.admission
            .gen_admission
            .as_ref()
            .map(|branch| &branch.image)
    }
}

/// Deterministic local model engine with protocol-faithful lifecycle state.
pub struct SimEngine {
    caps: WorkerCapabilities,
    text_len: usize,
    fake_eos: u32,
    vocab: usize,
    sessions: HashMap<RequestId, SimSession>,
}

impl SimEngine {
    pub fn new() -> Self {
        let caps = WorkerCapabilities {
            supported_work: vec![
                WorkVariant::TokenExtend,
                WorkVariant::TokenDecode,
                WorkVariant::EncodeVision,
                WorkVariant::EncodeLatent,
                WorkVariant::GenTransition,
                WorkVariant::GenFlow,
                WorkVariant::Materialize,
                WorkVariant::TransferProduct,
                WorkVariant::TransferKvPublish,
                WorkVariant::TransferKvInstall,
            ],
            latent_page_units: 64,
            num_latent_pages: 1_025,
            latent_width: 16,
            latent_dtype: "bfloat16".into(),
            latent_downsample: 16,
            max_vae_grid_tokens: 1_024,
            max_vit_grid_tokens: 64,
            max_latent_feature_bytes: 1 << 20,
            max_vision_feature_bytes: 1 << 20,
            encoder_cache_budget: 256,
            supported_controls: vec![
                RequestKind::DropSession,
                RequestKind::CopyKv,
                RequestKind::ReleaseProducts,
            ],
            max_batch_operations: 1024,
            max_unresolved_window: 2,
            mixed_buckets: (1..=128)
                .flat_map(|decode_rows| {
                    (1..=3).map(move |cfg_branches| MixedExecutionCapability {
                        decode_rows,
                        flow_rows: 1,
                        height: DEFAULT_IMAGE_HW.0,
                        width: DEFAULT_IMAGE_HW.1,
                        cfg_branches,
                    })
                })
                .collect(),
            resource_classes: vec![ResourceClass::ImageLatent, ResourceClass::Scratch],
            model_identity: "0".repeat(64),
            weight_digest: "1".repeat(64),
            ..WorkerCapabilities::default()
        };
        Self {
            caps,
            text_len: DEFAULT_TEXT_LEN,
            fake_eos: FAKE_EOS_TOKEN,
            vocab: SYNTH_VOCAB_SIZE,
            sessions: HashMap::new(),
        }
    }

    fn synth_logits(&self, session_id: RequestId, index: usize) -> Vec<f32> {
        let mut logits = vec![0.0; self.vocab];
        let natural = if index >= self.text_len {
            self.fake_eos
        } else {
            1_000 + ((session_id.0 as u32 * 7 + index as u32) % 5_000)
        };
        logits[natural as usize] = 10.0;
        let alternate_one = 1_000 + ((session_id.0 as u32 * 13 + index as u32 + 1) % 5_000);
        let alternate_two = 1_000 + ((session_id.0 as u32 * 29 + index as u32 + 2) % 5_000);
        if alternate_one != natural {
            logits[alternate_one as usize] = 8.0;
        }
        if alternate_two != natural {
            logits[alternate_two as usize] = 6.0;
        }
        logits[self.fake_eos as usize] = if index >= self.text_len { 100.0 } else { 1.0 };
        logits
    }

    fn sample(
        &self,
        operation: &Operation,
        session: &SimSession,
        index: usize,
        state: Option<&SamplingState>,
    ) -> anyhow::Result<Option<SampleOutput>> {
        let session_id = operation.request_key.session_id;
        let mut logits = self.synth_logits(session_id, index);
        match session.sampling() {
            Some(sampling) => {
                let draw = if sampling.temperature > 0.0 {
                    let rng = operation.rng.ok_or_else(|| {
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
                        session_id.0,
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
                let recent_counts = session.recent_counts();
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
                token: if index >= self.text_len {
                    self.fake_eos
                } else {
                    1_000 + ((session_id.0 as u32 * 7 + index as u32) % 5_000)
                },
                logprob: 0.0,
                top: Vec::new(),
            })),
        }
    }

    /// Execute one operation against its session, producing the terminal
    /// [`CompletionRecord`] and any resolved output-product values.
    fn execute_operation(
        &self,
        operation: &Operation,
        session: &mut SimSession,
        input_products: &[ProductPayload],
    ) -> anyhow::Result<(CompletionRecord, Vec<ProductPayload>)> {
        let point_index = session.point_index;
        let selected_point = u32::from(operation.advances_state);
        let parent_semantic = session.committed_semantic.clone();

        let mut record = CompletionRecord {
            request_key: operation.request_key,
            op_id: operation.op_id,
            completion_slot_generation: ((operation.op_id.0 - 1) % u64::from(u32::MAX) + 1) as u32,
            status: OpStatus::Ok,
            selected_point,
            logical_lengths: LogicalLengths::default(),
            token_span: TokenSpan::default(),
            committed_tokens: Vec::new(),
            finish_flags: FinishFlags::default(),
            product_generations: operation.outputs.iter().map(|out| out.generation).collect(),
            semantic_digest: String::new(),
            error_code: None,
            timing_counters: TimingCounters::default(),
        };
        record.logical_lengths.token_len = session.logical_position;
        set_kv_lengths(
            &mut record.logical_lengths,
            session.kv_visible_len,
            session.kv_visible_len,
            session.kv_published_len,
        );
        let mut products = Vec::new();

        match operation.work {
            Work::Token(mode) => {
                let visual_state = operation.inputs.iter().any(|input| {
                    matches!(
                        input.kind,
                        ProductKind::VisionFeature | ProductKind::LatentFeature
                    )
                });
                let samples_token = operation
                    .outputs
                    .iter()
                    .any(|output| output.kind == ProductKind::Token);
                if visual_state {
                    session.kv_visible_len = session
                        .kv_visible_len
                        .saturating_add(operation.bounds.max_tokens);
                    set_kv_lengths(
                        &mut record.logical_lengths,
                        session.kv_visible_len,
                        session.kv_visible_len,
                        session.kv_published_len,
                    );
                    if operation
                        .outputs
                        .iter()
                        .any(|output| output.kind == ProductKind::Completion)
                    {
                        session.emitted = 0;
                    }
                }
                if !visual_state && !samples_token {
                    session.kv_visible_len = session
                        .kv_visible_len
                        .saturating_add(operation.bounds.max_tokens);
                    record.logical_lengths.token_len = session.logical_position;
                    set_kv_lengths(
                        &mut record.logical_lengths,
                        session.kv_visible_len,
                        session.kv_visible_len,
                        session.kv_published_len,
                    );
                } else if samples_token {
                    let index = session.emitted;
                    let sampling_state = operation_sampling_state(operation, input_products)?;
                    let Some(output) =
                        self.sample(operation, session, index, sampling_state.as_ref())?
                    else {
                        record.status = OpStatus::Error;
                        record.selected_point = point_index;
                        record.product_generations.clear();
                        record.error_code = Some(ErrorCode::InvalidOperation);
                        record.semantic_digest = record
                            .compute_semantic_digest(&parent_semantic, &operation.plan_digest);
                        return Ok((record, Vec::new()));
                    };
                    record.finish_flags.eos = output.token == self.fake_eos;
                    if let Some(state) = sampling_state.as_ref() {
                        record.finish_flags.stop =
                            state.finish_token_ids.binary_search(&output.token).is_ok()
                                && !record.finish_flags.eos;
                        record.finish_flags.length = state.force_finish;
                    }
                    let admitted_stops = session
                        .admission
                        .und
                        .as_ref()
                        .map_or(&[][..], |und| und.finish_token_ids.as_slice());
                    let continuation = admitted_stops.binary_search(&output.token).is_err()
                        && sampling_state.as_ref().is_none_or(|state| {
                            !state.force_finish
                                && state.finish_token_ids.binary_search(&output.token).is_err()
                        });
                    if let Some(token_product) = operation
                        .outputs
                        .iter()
                        .find(|output| output.kind == ProductKind::Token)
                    {
                        session
                            .predicate_values
                            .insert(token_product.clone(), continuation);
                    }
                    let transition = sampling_state.as_ref().is_some_and(|state| {
                        state
                            .transition_token_ids
                            .binary_search(&output.token)
                            .is_ok()
                    });
                    for completion in operation.outputs.iter().filter(|candidate| {
                        candidate.kind == ProductKind::Completion
                            && matches!(candidate.output_index, 4 | 6)
                    }) {
                        session
                            .predicate_values
                            .insert(completion.clone(), transition);
                    }
                    record.token_span = TokenSpan {
                        base: index as u32,
                        len: 1,
                    };
                    record.logical_lengths.token_len = 1;
                    if !visual_state {
                        let query_tokens = match mode {
                            TokenMode::Extend => operation.bounds.max_tokens,
                            TokenMode::Decode | TokenMode::Verify => 1,
                        };
                        session.logical_position =
                            session.logical_position.saturating_add(query_tokens);
                        session.kv_visible_len =
                            session.kv_visible_len.saturating_add(query_tokens);
                        record.logical_lengths.token_len = session.logical_position;
                        set_kv_lengths(
                            &mut record.logical_lengths,
                            session.kv_visible_len,
                            session.kv_visible_len,
                            session.kv_published_len,
                        );
                    }
                    match mode {
                        TokenMode::Extend => session.emitted = session.emitted.max(1),
                        TokenMode::Decode | TokenMode::Verify => {
                            session.emitted = session.emitted.saturating_add(1)
                        }
                    }
                    record.committed_tokens = vec![output.token];
                    // Fold the generated token into the device-resident penalty
                    // base so the next operation's penalties see it before this
                    // one is host-observed. A false-predicate no-op never reaches
                    // this branch, so a retracted point is never folded.
                    session.fold_penalty_token(output.token);
                    let blob = LogprobBlob {
                        sampled_logprob: session
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
                            bytes: blob.encode(),
                        });
                    }
                }
            }
            Work::Draft => {}
            Work::Encode(_) => {}
            Work::Transfer(mode) => {
                if mode == TransferMode::KvPublish {
                    session.kv_published_len = session.kv_visible_len;
                }
                set_kv_lengths(
                    &mut record.logical_lengths,
                    session.kv_visible_len,
                    session.kv_visible_len,
                    session.kv_published_len,
                );
            }
            Work::Gen(GenMode::Transition) => {}
            Work::Gen(GenMode::Flow) => {
                let steps = operation.bounds.max_tokens.max(1) as u16;
                session.flow_step = session.flow_step.saturating_add(steps);
                let total = session
                    .image()
                    .map(|image| image.steps)
                    .unwrap_or(DEFAULT_DENOISE_STEPS);
                record.logical_lengths.latent_len = u32::from(session.flow_step);
                // Denoise completion surfaces as a length finish once the flow has
                // advanced through every scheduled step.
                record.finish_flags.length = session.flow_step >= total;
            }
            Work::Materialize => {
                session.flow_step = 0;
                if let Some(image) = session.image().cloned() {
                    let (height, width) = if image.height > 0 && image.width > 0 {
                        (image.height, image.width)
                    } else {
                        DEFAULT_IMAGE_HW
                    };
                    let png_base64 = synthetic_png_b64(width, height)?;
                    products.push(ProductPayload {
                        product: output_ref(operation, ProductKind::Artifact)?,
                        bytes: png_base64.into_bytes(),
                    });
                }
            }
        }

        for completion in operation
            .outputs
            .iter()
            .filter(|output| output.kind == ProductKind::Completion)
        {
            session
                .predicate_values
                .entry(completion.clone())
                .or_insert(true);
        }

        // Every declared output surfaces as a resolvable product for the host.
        // The Artifact/Logprob payloads above carry real bytes; the remaining
        // declared KV, token, latent, and feature references surface as
        // presence-only entries addressed by identity, since the host does not
        // consume their values.
        for output in &operation.outputs {
            if products
                .iter()
                .any(|payload| payload.product.output_index == output.output_index)
            {
                continue;
            }
            products.push(ProductPayload {
                product: output.clone(),
                bytes: Vec::new(),
            });
        }

        record.semantic_digest =
            record.compute_semantic_digest(&parent_semantic, &operation.plan_digest);
        Ok((record, products))
    }

    fn predicated_completion(operation: &Operation, session: &SimSession) -> CompletionRecord {
        let mut record = CompletionRecord {
            request_key: operation.request_key,
            op_id: operation.op_id,
            completion_slot_generation: ((operation.op_id.0 - 1) % u64::from(u32::MAX) + 1) as u32,
            status: OpStatus::Predicated,
            selected_point: session.point_index,
            logical_lengths: LogicalLengths {
                token_len: session.logical_position,
                kv_visible_len: session.kv_visible_len,
                kv_reserved_len: session.kv_visible_len,
                kv_initialized_len: session.kv_visible_len,
                kv_committed_len: session.kv_visible_len,
                kv_published_len: session.kv_published_len,
                ..LogicalLengths::default()
            },
            token_span: TokenSpan {
                base: session.emitted.min(u32::MAX as usize) as u32,
                len: 0,
            },
            committed_tokens: Vec::new(),
            finish_flags: FinishFlags::default(),
            product_generations: Vec::new(),
            semantic_digest: String::new(),
            error_code: None,
            timing_counters: TimingCounters::default(),
        };
        record.semantic_digest = session.committed_semantic.clone();
        record
    }

    pub fn set_pipeline_depth(&mut self, depth: u32) {
        self.caps.pipeline_depth = depth.max(1);
    }

    pub fn set_text_len(&mut self, length: usize) {
        self.text_len = length;
    }

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

    pub fn set_groups(&mut self, groups: Vec<uniserve_core::KvCacheGroupSpec>) {
        self.caps.groups = groups;
    }

    pub fn set_num_blocks(&mut self, count: u32) {
        self.caps.num_blocks = count;
    }

    pub fn set_block_size(&mut self, size: u32) {
        self.caps.block_size = size;
    }

    pub fn mut_caps_for_test(&mut self) -> &mut WorkerCapabilities {
        &mut self.caps
    }
}

/// Decode the branch-local state declared for one sampling operation.
fn operation_sampling_state(
    operation: &Operation,
    input_products: &[ProductPayload],
) -> anyhow::Result<Option<SamplingState>> {
    let mut references = operation
        .inputs
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
    decode_sampling_state_bytes(&payload.bytes).map(Some)
}

fn set_kv_lengths(lengths: &mut LogicalLengths, visible: u32, committed: u32, published: u32) {
    lengths.kv_reserved_len = visible;
    lengths.kv_initialized_len = visible;
    lengths.kv_visible_len = visible;
    lengths.kv_committed_len = committed;
    lengths.kv_published_len = published;
}

/// The declared output-product reference for a value packed into a completion.
fn output_ref(operation: &Operation, kind: ProductKind) -> anyhow::Result<ProductRef> {
    operation
        .outputs
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
    fn default() -> Self {
        Self::new()
    }
}

impl ModelEngine for SimEngine {
    fn caps(&self) -> WorkerCapabilities {
        self.caps.clone()
    }

    fn execute(&mut self, batch: Batch) -> anyhow::Result<CompletionReport> {
        batch.validate()?;
        let step_id = batch.step_id;
        for admission in batch.admissions {
            match self.sessions.get(&admission.request_key.session_id) {
                Some(session) => anyhow::ensure!(
                    session.admission == admission,
                    "session {} was readmitted with a different descriptor",
                    admission.request_key.session_id.0
                ),
                None => {
                    self.sessions
                        .insert(admission.request_key.session_id, SimSession::new(admission));
                }
            }
        }
        let input_products = batch.input_products;
        let mut partition_reports = Vec::with_capacity(batch.partitions.len());
        for partition in batch.partitions {
            let mut completions = Vec::with_capacity(partition.operations.len());
            let mut products = Vec::new();
            for operation in partition.operations {
                let mut session = self
                    .sessions
                    .get(&operation.request_key.session_id)
                    .cloned()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "session {} has no admission",
                            operation.request_key.session_id.0
                        )
                    })?;
                if let Some(recorded) = session.terminal.get(&operation.op_id.0) {
                    anyhow::ensure!(
                        recorded.plan_digest == operation.plan_digest,
                        "operation {} plan digest conflicts with its terminal record",
                        operation.op_id.0
                    );
                    completions.push(recorded.completion.clone());
                    products.extend(recorded.products.clone());
                    continue;
                }
                anyhow::ensure!(
                    operation.request_key == session.admission.request_key,
                    "operation identity {:?} does not match its admitted lineage",
                    operation.request_key
                );
                match &operation.parent.point {
                    Point::Fixed {
                        point_index,
                        semantic_digest,
                    } => {
                        anyhow::ensure!(
                            *point_index == session.point_index,
                            "operation {} ({}) parent point {} does not match session point {}",
                            operation.op_id.0,
                            operation.work.variant().as_wire_str(),
                            point_index,
                            session.point_index
                        );
                        anyhow::ensure!(
                            *semantic_digest == session.committed_semantic,
                            "operation {} ({}) parent semantic digest does not match the committed point",
                            operation.op_id.0,
                            operation.work.variant().as_wire_str()
                        );
                    }
                    Point::Device {
                        point_index,
                        selected_point,
                        producer_plan_digest,
                    } => {
                        // A device-relay successor roots on its predecessor's
                        // selected point before host observation. By the time it
                        // runs, the predecessor has committed and advanced this
                        // session, so its point is the current session point and its
                        // terminal record names the referenced producer. The base
                        // point is the session's tracked point, not the absent
                        // device point index.
                        let producer_op_id = operation.parent.producer_op_id.0;
                        let recorded = session.terminal.get(&producer_op_id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "device parent names unknown predecessor op {producer_op_id}"
                            )
                        })?;
                        anyhow::ensure!(
                            recorded.plan_digest == *producer_plan_digest,
                            "device parent producer plan digest does not match its predecessor"
                        );
                        if let Some(selected_point) = selected_point {
                            anyhow::ensure!(
                                selected_point.producer_op_id.0 == producer_op_id,
                                "device parent selected point is not produced by its named predecessor"
                            );
                        } else {
                            anyhow::ensure!(
                                *point_index == recorded.completion.selected_point,
                                "static device parent point does not match its predecessor"
                            );
                        }
                        anyhow::ensure!(
                            recorded.completion.selected_point == session.point_index,
                            "device parent predecessor is not the session's committed point"
                        );
                    }
                }

                if let Some(predicate) = operation.predicate.as_ref() {
                    let predicate_value = session
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
                        for completion in operation
                            .outputs
                            .iter()
                            .filter(|output| output.kind == ProductKind::Completion)
                        {
                            session.predicate_values.insert(completion.clone(), false);
                        }
                        let completion = Self::predicated_completion(&operation, &session);
                        session.terminal.insert(
                            operation.op_id.0,
                            RecordedCompletion {
                                plan_digest: operation.plan_digest.clone(),
                                completion: completion.clone(),
                                products: Vec::new(),
                            },
                        );
                        self.sessions
                            .insert(operation.request_key.session_id, session);
                        completions.push(completion);
                        continue;
                    }
                }

                let (completion, op_products) =
                    self.execute_operation(&operation, &mut session, &input_products)?;
                if operation.advances_state && completion.status == OpStatus::Ok {
                    session.point_index = completion.selected_point;
                    session.committed_semantic = completion.semantic_digest.clone();
                }
                session.terminal.insert(
                    operation.op_id.0,
                    RecordedCompletion {
                        plan_digest: operation.plan_digest.clone(),
                        completion: completion.clone(),
                        products: op_products.clone(),
                    },
                );
                self.sessions
                    .insert(operation.request_key.session_id, session);
                completions.push(completion);
                products.extend(op_products);
            }
            partition_reports.push(PartitionCompletion {
                partition_id: partition.partition_id,
                completions,
                products,
                registration: RegistrationAck { visible: true },
                worker_exec_us: None,
                forward_stats: None,
            });
        }
        Ok(CompletionReport {
            step_id,
            partitions: partition_reports,
        })
    }

    fn drop_session(&mut self, session_id: RequestId) -> anyhow::Result<()> {
        self.sessions.remove(&session_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_worker_wire::{
        AttentionRegime, BatchPartition, Bounds, DType, Domain, ExecutionCapability, KvAdmission,
        OpId, PointRange, ProductRef, RequestKey, RouteId, ShapeBound, StorageClass, UndAdmission,
        VersionRef,
    };

    fn request_key() -> RequestKey {
        RequestKey::new(0, RequestId(9), 1)
    }

    fn admission() -> Admission {
        Admission::new(
            request_key(),
            1,
            Some(UndAdmission {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                finish_token_ids: Vec::new(),
                kv: KvAdmission::default(),
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
            storage_class: StorageClass::DeviceTensor,
            dtype,
            shape_bound: ShapeBound { dims },
            point_range: PointRange {
                base_point: 0,
                max_points: 1,
            },
        })
        .collect()
    }

    fn batch(step_id: u64, op_id: u64) -> Batch {
        let request_key = request_key();
        let admission = admission();
        let parent = VersionRef::admission_root(request_key, OpId(1), admission.digest.clone());
        let operation = Operation::registered(
            request_key,
            OpId(op_id),
            parent,
            Work::Token(TokenMode::Extend),
            RouteId(0),
            Domain::Prefill,
            Bounds {
                max_points: 1,
                max_tokens: 2,
                ..Bounds::default()
            },
            Vec::new(),
            token_outputs(OpId(op_id)),
            0,
            None,
            None,
            0,
        );
        Batch::new(
            step_id,
            vec![admission],
            vec![BatchPartition {
                partition_id: 1,
                submission_group: 1,
                collective_seq: step_id.max(1),
                domain: Domain::Prefill,
                route: RouteId(0),
                execution: ExecutionCapability::DomainHomogeneous,
                attention: AttentionRegime::Causal,
                shape_class: 0,
                operations: vec![operation],
                request_pool_indices: vec![1],
                kv_placements: Vec::new(),
                kv_branch_placements: Vec::new(),
                latent_placements: Vec::new(),
            }],
        )
    }

    #[test]
    fn duplicate_operation_replays_one_committed_record() {
        let mut engine = SimEngine::new();
        let first = engine.execute(batch(4, 17)).expect("first execution");
        let replay = engine.execute(batch(5, 17)).expect("replay execution");
        assert_eq!(first.partitions, replay.partitions);
    }

    #[test]
    fn extend_commits_the_greedy_synthetic_token() {
        // The synthetic distribution's natural token for session 9 at index 0 is
        // `1000 + (9*7 + 0) % 5000 = 1063`; greedy default sampling commits it.
        let mut engine = SimEngine::new();
        let report = engine.execute(batch(1, 3)).expect("execution");
        let completion = report.completions().next().unwrap();
        assert_eq!(report.completions().count(), 1);
        assert_eq!(completion.committed_tokens, vec![1063]);
        assert_eq!(completion.selected_point, 1);
        assert!(report.partitions[0].registration.visible);
    }

    #[test]
    fn default_capabilities_admit_public_image_geometry() {
        let caps = SimEngine::new().caps();
        let latent_units = (2_048 / caps.latent_downsample) * (1_152 / caps.latent_downsample);
        assert!(u64::from(latent_units) <= caps.latent_capacity_units());
    }
}
