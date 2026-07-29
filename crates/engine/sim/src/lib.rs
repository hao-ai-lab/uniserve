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
use uniserve_core::product_blob::{LogprobBlob, RankedToken};
use uniserve_core::{ImageParams, RequestId, SampleOutput, SamplingParams, apply_sampling};
use uniserve_executor::{ControlAck, ControlOp, Executor, ModelEngine};
use uniserve_worker_wire::{
    Admission, Batch, CompletionRecord, CompletionReport, DType, Digest, EngineCaps, FinishFlags,
    GenMode, LogicalLengths, OpStatus, Operation, Point, PointRange, ProductKind, ProductPayload,
    ProductRef, RegistrationAck, RequestKind, ShapeBound, StorageClass, TimingCounters, TokenMode,
    TokenSpan, TransferMode, Work, WorkVariant,
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
    caps: EngineCaps,
    depth: usize,
    to_worker: Sender<Job>,
    from_worker: Receiver<anyhow::Result<CompletionReport>>,
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
            ControlOp::CopyKv(_)
            | ControlOp::ReleaseProducts(_)
            | ControlOp::LoadAdapter { .. }
            | ControlOp::UnloadAdapter { .. }
            | ControlOp::ResetPrefixCache => {}
            ControlOp::SnapshotSession(_) | ControlOp::RestoreSession(_) => {
                unreachable!("unsupported controls are rejected before dispatch")
            }
        }
        Ok(())
    }
}

impl Executor for SimExecutor {
    fn caps(&self) -> EngineCaps {
        self.caps.clone()
    }

    fn pipeline_depth(&self) -> usize {
        self.depth
    }

    fn in_flight(&self) -> usize {
        self.in_flight
    }

    fn generated_image_commit_capabilities(
        &self,
    ) -> uniserve_core::GeneratedImageCommitCapabilities {
        uniserve_core::GeneratedImageCommitCapabilities {
            inline: true,
            separate_writeback: true,
        }
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
    emitted: usize,
    flow_step: u16,
    terminal: BTreeMap<u64, RecordedCompletion>,
}

impl SimSession {
    fn new(admission: Admission) -> Self {
        let committed_semantic = admission.digest.clone();
        Self {
            admission,
            point_index: 0,
            committed_semantic,
            emitted: 0,
            flow_step: 0,
            terminal: BTreeMap::new(),
        }
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
    caps: EngineCaps,
    text_len: usize,
    fake_eos: u32,
    vocab: usize,
    sessions: HashMap<RequestId, SimSession>,
}

impl SimEngine {
    pub fn new() -> Self {
        let mut caps = EngineCaps {
            supported_work: vec![
                WorkVariant::TokenExtend,
                WorkVariant::TokenDecode,
                WorkVariant::TokenVerify,
                WorkVariant::Draft,
                WorkVariant::EncodeVision,
                WorkVariant::EncodeLatent,
                WorkVariant::GenTransition,
                WorkVariant::GenFlow,
                WorkVariant::Materialize,
                WorkVariant::TransferProduct,
                WorkVariant::TransferKvPublish,
                WorkVariant::TransferKvInstall,
            ],
            max_latent_size: 65_536,
            latent_downsample: 16,
            max_vae_grid_tokens: 1_024,
            max_vit_grid_tokens: 64,
            encoder_cache_budget: 256,
            supported_controls: vec![
                RequestKind::DropSession,
                RequestKind::CopyKv,
                RequestKind::ReleaseProducts,
                RequestKind::LoadAdapter,
                RequestKind::UnloadAdapter,
                RequestKind::ResetPrefixCache,
            ],
            model_spec_digest: "0".repeat(64),
            weight_digest: "1".repeat(64),
            ..EngineCaps::default()
        };
        caps.route_capability_digest = caps.compute_route_capability_digest();
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

    fn sample(&self, session_id: RequestId, session: &SimSession, index: usize) -> SampleOutput {
        let mut logits = self.synth_logits(session_id, index);
        match session.sampling() {
            Some(sampling) => apply_sampling(
                &mut logits,
                sampling,
                &[],
                sampling.allowed_token_ids.as_deref(),
                None,
                sampling.n_logprobs as usize,
            ),
            None => SampleOutput {
                token: if index >= self.text_len {
                    self.fake_eos
                } else {
                    1_000 + ((session_id.0 as u32 * 7 + index as u32) % 5_000)
                },
                logprob: 0.0,
                top: Vec::new(),
            },
        }
    }

    /// Execute one operation against its session, producing the terminal
    /// [`CompletionRecord`] and any resolved output-product values.
    fn execute_operation(
        &self,
        operation: &Operation,
        session: &mut SimSession,
    ) -> anyhow::Result<(CompletionRecord, Vec<ProductPayload>)> {
        let point_index = session.point_index;
        let selected_point = if operation.advances_state {
            point_index.saturating_add(1)
        } else {
            point_index
        };
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
        let mut products = Vec::new();

        match operation.work {
            Work::Token(mode) => {
                let index = session.emitted;
                let output = self.sample(operation.request_key.session_id, session, index);
                // A committed EOS reproduces the natural-termination behavior the
                // synthetic distribution enforces past `text_len`.
                record.finish_flags.eos = output.token == self.fake_eos;
                record.token_span = TokenSpan {
                    base: index as u32,
                    len: 1,
                };
                record.logical_lengths = LogicalLengths {
                    token_len: 1,
                    kv_visible_len: operation.bounds.max_tokens,
                    latent_len: 0,
                };
                match mode {
                    TokenMode::Extend => session.emitted = session.emitted.max(1),
                    TokenMode::Decode | TokenMode::Verify => {
                        session.emitted = session.emitted.saturating_add(1)
                    }
                }
                record.committed_tokens = vec![output.token];
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
                        product: output_ref(
                            operation,
                            ProductKind::Logprob,
                            StorageClass::CompletionArena,
                        ),
                        bytes: blob.encode(),
                    });
                }
            }
            Work::Draft => {}
            Work::Encode(_) => {
                record.logical_lengths.kv_visible_len = 1;
            }
            Work::Transfer(mode) => {
                record.logical_lengths.kv_visible_len =
                    matches!(mode, TransferMode::KvPublish | TransferMode::KvInstall) as u32;
            }
            Work::Gen(GenMode::Transition) => {}
            Work::Gen(GenMode::Flow) => {
                let steps = operation.bounds.max_points.max(1) as u16;
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
                session.emitted = 0;
                if let Some(image) = session.image().cloned() {
                    let (height, width) = if image.height > 0 && image.width > 0 {
                        (image.height, image.width)
                    } else {
                        DEFAULT_IMAGE_HW
                    };
                    if image.retain_images {
                        let downsample = self.caps.latent_downsample.max(1);
                        record.logical_lengths.kv_visible_len = (image.height / downsample)
                            .saturating_mul(image.width / downsample)
                            .saturating_add(self.caps.commit_marker_tokens.max(1));
                    }
                    let png_base64 = synthetic_png_b64(width, height)?;
                    products.push(ProductPayload {
                        product: output_ref(
                            operation,
                            ProductKind::Artifact,
                            StorageClass::HostStaging,
                        ),
                        bytes: png_base64.into_bytes(),
                    });
                }
            }
        }

        // Every declared output surfaces as a resolvable product for the host.
        // The Artifact/Logprob payloads above carry real bytes; the remaining
        // declared references (Kv writeback locators, token/latent/feature
        // handles) surface as presence-only entries addressed by their
        // reference, since the host consumes those by identity, not by value.
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

    pub fn mut_caps_for_test(&mut self) -> &mut EngineCaps {
        &mut self.caps
    }
}

/// The resolved output-product reference for a value the worker packs into the
/// completion report: the operation's declared output of that kind when present,
/// otherwise a freshly named completion-arena reference the host can address.
fn output_ref(operation: &Operation, kind: ProductKind, storage_class: StorageClass) -> ProductRef {
    operation
        .outputs
        .iter()
        .find(|output| output.kind == kind)
        .cloned()
        .unwrap_or_else(|| ProductRef {
            request_key: operation.request_key,
            producer_op_id: operation.op_id,
            output_index: operation.outputs.len() as u16,
            generation: 0,
            kind,
            storage_class,
            dtype: DType::U8,
            shape_bound: ShapeBound::default(),
            point_range: PointRange::default(),
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
    fn caps(&self) -> EngineCaps {
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

        let mut completions = Vec::with_capacity(batch.operations.len());
        let mut products = Vec::new();
        for operation in batch.operations {
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
                        "operation parent point {} does not match session point {}",
                        point_index,
                        session.point_index
                    );
                    anyhow::ensure!(
                        *semantic_digest == session.committed_semantic,
                        "operation parent semantic digest does not match the committed point"
                    );
                }
                Point::Device {
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
                    anyhow::ensure!(
                        selected_point.producer_op_id.0 == producer_op_id,
                        "device parent selected point is not produced by its named predecessor"
                    );
                    anyhow::ensure!(
                        recorded.completion.selected_point == session.point_index,
                        "device parent predecessor is not the session's committed point"
                    );
                }
            }

            let (completion, op_products) = self.execute_operation(&operation, &mut session)?;
            if operation.advances_state {
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
        Ok(CompletionReport {
            step_id,
            completions,
            products,
            registration: RegistrationAck { visible: true },
            worker_exec_us: None,
            forward_stats: None,
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
        Bounds, Domain, KvAllocation, OpId, RequestKey, RouteId, UndAdmission, VersionRef,
    };

    fn request_key() -> RequestKey {
        RequestKey::new(0, RequestId(9), 1)
    }

    fn admission() -> Admission {
        Admission::new(
            request_key(),
            Some(UndAdmission {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                kv: KvAllocation::default(),
            }),
            None,
            None,
        )
        .expect("admission")
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
            Domain::Und,
            Bounds {
                max_points: 1,
                max_tokens: 2,
                ..Bounds::default()
            },
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            None,
            0,
        );
        Batch::new(step_id, vec![admission], vec![operation])
    }

    #[test]
    fn duplicate_operation_replays_one_committed_record() {
        let mut engine = SimEngine::new();
        let first = engine.execute(batch(4, 17)).expect("first execution");
        let replay = engine.execute(batch(5, 17)).expect("replay execution");
        assert_eq!(first.completions, replay.completions);
        assert_eq!(first.products, replay.products);
    }

    #[test]
    fn extend_commits_the_greedy_synthetic_token() {
        // The synthetic distribution's natural token for session 9 at index 0 is
        // `1000 + (9*7 + 0) % 5000 = 1063`; greedy default sampling commits it.
        let mut engine = SimEngine::new();
        let report = engine.execute(batch(1, 3)).expect("execution");
        assert_eq!(report.completions.len(), 1);
        assert_eq!(report.completions[0].committed_tokens, vec![1063]);
        assert_eq!(report.completions[0].selected_point, 1);
        assert!(report.registration.visible);
    }

    #[test]
    fn control_wait_acknowledges_the_closed_control_algebra() {
        let mut executor = SimExecutor::new(Box::new(SimEngine::new()));
        for operation in [
            ControlOp::DropSession(RequestId(1)),
            ControlOp::CopyKv(Vec::new()),
            ControlOp::ReleaseProducts(Vec::new()),
            ControlOp::LoadAdapter {
                adapter_id: 7,
                path: "/tmp/adapter".into(),
            },
            ControlOp::UnloadAdapter { adapter_id: 7 },
            ControlOp::ResetPrefixCache,
        ] {
            let acknowledgements = executor
                .control_wait(operation, None)
                .expect("control acknowledgement");
            assert_eq!(acknowledgements.len(), 1);
            assert!(acknowledgements[0].ok);
        }
    }

    #[test]
    fn default_capabilities_admit_public_image_geometry() {
        let caps = SimEngine::new().caps();
        let latent_units = (2_048 / caps.latent_downsample) * (1_152 / caps.latent_downsample);
        assert!(latent_units <= caps.max_latent_size);
    }
}
