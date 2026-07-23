//! GPU-free execution engine for scheduler and frontend conformance tests.
//!
//! The simulator is a strict peer of the production execution protocol: it
//! consumes typed admissions and operations, enforces lifecycle/version/replay
//! invariants, and returns typed deltas without a second simulation-only schema.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{BTreeMap, HashMap};
use std::thread::JoinHandle;
use std::time::Duration;

use base64::Engine as _;
use crossbeam_channel::{Receiver, Sender};
use uniserve_core::{
    ImageParams, RequestId, SampleOutput, SamplingParams, apply_sampling, score_token_logprobs,
};
use uniserve_executor::{ControlAck, ControlOp, Executor, ModelEngine};
use uniserve_worker_wire::{
    Admission, Batch, EncodeDelta, EncodeInput, EngineCaps, ExecutionResult, FlowDelta,
    ImageArtifact, MaterializeDelta, MaterializedProduct, Operation, OperationEnvelope,
    OperationResult, OperationType, RequestKind, ResultDelta, SequenceDelta, SequenceEffect,
    SequenceInput, SequenceMode, TokenLogprob, TransferDelta,
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
    from_worker: Receiver<anyhow::Result<ExecutionResult>>,
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

    fn poll(&mut self) -> anyhow::Result<Option<ExecutionResult>> {
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
    ) -> anyhow::Result<Option<ExecutionResult>> {
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

    fn next_result(&mut self) -> anyhow::Result<ExecutionResult> {
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

#[derive(Clone)]
struct RecordedResult {
    digest: String,
    result: OperationResult,
}

#[derive(Clone)]
struct SimSession {
    admission: Admission,
    epoch: Option<u64>,
    version: u64,
    emitted: usize,
    flow_step: u16,
    terminal: BTreeMap<u64, RecordedResult>,
}

impl SimSession {
    fn new(admission: Admission) -> Self {
        Self {
            admission,
            epoch: None,
            version: 0,
            emitted: 0,
            flow_step: 0,
            terminal: BTreeMap::new(),
        }
    }

    fn sampling(&self) -> Option<&SamplingParams> {
        self.admission
            .sequence
            .as_ref()
            .map(|sequence| &sequence.sampling)
    }

    fn image(&self) -> Option<&ImageParams> {
        self.admission.flow.as_ref().map(|flow| &flow.image)
    }
}

/// Deterministic local model engine with protocol-faithful lifecycle state.
pub struct SimEngine {
    caps: EngineCaps,
    text_len: usize,
    fake_eos: u32,
    vocab: usize,
    commit_token: Option<u32>,
    sessions: HashMap<RequestId, SimSession>,
}

impl SimEngine {
    pub fn new() -> Self {
        Self {
            caps: EngineCaps {
                supported_operation_types: vec![
                    OperationType::SequenceExtend,
                    OperationType::SequenceDecode,
                    OperationType::SequenceVerify,
                    OperationType::SequenceSample,
                    OperationType::Flow,
                    OperationType::EncodeVision,
                    OperationType::EncodeLatent,
                    OperationType::MaterializeImage,
                    OperationType::MaterializeFrame,
                    OperationType::TransferProduct,
                    OperationType::TransferKv,
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
            },
            text_len: DEFAULT_TEXT_LEN,
            fake_eos: FAKE_EOS_TOKEN,
            vocab: SYNTH_VOCAB_SIZE,
            commit_token: None,
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
        session_id: RequestId,
        session: &SimSession,
        sequence: &uniserve_worker_wire::SequenceOperation,
        index: usize,
    ) -> SampleOutput {
        let mut logits = self.synth_logits(session_id, index);
        match session.sampling() {
            Some(sampling) => apply_sampling(
                &mut logits,
                sampling,
                &sequence.policy.recent_tokens,
                (!sequence.policy.allowed_tokens.is_empty())
                    .then_some(sequence.policy.allowed_tokens.as_slice()),
                (!sequence.policy.suppress_tokens.is_empty())
                    .then_some(sequence.policy.suppress_tokens.as_slice()),
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

    fn sequence_effect(
        &self,
        envelope: &OperationEnvelope,
        session: &mut SimSession,
        sequence: &uniserve_worker_wire::SequenceOperation,
    ) -> anyhow::Result<SequenceEffect> {
        let SequenceInput::Tokens(input) = &sequence.input else {
            let output = self.sample(envelope.session_id, session, sequence, session.emitted);
            return Ok(sample_effect(session.sampling(), output, 1));
        };
        let mut effect = SequenceEffect::default();
        if input.return_all_logits {
            let sampling = session
                .sampling()
                .ok_or_else(|| anyhow::anyhow!("sequence session has no sampling admission"))?;
            let skip = usize::from(sequence.position.0 == 0);
            effect.prompt_logprobs = input
                .token_ids
                .iter()
                .enumerate()
                .skip(skip)
                .map(|(offset, token)| {
                    let logits = self.synth_logits(
                        envelope.session_id,
                        (sequence.position.0 as usize)
                            .saturating_add(offset)
                            .saturating_sub(1),
                    );
                    score_token_logprobs(
                        &logits,
                        *token,
                        sampling.n_prompt_logprobs as usize,
                        &sampling.logprob_token_ids,
                    )
                    .into_iter()
                    .map(|(token, logprob, rank)| TokenLogprob(token, logprob, rank))
                    .collect()
                })
                .collect();
        }
        let output = self.sample(envelope.session_id, session, sequence, session.emitted);
        match sequence.mode {
            SequenceMode::Extend => {
                effect = merge_sample(effect, session.sampling(), output);
                session.emitted = session.emitted.max(1);
            }
            SequenceMode::Decode => {
                effect = merge_sample(effect, session.sampling(), output);
                session.emitted = session.emitted.saturating_add(1);
            }
            SequenceMode::Verify => {
                effect.sampled_token_ids = input.draft_token_ids.clone();
                effect.sampled_token_ids.push(output.token);
                effect.accepted_draft_tokens = Some(input.draft_token_ids.len() as u32);
                effect.sampled_logprob = session
                    .sampling()
                    .is_some_and(SamplingParams::generated_logprobs_requested)
                    .then_some(output.logprob);
                effect.top_logprobs = output
                    .top
                    .into_iter()
                    .map(|(token, logprob, rank)| TokenLogprob(token, logprob, rank))
                    .collect();
                session.emitted = session
                    .emitted
                    .saturating_add(effect.sampled_token_ids.len());
            }
            SequenceMode::Sample => anyhow::bail!("sample sequence requires published logits"),
        }
        effect.kv_tokens = Some(input.token_ids.len() as u32);
        Ok(effect)
    }

    fn execute_operation(
        &self,
        envelope: &OperationEnvelope,
        session: &mut SimSession,
    ) -> anyhow::Result<ResultDelta> {
        Ok(match &envelope.operation {
            Operation::Sequence(sequence) => ResultDelta::Sequence(SequenceDelta {
                effect: self.sequence_effect(envelope, session, sequence)?,
            }),
            Operation::Flow(flow) => {
                let completed = flow.start_step.saturating_add(flow.step_count);
                session.flow_step = completed;
                let total = session
                    .image()
                    .map(|image| image.steps)
                    .unwrap_or(DEFAULT_DENOISE_STEPS);
                ResultDelta::Flow(FlowDelta {
                    steps_completed: completed,
                    done: completed >= total,
                })
            }
            Operation::Encode(encode) => {
                let content_hash = match &encode.input {
                    EncodeInput::InlineImage { content_hash, .. }
                    | EncodeInput::StagedProduct { content_hash, .. }
                    | EncodeInput::CachedProduct { content_hash } => *content_hash,
                };
                let handle = content_hash.wrapping_mul(0x9E37_79B1) | 1;
                ResultDelta::Encode(EncodeDelta {
                    product_handle: handle,
                    kv_tokens: 1,
                    image_size: session.image().map(|image| (image.height, image.width)),
                })
            }
            Operation::Materialize(materialize) => match materialize.kind {
                uniserve_worker_wire::MaterializeKind::Image => {
                    session.flow_step = 0;
                    session.emitted = 0;
                    let (height, width) = session
                        .image()
                        .map(|image| (image.height, image.width))
                        .unwrap_or(DEFAULT_IMAGE_HW);
                    let png_base64 = synthetic_png_b64(width, height)?;
                    let kv_tokens =
                        session
                            .image()
                            .filter(|image| image.retain_images)
                            .map(|image| {
                                let downsample = self.caps.latent_downsample.max(1);
                                (image.height / downsample)
                                    .saturating_mul(image.width / downsample)
                                    .saturating_add(self.caps.commit_marker_tokens.max(1))
                            });
                    let sequence = self.commit_token.map(|token| SequenceEffect {
                        sampled_token_ids: vec![token],
                        sampled_logprob: Some(0.0),
                        ..SequenceEffect::default()
                    });
                    ResultDelta::Materialize(MaterializeDelta {
                        product: MaterializedProduct::Image(ImageArtifact {
                            png_base64,
                            height,
                            width,
                            handle: envelope.session_id.0.max(1),
                            locator: format!("sim-image-{}", envelope.session_id.0),
                        }),
                        kv_tokens,
                        sequence,
                    })
                }
                uniserve_worker_wire::MaterializeKind::Frame => {
                    ResultDelta::Materialize(MaterializeDelta {
                        product: MaterializedProduct::Frame { count: 1 },
                        kv_tokens: None,
                        sequence: None,
                    })
                }
            },
            Operation::Transfer(transfer) => ResultDelta::Transfer(TransferDelta {
                product: (transfer.kind == uniserve_worker_wire::TransferKind::Product)
                    .then_some(transfer.source.clone()),
                kv_tokens: (transfer.kind == uniserve_worker_wire::TransferKind::Kv).then_some(1),
                sequence: None,
            }),
        })
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

    pub fn set_commit_token(&mut self, token: Option<u32>) {
        self.commit_token = token;
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

fn merge_sample(
    mut effect: SequenceEffect,
    sampling: Option<&SamplingParams>,
    output: SampleOutput,
) -> SequenceEffect {
    effect.sampled_token_ids.push(output.token);
    effect.sampled_logprob = sampling
        .is_some_and(SamplingParams::generated_logprobs_requested)
        .then_some(output.logprob);
    effect.top_logprobs = output
        .top
        .into_iter()
        .map(|(token, logprob, rank)| TokenLogprob(token, logprob, rank))
        .collect();
    effect
}

fn sample_effect(
    sampling: Option<&SamplingParams>,
    output: SampleOutput,
    kv_tokens: u32,
) -> SequenceEffect {
    let mut effect = merge_sample(SequenceEffect::default(), sampling, output);
    effect.kv_tokens = Some(kv_tokens);
    effect
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

    fn execute(&mut self, batch: Batch) -> anyhow::Result<ExecutionResult> {
        batch.validate()?;
        for admission in batch.admissions {
            match self.sessions.get(&admission.session_id) {
                Some(session) => anyhow::ensure!(
                    session.admission == admission,
                    "session {} was readmitted with a different descriptor",
                    admission.session_id.0
                ),
                None => {
                    self.sessions
                        .insert(admission.session_id, SimSession::new(admission));
                }
            }
        }

        let mut results = Vec::with_capacity(batch.operations.len());
        for envelope in batch.operations {
            let mut session = self
                .sessions
                .get(&envelope.session_id)
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!("session {} has no admission", envelope.session_id.0)
                })?;
            if let Some(recorded) = session.terminal.get(&envelope.op_id) {
                anyhow::ensure!(
                    recorded.digest == envelope.digest,
                    "operation {} digest conflicts with its terminal record",
                    envelope.op_id
                );
                results.push(recorded.result.clone());
                continue;
            }
            match session.epoch {
                None => session.epoch = Some(envelope.epoch),
                Some(epoch) => anyhow::ensure!(
                    epoch == envelope.epoch,
                    "operation epoch {} does not match session epoch {epoch}",
                    envelope.epoch
                ),
            }
            anyhow::ensure!(
                envelope.base_version == session.version,
                "operation base version {} does not match session version {}",
                envelope.base_version,
                session.version
            );
            let delta = self.execute_operation(&envelope, &mut session)?;
            let result = OperationResult {
                session_id: envelope.session_id,
                epoch: envelope.epoch,
                op_id: envelope.op_id,
                base_version: envelope.base_version,
                result_version: envelope.base_version.saturating_add(1),
                delta,
            };
            result.validate_for(&envelope)?;
            session.version = result.result_version;
            session.terminal.insert(
                envelope.op_id,
                RecordedResult {
                    digest: envelope.digest,
                    result: result.clone(),
                },
            );
            self.sessions.insert(envelope.session_id, session);
            results.push(result);
        }
        Ok(ExecutionResult {
            step_id: batch.step_id,
            operations: results,
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
        KvAllocation, KvLeaseDelta, SequenceAdmission, SequenceOperation, TokenInput, TokenPolicy,
        TokenSource,
    };

    fn batch(step_id: u64, op_id: u64) -> Batch {
        let admission = Admission::new(
            RequestId(9),
            Some(SequenceAdmission {
                sampling: SamplingParams::default(),
                negative_token_ids: Vec::new(),
                kv: KvAllocation {
                    block_ids: Vec::new(),
                    prefix_len: 0,
                    group_id: 0,
                },
            }),
            None,
            None,
        )
        .expect("admission");
        let mut operation = OperationEnvelope::unsealed(
            RequestId(9),
            Operation::Sequence(SequenceOperation {
                mode: SequenceMode::Extend,
                lease: KvLeaseDelta::default(),
                position: (0, 2),
                policy: TokenPolicy::default(),
                input: SequenceInput::Tokens(TokenInput {
                    token_ids: vec![1, 2],
                    source: TokenSource::Wire,
                    draft_token_ids: Vec::new(),
                    burst_tokens: 1,
                    stop_token_ids: Vec::new(),
                    stop_terminal: false,
                    return_all_logits: false,
                }),
            }),
        );
        operation.admission_digest = admission.digest.clone();
        operation.model_spec_digest = "0".repeat(64);
        operation.weight_digest = "1".repeat(64);
        operation.seal(3, op_id, 0);
        Batch::new(step_id, vec![admission], vec![operation])
    }

    #[test]
    fn duplicate_operation_replays_one_typed_effect() {
        let mut engine = SimEngine::new();
        let first = engine.execute(batch(4, 17)).expect("first execution");
        let replay = engine.execute(batch(5, 17)).expect("replay execution");
        assert_eq!(first.operations, replay.operations);
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
