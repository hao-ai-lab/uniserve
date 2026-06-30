//! GPU-free CPU model engine that fabricates text tokens and denoise-step
//! progress against the same [`ModelEngine`] trait, so the scheduler, FSM, block
//! manager, and frontend can be exercised without a GPU or Python worker.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::collections::HashMap;
use std::thread::JoinHandle;
use std::time::Duration;

use base64::Engine as _;
use crossbeam_channel::{Receiver, Sender};
use uniserve_core::{ImageParams, RequestId, SampleOutput, SamplingParams, apply_sampling};
use uniserve_executor::{ControlAck, ControlOp, Executor, ModelEngine};
use uniserve_worker_wire::{EngineCaps, ForwardBatch, ForwardResult, OpKind, SeqResult};

/// Default fabricated-text length before a synthetic EOS (overridable via
/// [`SimEngine::set_text_len`]).
const DEFAULT_TEXT_LEN: usize = 8;
/// Synthetic end-of-sequence token id. Matches Qwen-family `<|im_end|>`
/// (151645) so fixtures exercising real control tokens line up with the sim.
const FAKE_EOS_TOKEN: u32 = 151645;
/// Synthetic vocab size: large enough to cover the fabricated token ids
/// (`1000 + (.. % 5000)`) plus [`FAKE_EOS_TOKEN`] itself.
const SYNTH_VOCAB_SIZE: usize = FAKE_EOS_TOKEN as usize + 1;
/// Denoise steps assumed when a request carries no [`ImageParams::steps`].
const DEFAULT_DENOISE_STEPS: u16 = 50;
/// Image dimensions (height, width) assumed when a request carries no
/// [`ImageParams`] height/width.
const DEFAULT_IMAGE_HW: (u32, u32) = (512, 512);

enum Job {
    Batch(ForwardBatch),
    Drop(RequestId),
    Shutdown,
}

/// Wraps a synchronous in-process [`ModelEngine`] on its own worker thread.
pub struct SimExecutor {
    caps: EngineCaps,
    depth: usize,
    to_worker: Sender<Job>,
    from_worker: Receiver<anyhow::Result<ForwardResult>>,
    in_flight: usize,
    next_call_id: u64,
    handle: Option<JoinHandle<()>>,
}

impl SimExecutor {
    pub fn new(engine: Box<dyn ModelEngine>) -> Self {
        let caps = engine.caps();
        let depth = (caps.pipeline_depth as usize).max(1);
        Self::with_depth(engine, depth)
    }

    pub fn with_depth(mut engine: Box<dyn ModelEngine>, depth: usize) -> Self {
        let caps = engine.caps();
        let depth = depth.max(1);
        let (to_worker, jobs) = crossbeam_channel::unbounded::<Job>();
        let (results_tx, from_worker) = crossbeam_channel::unbounded();
        let handle = std::thread::Builder::new()
            .name("uniserve-sim-executor".into())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    match job {
                        Job::Batch(batch) => {
                            let result = engine.execute(batch);
                            if results_tx.send(result).is_err() {
                                break;
                            }
                        }
                        Job::Drop(id) => {
                            let _ = engine.drop_request(id);
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

 /// Apply a control op against the in-process engine.

 /// Only [`ControlOp::DropRequest`] has observable engine state in the sim:
 /// it forwards a [`Job::Drop`] so the worker thread evicts the request's
 /// records (mirroring the real worker's stateful-diff contract). The
 /// remaining ops address resources the CPU sim does not model — there is no
 /// real KV pool to copy blocks within, no GPU encoder cache to free, no
 /// LoRA registry, no prefix cache, and no device to sleep/wake — so they are
 /// intentional no-ops. The exhaustive match (rather than a single `if let`)
 /// is deliberate: a newly added `ControlOp` variant fails to compile here,
 /// forcing a conscious decision instead of silently dropping the op.
    fn apply_control(&mut self, op: &ControlOp) {
        match op {
            ControlOp::DropRequest(id) => {
                let _ = self.to_worker.send(Job::Drop(*id));
            }
            ControlOp::CopyBlocks(_)
            | ControlOp::FreeEncoder(_)
            | ControlOp::LoadLora { .. }
            | ControlOp::UnloadLora { .. }
            | ControlOp::ResetPrefixCache
            | ControlOp::Sleep
            | ControlOp::WakeUp => {}
        }
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

    fn submit(&mut self, batch: ForwardBatch) -> anyhow::Result<()> {
        self.to_worker
            .send(Job::Batch(batch))
            .map_err(|_| anyhow::anyhow!("sim executor thread gone"))?;
        self.in_flight += 1;
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
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

    fn wait_result_timeout(&mut self, timeout: Duration) -> anyhow::Result<Option<ForwardResult>> {
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

    fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
        let result = self
            .from_worker
            .recv()
            .map_err(|_| anyhow::anyhow!("sim executor thread disconnected"))?;
        self.in_flight = self.in_flight.saturating_sub(1);
        result
    }

    fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
        let call_id = self.next_call_id;
        self.next_call_id += 1;
        self.apply_control(&op);
        Ok(call_id)
    }

    fn control_wait(
        &mut self,
        op: ControlOp,
        _targets: Option<&[u32]>,
    ) -> anyhow::Result<Vec<ControlAck>> {
 // The sim is a single synchronous in-process rank: `apply_control`
 // runs the op (or no-ops it; see `apply_control`) to completion before
 // returning, so the rank-0 ack is genuinely satisfied here rather than
 // optimistically fabricated. `_targets` is meaningless for one rank.
        self.apply_control(&op);
        Ok(vec![ControlAck {
            rank: 0,
            ok: true,
            message: None,
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

/// Per-request control record seeded by `NewRequestData` (the stateful-diff
/// contract): the sim mirrors the real worker's statefulness so the contract
/// is exercised in CI exactly as it is on the GPU path.
#[derive(Default)]
struct SimRecord {
    sampling: Option<SamplingParams>,
    image: Option<ImageParams>,
}

/// CPU model engine for deterministic local testing.
pub struct SimEngine {
    caps: EngineCaps,
    text_len: usize,
    fake_eos: u32,
    vocab: usize,
    commit_token: Option<u32>,
    emitted: HashMap<RequestId, usize>,
    steps: HashMap<RequestId, u16>,
    records: HashMap<RequestId, SimRecord>,
}

impl SimEngine {
 /// Synthetic logit distribution for request `r` at generation index `n`.
 /// The "natural" next token (matching the deterministic fake sampler) gets the
 /// top logit, two alternatives get descending logits, and EOS dominates once
 /// `n >= text_len`. The host-side sampling pipeline (bias, penalties, masks,
 /// min-p/top-k/p) can change the chosen token from the natural argmax.
    fn synth_logits(&self, r: RequestId, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; self.vocab];
        let nat = if n >= self.text_len {
            self.fake_eos
        } else {
            1000 + ((r.0 as u32 * 7 + n as u32) % 5000)
        };
        v[nat as usize] = 10.0;
        let alt1 = 1000 + ((r.0 as u32 * 13 + n as u32 + 1) % 5000);
        let alt2 = 1000 + ((r.0 as u32 * 29 + n as u32 + 2) % 5000);
        if (alt1 as usize) < self.vocab && alt1 != nat {
            v[alt1 as usize] = 8.0;
        }
        if (alt2 as usize) < self.vocab && alt2 != nat {
            v[alt2 as usize] = 6.0;
        }
        v[self.fake_eos as usize] = if n >= self.text_len { 12.0 } else { 1.0 };
        v
    }
}

impl SimEngine {
    pub fn new() -> Self {
        Self {
            caps: EngineCaps {
                supported_ops: vec![
                    "prefill_und".into(),
                    "decode_und".into(),
                    "denoise_gen".into(),
                    "commit_gen".into(),
                    "vit_encode".into(),
                    "vae_encode".into(),
                ],
                ..Default::default()
            },
            text_len: DEFAULT_TEXT_LEN,
            fake_eos: FAKE_EOS_TOKEN,
            vocab: SYNTH_VOCAB_SIZE,
            commit_token: None,
            emitted: HashMap::new(),
            steps: HashMap::new(),
            records: HashMap::new(),
        }
    }
}

fn synthetic_png_b64(width: u32, height: u32) -> anyhow::Result<String> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        let row_bytes = width as usize * 3;
        let mut image = vec![0_u8; row_bytes * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = y * row_bytes + x * 3;
                image[idx] = ((x * 255) / (width.max(1) as usize)) as u8;
                image[idx + 1] = ((y * 255) / (height.max(1) as usize)) as u8;
                image[idx + 2] = 128;
            }
        }
        writer.write_image_data(&image)?;
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

impl SimEngine {
 /// Advertise a batch-queue depth so the executor keeps that many op-batches
 /// in flight.
    pub fn set_pipeline_depth(&mut self, depth: u32) {
        self.caps.pipeline_depth = depth.max(1);
    }
 /// Number of fabricated text tokens before a synthetic EOS (test knob).
    pub fn set_text_len(&mut self, n: usize) {
        self.text_len = n;
    }
 /// Synthetic sampled token to return from commit_gen (test knob).
    pub fn set_commit_token(&mut self, token: Option<u32>) {
        self.commit_token = token;
    }
 /// Advertise hybrid KV-cache groups at the handshake (test knob).
    pub fn set_groups(&mut self, groups: Vec<uniserve_core::KvCacheGroupSpec>) {
        self.caps.groups = groups;
    }
 /// Shrink the KV pool to force admission/preemption pressure (test knob).
    pub fn set_num_blocks(&mut self, n: u32) {
        self.caps.num_blocks = n;
    }
 /// Shrink the block size so growth pressure happens sooner (test knob).
    pub fn set_block_size(&mut self, n: u32) {
        self.caps.block_size = n;
    }
 /// Test hook for scheduler/worker capability negotiation.
    pub fn mut_caps_for_test(&mut self) -> &mut EngineCaps {
        &mut self.caps
    }
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

    fn execute(&mut self, batch: ForwardBatch) -> anyhow::Result<ForwardResult> {
 // Seed per-request control records before running ops (the worker's
 // half of the stateful-diff contract).
        for nr in &batch.new_reqs {
            self.records.insert(
                nr.req_id,
                SimRecord {
                    sampling: nr.sampling.clone(),
                    image: nr.image.clone(),
                },
            );
        }
        let mut per_seq = Vec::new();
        for op in batch.ops {
            let r = op.req_id;
 // Echo the op's op_id on its result, exactly as the GPU worker does,
 // so the host's op_id-correlated completion path is exercised here.
            let op_id = op.op_id;
            match op.kind {
                OpKind::PrefillUnd | OpKind::DecodeUnd => {
                    let n = *self.emitted.get(&r).unwrap_or(&0);
 // Run the host-side sampling pipeline over synthetic logits so
 // the op's params and masks can change the chosen token.
                    let sampling = self.records.get(&r).and_then(|rec| rec.sampling.clone());
                    let out: SampleOutput = match &sampling {
                        Some(sp) => {
                            let mut logits = self.synth_logits(r, n);
                            apply_sampling(
                                &mut logits,
                                sp,
                                op.recent_tokens.as_deref().unwrap_or(&[]),
                                op.allowed_tokens.as_deref(),
                                op.suppress_tokens.as_deref(),
                                sp.n_logprobs as usize,
                            )
                        }
                        None => {
                            let nat = if n >= self.text_len {
                                self.fake_eos
                            } else {
                                1000 + ((r.0 as u32 * 7 + n as u32) % 5000)
                            };
                            SampleOutput {
                                token: nat,
                                logprob: 0.0,
                                top: Vec::new(),
                            }
                        }
                    };
 // Only decode advances the generation counter — intermediate
 // (chunked) prefill ops produce no kept token, matching a real
 // model where prefill yields logits only for the last position.
                    if op.kind == OpKind::DecodeUnd {
                        self.emitted.insert(r, n + 1);
                    }
                    per_seq.push(SeqResult {
                        req_id: r,
                        op_id,
                        sampled_token_id: Some(out.token),
                        sampled_logprob: Some(out.logprob),
                        top_logprobs: if out.top.is_empty() {
                            None
                        } else {
                            Some(out.top)
                        },
                        ..Default::default()
                    });
                }
                OpKind::DenoiseGen => {
                    let s = self.steps.get(&r).unwrap_or(&0) + 1;
                    self.steps.insert(r, s);
                    let total = self
                        .records
                        .get(&r)
                        .and_then(|rec| rec.image.as_ref())
                        .map(|i| i.steps)
                        .unwrap_or(DEFAULT_DENOISE_STEPS);
                    per_seq.push(SeqResult {
                        req_id: r,
                        op_id,
                        denoise_done: s >= total,
                        num_steps_done: Some(s),
                        ..Default::default()
                    });
                }
                OpKind::CommitGen => {
                    self.steps.remove(&r);
 // After committing an image, reset the text counter so a
 // round-trip (text → image → text …) generates fresh text.
                    self.emitted.insert(r, 0);
                    let hw = self
                        .records
                        .get(&r)
                        .and_then(|rec| rec.image.as_ref())
                        .map(|i| (i.height, i.width))
                        .unwrap_or(DEFAULT_IMAGE_HW);
                    let png = synthetic_png_b64(hw.1, hw.0)?;
                    per_seq.push(SeqResult {
                        req_id: r,
                        op_id,
                        sampled_token_id: self.commit_token,
                        sampled_logprob: self.commit_token.map(|_| 0.0),
                        image_png_b64: Some(png),
                        image_hw: Some(hw),
                        ..Default::default()
                    });
                }
                OpKind::VitEncode | OpKind::VaeEncode => {
 // Fabricate a deterministic worker-side encoder handle from the
 // image content hash. The embedding stays on the worker side.
                    let handle = op.mm_hash.unwrap_or(0).wrapping_mul(0x9E3779B1) | 1;
                    let image_hw = self
                        .records
                        .get(&r)
                        .and_then(|rec| rec.image.as_ref())
                        .map(|i| (i.height, i.width));
                    per_seq.push(SeqResult {
                        req_id: r,
                        op_id,
                        encoder_handle: Some(handle),
                        num_tokens: Some(1),
                        image_hw,
                        ..Default::default()
                    });
                }
                _ => per_seq.push(SeqResult {
                    req_id: r,
                    op_id,
                    ..Default::default()
                }),
            }
        }
        Ok(ForwardResult {
            step_id: batch.step_id,
            per_seq,
            worker_exec_us: None,
            forward_stats: None,
        })
    }

    fn drop_request(&mut self, id: RequestId) -> anyhow::Result<()> {
        self.emitted.remove(&id);
        self.steps.remove(&id);
        self.records.remove(&id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_executor::Executor as _;

    #[test]
    fn control_wait_acks_every_control_op() {
        let mut exec = SimExecutor::new(Box::new(SimEngine::new()));
 // Every op variant must produce exactly one ok ack from the single
 // simulated rank — none may be silently dropped.
        for op in [
            ControlOp::DropRequest(RequestId(1)),
            ControlOp::CopyBlocks(Vec::new()),
            ControlOp::FreeEncoder(Vec::new()),
            ControlOp::LoadLora {
                lora_id: 7,
                path: "/tmp/adapter".into(),
            },
            ControlOp::UnloadLora { lora_id: 7 },
            ControlOp::ResetPrefixCache,
            ControlOp::Sleep,
            ControlOp::WakeUp,
        ] {
            let acks = exec.control_wait(op, None).expect("control_wait");
            assert_eq!(acks.len(), 1);
            assert_eq!(acks[0].rank, 0);
            assert!(acks[0].ok);
        }
        exec.shutdown();
    }
}
