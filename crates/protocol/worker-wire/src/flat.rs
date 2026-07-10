use std::collections::BTreeMap;

use anyhow::{Context, bail};
use flatbuffers::FlatBufferBuilder;
use uniserve_core::{
    BlockId, CfgParams, ImageParams, KvCacheGroupSpec, KvGroupKind, Modality, RankInfo, RequestId,
    SamplingParams,
};

use crate::resources::{ResourceClass, ResourcePressure};
use crate::schema::uniserve::wire as fbs;
use crate::{
    AdapterMode, EngineCaps, ExecutionConstraints, ForwardBatch, ForwardOp, ForwardResult,
    NewRequestData, OpKind, RequestKind, SeqResult, TokenLogprob, TokenSource, WorkerForwardStats,
    WorkerMetrics, WorkerRequest, WorkerResponse,
};

pub fn encode_request(req: &WorkerRequest) -> anyhow::Result<Vec<u8>> {
    let native = request_to_fb(req)?;
    let mut fbb = FlatBufferBuilder::new();
    let root = native.pack(&mut fbb);
    fbb.finish(root, None);
    Ok(fbb.finished_data().to_vec())
}

pub fn decode_request(bytes: &[u8]) -> anyhow::Result<WorkerRequest> {
    let root = fbs::root_as_worker_request(bytes).context("invalid WorkerRequest flatbuffer")?;
    request_from_fb(root.unpack())
}

pub fn encode_response(resp: &WorkerResponse) -> anyhow::Result<Vec<u8>> {
    let native = response_to_fb(resp)?;
    let mut fbb = FlatBufferBuilder::new();
    let root = native.pack(&mut fbb);
    fbb.finish(root, None);
    Ok(fbb.finished_data().to_vec())
}

pub fn decode_response(bytes: &[u8]) -> anyhow::Result<WorkerResponse> {
    let root = flatbuffers::root::<fbs::WorkerResponse>(bytes)
        .context("invalid WorkerResponse flatbuffer")?;
    response_from_fb(root.unpack())
}

fn request_to_fb(req: &WorkerRequest) -> anyhow::Result<fbs::WorkerRequestT> {
    Ok(fbs::WorkerRequestT {
        kind: req_kind_to_fb(req.kind),
        call_id: req.call_id,
        batch: req
            .batch
            .as_ref()
            .map(batch_to_fb)
            .transpose()?
            .map(Box::new),
        req_id: req.req_id.map(|id| id.0),
        copies: req.copies.as_ref().map(|copies| {
            copies
                .iter()
                .map(|(src, dst)| fbs::BlockPairT {
                    src: src.0,
                    dst: dst.0,
                })
                .collect()
        }),
        lora_id: req.lora_id,
        lora_path: req.lora_path.clone(),
        free_handles: req.free_handles.clone(),
    })
}

fn request_from_fb(req: fbs::WorkerRequestT) -> anyhow::Result<WorkerRequest> {
    Ok(WorkerRequest {
        kind: req_kind_from_fb(req.kind)?,
        call_id: req.call_id,
        batch: req.batch.map(|b| batch_from_fb(*b)).transpose()?,
        req_id: req.req_id.map(RequestId),
        copies: req.copies.map(|copies| {
            copies
                .into_iter()
                .map(|p| (BlockId(p.src), BlockId(p.dst)))
                .collect()
        }),
        lora_id: req.lora_id,
        lora_path: req.lora_path,
        free_handles: req.free_handles,
    })
}

fn response_to_fb(resp: &WorkerResponse) -> anyhow::Result<fbs::WorkerResponseT> {
    Ok(fbs::WorkerResponseT {
        kind: resp_kind_to_fb(&resp.kind)?,
        call_id: resp.call_id,
        caps: resp
            .caps
            .as_ref()
            .map(caps_to_fb)
            .transpose()?
            .map(Box::new),
        result: resp
            .result
            .as_ref()
            .map(result_to_fb)
            .transpose()?
            .map(Box::new),
        metrics: resp.metrics.as_ref().map(metrics_to_fb).map(Box::new),
        pressure: resp
            .pressure
            .as_ref()
            .map(|items| items.iter().map(pressure_to_fb).collect()),
        message: resp.message.clone(),
        code: resp.code.clone(),
        retryable: resp.retryable,
        fatal: resp.fatal,
    })
}

fn response_from_fb(resp: fbs::WorkerResponseT) -> anyhow::Result<WorkerResponse> {
    Ok(WorkerResponse {
        kind: resp_kind_from_fb(resp.kind)?.to_string(),
        call_id: resp.call_id,
        caps: resp.caps.map(|c| caps_from_fb(*c)).transpose()?,
        result: resp.result.map(|r| result_from_fb(*r)).transpose()?,
        metrics: resp.metrics.map(|m| metrics_from_fb(*m)),
        pressure: resp
            .pressure
            .map(|items| items.into_iter().map(pressure_from_fb).collect())
            .transpose()?,
        message: resp.message,
        code: resp.code,
        retryable: resp.retryable,
        fatal: resp.fatal,
    })
}

fn batch_to_fb(batch: &ForwardBatch) -> anyhow::Result<fbs::ForwardBatchT> {
    Ok(fbs::ForwardBatchT {
        step_id: batch.step_id,
        new_reqs: Some(
            batch
                .new_reqs
                .iter()
                .map(new_request_to_fb)
                .collect::<anyhow::Result<_>>()?,
        ),
        ops: Some(
            batch
                .ops
                .iter()
                .map(op_to_fb)
                .collect::<anyhow::Result<_>>()?,
        ),
    })
}

fn batch_from_fb(batch: fbs::ForwardBatchT) -> anyhow::Result<ForwardBatch> {
    Ok(ForwardBatch {
        step_id: batch.step_id,
        new_reqs: batch
            .new_reqs
            .unwrap_or_default()
            .into_iter()
            .map(new_request_from_fb)
            .collect::<anyhow::Result<_>>()?,
        ops: batch
            .ops
            .unwrap_or_default()
            .into_iter()
            .map(op_from_fb)
            .collect::<anyhow::Result<_>>()?,
    })
}

fn new_request_to_fb(req: &NewRequestData) -> anyhow::Result<fbs::NewRequestDataT> {
    Ok(fbs::NewRequestDataT {
        req_id: req.req_id.0,
        sampling: req
            .sampling
            .as_ref()
            .map(sampling_to_fb)
            .transpose()?
            .map(Box::new),
        image: req.image.as_ref().map(image_to_fb).map(Box::new),
        neg_token_ids: req.neg_token_ids.clone(),
        lora_id: req.lora_id,
        block_ids: Some(req.block_ids.iter().map(|id| id.0).collect()),
        group_id: req.group_id,
    })
}

fn new_request_from_fb(req: fbs::NewRequestDataT) -> anyhow::Result<NewRequestData> {
    Ok(NewRequestData {
        req_id: RequestId(req.req_id),
        sampling: req.sampling.map(|s| sampling_from_fb(*s)).transpose()?,
        image: req.image.map(|i| image_from_fb(*i)),
        neg_token_ids: req.neg_token_ids,
        lora_id: req.lora_id,
        block_ids: req
            .block_ids
            .unwrap_or_default()
            .into_iter()
            .map(BlockId)
            .collect(),
        group_id: req.group_id,
    })
}

fn op_to_fb(op: &ForwardOp) -> anyhow::Result<fbs::ForwardOpT> {
    Ok(fbs::ForwardOpT {
        req_id: op.req_id.0,
        kind: op_kind_to_fb(op.kind),
        modality: modality_to_fb(op.modality),
        new_block_ids: Some(op.new_block_ids.iter().map(|id| id.0).collect()),
        pos_lo: op.pos_range.0,
        pos_hi: op.pos_range.1,
        token_ids: op.token_ids.clone(),
        token_source: token_source_to_fb(op.token_source),
        timestep_idx: op.timestep_idx,
        cond_pos: op.cond_pos,
        cfg: op.cfg.as_ref().map(cfg_to_fb).map(Box::new),
        image_in: op.image_in,
        image_prompt: op.image_prompt.clone(),
        image_b64: op.image_b64.clone(),
        group_id: op.group_id,
        allowed_tokens: op.allowed_tokens.clone(),
        suppress_tokens: op.suppress_tokens.clone(),
        recent_tokens: op.recent_tokens.clone(),
        mm_hash: op.mm_hash,
        spec_token_ids: op.spec_token_ids.clone(),
        denoise_step_count: op.denoise_step_count,
        decode_token_count: op.decode_token_count,
        decode_stop_token_ids: op.decode_stop_token_ids.clone(),
        decode_stop_terminal: op.decode_stop_terminal,
        return_all_logits: op.return_all_logits,
        op_id: op.op_id,
        logits_handle: op.logits_handle,
        locator: op.locator.clone(),
    })
}

fn op_from_fb(op: fbs::ForwardOpT) -> anyhow::Result<ForwardOp> {
    Ok(ForwardOp {
        req_id: RequestId(op.req_id),
        kind: op_kind_from_fb(op.kind)?,
        modality: modality_from_fb(op.modality)?,
        new_block_ids: op
            .new_block_ids
            .unwrap_or_default()
            .into_iter()
            .map(BlockId)
            .collect(),
        pos_range: (op.pos_lo, op.pos_hi),
        token_ids: op.token_ids,
        token_source: token_source_from_fb(op.token_source)?,
        timestep_idx: op.timestep_idx,
        cond_pos: op.cond_pos,
        cfg: op.cfg.map(|cfg| cfg_from_fb(*cfg)).transpose()?,
        image_in: op.image_in,
        image_prompt: op.image_prompt,
        image_b64: op.image_b64,
        group_id: op.group_id,
        allowed_tokens: op.allowed_tokens,
        suppress_tokens: op.suppress_tokens,
        recent_tokens: op.recent_tokens,
        mm_hash: op.mm_hash,
        spec_token_ids: op.spec_token_ids,
        denoise_step_count: op.denoise_step_count,
        decode_token_count: op.decode_token_count,
        decode_stop_token_ids: op.decode_stop_token_ids,
        decode_stop_terminal: op.decode_stop_terminal,
        return_all_logits: op.return_all_logits,
        op_id: op.op_id,
        logits_handle: op.logits_handle,
        locator: op.locator,
    })
}

fn result_to_fb(result: &ForwardResult) -> anyhow::Result<fbs::ForwardResultT> {
    Ok(fbs::ForwardResultT {
        step_id: result.step_id,
        per_seq: Some(
            result
                .per_seq
                .iter()
                .map(seq_result_to_fb)
                .collect::<anyhow::Result<Vec<_>>>()?,
        ),
        worker_exec_us: result.worker_exec_us,
        forward_stats: result
            .forward_stats
            .as_ref()
            .map(forward_stats_to_fb)
            .map(Box::new),
    })
}

fn result_from_fb(result: fbs::ForwardResultT) -> anyhow::Result<ForwardResult> {
    Ok(ForwardResult {
        step_id: result.step_id,
        per_seq: result
            .per_seq
            .unwrap_or_default()
            .into_iter()
            .map(seq_result_from_fb)
            .collect::<anyhow::Result<Vec<_>>>()?,
        worker_exec_us: result.worker_exec_us,
        forward_stats: result
            .forward_stats
            .map(|stats| forward_stats_from_fb(*stats)),
    })
}

fn forward_stats_to_fb(stats: &WorkerForwardStats) -> fbs::WorkerForwardStatsT {
    fbs::WorkerForwardStatsT {
        mode_counts: Some(map_to_fb(&stats.mode_counts)),
        mode_tokens: Some(map_to_fb(&stats.mode_tokens)),
        mode_us: Some(map_to_fb(&stats.mode_us)),
        component_us: Some(map_to_fb(&stats.component_us)),
        attention_launches: stats.attention_launches,
        attention_us: stats.attention_us,
        attention_backend_counts: Some(map_to_fb(&stats.attention_backend_counts)),
        cuda_graph_captures: stats.cuda_graph_captures,
        cuda_graph_replays: stats.cuda_graph_replays,
        cuda_graph_misses: stats.cuda_graph_misses,
        cuda_graph_fallbacks: stats.cuda_graph_fallbacks,
        cuda_graph_unpadded_tokens: stats.cuda_graph_unpadded_tokens,
        cuda_graph_padded_tokens: stats.cuda_graph_padded_tokens,
        cuda_graph_runtime_mode_counts: Some(map_to_fb(&stats.cuda_graph_runtime_mode_counts)),
        text_decode_token_relay_hits: stats.text_decode_token_relay_hits,
        text_decode_token_relay_misses: stats.text_decode_token_relay_misses,
        text_decode_position_relay_hits: stats.text_decode_position_relay_hits,
        text_decode_position_relay_misses: stats.text_decode_position_relay_misses,
        flashinfer_decode_plan_calls: stats.flashinfer_decode_plan_calls,
        flashinfer_decode_plan_reuses: stats.flashinfer_decode_plan_reuses,
        flashinfer_decode_plan_rows: stats.flashinfer_decode_plan_rows,
        flashinfer_decode_plan_indices: stats.flashinfer_decode_plan_indices,
        flashinfer_decode_graph_plan_calls: stats.flashinfer_decode_graph_plan_calls,
        flashinfer_decode_graph_plan_reuses: stats.flashinfer_decode_graph_plan_reuses,
        spec_verify_rows: stats.spec_verify_rows,
        spec_verify_draft_tokens: stats.spec_verify_draft_tokens,
        spec_verify_accepted_tokens: stats.spec_verify_accepted_tokens,
        spec_verify_rejected_tokens: stats.spec_verify_rejected_tokens,
        spec_verify_committed_tokens: stats.spec_verify_committed_tokens,
        spec_verify_path_counts: Some(map_to_fb(&stats.spec_verify_path_counts)),
    }
}

fn forward_stats_from_fb(stats: fbs::WorkerForwardStatsT) -> WorkerForwardStats {
    WorkerForwardStats {
        mode_counts: map_from_fb(stats.mode_counts),
        mode_tokens: map_from_fb(stats.mode_tokens),
        mode_us: map_from_fb(stats.mode_us),
        component_us: map_from_fb(stats.component_us),
        attention_launches: stats.attention_launches,
        attention_us: stats.attention_us,
        attention_backend_counts: map_from_fb(stats.attention_backend_counts),
        cuda_graph_captures: stats.cuda_graph_captures,
        cuda_graph_replays: stats.cuda_graph_replays,
        cuda_graph_misses: stats.cuda_graph_misses,
        cuda_graph_fallbacks: stats.cuda_graph_fallbacks,
        cuda_graph_unpadded_tokens: stats.cuda_graph_unpadded_tokens,
        cuda_graph_padded_tokens: stats.cuda_graph_padded_tokens,
        cuda_graph_runtime_mode_counts: map_from_fb(stats.cuda_graph_runtime_mode_counts),
        text_decode_token_relay_hits: stats.text_decode_token_relay_hits,
        text_decode_token_relay_misses: stats.text_decode_token_relay_misses,
        text_decode_position_relay_hits: stats.text_decode_position_relay_hits,
        text_decode_position_relay_misses: stats.text_decode_position_relay_misses,
        flashinfer_decode_plan_calls: stats.flashinfer_decode_plan_calls,
        flashinfer_decode_plan_reuses: stats.flashinfer_decode_plan_reuses,
        flashinfer_decode_plan_rows: stats.flashinfer_decode_plan_rows,
        flashinfer_decode_plan_indices: stats.flashinfer_decode_plan_indices,
        flashinfer_decode_graph_plan_calls: stats.flashinfer_decode_graph_plan_calls,
        flashinfer_decode_graph_plan_reuses: stats.flashinfer_decode_graph_plan_reuses,
        spec_verify_rows: stats.spec_verify_rows,
        spec_verify_draft_tokens: stats.spec_verify_draft_tokens,
        spec_verify_accepted_tokens: stats.spec_verify_accepted_tokens,
        spec_verify_rejected_tokens: stats.spec_verify_rejected_tokens,
        spec_verify_committed_tokens: stats.spec_verify_committed_tokens,
        spec_verify_path_counts: map_from_fb(stats.spec_verify_path_counts),
    }
}

fn seq_result_to_fb(sr: &SeqResult) -> anyhow::Result<fbs::SeqResultT> {
    validate_seq_result_logprobs(sr)?;
    Ok(fbs::SeqResultT {
        req_id: sr.req_id.0,
        sampled_token_id: sr.sampled_token_id,
        denoise_done: sr.denoise_done,
        num_steps_done: sr.num_steps_done,
        image_png_b64: sr.image_png_b64.clone(),
        // `SeqResult.image_hw` is canonically `(height, width)`; store each
        // component into the matching schema field so the field names do not lie.
        image_hw_h: sr.image_hw.map(|(h, _)| h),
        image_hw_w: sr.image_hw.map(|(_, w)| w),
        sampled_logprob: sr.sampled_logprob,
        top_logprobs: sr.top_logprobs.as_ref().map(|items| {
            items
                .iter()
                .map(|item| fbs::TokenLogprobT {
                    token_id: item.0,
                    logprob: item.1,
                    rank: item.2,
                })
                .collect()
        }),
        prompt_logprobs: sr.prompt_logprobs.as_ref().map(|positions| {
            positions
                .iter()
                .map(|entries| fbs::PositionLogprobsT {
                    entries: Some(
                        entries
                            .iter()
                            .map(|item| fbs::TokenLogprobT {
                                token_id: item.0,
                                logprob: item.1,
                                rank: item.2,
                            })
                            .collect(),
                    ),
                })
                .collect()
        }),
        sampled_token_ids: sr.sampled_token_ids.clone(),
        encoder_handle: sr.encoder_handle,
        num_tokens: sr.num_tokens,
        num_accepted_tokens: sr.num_accepted_tokens,
        op_id: sr.op_id,
        logits_handle: sr.logits_handle,
        locator: sr.locator.clone(),
        op_kind: sr
            .op_kind
            .map(op_kind_to_fb)
            .unwrap_or(fbs::OpKind::PrefillUnd),
        has_op_kind: sr.op_kind.is_some(),
    })
}

fn seq_result_from_fb(sr: fbs::SeqResultT) -> anyhow::Result<SeqResult> {
    let result = SeqResult {
        req_id: RequestId(sr.req_id),
        op_kind: sr
            .has_op_kind
            .then(|| op_kind_from_fb(sr.op_kind))
            .transpose()?,
        sampled_token_id: sr.sampled_token_id,
        denoise_done: sr.denoise_done,
        num_steps_done: sr.num_steps_done,
        image_png_b64: sr.image_png_b64,
        // Reassemble the canonical `(height, width)` tuple from the matching fields.
        image_hw: match (sr.image_hw_h, sr.image_hw_w) {
            (Some(h), Some(w)) => Some((h, w)),
            _ => None,
        },
        sampled_logprob: sr.sampled_logprob,
        top_logprobs: sr.top_logprobs.map(|items| {
            items
                .into_iter()
                .map(|item| crate::TokenLogprob(item.token_id, item.logprob, item.rank))
                .collect()
        }),
        prompt_logprobs: sr.prompt_logprobs.map(|positions| {
            positions
                .into_iter()
                .map(|position| {
                    position
                        .entries
                        .unwrap_or_default()
                        .into_iter()
                        .map(|item| crate::TokenLogprob(item.token_id, item.logprob, item.rank))
                        .collect()
                })
                .collect()
        }),
        sampled_token_ids: sr.sampled_token_ids,
        encoder_handle: sr.encoder_handle,
        num_tokens: sr.num_tokens,
        num_accepted_tokens: sr.num_accepted_tokens,
        op_id: sr.op_id,
        logits_handle: sr.logits_handle,
        locator: sr.locator,
    };
    validate_seq_result_logprobs(&result)?;
    Ok(result)
}

fn validate_seq_result_logprobs(result: &SeqResult) -> anyhow::Result<()> {
    if result
        .sampled_logprob
        .is_some_and(|value| !value.is_finite())
    {
        bail!("sampled_logprob must be finite");
    }
    if let Some(entries) = &result.top_logprobs {
        for (index, entry) in entries.iter().enumerate() {
            validate_token_logprob(entry, &format!("top_logprobs[{index}]"))?;
        }
    }
    if let Some(positions) = &result.prompt_logprobs {
        for (position, entries) in positions.iter().enumerate() {
            for (index, entry) in entries.iter().enumerate() {
                validate_token_logprob(entry, &format!("prompt_logprobs[{position}][{index}]"))?;
            }
        }
    }
    Ok(())
}

fn validate_token_logprob(entry: &TokenLogprob, where_: &str) -> anyhow::Result<()> {
    if !entry.1.is_finite() {
        bail!("{where_}.logprob must be finite");
    }
    if entry.2 == 0 {
        bail!("{where_}.rank must be at least 1");
    }
    Ok(())
}

fn token_source_to_fb(source: TokenSource) -> fbs::TokenSource {
    match source {
        TokenSource::Wire => fbs::TokenSource::Wire,
        TokenSource::LastSampled => fbs::TokenSource::LastSampled,
    }
}

fn token_source_from_fb(source: fbs::TokenSource) -> anyhow::Result<TokenSource> {
    Ok(match source {
        fbs::TokenSource::Wire => TokenSource::Wire,
        fbs::TokenSource::LastSampled => TokenSource::LastSampled,
        _ => bail!("unknown TokenSource {:?}", source),
    })
}

fn caps_to_fb(caps: &EngineCaps) -> anyhow::Result<fbs::EngineCapsT> {
    Ok(fbs::EngineCapsT {
        block_size: caps.block_size,
        num_blocks: caps.num_blocks,
        num_layers: caps.num_layers,
        scratch_capacity_tokens: caps.scratch_capacity_tokens,
        supported_ops: Some(
            caps.supported_ops
                .iter()
                .copied()
                .map(op_kind_to_fb)
                .collect(),
        ),
        max_latent_size: caps.max_latent_size,
        latent_downsample: caps.latent_downsample,
        max_vae_grid_tokens: caps.max_vae_grid_tokens,
        max_vit_grid_tokens: caps.max_vit_grid_tokens,
        commit_marker_tokens: caps.commit_marker_tokens,
        gen_rope_advance: caps.gen_rope_advance,
        max_cfg_branches: caps.max_cfg_branches,
        bytes_per_token: caps.bytes_per_token,
        groups: Some(caps.groups.iter().map(kv_group_to_fb).collect()),
        kv_dtype: Some(caps.kv_dtype.clone()),
        attention_backend: Some(caps.attention_backend.clone()),
        quantization: caps.quantization.clone(),
        rank: Some(Box::new(rank_to_fb(caps.rank))),
        pipeline_depth: caps.pipeline_depth,
        encoder_cache_budget: caps.encoder_cache_budget,
        supported_controls: Some(
            caps.supported_controls
                .iter()
                .map(|control| {
                    RequestKind::from_wire_str(control)
                        .map(req_kind_to_fb)
                        .ok_or_else(|| anyhow::anyhow!("unknown control kind {control:?}"))
                })
                .collect::<anyhow::Result<_>>()?,
        ),
        adapter_mode: adapter_mode_to_fb(caps.adapter_mode),
        execution_constraints: Some(Box::new(execution_constraints_to_fb(
            &caps.execution_constraints,
        ))),
        resource_classes: Some(
            caps.resource_classes
                .iter()
                .copied()
                .map(resource_class_to_fb)
                .collect(),
        ),
    })
}

fn caps_from_fb(caps: fbs::EngineCapsT) -> anyhow::Result<EngineCaps> {
    Ok(EngineCaps {
        block_size: caps.block_size,
        num_blocks: caps.num_blocks,
        num_layers: caps.num_layers,
        scratch_capacity_tokens: caps.scratch_capacity_tokens,
        supported_ops: caps
            .supported_ops
            .unwrap_or_default()
            .into_iter()
            .map(op_kind_from_fb)
            .collect::<anyhow::Result<_>>()?,
        max_latent_size: caps.max_latent_size,
        latent_downsample: caps.latent_downsample,
        max_vae_grid_tokens: caps.max_vae_grid_tokens,
        max_vit_grid_tokens: caps.max_vit_grid_tokens,
        commit_marker_tokens: if caps.commit_marker_tokens == 0 {
            2
        } else {
            caps.commit_marker_tokens
        },
        gen_rope_advance: if caps.gen_rope_advance == 0 {
            2
        } else {
            caps.gen_rope_advance
        },
        max_cfg_branches: if caps.max_cfg_branches == 0 {
            3
        } else {
            caps.max_cfg_branches
        },
        bytes_per_token: caps.bytes_per_token,
        groups: caps
            .groups
            .unwrap_or_default()
            .into_iter()
            .map(kv_group_from_fb)
            .collect::<anyhow::Result<_>>()?,
        kv_dtype: caps.kv_dtype.unwrap_or_else(|| "bf16".to_string()),
        attention_backend: caps
            .attention_backend
            .unwrap_or_else(|| "flashinfer".to_string()),
        quantization: caps.quantization,
        rank: caps.rank.map(|r| rank_from_fb(*r)).unwrap_or_default(),
        pipeline_depth: caps.pipeline_depth.max(1),
        encoder_cache_budget: caps.encoder_cache_budget,
        supported_controls: caps
            .supported_controls
            .unwrap_or_default()
            .into_iter()
            .map(|kind| req_kind_from_fb(kind).map(|k| k.as_wire_str().to_string()))
            .collect::<anyhow::Result<_>>()?,
        adapter_mode: adapter_mode_from_fb(caps.adapter_mode)?,
        execution_constraints: caps
            .execution_constraints
            .map(|ec| execution_constraints_from_fb(*ec))
            .unwrap_or_default(),
        resource_classes: caps
            .resource_classes
            .unwrap_or_default()
            .into_iter()
            .map(resource_class_from_fb)
            .collect::<anyhow::Result<_>>()?,
    })
}

fn sampling_to_fb(s: &SamplingParams) -> anyhow::Result<fbs::SamplingParamsT> {
    Ok(fbs::SamplingParamsT {
        temperature: s.temperature,
        top_k: s.top_k,
        top_p: s.top_p,
        ignore_eos: s.ignore_eos,
        seed: s.seed,
        min_p: s.min_p,
        repetition_penalty: s.repetition_penalty,
        frequency_penalty: s.frequency_penalty,
        presence_penalty: s.presence_penalty,
        logit_bias: Some(
            s.logit_bias
                .iter()
                .map(|(token_id, bias)| fbs::TokenBiasT {
                    token_id: *token_id,
                    bias: *bias,
                })
                .collect(),
        ),
        min_tokens: s.min_tokens as u64,
        return_logprobs: s.return_logprobs,
        n_logprobs: s.n_logprobs,
        return_prompt_logprobs: s.return_prompt_logprobs,
        n_prompt_logprobs: s.n_prompt_logprobs,
        logprob_token_ids: Some(s.logprob_token_ids.clone()),
        bad_words_ids: Some(
            s.bad_words_ids
                .iter()
                .map(|items| fbs::U32ListT {
                    items: Some(items.clone()),
                })
                .collect(),
        ),
        allowed_token_ids: s.allowed_token_ids.clone(),
    })
}

fn sampling_from_fb(s: fbs::SamplingParamsT) -> anyhow::Result<SamplingParams> {
    for (name, value) in [
        ("temperature", s.temperature),
        ("top_p", s.top_p),
        ("min_p", s.min_p),
        ("repetition_penalty", s.repetition_penalty),
        ("frequency_penalty", s.frequency_penalty),
        ("presence_penalty", s.presence_penalty),
    ] {
        if !value.is_finite() {
            bail!("non-finite sampling param {name}: {value}");
        }
    }
    Ok(SamplingParams {
        temperature: s.temperature,
        top_k: s.top_k,
        top_p: s.top_p,
        ignore_eos: s.ignore_eos,
        seed: s.seed,
        min_p: s.min_p,
        repetition_penalty: s.repetition_penalty,
        frequency_penalty: s.frequency_penalty,
        presence_penalty: s.presence_penalty,
        logit_bias: s
            .logit_bias
            .unwrap_or_default()
            .into_iter()
            .map(|item| (item.token_id, item.bias))
            .collect(),
        min_tokens: usize::try_from(s.min_tokens).context("min_tokens does not fit usize")?,
        return_logprobs: s.return_logprobs,
        n_logprobs: s.n_logprobs,
        return_prompt_logprobs: s.return_prompt_logprobs,
        n_prompt_logprobs: s.n_prompt_logprobs,
        logprob_token_ids: s.logprob_token_ids.unwrap_or_default(),
        bad_words_ids: s
            .bad_words_ids
            .unwrap_or_default()
            .into_iter()
            .map(|items| items.items.unwrap_or_default())
            .collect(),
        allowed_token_ids: s.allowed_token_ids,
    })
}

fn image_to_fb(i: &ImageParams) -> fbs::ImageParamsT {
    fbs::ImageParamsT {
        steps: i.steps,
        cfg_text_scale: i.cfg_text_scale,
        cfg_img_scale: i.cfg_img_scale,
        cfg_renorm_type: Some(i.cfg_renorm_type.clone()),
        cfg_renorm_min: i.cfg_renorm_min,
        cfg_interval_lo: i.cfg_interval.0,
        cfg_interval_hi: i.cfg_interval.1,
        timestep_shift: i.timestep_shift,
        height: i.height,
        width: i.width,
        seed: i.seed,
        negative_prompt: Some(i.negative_prompt.clone()),
        max_images: i.max_images,
        image_prompts: Some(i.image_prompts.clone()),
        retain_images: i.retain_images,
    }
}

fn image_from_fb(i: fbs::ImageParamsT) -> ImageParams {
    ImageParams {
        steps: i.steps,
        cfg_text_scale: i.cfg_text_scale,
        cfg_img_scale: i.cfg_img_scale,
        cfg_renorm_type: i.cfg_renorm_type.unwrap_or_else(|| "global".to_string()),
        cfg_renorm_min: i.cfg_renorm_min,
        cfg_interval: (i.cfg_interval_lo, i.cfg_interval_hi),
        timestep_shift: i.timestep_shift,
        height: i.height,
        width: i.width,
        seed: i.seed,
        negative_prompt: i.negative_prompt.unwrap_or_default(),
        max_images: i.max_images,
        image_prompts: i.image_prompts.unwrap_or_default(),
        retain_images: i.retain_images,
    }
}

fn cfg_to_fb(cfg: &CfgParams) -> fbs::CfgParamsT {
    fbs::CfgParamsT {
        branch_count: cfg.branch_count,
        text_scale: cfg.text_scale,
        img_scale: cfg.img_scale,
        renorm_type: Some(cfg.renorm_type.clone()),
        renorm_min: cfg.renorm_min,
        interval_lo: cfg.interval.0,
        interval_hi: cfg.interval.1,
    }
}

fn cfg_from_fb(cfg: fbs::CfgParamsT) -> anyhow::Result<CfgParams> {
    for (name, value) in [
        ("text_scale", cfg.text_scale),
        ("img_scale", cfg.img_scale),
        ("renorm_min", cfg.renorm_min),
        ("interval_lo", cfg.interval_lo),
        ("interval_hi", cfg.interval_hi),
    ] {
        if !value.is_finite() {
            bail!("non-finite cfg param {name}: {value}");
        }
    }
    Ok(CfgParams {
        branch_count: cfg.branch_count,
        text_scale: cfg.text_scale,
        img_scale: cfg.img_scale,
        renorm_type: cfg.renorm_type.unwrap_or_else(|| "global".to_string()),
        renorm_min: cfg.renorm_min,
        interval: (cfg.interval_lo, cfg.interval_hi),
    })
}

fn metrics_to_fb(m: &WorkerMetrics) -> fbs::WorkerMetricsT {
    fbs::WorkerMetricsT {
        executes: m.executes,
        ops_total: m.ops_total,
        exec_us_total: m.exec_us_total,
        last_exec_us: m.last_exec_us,
        op_kind_counts: Some(map_to_fb(&m.op_kind_counts)),
        op_kind_us: Some(map_to_fb(&m.op_kind_us)),
        control_ok: Some(map_to_fb(&m.control_ok)),
        control_err: Some(map_to_fb(&m.control_err)),
        error_counts: Some(map_to_fb(&m.error_counts)),
        cuda_graph_captures: m.cuda_graph_captures,
        cuda_graph_replays: m.cuda_graph_replays,
        cuda_graph_misses: m.cuda_graph_misses,
        cuda_graph_fallbacks: m.cuda_graph_fallbacks,
        cuda_graph_unpadded_tokens: m.cuda_graph_unpadded_tokens,
        cuda_graph_padded_tokens: m.cuda_graph_padded_tokens,
        cuda_graph_runtime_mode_counts: Some(map_to_fb(&m.cuda_graph_runtime_mode_counts)),
        forward: m.forward.as_ref().map(forward_stats_to_fb).map(Box::new),
    }
}

fn metrics_from_fb(m: fbs::WorkerMetricsT) -> WorkerMetrics {
    WorkerMetrics {
        executes: m.executes,
        ops_total: m.ops_total,
        exec_us_total: m.exec_us_total,
        last_exec_us: m.last_exec_us,
        op_kind_counts: map_from_fb(m.op_kind_counts),
        op_kind_us: map_from_fb(m.op_kind_us),
        control_ok: map_from_fb(m.control_ok),
        control_err: map_from_fb(m.control_err),
        error_counts: map_from_fb(m.error_counts),
        cuda_graph_captures: m.cuda_graph_captures,
        cuda_graph_replays: m.cuda_graph_replays,
        cuda_graph_misses: m.cuda_graph_misses,
        cuda_graph_fallbacks: m.cuda_graph_fallbacks,
        cuda_graph_unpadded_tokens: m.cuda_graph_unpadded_tokens,
        cuda_graph_padded_tokens: m.cuda_graph_padded_tokens,
        cuda_graph_runtime_mode_counts: map_from_fb(m.cuda_graph_runtime_mode_counts),
        forward: m.forward.map(|stats| forward_stats_from_fb(*stats)),
    }
}

fn map_to_fb(map: &BTreeMap<String, u64>) -> Vec<fbs::StringU64PairT> {
    map.iter()
        .map(|(key, value)| fbs::StringU64PairT {
            key: Some(key.clone()),
            value: *value,
        })
        .collect()
}

fn map_from_fb(items: Option<Vec<fbs::StringU64PairT>>) -> BTreeMap<String, u64> {
    items
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| item.key.map(|key| (key, item.value)))
        .collect()
}

fn kv_group_to_fb(group: &KvCacheGroupSpec) -> fbs::KvGroupSpecT {
    match group.kind {
        KvGroupKind::Full => fbs::KvGroupSpecT {
            group_id: group.group_id,
            block_offset: group.block_offset,
            num_blocks: group.num_blocks,
            kind: fbs::KvGroupKind::Full,
            window: 0,
            sink: 0,
        },
        KvGroupKind::SlidingWindow { window, sink } => fbs::KvGroupSpecT {
            group_id: group.group_id,
            block_offset: group.block_offset,
            num_blocks: group.num_blocks,
            kind: fbs::KvGroupKind::SlidingWindow,
            window,
            sink,
        },
    }
}

fn kv_group_from_fb(group: fbs::KvGroupSpecT) -> anyhow::Result<KvCacheGroupSpec> {
    let kind = if group.kind == fbs::KvGroupKind::Full {
        KvGroupKind::Full
    } else if group.kind == fbs::KvGroupKind::SlidingWindow {
        KvGroupKind::SlidingWindow {
            window: group.window,
            sink: group.sink,
        }
    } else {
        bail!("unknown KV group kind {}", group.kind.0);
    };
    Ok(KvCacheGroupSpec {
        group_id: group.group_id,
        block_offset: group.block_offset,
        num_blocks: group.num_blocks,
        kind,
    })
}

fn rank_to_fb(rank: RankInfo) -> fbs::RankInfoT {
    fbs::RankInfoT {
        tp_rank: rank.tp_rank,
        tp_size: rank.tp_size,
        pp_rank: rank.pp_rank,
        pp_size: rank.pp_size,
        dp_rank: rank.dp_rank,
        dp_size: rank.dp_size,
    }
}

fn rank_from_fb(rank: fbs::RankInfoT) -> RankInfo {
    RankInfo {
        tp_rank: rank.tp_rank,
        tp_size: rank.tp_size,
        pp_rank: rank.pp_rank,
        pp_size: rank.pp_size,
        dp_rank: rank.dp_rank,
        dp_size: rank.dp_size,
    }
}

fn execution_constraints_to_fb(ec: &ExecutionConstraints) -> fbs::ExecutionConstraintsT {
    fbs::ExecutionConstraintsT {
        max_batch_ops: ec.max_batch_ops,
    }
}

fn execution_constraints_from_fb(ec: fbs::ExecutionConstraintsT) -> ExecutionConstraints {
    ExecutionConstraints {
        max_batch_ops: ec.max_batch_ops,
    }
}

fn pressure_to_fb(p: &ResourcePressure) -> fbs::ResourcePressureT {
    fbs::ResourcePressureT {
        class: resource_class_to_fb(p.class),
        total: p.total,
        used: p.used,
        evictable: p.evictable,
        free: p.free,
    }
}

fn pressure_from_fb(p: fbs::ResourcePressureT) -> anyhow::Result<ResourcePressure> {
    Ok(ResourcePressure {
        class: resource_class_from_fb(p.class)?,
        total: p.total,
        used: p.used,
        evictable: p.evictable,
        free: p.free,
    })
}

fn op_kind_to_fb(kind: OpKind) -> fbs::OpKind {
    match kind {
        OpKind::PrefillUnd => fbs::OpKind::PrefillUnd,
        OpKind::DecodeUnd => fbs::OpKind::DecodeUnd,
        OpKind::TargetVerifyUnd => fbs::OpKind::TargetVerifyUnd,
        OpKind::DenoiseGen => fbs::OpKind::DenoiseGen,
        OpKind::CommitGen => fbs::OpKind::CommitGen,
        OpKind::CommitWriteback => fbs::OpKind::CommitWriteback,
        OpKind::VaeEncode => fbs::OpKind::VaeEncode,
        OpKind::VitEncode => fbs::OpKind::VitEncode,
        OpKind::Sample => fbs::OpKind::Sample,
        OpKind::EncodeFrame => fbs::OpKind::EncodeFrame,
    }
}

fn op_kind_from_fb(kind: fbs::OpKind) -> anyhow::Result<OpKind> {
    if kind == fbs::OpKind::PrefillUnd {
        Ok(OpKind::PrefillUnd)
    } else if kind == fbs::OpKind::DecodeUnd {
        Ok(OpKind::DecodeUnd)
    } else if kind == fbs::OpKind::TargetVerifyUnd {
        Ok(OpKind::TargetVerifyUnd)
    } else if kind == fbs::OpKind::DenoiseGen {
        Ok(OpKind::DenoiseGen)
    } else if kind == fbs::OpKind::CommitGen {
        Ok(OpKind::CommitGen)
    } else if kind == fbs::OpKind::CommitWriteback {
        Ok(OpKind::CommitWriteback)
    } else if kind == fbs::OpKind::VaeEncode {
        Ok(OpKind::VaeEncode)
    } else if kind == fbs::OpKind::VitEncode {
        Ok(OpKind::VitEncode)
    } else if kind == fbs::OpKind::Sample {
        Ok(OpKind::Sample)
    } else if kind == fbs::OpKind::EncodeFrame {
        Ok(OpKind::EncodeFrame)
    } else {
        bail!("unknown op kind {}", kind.0)
    }
}

fn modality_to_fb(modality: Modality) -> fbs::Modality {
    match modality {
        Modality::Und => fbs::Modality::Und,
        Modality::Gen => fbs::Modality::Gen,
    }
}

fn modality_from_fb(modality: fbs::Modality) -> anyhow::Result<Modality> {
    if modality == fbs::Modality::Und {
        Ok(Modality::Und)
    } else if modality == fbs::Modality::Gen {
        Ok(Modality::Gen)
    } else {
        bail!("unknown modality {}", modality.0)
    }
}

/// Single source of truth for the request-kind `<->` FlatBuffers enum mapping.
///
/// The kind taxonomy is centralized in this table so encode/decode cannot drift.
/// `kind_table_round_trips` (below) pins it against the `request_kind_code` header
/// byte so no encoding can disagree.
const REQUEST_KINDS: &[(RequestKind, fbs::ReqKind)] = &[
    (RequestKind::GetCaps, fbs::ReqKind::GetCaps),
    (RequestKind::Execute, fbs::ReqKind::Execute),
    (RequestKind::DropRequest, fbs::ReqKind::DropRequest),
    (RequestKind::Shutdown, fbs::ReqKind::Shutdown),
    (RequestKind::CopyBlocks, fbs::ReqKind::CopyBlocks),
    (RequestKind::LoadLora, fbs::ReqKind::LoadLora),
    (RequestKind::UnloadLora, fbs::ReqKind::UnloadLora),
    (RequestKind::FreeEncoder, fbs::ReqKind::FreeEncoder),
    (
        RequestKind::ResetPrefixCache,
        fbs::ReqKind::ResetPrefixCache,
    ),
    (RequestKind::Sleep, fbs::ReqKind::Sleep),
    (RequestKind::WakeUp, fbs::ReqKind::WakeUp),
    (RequestKind::GetMetrics, fbs::ReqKind::GetMetrics),
    (RequestKind::GetPressure, fbs::ReqKind::GetPressure),
];

const RESPONSE_KINDS: &[(&str, fbs::RespKind)] = &[
    ("caps", fbs::RespKind::Caps),
    ("result", fbs::RespKind::Result),
    ("ok", fbs::RespKind::Ok),
    ("error", fbs::RespKind::Error),
    ("metrics", fbs::RespKind::Metrics),
    ("pressure", fbs::RespKind::Pressure),
];

/// The canonical request-kind names accepted by the wire, in declaration order.
///
/// Exposed so the lower-level header codec (`worker-ipc-core`) can pin its own
/// `request_kind_code` table against this single source rather than maintaining
/// an independent, drift-prone copy.
pub fn request_kind_names() -> impl Iterator<Item = &'static str> {
    REQUEST_KINDS.iter().map(|(kind, _)| kind.as_wire_str())
}

fn req_kind_to_fb(kind: RequestKind) -> fbs::ReqKind {
    match kind {
        RequestKind::GetCaps => fbs::ReqKind::GetCaps,
        RequestKind::Execute => fbs::ReqKind::Execute,
        RequestKind::DropRequest => fbs::ReqKind::DropRequest,
        RequestKind::Shutdown => fbs::ReqKind::Shutdown,
        RequestKind::CopyBlocks => fbs::ReqKind::CopyBlocks,
        RequestKind::LoadLora => fbs::ReqKind::LoadLora,
        RequestKind::UnloadLora => fbs::ReqKind::UnloadLora,
        RequestKind::FreeEncoder => fbs::ReqKind::FreeEncoder,
        RequestKind::ResetPrefixCache => fbs::ReqKind::ResetPrefixCache,
        RequestKind::Sleep => fbs::ReqKind::Sleep,
        RequestKind::WakeUp => fbs::ReqKind::WakeUp,
        RequestKind::GetMetrics => fbs::ReqKind::GetMetrics,
        RequestKind::GetPressure => fbs::ReqKind::GetPressure,
    }
}

fn req_kind_from_fb(kind: fbs::ReqKind) -> anyhow::Result<RequestKind> {
    REQUEST_KINDS
        .iter()
        .find(|(_, fb)| *fb == kind)
        .map(|(rk, _)| *rk)
        .ok_or_else(|| anyhow::anyhow!("unknown request kind {}", kind.0))
}

fn resp_kind_to_fb(kind: &str) -> anyhow::Result<fbs::RespKind> {
    RESPONSE_KINDS
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, fb)| *fb)
        .ok_or_else(|| anyhow::anyhow!("unknown response kind {kind:?}"))
}

fn resp_kind_from_fb(kind: fbs::RespKind) -> anyhow::Result<&'static str> {
    RESPONSE_KINDS
        .iter()
        .find(|(_, fb)| *fb == kind)
        .map(|(name, _)| *name)
        .ok_or_else(|| anyhow::anyhow!("unknown response kind {}", kind.0))
}

fn adapter_mode_to_fb(mode: AdapterMode) -> fbs::AdapterMode {
    match mode {
        AdapterMode::None => fbs::AdapterMode::None,
        AdapterMode::EngineWide => fbs::AdapterMode::EngineWide,
        AdapterMode::PerRequest => fbs::AdapterMode::PerRequest,
        AdapterMode::MultiAdapter => fbs::AdapterMode::MultiAdapter,
    }
}

fn adapter_mode_from_fb(mode: fbs::AdapterMode) -> anyhow::Result<AdapterMode> {
    if mode == fbs::AdapterMode::None {
        Ok(AdapterMode::None)
    } else if mode == fbs::AdapterMode::EngineWide {
        Ok(AdapterMode::EngineWide)
    } else if mode == fbs::AdapterMode::PerRequest {
        Ok(AdapterMode::PerRequest)
    } else if mode == fbs::AdapterMode::MultiAdapter {
        Ok(AdapterMode::MultiAdapter)
    } else {
        bail!("unknown adapter mode {}", mode.0)
    }
}

fn resource_class_to_fb(class: ResourceClass) -> fbs::ResourceClass {
    match class {
        ResourceClass::KvBlock => fbs::ResourceClass::KvBlock,
        ResourceClass::EncoderOutput => fbs::ResourceClass::EncoderOutput,
        ResourceClass::ImageLatent => fbs::ResourceClass::ImageLatent,
        ResourceClass::Scratch => fbs::ResourceClass::Scratch,
        ResourceClass::Adapter => fbs::ResourceClass::Adapter,
    }
}

fn resource_class_from_fb(class: fbs::ResourceClass) -> anyhow::Result<ResourceClass> {
    if class == fbs::ResourceClass::KvBlock {
        Ok(ResourceClass::KvBlock)
    } else if class == fbs::ResourceClass::EncoderOutput {
        Ok(ResourceClass::EncoderOutput)
    } else if class == fbs::ResourceClass::ImageLatent {
        Ok(ResourceClass::ImageLatent)
    } else if class == fbs::ResourceClass::Scratch {
        Ok(ResourceClass::Scratch)
    } else if class == fbs::ResourceClass::Adapter {
        Ok(ResourceClass::Adapter)
    } else {
        bail!("unknown resource class {}", class.0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn seq_result_image_hw_field_labels_match_height_width() {
        // `SeqResult.image_hw` is canonically `(height, width)`. Use asymmetric
        // dims so a height/width swap in either lambda would be caught.
        let height = 720u32;
        let width = 1280u32;
        let native = SeqResult {
            req_id: RequestId(1),
            image_hw: Some((height, width)),
            ..Default::default()
        };

        let fb = seq_result_to_fb(&native).expect("valid sequence result");
        // The schema field named `image_hw_h` must carry the height component,
        // and `image_hw_w` the width component.
        assert_eq!(fb.image_hw_h, Some(height));
        assert_eq!(fb.image_hw_w, Some(width));

        let back = seq_result_from_fb(fb).expect("valid sequence result");
        assert_eq!(back.image_hw, Some((height, width)));
    }

    #[test]
    fn seq_result_rejects_non_finite_logprobs_and_zero_ranks() {
        let non_finite = SeqResult {
            req_id: RequestId(1),
            sampled_logprob: Some(f32::NAN),
            ..Default::default()
        };
        assert!(seq_result_to_fb(&non_finite).is_err());

        let zero_rank = fbs::SeqResultT {
            req_id: 1,
            top_logprobs: Some(vec![fbs::TokenLogprobT {
                token_id: 7,
                logprob: -0.5,
                rank: 0,
            }]),
            ..Default::default()
        };
        assert!(seq_result_from_fb(zero_rank).is_err());
    }

    #[test]
    fn kind_tables_round_trip_and_reject_dead_cancel() {
        // both directions derive from the single REQUEST_KINDS /
        // RESPONSE_KINDS tables, so a `name -> fb -> name` round-trip over the
        // whole table catches any drift between the two lookups.
        for (kind, fb) in REQUEST_KINDS {
            assert_eq!(req_kind_to_fb(*kind), *fb, "to_fb drift: {kind:?}");
            assert_eq!(
                req_kind_from_fb(*fb).unwrap(),
                *kind,
                "from_fb drift: {kind:?}"
            );
        }
        for (name, fb) in RESPONSE_KINDS {
            assert_eq!(resp_kind_to_fb(name).unwrap(), *fb, "to_fb drift: {name}");
            assert_eq!(
                resp_kind_from_fb(*fb).unwrap(),
                *name,
                "from_fb drift: {name}"
            );
        }
        // the `cancel` request kind is not in the wire vocabulary.
        assert!(RequestKind::from_wire_str("cancel").is_none());
    }

    #[test]
    fn sampling_from_fb_rejects_non_finite_floats() {
        let s = fbs::SamplingParamsT {
            temperature: f32::NAN,
            ..Default::default()
        };
        assert!(sampling_from_fb(s).is_err());

        let s = fbs::SamplingParamsT {
            top_p: f32::INFINITY,
            ..Default::default()
        };
        assert!(sampling_from_fb(s).is_err());
    }

    #[test]
    fn cfg_from_fb_rejects_non_finite_floats() {
        let cfg = fbs::CfgParamsT {
            text_scale: f32::NAN,
            ..Default::default()
        };
        assert!(cfg_from_fb(cfg).is_err());

        let cfg = fbs::CfgParamsT {
            interval_hi: f32::NEG_INFINITY,
            ..Default::default()
        };
        assert!(cfg_from_fb(cfg).is_err());

        // A fully-finite cfg still decodes.
        let cfg = fbs::CfgParamsT::default();
        assert!(cfg_from_fb(cfg).is_ok());
    }
}
