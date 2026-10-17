//! Startup warmup through the scheduler.
//!
//! Once the workers have reported, the engine admits real requests of the
//! shapes they prepared for, so each captured row count, guidance-branch
//! count and canvas chunk has run through the serving path before the first
//! client request. The scheduler and its allocators form every batch; a
//! worker states only the shapes (`WorkerInfo::image_warmups`,
//! `canvas_step_rows`) and takes no part in forming them.
//!
//! The requests are ordinary generation requests with a one-token prompt:
//!
//! - a token request of two tokens, which runs a prefill and a decode that
//!   consumes the prefill's token;
//! - for each row count from `canvas_step_rows` down to one, that many
//!   canvas requests of one block, submitted together: they prefill their
//!   prompts in one batch and then step together, so every canvas step of
//!   the group runs with exactly that many rows. The one-row request
//!   generates a second block, so its committed block runs the prefill
//!   kernels of a commit;
//! - for each image shape, one image request of two denoising steps at the
//!   shape's guidance scales, which also decodes its image. The scheduler's
//!   denoiser lane runs one request at a time, so serving never batches
//!   denoising calls and one request reaches every shape it can run.
//!
//! No request reads or populates the prefix or encoder caches. A shape the
//! deployment's generation limits would refuse a client, such as a guidance
//! branch count above the model's bound, is not one serving can reach and is
//! left unwarmed.

use std::time::Duration;

use anyhow::Context as _;
use uniserve_core::{
    CachePolicy, EngineCoreOutput, FeedbackNextToken, FinishReason, GenerationConstraint,
    GenerationRequest, ImageGenerationConfig, ImageParams, ImageTrigger, MultimodalInputs, Request,
    SamplingParams,
};
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

use crate::core::{EngineConfig, EngineCore};
use crate::handle::EventRx;

/// Denoising steps of a warmup image: the fewest that run a step from a
/// prepared latent and a step from a denoised one.
const IMAGE_STEPS: u16 = 2;

/// Runs the warmup requests the workers' reports call for on `core`.
///
/// Returns once every request has finished. Fails, naming the group, when a
/// request is rejected, ends in an error, or the engine dies first.
///
/// Warmup runs only where the workers execute on CUDA devices: a CPU or stub
/// worker has no kernels or graphs to warm.
pub(crate) fn warm(core: &EngineCore, config: &EngineConfig) -> anyhow::Result<()> {
    let on_cuda = config
        .workers
        .iter()
        .flat_map(|worker| &worker.ranks)
        .any(|rank| rank.device == "cuda" || rank.device.starts_with("cuda:"));
    if config.worker_process.stub || !on_cuda {
        return Ok(());
    }

    let info = core.info();
    let limits = core.generation_limits();
    let supports = |kind| info.supported_calls.contains(&kind);
    let token = |tokens: usize| {
        warmup_request(
            config,
            core.next_request_id(),
            GenerationConstraint::UndOnly,
            tokens,
            ImageParams::default(),
            Vec::new(),
        )
    };
    // Keeps the requests clients could submit.
    let servable = |requests: Vec<GenerationRequest>| {
        requests
            .into_iter()
            .filter(|request| {
                limits.features.contains(request.required_features())
                    && request.validate_resources(&limits).is_ok()
            })
            .collect::<Vec<_>>()
    };

    // The token scenario: a prefill followed by a decode over its token. A
    // canvas deployment generates in blocks instead, which the canvas
    // requests below run.
    if supports(CallKind::Forward(ForwardMode::Prefill)) && info.canvas_step_rows == 0 {
        let requests = servable(vec![token(2)]);
        if !requests.is_empty() {
            run(core, "token", requests)?;
            tracing::info!("completed token runtime warmup");
        }
    }

    if info.canvas_step_rows > 0 {
        let canvas = config
            .worker_process
            .canvas_sampling
            .context("workers report canvas rows but the deployment has no canvas sampling")?;
        let length = canvas.canvas_length as usize;
        // Requests of different lengths stop stepping together once the
        // shortest finishes: each commits its block in its own batch, and
        // whether two of them step together again depends on timing. So each
        // row count is its own group of equal requests.
        let mut warmed = false;
        for rows in (1..=info.canvas_step_rows as usize).rev() {
            let blocks = if rows == 1 { 2 } else { 1 };
            let requests = servable((0..rows).map(|_| token(blocks * length)).collect());
            if !requests.is_empty() {
                run(core, "canvas", requests)?;
                warmed = true;
            }
        }
        if warmed {
            tracing::info!("completed canvas generation warmup");
        }
    }

    let images = supports(CallKind::Media(MediaCall::LatentPreparation))
        && supports(CallKind::Media(MediaCall::Denoising));
    if images {
        let mut warmed = false;
        for shape in &info.image_warmups {
            let image = ImageParams {
                steps: IMAGE_STEPS,
                cfg_text_scale: shape.cfg_text_scale,
                cfg_img_scale: shape.cfg_img_scale,
                height: shape.height,
                width: shape.width,
                seed: Some(0),
                ..ImageParams::default()
            };
            let requests = servable(vec![warmup_request(
                config,
                core.next_request_id(),
                GenerationConstraint::GenOnly,
                0,
                image,
                shape.negative_token_ids.clone(),
            )]);
            if !requests.is_empty() {
                run(core, "image", requests)?;
                warmed = true;
            }
        }
        if warmed {
            tracing::info!("completed flow runtime warmup");
        }
    }
    Ok(())
}

/// Builds one warmup generation request.
///
/// `tokens` is the number of understanding tokens it generates; an image
/// request generates none and starts its image directly after the prompt.
/// `negative_token_ids` is the negative prompt the worker encoded for its
/// guidance branches, whose length sizes the alternative prefix's KV.
fn warmup_request(
    config: &EngineConfig,
    request_id: uniserve_core::RequestId,
    constraint: GenerationConstraint,
    tokens: usize,
    image: ImageParams,
    negative_token_ids: Vec<u32>,
) -> GenerationRequest {
    GenerationRequest {
        request_id,
        prompt_token_ids: vec![config.bos],
        negative_prompt_token_ids: negative_token_ids,
        multimodal_inputs: MultimodalInputs::default(),
        constraint,
        sampling: SamplingParams {
            temperature: 0.0,
            ignore_eos: true,
            // A canvas request needs a seed; the rest ignore it.
            seed: Some(0),
            ..SamplingParams::default()
        },
        image,
        max_images: 1,
        max_und_tokens: tokens,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: CachePolicy {
            read: false,
            write: false,
            isolation_key: None,
        },
        include_stop_token: false,
        image_generation: ImageGenerationConfig {
            trigger: ImageTrigger::Disabled,
            requires_text_for_image: false,
            feedback_source: None,
            feedback_encoders: Vec::new(),
            num_feedback_positions: 0,
            feedback_next_token: FeedbackNextToken::None,
            sample_feedback_continuation: false,
        },
        readout: Vec::new(),
    }
}

/// Submits `requests` together and waits for each to finish.
///
/// The requests reach the scheduler in one command, so it queues all of them
/// before it forms its next batch, as it would concurrent clients that all
/// arrived before that batch.
fn run(core: &EngineCore, group: &str, requests: Vec<GenerationRequest>) -> anyhow::Result<()> {
    let mut streams = core
        .submit_all(requests.into_iter().map(Request::from).collect())
        .map_err(|error| anyhow::anyhow!("{group} warmup was not accepted: {error}"))?;

    for stream in &mut streams {
        await_finish(core, group, stream)?;
    }
    Ok(())
}

/// Reads one request's events until its terminal event.
fn await_finish(core: &EngineCore, group: &str, stream: &mut EventRx) -> anyhow::Result<()> {
    loop {
        match stream.try_recv() {
            Ok(EngineCoreOutput::Finished { reason, .. }) => {
                anyhow::ensure!(
                    !matches!(reason, FinishReason::Error | FinishReason::Aborted),
                    "{group} warmup finished with {reason:?}"
                );
                return Ok(());
            }
            Ok(EngineCoreOutput::Rejected { message, .. }) => {
                anyhow::bail!("{group} warmup was rejected: {message}")
            }
            Ok(EngineCoreOutput::Error { message }) => {
                anyhow::bail!("{group} warmup failed: {message}")
            }
            Ok(_) => {}
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                anyhow::ensure!(!core.is_dead(), "the engine died during {group} warmup");
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                anyhow::bail!("the {group} warmup stream closed before its request finished")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{BatchEvent, SimEngine, SimExecutor};
    use uniserve_worker_ipc::ImageWarmup;

    /// Runs `warm` on a simulated worker reported to run on a CUDA device and
    /// returns the kind of every batch the executor accepted with its call
    /// count.
    fn warmed_batches(
        sim: SimEngine,
        canvas: Option<uniserve_core::CanvasSampling>,
    ) -> Vec<(CallKind, usize)> {
        warmed_batches_within(sim, canvas, |_| {})
    }

    /// As `warmed_batches`, with `limits` adjusting the deployment's
    /// generation limits.
    fn warmed_batches_within(
        sim: SimEngine,
        canvas: Option<uniserve_core::CanvasSampling>,
        limits: impl FnOnce(&mut uniserve_core::GenerationLimits),
    ) -> Vec<(CallKind, usize)> {
        warmed_submissions(sim, canvas, limits)
            .into_iter()
            .filter_map(|batch| {
                Some(batch.requests.first()?.0.kind).map(|kind| (kind, batch.requests.len()))
            })
            .collect()
    }

    /// As `warmed_batches`, returning the batches themselves.
    fn warmed_submissions(
        sim: SimEngine,
        canvas: Option<uniserve_core::CanvasSampling>,
        limits: impl FnOnce(&mut uniserve_core::GenerationLimits),
    ) -> Vec<crate::executor::ExecutionBatch> {
        let mut executor = SimExecutor::new(sim);
        let boundary = executor.observe();
        let mut config = EngineConfig::sim("warmup");
        config.workers[0].ranks[0].device = "cuda:0".to_owned();
        config.worker_process.canvas_sampling = canvas;
        limits(&mut config.generation_limits);
        let core = EngineCore::with_executor(config.clone(), Box::new(executor)).unwrap();

        warm(&core, &config).unwrap();
        core.shutdown();

        boundary
            .try_iter()
            .filter_map(|event| match event {
                BatchEvent::Submitted(batch) => Some(batch),
                BatchEvent::Resolved { .. } => None,
            })
            .collect()
    }

    #[test]
    fn a_token_deployment_runs_a_prefill_and_a_decode() {
        let batches = warmed_batches(SimEngine::new(), None);
        assert!(batches.contains(&(CallKind::Forward(ForwardMode::Prefill), 1)));
        assert!(batches.contains(&(CallKind::Forward(ForwardMode::Decode), 1)));
        assert!(
            !batches
                .iter()
                .any(|(kind, _)| matches!(kind, CallKind::Media(MediaCall::Denoising)))
        );
    }

    #[test]
    fn each_image_shape_runs_both_denoising_steps_and_decodes_its_image() {
        let mut sim = SimEngine::new();
        sim.mut_info_for_test().image_warmups = vec![
            ImageWarmup {
                height: 64,
                width: 64,
                cfg_text_scale: 1.0,
                cfg_img_scale: 1.0,
                negative_token_ids: Vec::new(),
            },
            ImageWarmup {
                height: 64,
                width: 128,
                cfg_text_scale: 4.0,
                cfg_img_scale: 2.0,
                negative_token_ids: Vec::new(),
            },
        ];

        let batches = warmed_batches(sim, None);
        let count = |media_call| {
            batches
                .iter()
                .filter(|(kind, _)| *kind == CallKind::Media(media_call))
                .count()
        };

        // Two steps and one decode per shape.
        assert_eq!(count(MediaCall::Denoising), 4);
        assert_eq!(count(MediaCall::ImageDecoding), 2);
    }

    #[test]
    fn a_guided_shape_carries_the_negative_prompt_the_worker_encoded() {
        let mut sim = SimEngine::new();
        sim.mut_info_for_test().image_warmups = vec![ImageWarmup {
            height: 64,
            width: 64,
            cfg_text_scale: 4.0,
            cfg_img_scale: 1.0,
            negative_token_ids: vec![11, 12, 13],
        }];

        // The worker reads its alternative prefix from the admission, so the
        // request states the tokens the worker encoded for it.
        let admitted = warmed_submissions(sim, None, |_| {})
            .iter()
            .flat_map(|batch| batch.admissions().cloned().collect::<Vec<_>>())
            .filter(|admission| admission.image.is_some())
            .map(|admission| admission.generation.unwrap().negative_token_ids)
            .collect::<Vec<_>>();

        assert_eq!(admitted, vec![vec![11, 12, 13]]);
    }

    #[test]
    fn a_shape_the_limits_refuse_is_not_warmed() {
        let mut sim = SimEngine::new();
        sim.mut_info_for_test().image_warmups = vec![
            ImageWarmup {
                height: 64,
                width: 64,
                cfg_text_scale: 1.0,
                cfg_img_scale: 1.0,
                negative_token_ids: Vec::new(),
            },
            ImageWarmup {
                height: 64,
                width: 64,
                cfg_text_scale: 4.0,
                cfg_img_scale: 2.0,
                negative_token_ids: Vec::new(),
            },
        ];

        // Two guidance branches at most: the three-branch shape cannot be
        // submitted by a client either.
        let batches = warmed_batches_within(sim, None, |limits| limits.max_cfg_branches = 2);
        let steps = batches
            .iter()
            .filter(|(kind, _)| *kind == CallKind::Media(MediaCall::Denoising))
            .count();
        assert_eq!(steps, 2);
    }

    #[test]
    fn a_canvas_chunk_steps_at_every_row_count() {
        let mut sim = SimEngine::new();
        sim.set_queue_depth(8);
        sim.mut_info_for_test().canvas_step_rows = 3;
        let canvas = uniserve_core::CanvasSampling {
            canvas_length: 4,
            max_steps: 2,
            entropy_bound: 0.0,
            t_min: 0.0,
            t_max: 0.0,
            confidence_threshold: 0.0,
            stability_threshold: 0,
        };
        sim.set_canvas_sampling(canvas);

        let batches = warmed_batches(sim, Some(canvas));
        let step = CallKind::Forward(ForwardMode::TokenDenoising);
        let steps = batches
            .iter()
            .filter(|(kind, _)| *kind == step)
            .map(|(_, rows)| *rows)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(steps, std::collections::BTreeSet::from([1, 2, 3]));

        // A block a canvas step produced is committed by a prefill.
        let first_step = batches.iter().position(|(kind, _)| *kind == step).unwrap();
        assert!(
            batches[first_step..]
                .iter()
                .any(|(kind, _)| *kind == CallKind::Forward(ForwardMode::Prefill))
        );
    }
}
