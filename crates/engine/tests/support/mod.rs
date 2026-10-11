//! Scheduler construction shared by the engine's integration tests.
//!
//! Serving states a deployment's model limits from the model's description;
//! these tests have no model, so a simulated worker's limits follow from the
//! call kinds it serves. Each test binary uses a subset of these functions.

#![allow(dead_code)]

use uniserve_core::{GenerationFeatures, GenerationLimits, ModelDtype};
use uniserve_engine::{
    Executor, Scheduler, SchedulerConfig, SchedulingPolicy, SimEngine, SpecialTokenIds,
};
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

/// Builds a scheduler whose model limits follow from `executor`'s call kinds.
///
/// A token-denoising worker without text decode serves canvas readouts over
/// prompts that may carry encoded images; one that prepares latents without
/// decoding text serves video diffusion, which needs no generation feature.
/// Any other denoising, vision-encoding or latent-encoding worker serves
/// interleaved text and images without bound, and the rest serve text.
pub(crate) fn scheduler_with(
    executor: Box<dyn Executor>,
    ctrl: SpecialTokenIds,
    config: SchedulerConfig,
) -> anyhow::Result<Scheduler> {
    let info = executor.info().runtime_info()?;
    let calls = &info.supported_calls;
    let decodes = calls.contains(&CallKind::Forward(ForwardMode::Decode));
    let limits = if calls.contains(&CallKind::Forward(ForwardMode::TokenDenoising)) && !decodes {
        GenerationLimits {
            features: GenerationFeatures::TOKEN_DENOISING | GenerationFeatures::VISION_ENCODE,
            latent_downsample: 1,
            max_vit_grid_tokens: u32::MAX,
            max_vision_feature_bytes: 256 << 20,
            max_cfg_branches: 1,
            encoder_cache_entries: 256,
            ..Default::default()
        }
    } else if calls.contains(&CallKind::Media(MediaCall::LatentPreparation)) && !decodes {
        GenerationLimits {
            latent_downsample: 1,
            max_cfg_branches: 1,
            ..Default::default()
        }
    } else if calls.contains(&CallKind::Media(MediaCall::Denoising))
        || calls.contains(&CallKind::Media(MediaCall::VisionEncoding))
        || calls.contains(&CallKind::Media(MediaCall::LatentEncoding))
    {
        SimEngine::generation_limits()
    } else {
        GenerationLimits {
            features: GenerationFeatures::UNDERSTANDING,
            latent_downsample: 1,
            max_cfg_branches: 1,
            ..Default::default()
        }
    };
    Scheduler::new(executor, ctrl, config, ModelDtype::BFloat16, limits)
}

/// Builds a scheduler that batches at most `max_batch` calls per pass.
pub(crate) fn scheduler(
    executor: Box<dyn Executor>,
    ctrl: SpecialTokenIds,
    max_batch: usize,
) -> anyhow::Result<Scheduler> {
    scheduler_with(
        executor,
        ctrl,
        SchedulerConfig {
            max_batch,
            ..Default::default()
        },
    )
}

/// Builds a scheduler that batches at most `max_batch` calls per pass and
/// orders waiting requests by `policy`.
pub(crate) fn scheduler_with_policy(
    executor: Box<dyn Executor>,
    ctrl: SpecialTokenIds,
    max_batch: usize,
    policy: SchedulingPolicy,
) -> anyhow::Result<Scheduler> {
    scheduler_with(
        executor,
        ctrl,
        SchedulerConfig {
            max_batch,
            policy,
            ..Default::default()
        },
    )
}
