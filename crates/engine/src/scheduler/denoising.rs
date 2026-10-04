//! Solver progress of one request's latent trajectory.
//!
//! Image generation inside a text request and the video pipeline keep their
//! own lifecycles, but both advance a trajectory the same way: latent
//! preparation opens it at step zero, denoising calls cover consecutive
//! intervals of its fixed schedule, and the worker reports the steps it has
//! accepted. [`Denoising`] owns that progress for both.

use uniserve_worker_ipc::{CallId, LatentParams, MediaCall, RequestKey, TensorRef};

/// Where a request's latent lives on its worker and the raster it describes.
///
/// Denoising progress carries it into every call's parameters without
/// interpreting it: the pages are the worker's physical storage and the raster
/// is the media the samples decode to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LatentPlacement {
    /// Pages the request owns in its worker's latent pool.
    pub(crate) page_table: Vec<u32>,
    /// Latent units the trajectory occupies in those pages, in the
    /// model-defined unit the worker sizes pages by (`latent_page_units`).
    pub(crate) latent_units: u32,
    /// Raster height in pixels: the requested image height for image
    /// generation, the request's canvas height for video.
    pub(crate) height: u32,
    /// Raster width, from the same source as `height`.
    pub(crate) width: u32,
}

/// Solver progress of one latent trajectory.
///
/// Only accepted progress is stored. The steps already submitted are derived
/// from the request's pending calls, so a call that fails or is invalidated
/// needs no counter repair.
#[derive(Debug, Clone)]
pub(crate) struct Denoising {
    /// Steps in the trajectory's fixed schedule.
    steps: u32,
    /// Steps the worker has accepted.
    completed: u32,
    /// The latest accepted latent product, when the worker publishes one per
    /// call.
    latent: Option<TensorRef>,
}

impl Denoising {
    /// Returns the progress of a trajectory of `steps` solver steps.
    pub(crate) const fn new(steps: u32) -> Self {
        Self {
            steps,
            completed: 0,
            latent: None,
        }
    }

    /// Returns the length of the trajectory's schedule.
    pub(crate) const fn steps(&self) -> u32 {
        self.steps
    }

    /// Returns the steps the worker has accepted.
    pub(crate) const fn completed(&self) -> u32 {
        self.completed
    }

    /// Returns whether every step has been accepted.
    pub(crate) const fn is_complete(&self) -> bool {
        self.completed >= self.steps
    }

    /// Returns the latest accepted latent product.
    pub(crate) const fn latent(&self) -> Option<&TensorRef> {
        self.latent.as_ref()
    }

    /// Returns the steps covered once the request's submitted calls complete.
    ///
    /// `pending` lists the request's submitted calls in submission order with
    /// their latent parameters: latent preparation opens a trajectory and
    /// image decoding closes one, both leaving it at step zero, and each
    /// denoising call adds its interval.
    pub(crate) fn scheduled<'a>(
        &self,
        pending: impl IntoIterator<Item = (MediaCall, Option<&'a LatentParams>)>,
    ) -> u32 {
        pending
            .into_iter()
            .fold(self.completed, |steps, (call, latent)| match call {
                MediaCall::LatentPreparation | MediaCall::ImageDecoding => 0,
                MediaCall::Denoising => {
                    steps.saturating_add(latent.map_or(0, |params| params.step_count))
                }
                _ => steps,
            })
    }

    /// Returns the next interval, as its start and step count, of at most
    /// `burst` steps once `scheduled` steps are submitted, or `None` when
    /// every step is.
    ///
    /// A `burst` of zero still yields one-step intervals.
    pub(crate) fn next(&self, scheduled: u32, burst: u32) -> Option<(u32, u32)> {
        let remaining = self.steps.saturating_sub(scheduled);
        (remaining > 0).then(|| (scheduled, burst.max(1).min(remaining)))
    }

    /// Returns whether an interval ends the trajectory.
    pub(crate) const fn ends(&self, start: u32, count: u32) -> bool {
        start.saturating_add(count) == self.steps
    }

    /// Returns whether a worker's accepted step count completes an interval.
    pub(crate) const fn completes(interval: &LatentParams, completed: u32) -> bool {
        interval.start_step.saturating_add(interval.step_count) == completed
    }

    /// Returns the parameters of one call on this trajectory.
    ///
    /// Preparation and decoding calls cover no steps; a denoising call covers
    /// `count` steps from `start`.
    pub(crate) fn params(
        request_key: RequestKey,
        call_id: CallId,
        placement: &LatentPlacement,
        start: u32,
        count: u32,
    ) -> LatentParams {
        LatentParams {
            request_key,
            call_id,
            page_table: placement.page_table.clone(),
            latent_units: placement.latent_units,
            height: placement.height,
            width: placement.width,
            start_step: start,
            step_count: count,
        }
    }

    /// Opens a trajectory at step zero with its prepared latent.
    pub(crate) fn open(&mut self, latent: Option<TensorRef>) {
        self.completed = 0;
        self.latent = latent;
    }

    /// Accepts a completed denoising call and its successor latent.
    ///
    /// Returns `false`, leaving progress unchanged, when the worker's accepted
    /// step count is not the end of the call's interval or exceeds the
    /// schedule. A `None` latent keeps the previously accepted product.
    pub(crate) fn accept(
        &mut self,
        interval: &LatentParams,
        completed: u32,
        latent: Option<TensorRef>,
    ) -> bool {
        if !Self::completes(interval, completed) || completed > self.steps {
            return false;
        }
        self.completed = completed;
        if latent.is_some() {
            self.latent = latent;
        }
        true
    }

    /// Closes the trajectory once its latent is released; the next one starts
    /// at step zero.
    pub(crate) fn close(&mut self) {
        self.completed = 0;
        self.latent = None;
    }
}
