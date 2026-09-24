//! Bounded process-local video jobs and immutable retained artifacts.
//!
//! `VideoJobs` is the job store behind the asynchronous video routes in
//! `http::routes::openai::videos`. It holds each job's public record, its
//! generated artifact once complete, and the cancellation token of its
//! detached generation task. Nothing persists across a restart.
//!
//! Three bounds apply, each advertised by the video `capabilities` route:
//!
//! - `MAX_VIDEO_JOBS` job slots. A slot is taken by `VideoJobs::reserve` before
//!   submission and stays taken while either the record or its generation task
//!   exists.
//! - `MAX_VIDEO_BYTES` of artifact bytes, counting artifacts retained by a job
//!   and artifacts held by an unfinished download.
//! - `VIDEO_RETENTION`, after which a completed or failed job's record is
//!   deleted.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use uniserve_core::SharedMedia;

/// Maximum number of job slots held at once.
pub(crate) const MAX_VIDEO_JOBS: usize = 128;
/// Retained-artifact byte budget.
///
/// The `artifact_capacity_exceeded` message in `VideoJobs::finish` states this
/// limit as "1 GiB"; the two must change together.
pub(crate) const MAX_VIDEO_BYTES: usize = 1024 * 1024 * 1024;
/// How long a completed or failed job is retained before it is deleted.
pub(crate) const VIDEO_RETENTION: Duration = Duration::from_secs(3600);

/// Public job record, serialized in the responses of the video job routes.
///
/// Timestamps are Unix seconds from `timestamp`. `status` is `queued` when
/// the video routes create the record, `in_progress` after
/// `VideoJobs::progress`, and `completed` or `failed` after
/// `VideoJobs::finish`.
#[derive(Clone, Serialize)]
pub(crate) struct VideoJob {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    /// Set with `completed_at`, to `VIDEO_RETENTION` after it.
    pub expires_at: Option<u64>,
    /// Requested duration in seconds, or the model default when the request
    /// omits it.
    pub seconds: f64,
    /// Duration in seconds of the frames the model generates.
    pub actual_seconds: f64,
    pub status: &'static str,
    /// `queued`, then the phases the generation task reports from runtime
    /// events, then `completed` or `failed`.
    pub phase: String,
    /// Completed inference steps; set to `total_steps` on success.
    pub completed_steps: u32,
    pub total_steps: u32,
    pub error: Option<VideoFailure>,
}

/// Failure reported on a `failed` job.
///
/// Codes are assigned by the generation task in the video routes, and by
/// `VideoJobs::finish` (`artifact_capacity_exceeded`).
#[derive(Clone, Serialize)]
pub(crate) struct VideoFailure {
    pub code: &'static str,
    pub message: String,
}

/// One entry of the job table.
struct RetainedJob {
    record: VideoJob,
    /// The generated artifact; set only on successful completion.
    media: Option<Arc<RetainedMedia>>,
    /// Shared with the job's generation task.
    cancellation: Arc<JobReservation>,
}

/// A job slot acquired before submission and not yet bound to a record.
///
/// Dropping it releases the slot, so a caller that abandons a request between
/// admission and job creation leaves no record behind.
pub(crate) struct JobSlot(OwnedSemaphorePermit);

/// A job's cancellation token together with its job slot.
///
/// `VideoJobs::insert` keeps one `Arc` in the record and returns another,
/// which the video routes move into the generation task, so the slot is
/// released only once the record is removed and the task has ended. A deleted
/// job therefore keeps its slot until its execution drains.
pub(crate) struct JobReservation {
    cancellation: CancellationToken,
    _slot: OwnedSemaphorePermit,
}

impl std::ops::Deref for JobReservation {
    type Target = CancellationToken;
    fn deref(&self) -> &Self::Target {
        &self.cancellation
    }
}

/// A completed artifact holding its share of the retained-byte budget.
///
/// The download route moves an `Arc` of it into the response body, so the
/// bytes stay counted against `MAX_VIDEO_BYTES` until the download ends, even
/// if the job is deleted or expires meanwhile.
pub(crate) struct RetainedMedia {
    media: Arc<SharedMedia>,
    _bytes: OwnedSemaphorePermit,
}

impl std::ops::Deref for RetainedMedia {
    type Target = SharedMedia;
    fn deref(&self) -> &Self::Target {
        &self.media
    }
}

impl AsRef<[u8]> for RetainedMedia {
    fn as_ref(&self) -> &[u8] {
        self.media.as_bytes()
    }
}

/// Process-local job table and its capacity budgets.
pub(crate) struct VideoJobs {
    /// Jobs keyed by public job ID.
    entries: Mutex<BTreeMap<String, RetainedJob>>,
    /// One permit per retained artifact byte, up to `MAX_VIDEO_BYTES`.
    bytes: Arc<Semaphore>,
    /// One permit per job slot, up to `MAX_VIDEO_JOBS`.
    slots: Arc<Semaphore>,
}

impl Default for VideoJobs {
    fn default() -> Self {
        Self {
            entries: Mutex::default(),
            bytes: Arc::new(Semaphore::new(MAX_VIDEO_BYTES)),
            slots: Arc::new(Semaphore::new(MAX_VIDEO_JOBS)),
        }
    }
}

/// Current Unix time in whole seconds, or `0` if the clock is before the epoch.
pub(crate) fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl VideoJobs {
    /// Locks the job table and recovers it after poisoning.
    fn entries(&self) -> MutexGuard<'_, BTreeMap<String, RetainedJob>> {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Acquires one bounded job slot before the request reaches the engine.
    ///
    /// Does not wait: returns an error message when every slot is taken.
    pub(crate) fn reserve(&self) -> Result<JobSlot, &'static str> {
        Arc::clone(&self.slots)
            .try_acquire_owned()
            .map(JobSlot)
            .map_err(|_| {
                "video job limit reached; delete retained jobs or wait for cancellations to drain"
            })
    }

    /// Binds a reserved slot to its record before detaching generation from its HTTP caller.
    ///
    /// Returns the reservation the generation task must hold until it ends.
    /// Fails without inserting when `record.id` is already present; the slot is
    /// then dropped and released.
    pub(crate) fn insert(
        &self,
        record: VideoJob,
        slot: JobSlot,
    ) -> Result<Arc<JobReservation>, &'static str> {
        let mut entries = self.entries();
        if entries.contains_key(&record.id) {
            return Err("video job already exists");
        }
        let cancellation = Arc::new(JobReservation {
            cancellation: CancellationToken::new(),
            _slot: slot.0,
        });
        entries.insert(
            record.id.clone(),
            RetainedJob {
                record,
                media: None,
                cancellation: Arc::clone(&cancellation),
            },
        );
        Ok(cancellation)
    }

    pub(crate) fn get(&self, id: &str) -> Option<VideoJob> {
        self.entries().get(id).map(|job| job.record.clone())
    }

    /// Returns a snapshot of every retained record, newest first, with ties on
    /// `created_at` ordered by descending ID.
    pub(crate) fn list(&self) -> Vec<VideoJob> {
        let entries = self.entries();
        let mut records: Vec<_> = entries.values().map(|job| job.record.clone()).collect();
        records.sort_by(|a, b| (b.created_at, &b.id).cmp(&(a.created_at, &a.id)));
        records
    }

    /// Marks a job `in_progress` at `phase` with `steps` completed steps.
    ///
    /// An unknown or deleted ID is ignored.
    pub(crate) fn progress(&self, id: &str, phase: &str, steps: u32) {
        if let Some(job) = self.entries().get_mut(id) {
            job.record.status = "in_progress";
            job.record.phase = phase.to_owned();
            job.record.completed_steps = steps;
        }
    }

    /// Publish completion atomically with ownership of the one generated artifact.
    ///
    /// A successful result is retained only if its bytes fit the remaining
    /// retained-byte budget; otherwise the job fails with
    /// `artifact_capacity_exceeded`. Either way the record gets `completed_at`
    /// and `expires_at`, and a timer task deletes it after `VIDEO_RETENTION`
    /// unless its token is cancelled first. If the job was already deleted,
    /// the result and any budget it took are dropped and no timer starts.
    ///
    /// Must run inside a Tokio runtime, because it spawns the timer task.
    pub(crate) fn finish(
        self: &Arc<Self>,
        id: &str,
        result: Result<Arc<SharedMedia>, VideoFailure>,
    ) {
        let mut entries = self.entries();

        // `try_acquire_many_owned` takes a `u32` count and does not wait, so an
        // artifact larger than `u32::MAX` bytes and an exhausted budget both
        // fail the job.
        let result = result.and_then(|media| {
            let bytes = u32::try_from(media.len())
                .ok()
                .and_then(|bytes| Arc::clone(&self.bytes).try_acquire_many_owned(bytes).ok());
            match bytes {
                Some(permit) => Ok(Arc::new(RetainedMedia {
                    media,
                    _bytes: permit,
                })),
                None => Err(VideoFailure {
                    code: "artifact_capacity_exceeded",
                    message: "retained videos and active downloads exceed the 1 GiB limit"
                        .to_owned(),
                }),
            }
        });
        let Some(job) = entries.get_mut(id) else {
            return;
        };

        let now = timestamp();
        job.record.completed_at = Some(now);
        job.record.expires_at = Some(now + VIDEO_RETENTION.as_secs());
        match result {
            Ok(media) => {
                job.record.status = "completed";
                job.record.completed_steps = job.record.total_steps;
                job.record.phase = "completed".to_owned();
                job.media = Some(media);
            }
            Err(error) => {
                job.record.status = "failed";
                job.record.phase = "failed".to_owned();
                job.record.error = Some(error);
            }
        }

        // The expiry timer holds only a clone of the token and a `Weak` handle,
        // so a pending timer neither keeps the job slot nor keeps `VideoJobs`
        // alive. `delete` and `cancel_all` cancel the token, which ends it.
        let expiry_cancellation = job.cancellation.cancellation.clone();
        let weak = Arc::downgrade(self);
        let id = id.to_owned();
        tokio::spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep(VIDEO_RETENTION) => {}
                _ = expiry_cancellation.cancelled() => return,
            }
            if let Some(jobs) = weak.upgrade() {
                jobs.delete(&id);
            }
        });
    }

    /// Returns the artifact of a successfully completed job.
    pub(crate) fn content(&self, id: &str) -> Option<Arc<RetainedMedia>> {
        self.entries().get(id).and_then(|job| job.media.clone())
    }

    /// Removal wins over a racing completion; active work drains through runtime cancellation.
    ///
    /// Removes the record and cancels its token, which makes a running
    /// generation task cancel its runtime request and ends a pending expiry
    /// timer. Returns `false` for an unknown ID.
    pub(crate) fn delete(&self, id: &str) -> bool {
        if let Some(job) = self.entries().remove(id) {
            job.cancellation.cancel();
            true
        } else {
            false
        }
    }

    /// Cancels every job's token without removing any record.
    ///
    /// Running generation tasks cancel their runtime requests, drain, and
    /// publish their results; the expiry timers of finished jobs end.
    /// `AppState::shutdown` calls this.
    pub(crate) fn cancel_all(&self) {
        for job in self.entries().values() {
            job.cancellation.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str) -> VideoJob {
        VideoJob {
            id: id.to_owned(),
            object: "video",
            model: "FastH3".to_owned(),
            created_at: timestamp(),
            completed_at: None,
            expires_at: None,
            seconds: 5.0,
            actual_seconds: 124.0 / 24.0,
            status: "queued",
            phase: "queued".to_owned(),
            completed_steps: 0,
            total_steps: 4,
            error: None,
        }
    }

    fn insert(jobs: &VideoJobs, id: &str) -> Result<Arc<JobReservation>, &'static str> {
        jobs.insert(record(id), jobs.reserve()?)
    }

    #[tokio::test]
    async fn an_abandoned_reservation_releases_its_slot_without_a_record() {
        let jobs = Arc::new(VideoJobs::default());
        let slots = (0..MAX_VIDEO_JOBS)
            .map(|_| jobs.reserve().unwrap())
            .collect::<Vec<_>>();
        assert!(jobs.reserve().is_err());
        drop(slots);
        assert!(jobs.reserve().is_ok());
        assert!(jobs.list().is_empty());
    }

    #[tokio::test]
    async fn deleted_work_is_cancelled_and_cannot_reappear() {
        // `cancellation` stands in for the generation task's reservation.
        let jobs = Arc::new(VideoJobs::default());
        let cancellation = insert(&jobs, "video-a").unwrap();
        jobs.progress("video-a", "denoising", 2);
        let running = jobs.get("video-a").unwrap();
        assert_eq!(
            (running.status, running.completed_steps),
            ("in_progress", 2)
        );

        assert!(jobs.delete("video-a"));
        assert!(cancellation.is_cancelled());

        // A completion racing the deletion must not recreate the record.
        jobs.finish(
            "video-a",
            Err(VideoFailure {
                code: "cancelled",
                message: "cancelled".into(),
            }),
        );
        assert!(jobs.get("video-a").is_none());
        assert!(jobs.list().is_empty());

        // The deleted job's slot stays taken while its task still holds the
        // reservation, so only `MAX_VIDEO_JOBS - 1` more jobs fit.
        for index in 0..127 {
            insert(&jobs, &format!("video-{index}")).unwrap();
        }
        assert!(insert(&jobs, "video-next").is_err());
        drop(cancellation);
        assert!(insert(&jobs, "video-next").is_ok());
    }

    #[tokio::test]
    async fn failures_have_retention_and_deletion_releases_record_capacity() {
        let jobs = Arc::new(VideoJobs::default());
        // Each returned reservation is dropped at once, so every slot is held
        // by its record alone.
        for index in 0..128 {
            insert(&jobs, &format!("video-{index}")).unwrap();
        }
        assert!(insert(&jobs, "video-overflow").is_err());

        jobs.finish(
            "video-0",
            Err(VideoFailure {
                code: "generation_failed",
                message: "decoder failed".into(),
            }),
        );
        let failed = jobs.get("video-0").unwrap();
        assert_eq!(failed.status, "failed");
        assert_eq!(failed.error.unwrap().code, "generation_failed");
        assert_eq!(
            failed.expires_at.unwrap() - failed.completed_at.unwrap(),
            3600
        );

        assert!(jobs.delete("video-0"));
        assert!(insert(&jobs, "video-next").is_ok());
        assert_eq!(jobs.list().len(), 128);
    }
}
