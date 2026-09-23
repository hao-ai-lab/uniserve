//! Bounded process-local video jobs and immutable retained artifacts.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use uniserve_core::SharedMedia;

pub(crate) const MAX_VIDEO_JOBS: usize = 128;
pub(crate) const MAX_VIDEO_BYTES: usize = 1024 * 1024 * 1024;
pub(crate) const VIDEO_RETENTION: Duration = Duration::from_secs(3600);

#[derive(Clone, Serialize)]
pub(crate) struct VideoJob {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub expires_at: Option<u64>,
    pub seconds: f64,
    pub actual_seconds: f64,
    pub status: &'static str,
    pub phase: String,
    pub completed_steps: u32,
    pub total_steps: u32,
    pub error: Option<VideoFailure>,
}

#[derive(Clone, Serialize)]
pub(crate) struct VideoFailure {
    pub code: &'static str,
    pub message: String,
}

struct RetainedJob {
    record: VideoJob,
    media: Option<Arc<RetainedMedia>>,
    cancellation: Arc<JobReservation>,
}

/// A job slot acquired before submission and not yet bound to a record.
///
/// Dropping it releases the slot, so a caller that abandons a request between
/// admission and job creation leaves no record behind.
pub(crate) struct JobSlot(OwnedSemaphorePermit);

/// A slot remains reserved through deletion until its physical execution drains.
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

pub(crate) struct VideoJobs {
    entries: Mutex<BTreeMap<String, RetainedJob>>,
    bytes: Arc<Semaphore>,
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
    pub(crate) fn reserve(&self) -> Result<JobSlot, &'static str> {
        Arc::clone(&self.slots)
            .try_acquire_owned()
            .map(JobSlot)
            .map_err(|_| {
                "video job limit reached; delete retained jobs or wait for cancellations to drain"
            })
    }

    /// Binds a reserved slot to its record before detaching generation from its HTTP caller.
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

    pub(crate) fn list(&self) -> Vec<VideoJob> {
        let entries = self.entries();
        let mut records: Vec<_> = entries.values().map(|job| job.record.clone()).collect();
        records.sort_by(|a, b| (b.created_at, &b.id).cmp(&(a.created_at, &a.id)));
        records
    }

    pub(crate) fn progress(&self, id: &str, phase: &str, steps: u32) {
        if let Some(job) = self.entries().get_mut(id) {
            job.record.status = "in_progress";
            job.record.phase = phase.to_owned();
            job.record.completed_steps = steps;
        }
    }

    /// Publish completion atomically with ownership of the one generated artifact.
    pub(crate) fn finish(
        self: &Arc<Self>,
        id: &str,
        result: Result<Arc<SharedMedia>, VideoFailure>,
    ) {
        let mut entries = self.entries();
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

    pub(crate) fn content(&self, id: &str) -> Option<Arc<RetainedMedia>> {
        self.entries().get(id).and_then(|job| job.media.clone())
    }

    /// Removal wins over a racing completion; active work drains through runtime cancellation.
    pub(crate) fn delete(&self, id: &str) -> bool {
        if let Some(job) = self.entries().remove(id) {
            job.cancellation.cancel();
            true
        } else {
            false
        }
    }

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
        jobs.finish(
            "video-a",
            Err(VideoFailure {
                code: "cancelled",
                message: "cancelled".into(),
            }),
        );
        assert!(jobs.get("video-a").is_none());
        assert!(jobs.list().is_empty());
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
