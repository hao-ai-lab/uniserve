//! Folds the runtime events of one video generation into its artifact.
//!
//! Every consumer of a video stream (the synchronous route, the job task and
//! the Dynamo worker) waits for the same outcome: exactly one artifact
//! followed by a normal finish. [`ArtifactCollector`] classifies the events
//! once; each consumer keeps its own read loop, with its own cancellation and
//! progress handling, and renders a [`VideoStreamError`] in its own error
//! envelope.

use std::ops::ControlFlow;

use uniserve_core::{ArtifactEvent, RejectionKind};

use crate::serving::{FinishStatus, RequestOutput};

/// Why a video generation stream did not end in an artifact.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoStreamError {
    /// The engine refused the request before executing it.
    Rejected {
        /// Whether the request is unservable or the engine is full.
        kind: RejectionKind,
        /// Caller-visible rejection description.
        message: String,
    },
    /// Generation failed.
    Failed {
        /// Description of the failure.
        message: String,
    },
    /// The request finished without completing normally.
    Ended {
        /// How the request finished.
        reason: FinishStatus,
    },
    /// The request was cancelled or aborted.
    Cancelled,
    /// The request completed normally but produced no artifact.
    MissingArtifact,
    /// The runtime emitted a second artifact.
    MultipleArtifacts,
    /// The runtime emitted an event a video generation never carries.
    Incompatible,
    /// The stream ended before the request finished.
    Closed,
}

/// Holds the artifact of a video generation while its events arrive.
#[derive(Debug, Default)]
pub struct ArtifactCollector {
    artifact: Option<ArtifactEvent>,
}

impl ArtifactCollector {
    /// Folds one stream event, `None` for the end of the stream.
    ///
    /// Acceptance, usage, scheduling and progress events carry nothing the
    /// outcome needs and continue. The first terminal event breaks with the
    /// artifact of a normal finish, or the error that ended the request.
    pub fn push(
        &mut self,
        event: Option<RequestOutput>,
    ) -> ControlFlow<Result<ArtifactEvent, VideoStreamError>> {
        let outcome = match event {
            Some(RequestOutput::Artifact(value)) => {
                if self.artifact.replace(value).is_some() {
                    Err(VideoStreamError::MultipleArtifacts)
                } else {
                    return ControlFlow::Continue(());
                }
            }
            Some(RequestOutput::Finished {
                reason: FinishStatus::Stop { .. },
                ..
            }) => self
                .artifact
                .take()
                .ok_or(VideoStreamError::MissingArtifact),
            Some(RequestOutput::Finished { reason, .. }) => Err(VideoStreamError::Ended { reason }),
            Some(RequestOutput::Rejected { kind, message, .. }) => {
                Err(VideoStreamError::Rejected { kind, message })
            }
            Some(RequestOutput::Failed { message, .. }) => {
                Err(VideoStreamError::Failed { message })
            }
            Some(RequestOutput::Cancelled { .. } | RequestOutput::Aborted { .. }) => {
                Err(VideoStreamError::Cancelled)
            }
            Some(
                RequestOutput::Accepted { .. }
                | RequestOutput::Usage { .. }
                | RequestOutput::Scheduled { .. }
                | RequestOutput::MediaProgress { .. },
            ) => return ControlFlow::Continue(()),
            Some(_) => Err(VideoStreamError::Incompatible),
            None => Err(VideoStreamError::Closed),
        };
        ControlFlow::Break(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::ServeRequestId;

    fn id() -> ServeRequestId {
        ServeRequestId::new("video")
    }

    fn outcome(event: Option<RequestOutput>) -> Result<ArtifactEvent, VideoStreamError> {
        match ArtifactCollector::default().push(event) {
            ControlFlow::Break(outcome) => outcome,
            ControlFlow::Continue(()) => panic!("the event does not end the stream"),
        }
    }

    #[test]
    fn lifecycle_and_progress_events_continue() {
        let mut collector = ArtifactCollector::default();
        for event in [
            RequestOutput::Scheduled {
                request_id: id(),
                queued_at: None,
                scheduled_at: None,
            },
            RequestOutput::MediaProgress {
                phase: "denoising".to_owned(),
                completed_steps: 1,
            },
        ] {
            assert!(collector.push(Some(event)).is_continue());
        }
    }

    #[test]
    fn a_normal_finish_without_an_artifact_is_missing_one() {
        let finished = RequestOutput::Finished {
            reason: FinishStatus::Stop { cause: None },
            finish_detail: None,
        };
        assert_eq!(
            outcome(Some(finished)).unwrap_err(),
            VideoStreamError::MissingArtifact
        );
    }

    #[test]
    fn terminal_events_name_why_the_request_ended() {
        let finished = RequestOutput::Finished {
            reason: FinishStatus::Length,
            finish_detail: None,
        };
        assert_eq!(
            outcome(Some(finished)).unwrap_err(),
            VideoStreamError::Ended {
                reason: FinishStatus::Length
            }
        );

        let rejected = RequestOutput::Rejected {
            request_id: id(),
            kind: RejectionKind::Overloaded,
            message: "full".to_owned(),
        };
        assert_eq!(
            outcome(Some(rejected)).unwrap_err(),
            VideoStreamError::Rejected {
                kind: RejectionKind::Overloaded,
                message: "full".to_owned()
            }
        );

        let failed = RequestOutput::Failed {
            request_id: id(),
            message: "boom".to_owned(),
        };
        assert_eq!(
            outcome(Some(failed)).unwrap_err(),
            VideoStreamError::Failed {
                message: "boom".to_owned()
            }
        );

        let cancelled = RequestOutput::Cancelled { request_id: id() };
        assert_eq!(
            outcome(Some(cancelled)).unwrap_err(),
            VideoStreamError::Cancelled
        );
        let aborted = RequestOutput::Aborted { request_id: id() };
        assert_eq!(
            outcome(Some(aborted)).unwrap_err(),
            VideoStreamError::Cancelled
        );
    }

    #[test]
    fn a_stream_that_ends_early_or_carries_text_is_refused() {
        assert_eq!(outcome(None).unwrap_err(), VideoStreamError::Closed);
        let text = RequestOutput::TextDelta {
            text: "hello".to_owned(),
            token_ids: Vec::new(),
            logprobs: None,
        };
        assert_eq!(
            outcome(Some(text)).unwrap_err(),
            VideoStreamError::Incompatible
        );
    }
}
