use std::collections::BTreeMap;

use tokio::sync::mpsc;
use uniserve_core::MediaEvent;
use uniserve_core::{RequestId, now_unix_secs};

/// Frontend-owned metadata and the compact fixed-profile media input.
#[derive(Debug, Clone)]
pub struct MediaSubmission {
    pub external_request_id: String,
    pub prompt: String,
    pub seed: u64,
    pub priority: i32,
    pub output_path: String,
    pub arrival_time: Option<f64>,
    pub data_parallel_rank: Option<u32>,
    pub trace_headers: Option<BTreeMap<String, String>>,
}

impl MediaSubmission {
    pub fn new(
        external_request_id: impl Into<String>,
        prompt: impl Into<String>,
        seed: u64,
        output_path: impl Into<String>,
    ) -> Self {
        Self {
            external_request_id: external_request_id.into(),
            prompt: prompt.into(),
            seed,
            priority: 0,
            output_path: output_path.into(),
            arrival_time: None,
            data_parallel_rank: None,
            trace_headers: None,
        }
    }

    pub(crate) fn into_envelope(
        self,
        client_index: u32,
    ) -> uniserve_core::codec::MediaRequestEnvelope {
        uniserve_core::codec::MediaRequestEnvelope {
            external_request_id: self.external_request_id,
            arrival_time: self.arrival_time.unwrap_or_else(now_unix_secs),
            client_index,
            data_parallel_rank: self.data_parallel_rank,
            trace_headers: self.trace_headers,
            request: uniserve_core::MediaRequest {
                request_id: RequestId(0),
                prompt: self.prompt,
                seed: self.seed,
                priority: self.priority,
                output_path: self.output_path,
            },
        }
    }
}

/// One terminal media event. Dropping a live receiver cancels the request.
pub struct MediaEventStream {
    rx: mpsc::Receiver<MediaEvent>,
    cancel: Option<Box<dyn FnOnce() + Send + 'static>>,
    finished: bool,
}

impl MediaEventStream {
    pub fn new(rx: mpsc::Receiver<MediaEvent>) -> Self {
        Self {
            rx,
            cancel: None,
            finished: false,
        }
    }

    pub fn with_cancel(
        rx: mpsc::Receiver<MediaEvent>,
        cancel: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self {
            rx,
            cancel: Some(Box::new(cancel)),
            finished: false,
        }
    }

    pub async fn next(&mut self) -> Option<MediaEvent> {
        let event = self.rx.recv().await?;
        self.finished = true;
        self.cancel = None;
        Some(event)
    }

    pub fn cancel(&mut self) {
        if !self.finished
            && let Some(cancel) = self.cancel.take()
        {
            cancel();
        }
    }
}

impl Drop for MediaEventStream {
    fn drop(&mut self) {
        if !self.finished
            && let Some(cancel) = self.cancel.take()
        {
            cancel();
        }
    }
}
