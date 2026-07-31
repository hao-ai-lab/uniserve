use std::ops::Deref;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use futures::stream::FusedStream;
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use crate::client::{StreamControl, StreamControlRequest};
use crate::protocol::{EngineCoreFinishReason, EngineCoreOutput};
use crate::{Error, Result, StreamCancelCause};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Running,
    Finished,
    ClosedWithError,
    UnexpectedClose,
}

/// One request-scoped engine output plus the enclosing batch metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineCoreStreamOutput {
    pub engine_index: u32,
    pub timestamp: f64,
    pub output: EngineCoreOutput,
}

impl Deref for EngineCoreStreamOutput {
    type Target = EngineCoreOutput;

    fn deref(&self) -> &Self::Target {
        &self.output
    }
}

/// Stream of raw engine outputs for one request.
///
/// The stream yields only [`EngineCoreStreamOutput`] values whose embedded
/// output `request_id` matches the originating `add_request` call. Normal
/// request completion is expected to include a final output object whose
/// `finish_reason` is non-`None`.
pub struct EngineCoreOutputStream {
    request_id: String,
    control_tx: mpsc::UnboundedSender<StreamControlRequest>,
    output_token_count: usize,
    acknowledge_on_receive: bool,
    state: State,
    rx: mpsc::Receiver<Result<EngineCoreStreamOutput>>,
}

impl EngineCoreOutputStream {
    pub const BUFFER_CAPACITY: usize = 64;

    pub fn new(
        request_id: String,
        control_tx: mpsc::UnboundedSender<StreamControlRequest>,
        rx: mpsc::Receiver<Result<EngineCoreStreamOutput>>,
        acknowledge_on_receive: bool,
    ) -> Self {
        Self {
            request_id,
            control_tx,
            output_token_count: 0,
            acknowledge_on_receive,
            state: State::Running,
            rx,
        }
    }

    /// Return the engine-wire `request_id` bound to this stream.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub(crate) fn cancel_at(&mut self, cause: StreamCancelCause, output_token_count: usize) {
        if self.is_terminated() {
            return;
        }
        let control_request = StreamControlRequest {
            request_id: self.request_id.clone(),
            control: StreamControl::Cancel {
                cause,
                output_token_count,
            },
        };
        if self.control_tx.send(control_request).is_err() {
            warn!(
                request_id = self.request_id,
                "stream-cancellation worker already shut down; skip cancellation"
            );
        }
        self.state = State::Finished;
    }

    pub(crate) fn acknowledge_at(&self, output_token_count: usize) {
        let control_request = StreamControlRequest {
            request_id: self.request_id.clone(),
            control: StreamControl::Acknowledge { output_token_count },
        };
        if self.control_tx.send(control_request).is_err() {
            warn!(
                request_id = self.request_id,
                "stream-control worker already shut down; skip semantic acknowledgement"
            );
        }
    }

    /// Transfer prefix acknowledgement to a downstream canonical decoder.
    pub(crate) fn delegate_acknowledgement(&mut self) {
        self.acknowledge_on_receive = false;
    }
}

impl Stream for EngineCoreOutputStream {
    type Item = Result<EngineCoreStreamOutput>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_terminated() {
            return Poll::Ready(None);
        }

        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(item)) => {
                match &item {
                    Ok(output) => {
                        self.output_token_count = self
                            .output_token_count
                            .saturating_add(output.output.new_token_ids.len());
                        if self.acknowledge_on_receive && !output.output.new_token_ids.is_empty() {
                            self.acknowledge_at(self.output_token_count);
                        }
                        // If the output indicates the request is finished, mark the stream as
                        // terminated with cleanly-finished state and expect no more outputs to
                        // come.
                        if output.finished() {
                            if output.finish_reason == Some(EngineCoreFinishReason::Error) {
                                error!(
                                    self.request_id,
                                    "request failed with an internal error during generation"
                                );
                            }
                            debug!(self.request_id, "request completed via final output");
                            self.state = State::Finished;
                        }
                    }
                    Err(error) => {
                        // If we get an error from the output stream, mark the stream as terminated
                        // with an error.
                        warn!(self.request_id, error = %error.as_report(), "request encountered an error");
                        self.state = State::ClosedWithError;
                    }
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                // If we get a `None` without seeing a finished output, this is an unexpected
                // close from the engine side. Mark the stream as terminated
                // with an unexpected close state and send an error down the
                // stream to notify the caller.
                warn!(self.request_id, "request stream closed unexpectedly");
                self.state = State::UnexpectedClose;

                Poll::Ready(Some(Err(Error::RequestStreamClosed {
                    request_id: self.request_id.clone(),
                })))
            }
        }
    }
}

impl FusedStream for EngineCoreOutputStream {
    fn is_terminated(&self) -> bool {
        !matches!(self.state, State::Running)
    }
}

impl Drop for EngineCoreOutputStream {
    fn drop(&mut self) {
        if self.is_terminated() {
            // If it's terminated, it means that the request either finished cleanly, or
            // encountered an error or unexpected close from the engine. In any
            // case, the request stream is already considered inactive and
            // there's no need to cancel it on the engine side.
            return;
        }

        self.cancel_at(StreamCancelCause::current(), self.output_token_count);
    }
}
