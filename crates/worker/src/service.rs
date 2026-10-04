//! Rank-channel admission, result delivery, and orderly worker shutdown.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use uniserve_worker_ipc::{
    Batch as BatchPlan, BatchOutput, IpcError, RankServer, RequestKind, WorkerInfo, WorkerRequest,
    WorkerResponse, WorkerResponseError,
};

use crate::{Backend, Batch, Error, Executor, Submission};

const EXPERT_POLL: Duration = Duration::from_micros(200);
// Keep a rank with open requests available for its engine's next batch before
// joining an empty expert step. The engine needs time to consume the result
// and schedule its next batch even when it still has admitted requests.
const OWN_STEP_WAIT: Duration = Duration::from_millis(8);

/// Numerical values and resource observations needed by the rank service.
/// Request correlation, admission, polling and shutdown remain in Service.
pub trait ServiceBackend: Backend<Output = BatchOutput> {
    fn batch(&self, plan: &BatchPlan) -> Result<Self::Batch, Self::Error>;
    fn response_error(
        &self,
        kind: RequestKind,
        error: Self::Error,
    ) -> Result<WorkerResponseError, Self::Error>;

    fn has_open_requests(&self) -> Result<bool, Self::Error>;
    fn awaiting_acknowledgment(&self) -> Result<bool, Self::Error>;

    /// Participate without an own forward. Returns (ran a step, group released).
    /// Closing ranks continue participating until every peer leaves the group.
    fn join_expert_step(&self, leaving: bool) -> Result<(bool, bool), Self::Error>;
}

enum PendingResponse {
    Ready(Box<WorkerResponse>),
    Batch {
        message_id: Option<u64>,
        submission: Arc<Submission>,
    },
}

/// Borrows a worker's executor and endpoint for one synchronous serving run.
/// All channel waits run on the caller's thread; completion wakes interrupt
/// them without advancing execution on a callback thread.
pub struct Service<'a, B: ServiceBackend> {
    executor: &'a mut Executor<B>,
    endpoint: &'a mut RankServer,
    info: WorkerInfo,
    experts: bool,
    pending: VecDeque<PendingResponse>,
    closing: bool,
    shutdown: Option<WorkerResponse>,
    last_result: Instant,
}

impl<'a, B: ServiceBackend> Service<'a, B> {
    pub fn new(
        executor: &'a mut Executor<B>,
        endpoint: &'a mut RankServer,
        info: WorkerInfo,
        experts: bool,
    ) -> Self {
        executor.set_completion_wake(Some(endpoint.completion_wake()));
        Self {
            executor,
            endpoint,
            info,
            experts,
            pending: VecDeque::new(),
            closing: false,
            shutdown: None,
            last_result: Instant::now(),
        }
    }

    /// Drain accepted results before acknowledging Close. A fatal result stops
    /// admission too, while preserving responses already owed to the engine.
    pub fn run(&mut self) -> Result<(), B::Error> {
        loop {
            let advanced = self.executor.advance()?;
            if self.send_ready()? || advanced {
                continue;
            }

            if self.closing && self.pending.is_empty() && !self.executor.has_work() {
                if self.experts && !self.executor.backend().join_expert_step(true)?.1 {
                    continue;
                }

                if let Some(response) = self.shutdown.take() {
                    self.respond(&response)?;
                }
                return Ok(());
            }

            if !self.closing && self.pending.len() < self.info.queue_depth as usize {
                let frame = self
                    .endpoint
                    .try_recv()
                    .map_err(|e| self.channel_error(e))?;
                if let Some(frame) = frame {
                    let request = frame.decode_request().map_err(|e| self.channel_error(e))?;
                    self.accept(request)?;
                    continue;
                }
            }

            if self.experts {
                if self.awaits_own_step()? {
                    self.wait(EXPERT_POLL)?;
                    continue;
                }

                if self.executor.backend().join_expert_step(false)?.0 {
                    continue;
                }
            }

            if !self.pending.is_empty() || self.executor.has_work() {
                // Remote acknowledgment words have no wake source. Poll only
                // while one is outstanding; ordinary work waits on its wake.
                let timeout = if self.experts {
                    EXPERT_POLL
                } else if self.executor.backend().awaiting_acknowledgment()? {
                    Duration::from_millis(1)
                } else {
                    Duration::from_secs(60)
                };
                self.wait(timeout)?;
            } else if self.experts {
                self.wait(EXPERT_POLL)?;
            } else {
                let frame = self.endpoint.recv().map_err(|e| self.channel_error(e))?;
                let request = frame.decode_request().map_err(|e| self.channel_error(e))?;
                self.accept(request)?;
            }
        }
    }

    fn awaits_own_step(&self) -> Result<bool, B::Error> {
        Ok(self.executor.backend().has_open_requests()?
            && (!self.pending.is_empty()
                || self.executor.has_work()
                || self.last_result.elapsed() < OWN_STEP_WAIT))
    }

    fn accept(&mut self, request: WorkerRequest) -> Result<(), B::Error> {
        let pending = match request {
            WorkerRequest::Close { message_id } => {
                self.closing = true;
                self.shutdown = Some(WorkerResponse::Ok { message_id });
                return Ok(());
            }
            WorkerRequest::Info { message_id } => {
                PendingResponse::Ready(Box::new(WorkerResponse::Info {
                    message_id,
                    info: self.info.clone(),
                }))
            }
            WorkerRequest::Submit { message_id, batch } => {
                let admitted =
                    self.executor.backend().batch(&batch).and_then(|data| {
                        self.executor.submit(Batch::from_plan(&batch, data), false)
                    });
                match admitted {
                    Ok(submission) => PendingResponse::Batch {
                        message_id,
                        submission,
                    },
                    Err(error) => {
                        PendingResponse::Ready(Box::new(self.error_response(message_id, error)?))
                    }
                }
            }
        };
        self.pending.push_back(pending);
        Ok(())
    }

    fn send_ready(&mut self) -> Result<bool, B::Error> {
        // A pending transfer or host task must not hold up independent results.
        for index in 0..self.pending.len() {
            let response = match &self.pending[index] {
                PendingResponse::Ready(response) => (**response).clone(),
                PendingResponse::Batch {
                    message_id,
                    submission,
                } => {
                    let message_id = *message_id;
                    let result = self.executor.poll(submission);
                    match result {
                        Ok(Some(result)) => WorkerResponse::Result { message_id, result },
                        Ok(None) => continue,
                        Err(error) => self.error_response(message_id, error)?,
                    }
                }
            };

            self.pending.remove(index);
            self.last_result = Instant::now();
            self.respond(&response)?;
            if matches!(response, WorkerResponse::Error { ref error, .. } if error.fatal) {
                self.closing = true;
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn error_response(
        &self,
        message_id: Option<u64>,
        error: B::Error,
    ) -> Result<WorkerResponse, B::Error> {
        Ok(WorkerResponse::Error {
            message_id,
            error: self
                .executor
                .backend()
                .response_error(RequestKind::Submit, error)?,
        })
    }

    fn respond(&mut self, response: &WorkerResponse) -> Result<(), B::Error> {
        self.endpoint
            .respond(response)
            .map_err(|e| self.channel_error(e))
    }

    fn wait(&mut self, timeout: Duration) -> Result<(), B::Error> {
        self.endpoint
            .wait_incoming(timeout)
            .map_err(|e| self.channel_error(e))
    }

    fn channel_error(&self, error: IpcError) -> B::Error {
        self.executor
            .backend()
            .error(Error::Transport(error.to_string()))
    }
}

impl<B: ServiceBackend> Drop for Service<'_, B> {
    fn drop(&mut self) {
        self.executor.set_completion_wake(None);
    }
}
