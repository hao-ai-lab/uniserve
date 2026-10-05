//! Assemble successful and failed native batch results for delivery.

use pyo3::prelude::*;
use uniserve_worker::{BatchResult, request_output};
use uniserve_worker_ipc::{BatchOutput, CallStatus, ErrorCode, RequestOutput};

use super::{BatchState, PythonBackend};
use crate::worker::error::native_error;
use crate::worker::pending::PendingOutput;

impl PythonBackend {
    pub(super) fn materialize(
        &self,
        py: Python<'_>,
        batch: &BatchState,
    ) -> PyResult<Option<BatchResult>> {
        let pending = batch.pending_outputs(py);
        for output in &pending {
            PendingOutput::submit_host_tasks(output)?;
        }
        for output in &pending {
            if !PendingOutput::ready(output)? {
                return Ok(None);
            }
        }

        // Resolve every row before advancing any request. Native results go
        // directly to the service; numerical owners stay until batch cleanup.
        let completions = pending
            .iter()
            .map(|output| {
                PendingOutput::resolve(output)?;
                output.borrow().result(py)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut media = Vec::new();
        for output in pending {
            let output = output.borrow();
            output.accept(py, &mut self.requests.borrow_mut(py))?;
            if let Some(source) = output.take_media(py)? {
                media.push(source);
            }
        }
        let output = self.output(py, batch, completions)?;
        Ok(Some(BatchResult { output, media }))
    }

    fn output(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        completions: Vec<RequestOutput>,
    ) -> PyResult<BatchOutput> {
        let mut products = Vec::new();
        for (pending, completion) in batch.pending_outputs(py).iter().zip(&completions) {
            if completion.status == CallStatus::Ok {
                products.extend_from_slice(pending.borrow().lock(py)?.products());
            }
        }
        Ok(BatchOutput {
            batch_id: batch.plan.batch_id,
            completions,
            products,
            worker_exec_us: batch.execution_us,
            forward_stats: batch.stats.clone(),
        })
    }

    pub(super) fn failed_output(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        failure: &Bound<'_, PyAny>,
    ) -> PyResult<BatchOutput> {
        let code: String = failure.getattr("code")?.getattr("value")?.extract()?;
        let code = match code.as_str() {
            "ResourceError" => ErrorCode::ResourceExhausted,
            "ComputeError" => ErrorCode::ComputeError,
            "InvariantViolation" | "FatalWorkerFailure" => ErrorCode::Internal,
            _ => ErrorCode::InvalidCall,
        };
        let requests = self.requests.borrow(py);
        let completions = batch
            .plan
            .calls
            .iter()
            .zip(&batch.predecessors)
            .map(|(call, predecessor)| {
                let key = call.request_key;
                // A failed first call has no accepted parent. A stale epoch must
                // never report the progress of a replacement request slot.
                let progress = requests
                    .pool
                    .peek(key.request_id.0)
                    .filter(|request| request.key() == key && predecessor.is_some())
                    .map(|request| request.progress())
                    .transpose()
                    .map_err(|error| native_error(py, error))?
                    .unwrap_or_default();
                let mut output = request_output(key, call.call_id, call.code, progress)
                    .map_err(|error| native_error(py, error))?;
                output.status = CallStatus::Error;
                output.error_code = Some(code);
                Ok(output)
            })
            .collect::<PyResult<_>>()?;
        self.output(py, batch, completions)
    }
}
