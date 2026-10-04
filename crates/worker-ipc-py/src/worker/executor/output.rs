//! Assemble successful and failed native batch results for delivery.

use std::collections::HashSet;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pythonize::depythonize;
use uniserve_worker_ipc::{BatchOutput, CallStatus, ErrorCode, RequestOutput};

use super::{BatchState, PythonBackend};
use crate::convert;
use crate::worker::error::native_error;
use crate::worker::pending::{PendingOutput, request_output};

impl PythonBackend {
    pub(super) fn materialize(
        &self,
        py: Python<'_>,
        batch: &BatchState,
    ) -> PyResult<Option<BatchOutput>> {
        let outputs = batch
            .numerical
            .bind(py)
            .getattr("outputs")?
            .cast_into::<PyList>()?;
        let pending = outputs
            .iter()
            .map(|output| output.cast_into::<PendingOutput>().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()?;
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
                Ok(output.borrow().result()?.clone())
            })
            .collect::<PyResult<Vec<_>>>()?;
        for output in pending {
            output
                .borrow()
                .accept(py, &mut self.requests.borrow_mut(py))?;
        }
        self.output(py, batch, completions).map(Some)
    }

    fn output(
        &self,
        py: Python<'_>,
        batch: &BatchState,
        completions: Vec<RequestOutput>,
    ) -> PyResult<BatchOutput> {
        let numerical = batch.numerical.bind(py);
        let successful = completions
            .iter()
            .filter(|value| value.status == CallStatus::Ok)
            .map(|value| (value.request_key, value.call_id))
            .collect::<HashSet<_>>();
        let mut products = Vec::new();
        for product in numerical.getattr("products")?.try_iter()? {
            let product =
                convert::tensor_publication_from_py(&product?.call_method0("to_mapping")?)
                    .ok_or_else(|| PyRuntimeError::new_err("invalid tensor output"))?;
            if successful.contains(&(
                product.product.request_key,
                product.product.producer_call_id,
            )) {
                products.push(product);
            }
        }
        let stats = numerical.getattr("stats")?;
        Ok(BatchOutput {
            batch_id: batch.plan.batch_id,
            completions,
            products,
            worker_exec_us: numerical.getattr("execution_us")?.extract()?,
            forward_stats: if stats.is_none() {
                None
            } else {
                Some(depythonize(&stats.call_method0("to_mapping")?)?)
            },
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
                let mut output = request_output(key, call.call_id, call.code, progress)?;
                output.status = CallStatus::Error;
                output.error_code = Some(code);
                Ok(output)
            })
            .collect::<PyResult<_>>()?;
        self.output(py, batch, completions)
    }
}
