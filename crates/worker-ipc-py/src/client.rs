//! Python access to the head's end of a rank channel.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pythonize::depythonize;
use uniserve_worker_ipc::{
    Outstanding, RankChannel, RequestKind, SHARED_STORAGE_CHANNEL, WorkerRequest,
};

use crate::{convert, py_runtime};

/// One caller owns sending and receiving. Results may arrive out of order.
#[pyclass(name = "Client")]
pub(crate) struct PyClient {
    channel: Option<RankChannel>,
    // Shared-storage loans are Send but not Sync. Python methods take an
    // exclusive mutable borrow, so get_mut accesses them without locking.
    pending: Mutex<VecDeque<(u64, Outstanding)>>,
}

#[pymethods]
impl PyClient {
    #[new]
    #[pyo3(signature = (endpoint, max_payload=1048576, max_inflight=1, transport=SHARED_STORAGE_CHANNEL, timeout=5.0))]
    fn new(
        py: Python<'_>,
        endpoint: &str,
        max_payload: usize,
        max_inflight: usize,
        transport: &str,
        timeout: f64,
    ) -> PyResult<Self> {
        let timeout = Duration::try_from_secs_f64(timeout).map_err(py_runtime)?;
        let channel = py
            .detach(|| {
                RankChannel::connect(transport, endpoint, max_payload, max_inflight, timeout)
            })
            .map_err(py_runtime)?;
        Ok(Self {
            channel: Some(channel),
            pending: Mutex::new(VecDeque::new()),
        })
    }

    /// Send a request containing a distinct message_id and, for Submit, a native batch.
    fn send(&mut self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let kind: RequestKind = depythonize(&request.get_item("kind")?)?;
        let request: WorkerRequest = if kind == RequestKind::Submit {
            WorkerRequest::Submit {
                message_id: request.get_item("message_id")?.extract()?,
                batch: Box::new(
                    request
                        .get_item("batch")?
                        .extract::<PyRef<'_, crate::batches::Batch>>()?
                        .inner
                        .as_ref()
                        .clone(),
                ),
            }
        } else {
            depythonize(request)?
        };
        let id = request.message_id().unwrap_or(0);
        let requests = self
            .pending
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        if requests.iter().any(|(pending, _)| *pending == id) {
            return Err(PyValueError::new_err("message_id already outstanding"));
        }

        let channel = self
            .channel
            .as_mut()
            .ok_or_else(|| py_runtime("IPC client is closed"))?;
        let pending = py
            .detach(|| channel.send_request_attempt(&request))
            .map_err(py_runtime)?;
        requests.push_back((id, pending));
        Ok(())
    }

    /// Receive any outstanding result. None means no result before the timeout.
    #[pyo3(signature = (timeout=0.0))]
    fn recv(&mut self, py: Python<'_>, timeout: f64) -> PyResult<Option<Py<PyAny>>> {
        let timeout = Duration::try_from_secs_f64(timeout).map_err(py_runtime)?;
        let channel = self
            .channel
            .as_mut()
            .ok_or_else(|| py_runtime("IPC client is closed"))?;
        let pending = self
            .pending
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        let response = py.detach(|| {
            let start = Instant::now();
            loop {
                for (index, (_, request)) in pending.iter().enumerate() {
                    if let Some(frame) = channel.try_recv_response(request).map_err(py_runtime)? {
                        pending.remove(index);
                        return frame.decode_response().map(Some).map_err(py_runtime);
                    }
                }

                let remaining = timeout.saturating_sub(start.elapsed());
                if pending.is_empty() || remaining.is_zero() {
                    return Ok(None);
                }
                channel.wait_wake(remaining).map_err(py_runtime)?;
            }
        })?;
        response
            .map(|response| convert::response_to_py(py, &response))
            .transpose()
    }

    fn close(&mut self) {
        // Shared request handles must drop before their channel's node.
        self.pending
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        self.channel.take();
    }
}

impl Drop for PyClient {
    fn drop(&mut self) {
        self.close();
    }
}
