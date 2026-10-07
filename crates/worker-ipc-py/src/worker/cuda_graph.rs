//! Captured numerical calls, segmented replay and their resource lifetime.

mod capture;
mod runner;

pub(super) use runner::CUDAGraphRunner;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use super::execution_context::ExecutionContext;
use super::host::with_context;
use super::stream::CUDAStream;
use capture::Capture;

pyo3::create_exception!(
    uniserve_worker._uniserve_ipc,
    CUDAGraphError,
    PyRuntimeError
);

/// One captured call and its borrowed numerical resources. Final readers must
/// retire before explicit close; an aborted close keeps backing until exit.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct CUDAGraph {
    resources: Option<Resources>,
}

struct Resources {
    context: Py<ExecutionContext>,
    pools: Py<PyDict>,
    computation: Py<PyAny>,
    capture_stream: Py<PyAny>,
    capture_owner: Py<CUDAStream>,
    captured: Option<Captured>,
}

struct Captured {
    sequence: Py<Capture>,
    output: Py<PyAny>,
    // The numerical closure retains input tensors and borrowed model weights.
    call: Py<PyAny>,
}

#[pymethods]
impl CUDAGraph {
    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        item: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cls.py()
            .import("types")?
            .call_method1("GenericAlias", (cls, item))
    }

    #[new]
    #[pyo3(signature = (*, context, pools=None))]
    pub(super) fn new(
        py: Python<'_>,
        context: Py<ExecutionContext>,
        pools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let numerical = context.bind(py);
        let device = numerical.borrow().device(py)?.into_bound(py);
        if device.getattr("type")?.extract::<String>()? != "cuda" {
            return Err(PyValueError::new_err(
                "CUDA graph execution requires a CUDA module",
            ));
        }

        let pools = match pools {
            Some(pools) => py
                .get_type::<PyDict>()
                .call1((pools,))?
                .cast_into::<PyDict>()?,
            None => PyDict::new(py),
        };
        let stream = numerical
            .borrow()
            .stream(py)
            .unwrap_or_else(|| py.None())
            .into_bound(py);
        let (computation, owner) = if stream.is_none() {
            let options = PyDict::new(py);
            options.set_item("device", &device)?;
            let computation = py
                .import("torch.cuda")?
                .getattr("Stream")?
                .call((), Some(&options))?;
            numerical
                .borrow()
                .graph_streams(py)?
                .bind(py)
                .add(&computation)?;
            (computation, py.None())
        } else {
            (
                stream.getattr("stream")?,
                stream.getattr("_native")?.unbind(),
            )
        };

        // PyTorch frees capture-stream workspaces at reset. A private sibling
        // prevents closing this graph from invalidating another graph's inputs.
        let (capture_owner, capture_stream) = py
            .import("uniserve.runtime.cuda")?
            .call_method1("create_sibling_stream", (&computation, owner))?
            .extract()?;
        Ok(Self {
            resources: Some(Resources {
                context,
                pools: pools.unbind(),
                computation: computation.unbind(),
                capture_stream,
                capture_owner,
                captured: None,
            }),
        })
    }

    #[getter]
    pub(super) fn context(&self, py: Python<'_>) -> PyResult<Py<ExecutionContext>> {
        Ok(self.resources()?.context.clone_ref(py))
    }

    /// Capture once, preserving PyTorch's allocator and generator bookkeeping.
    /// Restore runs after every capture attempt, including numerical failures.
    #[pyo3(signature = (call, *, restore=None))]
    pub(super) fn capture(
        &mut self,
        py: Python<'_>,
        call: Py<PyAny>,
        restore: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| CUDAGraphError::new_err("capture requires an open uncaptured graph"))?;
        if resources.captured.is_some() {
            return Err(CUDAGraphError::new_err(
                "capture requires an open uncaptured graph",
            ));
        }
        let context = resources.context.bind(py);
        let device = context.borrow().device(py)?.into_bound(py);
        let pool = resources.pools.bind(py).get_item(&device)?;
        let pool = pool
            .map(|pool| {
                py.import("torch.cuda")?
                    .call_method1("_POOL_HANDLE", (pool.getattr("id")?,))
            })
            .transpose()?;
        let sequence = Py::new(
            py,
            Capture::new(
                resources.capture_stream.clone_ref(py),
                resources.computation.clone_ref(py),
                pool.map(Bound::unbind),
            ),
        )?;
        let backend = py.import("uniserve.runtime.cuda_graph")?;
        let scope = backend.call_method1(
            "_capture_scope",
            (
                context,
                &resources.pools,
                &resources.capture_stream,
                &resources.computation,
            ),
        )?;
        let result = with_context(&scope, || {
            let result = (|| {
                backend.call_method1("_prepare_capture", (&device, &resources.computation))?;
                capture::run(sequence.bind(py), context, call.bind(py))
            })();
            let restored = restore
                .as_ref()
                .map(|restore| {
                    ExecutionContext::with_active(context, || restore.call0(py).map(drop))
                })
                .transpose();

            // A failed restoration takes precedence, as for a Python finally.
            restored?;
            result
        });
        match result {
            Ok(output) => {
                resources.captured = Some(Captured {
                    sequence,
                    output,
                    call,
                });
                Ok(())
            }
            Err(error) => {
                if let Err(cleanup) = sequence.borrow_mut(py).reset(py) {
                    let _ = error.value(py).call_method1(
                        "add_note",
                        (format!("CUDA graph cleanup failed: {cleanup}"),),
                    );
                }
                Err(failure(
                    py,
                    format!("CUDA graph capture failed on {device}: {error}"),
                    error,
                ))
            }
        }
    }

    /// Submit all segments in capture order and return retained output views.
    /// Torch replay also advances captured RNG state; a raw CUDA launch cannot
    /// replace it without transferring that bookkeeping to the native backend.
    pub(super) fn replay(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let resources = self.resources()?;
        let captured = resources
            .captured
            .as_ref()
            .ok_or_else(|| CUDAGraphError::new_err("replay requires an open captured graph"))?;
        let context = resources.context.bind(py);
        ExecutionContext::with_active(context, || captured.sequence.borrow(py).replay(py))
            .map_err(|error| failure(py, format!("CUDA graph replay failed: {error}"), error))?;
        Ok(captured.output.clone_ref(py))
    }

    #[pyo3(signature = (*, aborted=false))]
    #[allow(clippy::mem_forget)] // Device accesses outlive failed rank shutdown.
    pub(super) fn close(&mut self, py: Python<'_>, aborted: bool) -> PyResult<()> {
        let Some(mut resources) = self.resources.take() else {
            return Ok(());
        };
        if aborted {
            // A failed peer may prevent completion forever. Keep graph, input,
            // context and stream ownership together until the process exits.
            std::mem::forget(resources);
            return Ok(());
        }
        if let Err(error) = resources.close(py) {
            std::mem::forget(resources);
            return Err(error);
        }
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.resources()?;
        Ok(slf)
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _kind: &Bound<'_, PyAny>,
        error: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let aborted =
            !error.is_none() && self.resources.as_ref().is_some_and(|value| !value.idle(py));
        if let Err(cleanup) = self.close(py, aborted) {
            if error.is_none() {
                return Err(cleanup);
            }
            let _ = error.call_method1(
                "add_note",
                (format!("CUDA graph cleanup failed: {cleanup}"),),
            );
        }
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(resources) = &self.resources {
            visit.call(&resources.context)?;
            visit.call(&resources.pools)?;
            visit.call(&resources.computation)?;
            visit.call(&resources.capture_stream)?;
            visit.call(&resources.capture_owner)?;
            if let Some(captured) = &resources.captured {
                visit.call(&captured.sequence)?;
                visit.call(&captured.output)?;
                visit.call(&captured.call)?;
            }
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.retire();
    }
}

impl CUDAGraph {
    fn resources(&self) -> PyResult<&Resources> {
        self.resources
            .as_ref()
            .ok_or_else(|| CUDAGraphError::new_err("CUDA graph is closed"))
    }

    fn retire(&mut self) {
        if self.resources.is_none() {
            return;
        }

        Python::try_attach(|py| {
            let aborted = self.resources.as_ref().is_some_and(|value| !value.idle(py));
            if let Err(error) = self.close(py, aborted) {
                error.write_unraisable(py, None);
            }
        });
    }
}

impl Drop for CUDAGraph {
    fn drop(&mut self) {
        self.retire();
    }
}

impl Resources {
    fn idle(&self, py: Python<'_>) -> bool {
        // A context without a dedicated stream replays on its caller's current
        // stream. The context also accounts for cross-device numerical copies.
        let context_idle = self.context.borrow(py).idle(py).unwrap_or(false);
        context_idle
            && [&self.computation, &self.capture_stream]
                .iter()
                .all(|stream| {
                    stream
                        .call_method0(py, "query")
                        .and_then(|value| value.extract(py))
                        .unwrap_or(false)
                })
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(captured) = &self.captured {
            captured.sequence.borrow_mut(py).reset(py)?;
        }
        self.captured = None;
        self.pools.bind(py).clear();
        self.context
            .borrow(py)
            .graph_streams(py)?
            .bind(py)
            .discard(&self.computation)?;
        self.capture_owner.borrow(py).close(py, false)
    }
}

fn failure(py: Python<'_>, message: String, cause: PyErr) -> PyErr {
    let error = CUDAGraphError::new_err(message);
    error.set_cause(py, Some(cause));
    error
}
