//! Fixed graph inputs and numerical warmup shared by worker graph families.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;

use super::super::host::with_context;
use super::CUDAGraph;

/// A graph executable and the tensor correspondence used to update its inputs.
/// Python's Inputs performs tensor copies; this owner controls its lifetime.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct CUDAGraphRunner {
    #[pyo3(get)]
    executable: Py<CUDAGraph>,
    #[pyo3(get)]
    inputs: Py<PyAny>,
}

#[pymethods]
impl CUDAGraphRunner {
    /// Warm a numerical bucket, restoring modified inputs after both warmup
    /// and capture. Partial graphs can supply a complete eager warmup call.
    #[staticmethod]
    #[pyo3(signature = (context, inputs, call, *, pools, restore=None, warm=true, warmup=None))]
    #[allow(clippy::too_many_arguments)]
    fn capture(
        py: Python<'_>,
        context: Py<PyAny>,
        inputs: Py<PyAny>,
        call: Py<PyAny>,
        pools: Option<&Bound<'_, PyAny>>,
        restore: Option<Py<PyAny>>,
        warm: bool,
        warmup: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if warm {
            with_context(&context.bind(py).call_method0("activate")?, || {
                let result = warmup.as_ref().unwrap_or(&call).call1(py, (&inputs,));
                if let Some(restore) = &restore {
                    restore.call0(py)?;
                }
                result.map(drop)
            })?;
        }

        let mut executable = CUDAGraph::new(py, context, pools)?;
        let call = py
            .import("functools")?
            .getattr("partial")?
            .call1((call, &inputs))?
            .unbind();
        if let Err(error) = executable.capture(py, call, restore) {
            if let Err(cleanup) = executable.close(py, false) {
                let _ = error.value(py).call_method1(
                    "add_note",
                    (format!("CUDA graph cleanup failed: {cleanup}"),),
                );
            }
            return Err(error);
        }
        let inputs = py
            .import("uniserve_worker.model_executor.cuda_graph")?
            .call_method1("Inputs", (inputs,))?
            .unbind();
        Ok(Self {
            executable: Py::new(py, executable)?,
            inputs,
        })
    }

    /// Update captured tensors, then return output views overwritten by replay.
    #[pyo3(signature = (live=None))]
    pub(in crate::worker) fn replay(
        &self,
        py: Python<'_>,
        live: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let executable = self.executable.borrow(py);
        let context = executable.context(py)?;
        with_context(&context.bind(py).call_method0("activate")?, || {
            if let Some(live) = live {
                self.inputs.call_method1(py, "copy", (live,))?;
            }
            executable.replay(py)
        })
    }

    pub(in super::super) fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.executable.borrow_mut(py).close(py, false)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.executable)?;
        visit.call(&self.inputs)?;
        Ok(())
    }
}
