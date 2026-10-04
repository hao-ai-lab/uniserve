//! Python access to native KV storage lifetime management.

use std::collections::HashSet;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker::KVCacheManager as NativeKVCacheManager;
use uniserve_worker_ipc::{BufferId, RequestKey};

use super::completion::{Completion, CompletionRef};
use super::error::native_error;
use super::protocol::{buffer_id, request_key};

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct KVCacheManager {
    inner: NativeKVCacheManager<CompletionRef>,
}

#[pymethods]
impl KVCacheManager {
    #[new]
    fn new() -> Self {
        Self {
            inner: NativeKVCacheManager::default(),
        }
    }

    #[getter]
    fn has_pending_accesses(&mut self) -> bool {
        self.inner.has_pending_accesses()
    }

    #[getter]
    fn has_transfers(&self) -> bool {
        self.inner.has_transfers()
    }

    fn retain_execution(
        &mut self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
        spans: Vec<(u32, u32, u32)>,
        completion: Py<Completion>,
    ) -> PyResult<()> {
        if completion.borrow(py).done() {
            completion.borrow(py).result(py, None)?;
            return Ok(());
        }
        self.inner.retain_execution(
            request_key(request)?,
            &spans,
            CompletionRef::new(py, completion),
        );
        Ok(())
    }

    fn reserve_export(
        &mut self,
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
        spans: Vec<(u32, u32, u32)>,
    ) -> PyResult<()> {
        self.inner
            .reserve_export(buffer_id(buffer)?, &spans)
            .map_err(|error| native_error(py, error))
    }

    fn retain_export(
        &mut self,
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
        retirement: Py<Completion>,
    ) -> PyResult<()> {
        self.inner
            .retain_export(buffer_id(buffer)?, CompletionRef::new(py, retirement))
            .map_err(|error| native_error(py, error))
    }

    fn release_exports(&mut self, buffers: &Bound<'_, PyAny>) -> PyResult<()> {
        let buffers = buffers
            .try_iter()?
            .map(|buffer| buffer_id(&buffer?))
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.release_exports(&buffers);
        Ok(())
    }

    fn exported_buffers(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let buffers = self
            .inner
            .exported_buffers()
            .into_iter()
            .map(|buffer| crate::convert::buffer_id_to_py(py, &buffer))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, buffers)?.unbind())
    }

    fn reserve_import(
        &mut self,
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
        spans: Vec<(u32, u32, u32)>,
        retirement: Py<Completion>,
    ) -> PyResult<()> {
        self.inner
            .reserve_import(
                buffer_id(buffer)?,
                &spans,
                CompletionRef::new(py, retirement),
            )
            .map_err(|error| native_error(py, error))
    }

    fn discard_import(&mut self, buffer: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.discard_import(buffer_id(buffer)?);
        Ok(())
    }

    fn write_dependencies(
        &mut self,
        py: Python<'_>,
        spans: Vec<(u32, u32, u32)>,
    ) -> PyResult<Py<PyTuple>> {
        let completions = self
            .inner
            .write_dependencies(&spans)
            .into_iter()
            .map(|completion| completion.owner.clone_ref(py));
        Ok(PyTuple::new(py, completions)?.unbind())
    }

    fn require_writable(&mut self, py: Python<'_>, spans: Vec<(u32, u32, u32)>) -> PyResult<()> {
        self.inner
            .require_writable(&spans)
            .map_err(|error| native_error(py, error))
    }

    fn require_reusable(&mut self, py: Python<'_>, spans: Vec<(u32, u32, u32)>) -> PyResult<()> {
        self.inner
            .require_reusable(&spans)
            .map_err(|error| native_error(py, error))
    }

    fn retirement_ready(
        &mut self,
        py: Python<'_>,
        buffers: &Bound<'_, PyAny>,
        requests: &Bound<'_, PyAny>,
        retained: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let buffers = buffer_set(buffers)?;
        let requests: HashSet<RequestKey> = requests
            .try_iter()?
            .map(|request| request_key(&request?))
            .collect::<PyResult<_>>()?;
        let retained = buffer_set(retained)?;

        for completion in self
            .inner
            .retirement_completions(&buffers, &requests, &retained)
        {
            if completion.done() {
                completion.owner.borrow(py).result(py, None)?;
            }
        }
        Ok(self.inner.retirement_ready(&buffers, &requests, &retained))
    }

    fn require_retired(&mut self, py: Python<'_>) -> PyResult<()> {
        self.inner
            .require_retired()
            .map_err(|error| native_error(py, error))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for completion in self.inner.completions() {
            visit.call(&completion.owner)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.inner = NativeKVCacheManager::default();
    }
}

fn buffer_set(buffers: &Bound<'_, PyAny>) -> PyResult<HashSet<BufferId>> {
    buffers
        .try_iter()?
        .map(|buffer| buffer_id(&buffer?))
        .collect()
}
