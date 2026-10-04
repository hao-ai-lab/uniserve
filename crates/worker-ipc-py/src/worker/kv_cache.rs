//! Numerical views of native KV transfer state and storage lifetimes.

use std::collections::HashSet;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use uniserve_worker::KVCacheManager as NativeKVCacheManager;
use uniserve_worker_ipc::{BufferId, KvTransfer, RequestKey};

use super::completion::{Completion, CompletionRef};
use super::error::{invalid, native_error};
use super::protocol::{buffer_id, call_id, request_key};
use crate::convert;

#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct KVCacheManager {
    pub(super) inner: NativeKVCacheManager<CompletionRef>,
}

#[pymethods]
impl KVCacheManager {
    #[new]
    fn new() -> Self {
        Self {
            inner: NativeKVCacheManager::default(),
        }
    }

    fn resident(&self, py: Python<'_>, buffer: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .resident(buffer_id(buffer)?)
            .map(|transfer| convert::kv_transfer_to_py(py, transfer).map(Bound::unbind))
            .transpose()
    }

    fn destination_base(
        &self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
        destination: &str,
    ) -> PyResult<Option<(Py<PyAny>, u32)>> {
        self.inner
            .destination_base(request_key(request)?, destination)
            .map(|(buffer, extent)| Ok((convert::buffer_id_to_py(py, &buffer)?.unbind(), extent)))
            .transpose()
    }

    fn validate_install(&self, py: Python<'_>, transfer: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner
            .validate_install(&transfer_from_py(transfer)?)
            .map_err(|error| native_error(py, error))
    }

    fn validate_publications(
        &self,
        py: Python<'_>,
        publications: &Bound<'_, PyAny>,
        installations: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.inner
            .validate_publications(
                &publications_from_py(publications)?,
                &installations_from_py(installations)?,
            )
            .map_err(|error| native_error(py, error))
    }

    fn apply_publications(
        &mut self,
        publications: &Bound<'_, PyAny>,
        installations: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.inner.apply_publications(
            publications_from_py(publications)?,
            installations_from_py(installations)?,
        );
        Ok(())
    }

    fn release_calls(&mut self, py: Python<'_>, calls: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
        let calls = calls
            .try_iter()?
            .map(|call| {
                let (request, call): (Bound<'_, PyAny>, Bound<'_, PyAny>) = call?.extract()?;
                Ok((request_key(&request)?, call_id(&call)?))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let buffers = self
            .inner
            .release_calls(&calls)
            .iter()
            .map(|buffer| convert::buffer_id_to_py(py, buffer))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, buffers)?.unbind())
    }

    fn drop_request(&mut self, request_id: u64) {
        self.inner.drop_request(request_id);
    }

    fn clear_resident(&mut self) {
        self.inner.clear_resident();
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

        self.retirement_ready_for(py, &buffers, &requests, &retained)
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

fn transfer_from_py(value: &Bound<'_, PyAny>) -> PyResult<KvTransfer> {
    convert::kv_transfer_from_py(&value.call_method0("to_mapping")?)
        .ok_or_else(|| invalid(value.py(), "invalid KV transfer"))
}

pub(super) fn publications_from_py(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<(BufferId, KvTransfer)>> {
    values
        .try_iter()?
        .map(|value| {
            let (buffer, transfer): (Bound<'_, PyAny>, Bound<'_, PyAny>) = value?.extract()?;
            Ok((buffer_id(&buffer)?, transfer_from_py(&transfer)?))
        })
        .collect()
}

pub(super) fn installations_from_py(
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<(BufferId, BufferId, KvTransfer)>> {
    values
        .try_iter()?
        .map(|value| {
            let (source, installed, transfer): (
                Bound<'_, PyAny>,
                Bound<'_, PyAny>,
                Bound<'_, PyAny>,
            ) = value?.extract()?;
            Ok((
                buffer_id(&source)?,
                buffer_id(&installed)?,
                transfer_from_py(&transfer)?,
            ))
        })
        .collect()
}

fn buffer_set(buffers: &Bound<'_, PyAny>) -> PyResult<HashSet<BufferId>> {
    buffers
        .try_iter()?
        .map(|buffer| buffer_id(&buffer?))
        .collect()
}

impl KVCacheManager {
    pub(super) fn retirement_ready_for(
        &mut self,
        py: Python<'_>,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> PyResult<bool> {
        for completion in self
            .inner
            .retirement_completions(buffers, requests, retained)
        {
            if completion.done() {
                completion.owner.borrow(py).result(py, None)?;
            }
        }
        Ok(self.inner.retirement_ready(buffers, requests, retained))
    }
}
