//! Python entry points for the native descriptor service.

use std::io;
use std::mem::ManuallyDrop;
use std::os::fd::{IntoRawFd, RawFd};
use std::sync::Arc;

use pyo3::prelude::*;
use uniserve_worker::DescriptorGrants as NativeGrants;

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct DescriptorGrants {
    pub(super) inner: ManuallyDrop<Arc<NativeGrants>>,
}

#[pymethods]
impl DescriptorGrants {
    #[new]
    pub(in crate::worker) fn new(py: Python<'_>, endpoint: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ManuallyDrop::new(Arc::new(py.detach(|| NativeGrants::new(endpoint))?)),
        })
    }

    pub(in crate::worker) fn register(
        &self,
        py: Python<'_>,
        export: &str,
        descriptor: RawFd,
    ) -> PyResult<()> {
        py.detach(|| self.inner.register(export, descriptor))?;
        Ok(())
    }

    pub(in crate::worker) fn release(&self, py: Python<'_>, export: &str) {
        py.detach(|| self.inner.release(export));
    }

    pub(in crate::worker) fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.close())?;
        Ok(())
    }
}

impl Drop for DescriptorGrants {
    fn drop(&mut self) {
        // A retired chunk can outlive this wrapper. Release only this owner,
        // and let the final native owner join the service outside the GIL.
        // SAFETY: Drop takes this field exactly once; it is not dropped again.
        let inner = unsafe { ManuallyDrop::take(&mut self.inner) };
        Python::try_attach(|py| py.detach(|| drop(inner)));
    }
}

#[pyfunction]
pub(crate) fn fetch_descriptor(py: Python<'_>, endpoint: &str, export: &str) -> PyResult<RawFd> {
    py.detach(|| uniserve_worker::fetch_descriptor(endpoint, export))
        .map(IntoRawFd::into_raw_fd)
        .map_err(|error| match error.kind() {
            io::ErrorKind::NotFound
            | io::ErrorKind::ConnectionRefused
            | io::ErrorKind::InvalidInput => {
                super::error::invalid(py, format!("CUDA descriptor grant failed: {error}"))
            }
            _ => PyErr::from(error),
        })
}
