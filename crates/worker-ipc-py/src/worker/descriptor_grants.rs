//! Python entry points for the native descriptor service.

use std::io;
use std::os::fd::{IntoRawFd, RawFd};

use pyo3::prelude::*;
use uniserve_worker::DescriptorGrants as NativeGrants;

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct DescriptorGrants {
    inner: NativeGrants,
}

#[pymethods]
impl DescriptorGrants {
    #[new]
    fn new(py: Python<'_>, endpoint: &str) -> PyResult<Self> {
        Ok(Self {
            inner: py.detach(|| NativeGrants::new(endpoint))?,
        })
    }

    fn register(&self, py: Python<'_>, publication: &str, descriptor: RawFd) -> PyResult<()> {
        py.detach(|| self.inner.register(publication, descriptor))?;
        Ok(())
    }

    fn release(&self, py: Python<'_>, publication: &str) {
        py.detach(|| self.inner.release(publication));
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.close())?;
        Ok(())
    }
}

impl Drop for DescriptorGrants {
    fn drop(&mut self) {
        Python::try_attach(|py| {
            if let Err(error) = py.detach(|| self.inner.close()) {
                PyErr::from(error).write_unraisable(py, None);
            }
        });
    }
}

#[pyfunction]
pub(crate) fn fetch_descriptor(
    py: Python<'_>,
    endpoint: &str,
    publication: &str,
) -> PyResult<RawFd> {
    py.detach(|| uniserve_worker::fetch_descriptor(endpoint, publication))
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
