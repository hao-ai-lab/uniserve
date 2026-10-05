//! Borrowed Python views of native shared host buffers.

use std::sync::{Arc, Mutex, MutexGuard};

use pyo3::exceptions::PyBufferError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use uniserve_worker::{SharedBuffer as NativeBuffer, SharedMapping};

use super::error::{invariant, resource};

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct SharedBuffer {
    inner: Mutex<NativeBuffer>,
}

#[pymethods]
impl SharedBuffer {
    #[new]
    #[pyo3(signature = (nbytes, consumers, device=None))]
    fn new(
        py: Python<'_>,
        nbytes: usize,
        consumers: Vec<usize>,
        device: Option<(i32, usize)>,
    ) -> PyResult<Self> {
        let inner = py
            .detach(|| NativeBuffer::new(nbytes, consumers, device))
            .map_err(|error| resource(py, error))?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[getter]
    fn name(&self, py: Python<'_>) -> PyResult<String> {
        self.lock(py)?
            .name()
            .map(str::to_owned)
            .map_err(|error| resource(py, error))
    }

    #[getter]
    fn nbytes(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.nbytes())
    }

    #[getter]
    fn is_cuda(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.is_cuda())
    }

    fn begin_copy(&self, py: Python<'_>) -> PyResult<()> {
        self.lock(py)?.begin_copy();
        Ok(())
    }

    fn mark_ready(&self, py: Python<'_>) -> PyResult<()> {
        let mut buffer = self.lock(py)?;
        let buffer = &mut *buffer;
        py.detach(|| buffer.mark_ready())
            .map_err(|error| resource(py, error))
    }

    fn settled(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.settled())
    }

    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        let buffer = self.lock(py)?;
        let buffer = &*buffer;
        py.detach(|| buffer.synchronize())
            .map_err(|error| resource(py, error))
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let mut buffer = self.lock(py)?;
        let buffer = &mut *buffer;
        py.detach(|| buffer.close())
            .map_err(|error| resource(py, error))
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::ffi::c_int,
    ) -> PyResult<()> {
        let mapping = slf
            .get()
            .lock(slf.py())?
            .mapping()
            .map_err(PyBufferError::new_err)?;
        // Each exported view keeps the mapping alive independently of close.
        // CUDA registration remains with the producer, whose close drains DMA.
        unsafe {
            if pyo3::ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                mapping.address() as *mut _,
                mapping.size() as isize,
                0,
                flags,
            ) != 0
            {
                return Err(PyErr::fetch(slf.py()));
            }
            (*view).internal = Arc::into_raw(mapping) as *mut _;
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        // SAFETY: __getbuffer__ placed one owning Arc in this view.
        drop(unsafe { Arc::from_raw((*view).internal as *const SharedMapping) });
    }
}

impl SharedBuffer {
    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NativeBuffer>> {
        self.inner
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "shared buffer lock is poisoned"))
    }
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        let buffer = self
            .inner
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        Python::try_attach(|py| {
            if let Err(error) = py.detach(|| buffer.close()) {
                resource(py, error).write_unraisable(py, None);
            }
        });
    }
}
