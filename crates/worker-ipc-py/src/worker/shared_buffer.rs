//! Borrowed Python views of native shared host buffers.

use std::os::fd::IntoRawFd;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use uniserve_worker::{SharedBuffer as NativeBuffer, SharedMapping, SharedRead as NativeRead};

use super::error::{invariant, native_error, resource};

/// Hand an open POSIX descriptor to a codec's ordinary file interface.
#[pyfunction]
pub(super) fn open_shared_memory(py: Python<'_>, name: &str) -> PyResult<i32> {
    py.detach(|| uniserve_core::SharedMemory::open(name, false).map(IntoRawFd::into_raw_fd))
        .map_err(PyErr::from)
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct SharedBuffer {
    pub(super) inner: Mutex<NativeBuffer>,
}

#[pymethods]
impl SharedBuffer {
    #[new]
    #[pyo3(signature = (nbytes, consumers, device=None))]
    pub(in crate::worker) fn new(
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
    pub(in crate::worker) fn name(&self, py: Python<'_>) -> PyResult<String> {
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

    pub(in crate::worker) fn begin_copy(&self, py: Python<'_>) -> PyResult<()> {
        self.lock(py)?.begin_copy();
        Ok(())
    }

    pub(in crate::worker) fn mark_ready(&self, py: Python<'_>) -> PyResult<()> {
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

    pub(in crate::worker) fn close(&self, py: Python<'_>) -> PyResult<()> {
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
        let size = mapping.size();
        unsafe { export_mapping(slf.into_any(), mapping, 0, size, view, flags) }
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

/// A shared read claims and waits without Python; numerical consumers borrow
/// only its payload range through the ordinary buffer protocol.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct SharedRead {
    inner: Mutex<NativeRead>,
}

#[pymethods]
impl SharedRead {
    #[new]
    #[pyo3(signature = (name, nbytes, slot, *, offset=0, timeout=120.0))]
    pub(in crate::worker) fn new(
        py: Python<'_>,
        name: &str,
        nbytes: usize,
        slot: usize,
        offset: usize,
        timeout: f64,
    ) -> PyResult<Self> {
        let timeout = Duration::try_from_secs_f64(timeout).map_err(|_| {
            PyValueError::new_err("readiness timeout must be finite and nonnegative")
        })?;
        Self::open(py, name, nbytes, slot, offset, timeout, || Ok(()))
    }

    #[getter]
    fn nbytes(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.nbytes())
    }

    fn truncate(&self, py: Python<'_>, nbytes: usize) -> PyResult<()> {
        self.lock(py)?
            .truncate(nbytes)
            .map_err(|error| native_error(py, error))
    }

    pub(super) fn release(&self, py: Python<'_>) -> PyResult<()> {
        self.lock(py)?.release();
        Ok(())
    }

    fn __enter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _kind: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.release(py)
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::ffi::c_int,
    ) -> PyResult<()> {
        let read = slf.get().lock(slf.py())?;
        let mapping = read
            .mapping()
            .map_err(|error| native_error(slf.py(), error))?;
        let offset = read.offset();
        let size = read.nbytes();
        drop(read);

        unsafe { export_mapping(slf.into_any(), mapping, offset, size, view, flags) }
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        // SAFETY: export_mapping retained one reference in this view.
        drop(unsafe { Arc::from_raw((*view).internal as *const SharedMapping) });
    }
}

impl SharedRead {
    /// Open a borrowed payload while observing the caller's cancellation.
    /// Waiting and failed-read cleanup do not require the interpreter.
    pub(super) fn open(
        py: Python<'_>,
        name: &str,
        nbytes: usize,
        slot: usize,
        offset: usize,
        timeout: Duration,
        check: impl FnMut() -> uniserve_worker::Result<()> + Send,
    ) -> PyResult<Self> {
        let inner = py
            .detach(|| {
                let read = NativeRead::open(name, offset, nbytes, slot)?;
                read.wait(timeout, check)?;
                Ok::<_, uniserve_worker::Error>(read)
            })
            .map_err(|error| native_error(py, error))?;

        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NativeRead>> {
        self.inner
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "shared read lock is poisoned"))
    }
}

/// Export an already bounded range while retaining both its Python owner and
/// mapping. A view remains valid even if explicit release unlinks its source.
unsafe fn export_mapping(
    owner: Bound<'_, PyAny>,
    mapping: Arc<SharedMapping>,
    offset: usize,
    size: usize,
    view: *mut pyo3::ffi::Py_buffer,
    flags: std::ffi::c_int,
) -> PyResult<()> {
    unsafe {
        if pyo3::ffi::PyBuffer_FillInfo(
            view,
            owner.as_ptr(),
            (mapping.address() + offset) as *mut _,
            size as isize,
            0,
            flags,
        ) != 0
        {
            return Err(PyErr::fetch(owner.py()));
        }
        (*view).internal = Arc::into_raw(mapping) as *mut _;
    }
    Ok(())
}
