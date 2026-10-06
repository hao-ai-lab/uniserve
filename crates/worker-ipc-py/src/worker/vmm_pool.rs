//! Numerical tensor views backed by the native export pool.

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::PyBytes;
use uniserve_worker::vmm_pool::{PoolChunk as NativeChunk, VmmPool as NativePool};

use super::descriptor_grants::DescriptorGrants;
use super::error::{invariant, native_error};
use super::events::{CUDAEvent, current_stream};

pyo3::create_exception!(
    uniserve_worker._uniserve_ipc,
    PoolExhaustedError,
    PyRuntimeError
);

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct PoolChunk {
    inner: Arc<NativeChunk>,
    #[pyo3(get)]
    storage: Py<PyAny>,
    #[pyo3(get)]
    acknowledgments: Py<PyAny>,
}

#[pymethods]
impl PoolChunk {
    #[getter]
    fn offset(&self) -> usize {
        self.inner.offset
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes
    }

    #[getter]
    fn payload_offset(&self) -> usize {
        self.inner.payload_offset()
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.storage)?;
        visit.call(&self.acknowledgments)
    }
}

#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct VmmPool {
    inner: Mutex<NativePool<PoolStorage>>,
    device: Py<PyAny>,
}

struct PoolStorage {
    mapping: Py<PyAny>,
    handle: Vec<u8>,
    _descriptor: Option<OwnedFd>,
}

#[pymethods]
impl VmmPool {
    #[new]
    #[pyo3(signature = (device, *, capacity_bytes))]
    fn new(py: Python<'_>, device: &Bound<'_, PyAny>, capacity_bytes: usize) -> PyResult<Self> {
        if capacity_bytes == 0 {
            return Err(PyValueError::new_err(
                "a VMM pool needs a positive capacity",
            ));
        }
        let device = py
            .import("uniserve.runtime.device")?
            .getattr("canonical_device")?
            .call1((device,))?;
        let (storage, handle): (Py<PyAny>, Vec<u8>) = backend(py)?
            .call_method1("_allocate", (&device, capacity_bytes))?
            .extract()?;
        let address = storage.bind(py).call_method0("data_ptr")?.extract()?;
        let capacity = storage.bind(py).call_method0("numel")?.extract()?;
        let index = device.getattr("index")?.extract()?;

        // The tensor retains the physical allocation. POSIX export also
        // creates a descriptor, whose ownership follows this pool separately.
        let descriptor = <[u8; size_of::<i32>()]>::try_from(handle.as_slice())
            .ok()
            .map(|bytes| unsafe { OwnedFd::from_raw_fd(i32::from_ne_bytes(bytes)) });
        let storage = PoolStorage {
            mapping: storage,
            handle,
            _descriptor: descriptor,
        };
        let inner = py
            .detach(|| unsafe { NativePool::new(storage, index, address, capacity) })
            .map_err(|error| native_error(py, error))?;
        Ok(Self {
            inner: Mutex::new(inner),
            device: device.unbind(),
        })
    }

    #[getter]
    fn handle<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let pool = self.lock(py)?;
        let storage = pool.backing().map_err(|error| native_error(py, error))?;
        Ok(PyBytes::new(py, &storage.handle))
    }

    #[getter]
    fn mapping(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.lock(py)?
            .backing()
            .map(|storage| storage.mapping.clone_ref(py))
            .map_err(|error| native_error(py, error))
    }

    #[getter]
    fn capacity(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.capacity())
    }

    fn reserve(&self, py: Python<'_>, nbytes: usize) -> PyResult<PoolChunk> {
        let stream = current_stream(py, self.device.bind(py))?;
        let (chunk, storage) = {
            let mut pool = self.lock(py)?;
            let pool = &mut *pool;
            let chunk = py
                .detach(|| pool.reserve(nbytes, stream))
                .map_err(|error| native_error(py, error))?
                .ok_or_else(|| {
                    PoolExhaustedError::new_err(format!(
                        "product of {nbytes} bytes does not fit the {} byte VMM pool",
                        pool.capacity()
                    ))
                })?;
            let storage = pool
                .backing()
                .map_err(|error| native_error(py, error))?
                .mapping
                .clone_ref(py);
            (chunk, storage)
        };

        // Tensor construction can call foreign allocators; do it outside the
        // pool lock. The reserved chunk retains its byte range meanwhile.
        let views = backend(py)?
            .call_method1("_views", (&storage, chunk.offset, nbytes))
            .and_then(|value| value.extract::<(Py<PyAny>, Py<PyAny>)>());
        match views {
            Ok((storage, acknowledgments)) => Ok(PoolChunk {
                inner: chunk,
                storage,
                acknowledgments,
            }),
            Err(error) => {
                py.detach(|| uniserve_worker::cuda::Stream::borrowed(stream).wait())
                    .map_err(PyRuntimeError::new_err)?;
                self.lock(py)?
                    .release(&chunk)
                    .map_err(|error| native_error(py, error))?;
                Err(error)
            }
        }
    }

    fn release(&self, py: Python<'_>, chunk: &PoolChunk) -> PyResult<()> {
        self.lock(py)?
            .release(&chunk.inner)
            .map_err(|error| native_error(py, error))
    }

    #[pyo3(signature = (chunk, consumers, producer, grants=None, export_id=""))]
    pub(super) fn retire(
        &self,
        py: Python<'_>,
        chunk: &PoolChunk,
        consumers: Vec<usize>,
        producer: &CUDAEvent,
        grants: Option<&DescriptorGrants>,
        export_id: &str,
    ) -> PyResult<()> {
        let grant = grants.map(|grants| (Arc::clone(&grants.inner), export_id.to_owned()));
        let mut pool = self.lock(py)?;
        let pool = &mut *pool;
        py.detach(|| pool.retire(&chunk.inner, &consumers, &producer.inner, grant))
            .map_err(|error| native_error(py, error))
    }

    fn reap(&self, py: Python<'_>) -> PyResult<()> {
        let mut pool = self.lock(py)?;
        let pool = &mut *pool;
        py.detach(|| pool.reap())
            .map_err(|error| native_error(py, error))
    }

    fn awaiting_acknowledgment(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.awaiting_acknowledgment())
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let backing = {
            let mut pool = self.lock(py)?;
            let pool = &mut *pool;
            py.detach(|| pool.close())
                .map_err(|error| native_error(py, error))?
        };
        drop(backing);
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.device)?;
        let pool = match self.inner.try_lock() {
            Ok(pool) => pool,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => return Ok(()),
        };
        if let Ok(storage) = pool.backing() {
            visit.call(&storage.mapping)?;
        }
        Ok(())
    }

    fn __clear__(&self, py: Python<'_>) {
        if let Err(error) = self.close(py) {
            error.write_unraisable(py, None);
        }
    }
}

impl VmmPool {
    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NativePool<PoolStorage>>> {
        self.inner
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "VMM pool lock is poisoned"))
    }
}

impl Drop for VmmPool {
    fn drop(&mut self) {
        let pool = self
            .inner
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        Python::try_attach(|py| {
            if let Err(error) = py.detach(|| pool.close()) {
                native_error(py, error).write_unraisable(py, None);
            }
        });
    }
}

fn backend(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyModule>> {
    py.import("uniserve_worker.transport.vmm_pool")
}
