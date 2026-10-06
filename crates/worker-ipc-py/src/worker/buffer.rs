//! PyTorch arenas and numerical views for the native physical buffer pool.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker::{BufferBinding as NativeBufferBinding, BufferPool as NativeBufferPool};
use uniserve_worker_ipc::BufferAllocation;

use super::error::{invalid, invariant, native_error};
use super::protocol::buffer_id;

/// A numerical view retaining the allocation of its native physical binding.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BufferBinding {
    pub(crate) binding: Arc<NativeBufferBinding>,
    #[pyo3(get)]
    buffer: Py<PyAny>,
    #[pyo3(get)]
    pub(crate) tensor: Py<PyAny>,
}

#[pymethods]
impl BufferBinding {
    #[getter]
    fn physical_offset(&self) -> u64 {
        self.binding.physical_offset()
    }

    #[getter]
    fn physical_bytes(&self) -> u64 {
        self.binding.physical_bytes()
    }

    #[getter]
    fn device_name(&self) -> &str {
        self.binding.device()
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.buffer)?;
        visit.call(&self.tensor)
    }
}

/// Fixed byte arenas with pool-issued views and nonoverlapping ranges.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BufferPool {
    #[pyo3(get)]
    devices: Py<PyTuple>,
    state: Mutex<NativeBufferPool<Py<PyAny>>>,
}

#[pymethods]
impl BufferPool {
    #[new]
    #[pyo3(signature = (*, byte_capacity, devices, compact=false))]
    fn new(
        py: Python<'_>,
        byte_capacity: i64,
        devices: Vec<Bound<'_, PyAny>>,
        compact: bool,
    ) -> PyResult<Self> {
        let byte_capacity = u64::try_from(byte_capacity).map_err(|_| {
            PyValueError::new_err("persistent buffer capacity must not be negative")
        })?;
        let canonical = py
            .import("uniserve.runtime.device")?
            .getattr("canonical_device")?;
        let torch = py.import("torch")?;
        let mut normalized = Vec::new();
        let mut arenas = HashMap::new();
        let devices = if devices.is_empty() {
            vec![torch.getattr("device")?.call1(("cpu",))?]
        } else {
            devices
        };

        for device in devices {
            let device = canonical.call1((device,))?;
            let name = device.str()?.to_str()?.to_owned();
            if arenas.contains_key(&name) {
                continue;
            }
            let empty = if device.getattr("type")?.extract::<String>()? == "cuda" {
                // CUDA peers borrow these VMM allocations directly, so retain
                // the established peer-storage allocator and its tensor owner.
                py.import("uniserve_kernels.peer_storage")?
                    .getattr("empty")?
            } else {
                torch.getattr("empty")?
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", torch.getattr("uint8")?)?;
            kwargs.set_item("device", &device)?;
            let tensor = empty.call(((byte_capacity,),), Some(&kwargs))?.unbind();

            arenas.insert(name, tensor);
            normalized.push(device);
        }

        Ok(Self {
            devices: PyTuple::new(py, normalized)?.unbind(),
            state: Mutex::new(NativeBufferPool::new(byte_capacity, compact, arenas)),
        })
    }

    #[getter]
    pub(crate) fn byte_capacity(&self, py: Python<'_>) -> PyResult<u64> {
        Ok(self.lock(py)?.byte_capacity())
    }

    #[getter]
    fn compact(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.lock(py)?.compact())
    }

    /// Reserve the physical range and construct its numerical view under the
    /// same owner lock. Failed view creation returns the range before retry.
    #[pyo3(signature = (reference, allocation, *, device, dtype, shape))]
    pub(crate) fn bind(
        &self,
        py: Python<'_>,
        reference: &Bound<'_, PyAny>,
        allocation: &Bound<'_, PyAny>,
        device: &Bound<'_, PyAny>,
        dtype: &Bound<'_, PyAny>,
        shape: Vec<i64>,
    ) -> PyResult<Py<BufferBinding>> {
        let device = py
            .import("uniserve.runtime.device")?
            .getattr("canonical_device")?
            .call1((device,))?;
        let device_name = device.str()?.to_str()?.to_owned();
        let buffer = allocation.getattr("buffer")?;
        let id = buffer_id(&buffer)?;
        if id != buffer_id(&reference.getattr("buffer_id")?)? {
            return Err(invalid(py, "buffer allocation does not name its output"));
        }

        let element_bytes = dtype.getattr("itemsize")?.extract::<u64>()?;
        let required = shape
            .iter()
            .try_fold(element_bytes, |bytes, &dim| {
                u64::try_from(dim)
                    .ok()
                    .filter(|&dim| dim > 0)
                    .and_then(|dim| bytes.checked_mul(dim))
            })
            .filter(|&bytes| bytes > 0)
            .ok_or_else(|| invalid(py, "buffer tensor shape has an invalid byte extent"))?;
        let allocation = BufferAllocation {
            buffer: id,
            bytes: allocation.getattr("bytes")?.extract()?,
            offset: allocation.getattr("offset")?.extract()?,
        };

        let mut state = self.lock(py)?;
        let (binding, arena) = state
            .bind(allocation, &device_name, required, element_bytes)
            .map_err(|error| native_error(py, error))?;
        let created = (|| {
            // Only the tensor's own bytes are viewed. Padding stays reserved
            // until the store returns this binding after its readers retire.
            let tensor = arena
                .bind(py)
                .call_method1("narrow", (0, binding.physical_offset(), required))?
                .call_method1("view", (dtype,))?
                .call_method1("reshape", (PyTuple::new(py, shape)?,))?
                .unbind();
            Py::new(
                py,
                BufferBinding {
                    binding: Arc::clone(&binding),
                    buffer: buffer.unbind(),
                    tensor,
                },
            )
        })();

        if created.is_err() {
            state
                .release(&binding)
                .map_err(|error| native_error(py, error))?;
        }
        created
    }

    /// Release the current pool-issued binding after physical uses retire.
    pub(crate) fn release(
        &self,
        py: Python<'_>,
        binding: &Bound<'_, BufferBinding>,
    ) -> PyResult<()> {
        self.release_binding(py, &binding.get().binding)
    }

    pub(super) fn close(&self, py: Python<'_>) -> PyResult<()> {
        let arenas = self.lock(py)?.close();
        // Tensor destruction can enter Python, so it follows unlocking.
        drop(arenas);
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.devices)?;
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            // An executing method retains the pool while it holds this lock.
            // GC must not wait for a thread that may need the interpreter.
            Err(TryLockError::WouldBlock) => return Ok(()),
        };

        for arena in state.backings() {
            visit.call(arena)?;
        }
        Ok(())
    }
}

impl BufferPool {
    pub(crate) fn release_binding(
        &self,
        py: Python<'_>,
        binding: &Arc<NativeBufferBinding>,
    ) -> PyResult<()> {
        self.lock(py)?
            .release(binding)
            .map_err(|error| native_error(py, error))
    }

    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, NativeBufferPool<Py<PyAny>>>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "buffer pool allocation lock is poisoned"))
    }
}
