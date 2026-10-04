//! Physical backing for scheduler-placed buffers.
//!
//! Placement and overlap checks run in Rust. PyTorch creates the numerical
//! views, retaining each arena's allocation. Consumers must finish device work
//! and transport reads before returning a binding to this pool.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, TryLockError};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::BufferId;

use crate::error::{invalid, invariant};
use crate::protocol::buffer_id;

/// A pool-issued tensor view of one physical byte range.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BufferBinding {
    id: BufferId,
    #[pyo3(get)]
    buffer: Py<PyAny>,
    #[pyo3(get)]
    physical_offset: u64,
    #[pyo3(get)]
    physical_bytes: u64,
    #[pyo3(get)]
    binding_id: u64,
    #[pyo3(get)]
    device_name: String,
    #[pyo3(get)]
    tensor: Py<PyAny>,
}

#[pymethods]
impl BufferBinding {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.buffer)?;
        visit.call(&self.tensor)
    }
}

struct Arena {
    tensor: Py<PyAny>,
    active: HashMap<BufferId, Py<BufferBinding>>,
}

/// Allocation state shared with asynchronous retirement callbacks.
struct PoolState {
    arenas: HashMap<String, Arena>,
    next_binding_id: u64,
}

/// Fixed byte arenas with generation-safe release and nonoverlapping bindings.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct BufferPool {
    #[pyo3(get)]
    byte_capacity: u64,
    #[pyo3(get)]
    compact: bool,
    #[pyo3(get)]
    devices: Py<PyTuple>,
    state: Mutex<PoolState>,
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
        if byte_capacity < 0 {
            return Err(PyValueError::new_err(
                "persistent buffer capacity must not be negative",
            ));
        }
        let byte_capacity = byte_capacity as u64;
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
            arenas.insert(
                name,
                Arena {
                    tensor,
                    active: HashMap::new(),
                },
            );
            normalized.push(device);
        }
        Ok(Self {
            byte_capacity,
            compact,
            devices: PyTuple::new(py, normalized)?.unbind(),
            state: Mutex::new(PoolState {
                arenas,
                next_binding_id: 1,
            }),
        })
    }

    /// Bind a scheduler allocation to a typed view. Compact pools use the
    /// first aligned physical gap; other pools preserve the scheduler offset.
    #[pyo3(signature = (reference, allocation, *, device, dtype, shape))]
    fn bind(
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
        let allocation_bytes = allocation.getattr("bytes")?.extract::<u64>()?;
        let offset = allocation.getattr("offset")?.extract::<u64>()?;
        if required > allocation_bytes {
            return Err(invalid(
                py,
                "buffer allocation is smaller than its output tensor",
            ));
        }
        if !offset.is_multiple_of(element_bytes) {
            return Err(invalid(
                py,
                "buffer allocation is not aligned for its output dtype",
            ));
        }
        let extent = if self.compact {
            align(required).ok_or_else(|| invalid(py, "buffer allocation byte extent overflows"))?
        } else {
            allocation_bytes
        };

        let mut state = self.lock(py)?;
        let arena = state
            .arenas
            .get(&device_name)
            .ok_or_else(|| invalid(py, "buffer allocation names an undeclared worker device"))?;
        if arena.active.contains_key(&id) {
            return Err(invalid(py, "buffer allocation is already bound"));
        }
        let start = if self.compact {
            self.compact_offset(py, arena, extent)?
        } else {
            offset
        };
        let end = start
            .checked_add(extent)
            .filter(|&end| end <= self.byte_capacity)
            .ok_or_else(|| invalid(py, "buffer allocation exceeds the worker buffer pool"))?;
        for active in arena.active.values() {
            let active = active.get();
            if start < active.physical_offset + active.physical_bytes
                && active.physical_offset < end
            {
                return Err(invalid(
                    py,
                    "buffer allocation overlaps a live worker buffer",
                ));
            }
        }

        // The tensor addresses only its own bytes. Any padding remains reserved
        // in the active range until its binding is released.
        let tensor = arena
            .tensor
            .bind(py)
            .call_method1("narrow", (0, start, required))?
            .call_method1("view", (dtype,))?
            .call_method1("reshape", (PyTuple::new(py, shape)?,))?
            .unbind();
        let binding_id = state.next_binding_id;
        let next_binding_id = binding_id
            .checked_add(1)
            .ok_or_else(|| invariant(py, "buffer binding generation exhausted"))?;
        let binding = Py::new(
            py,
            BufferBinding {
                id,
                buffer: buffer.unbind(),
                physical_offset: start,
                physical_bytes: extent,
                binding_id,
                device_name: device_name.clone(),
                tensor,
            },
        )?;
        state
            .arenas
            .get_mut(&device_name)
            .ok_or_else(|| invariant(py, "buffer arena disappeared during allocation"))?
            .active
            .insert(id, binding.clone_ref(py));
        state.next_binding_id = next_binding_id;
        Ok(binding)
    }

    /// Release the exact pool-issued binding after every use has retired.
    /// Stale bindings and bindings issued by another pool are rejected.
    fn release(&self, py: Python<'_>, binding: &Bound<'_, BufferBinding>) -> PyResult<()> {
        let value = binding.get();
        let mut state = self.lock(py)?;
        let arena = state
            .arenas
            .get_mut(&value.device_name)
            .ok_or_else(|| invariant(py, "stale persistent buffer binding"))?;
        if !arena
            .active
            .get(&value.id)
            .is_some_and(|active| active.bind(py).is(binding))
        {
            return Err(invariant(py, "stale persistent buffer binding"));
        }
        let released = arena.active.remove(&value.id);
        drop(state);
        drop(released);
        Ok(())
    }

    /// Drop arenas after their device uses and remote readers have retired.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let arenas = std::mem::take(&mut self.lock(py)?.arenas);
        // Tensor destruction may enter Python. Never hold the allocation lock
        // while destroying a pool's numerical storage.
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
        for arena in state.arenas.values() {
            visit.call(&arena.tensor)?;
            for binding in arena.active.values() {
                visit.call(binding)?;
            }
        }
        Ok(())
    }
}

impl BufferPool {
    fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, PoolState>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| invariant(py, "buffer pool allocation lock is poisoned"))
    }

    fn compact_offset(&self, py: Python<'_>, arena: &Arena, extent: u64) -> PyResult<u64> {
        let mut spans: Vec<_> = arena
            .active
            .values()
            .map(|binding| {
                let binding = binding.get();
                (binding.physical_offset, binding.physical_bytes)
            })
            .collect();
        spans.sort_unstable();
        let mut cursor = 0;
        for &(offset, bytes) in &spans {
            if let Some(start) = align(cursor)
                && start.checked_add(extent).is_some_and(|end| end <= offset)
            {
                return Ok(start);
            }
            cursor = cursor.max(offset + bytes);
        }
        let start = align(cursor).filter(|&start| {
            start
                .checked_add(extent)
                .is_some_and(|end| end <= self.byte_capacity)
        });
        start.ok_or_else(|| {
            invalid(
                py,
                format!(
                    "physical buffer allocation exceeds the worker buffer pool: \
                     {extent} bytes requested from {} bytes with live spans {spans:?}",
                    self.byte_capacity,
                ),
            )
        })
    }
}

/// Match the 256-byte product alignment used by startup capacity accounting.
fn align(bytes: u64) -> Option<u64> {
    bytes.checked_add(255).map(|bytes| bytes / 256 * 256)
}
