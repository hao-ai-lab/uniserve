//! Collective allocation and ownership of ordered CUDA peer mappings.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use uniserve_worker::{DescriptorGrants, fetch_descriptor};
use uuid::Uuid;

/// The handle and local allocation outlive every borrowed peer view.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct SymmetricStorage {
    #[pyo3(get)]
    coordinator: Py<PyAny>,
    #[pyo3(get)]
    pub(super) local: Py<PyAny>,
    #[pyo3(get)]
    pub(super) peers: Py<PyTuple>,
    #[pyo3(get)]
    handle: Py<PyAny>,
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get)]
    size: usize,
}

#[pymethods]
impl SymmetricStorage {
    /// Order peer access with the numerical group's stream-ordered gather.
    fn fence(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        output: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if input.getattr("shape")?.extract::<Vec<usize>>()? != [1]
            || output.getattr("shape")?.extract::<Vec<usize>>()? != [self.size]
        {
            return Err(PyValueError::new_err(
                "symmetric-storage fence buffers do not match group membership",
            ));
        }
        self.coordinator
            .call_method1(py, "all_gather_into", (output, input))?;
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.coordinator)?;
        visit.call(&self.local)?;
        visit.call(&self.peers)?;
        visit.call(&self.handle)
    }
}

/// Allocate one physical shard and map all shards in logical membership order.
#[pyfunction]
#[pyo3(signature = (group, shape, *, dtype))]
fn allocate_peer_tensor(
    py: Python<'_>,
    group: &Bound<'_, PyAny>,
    shape: &Bound<'_, PyTuple>,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let device = group.getattr("device")?;
    let backend = py.import("uniserve_kernels.peer_storage")?;
    let options = PyDict::new(py);
    options.set_item("dtype", dtype)?;
    options.set_item("device", &device)?;
    let allocation = backend.call_method("allocate", (shape,), Some(&options))?;
    let index = match device.getattr("index")?.extract::<Option<usize>>()? {
        Some(index) => index,
        None => py
            .import("torch.cuda")?
            .call_method0("current_device")?
            .extract()?,
    };
    let fabric = backend
        .call_method1("exports_fabric_handles", (index,))?
        .is_truthy()?;
    let exported: Vec<u8> = allocation.call_method0("export_handle")?.extract()?;
    let descriptor = if fabric {
        None
    } else {
        let bytes: [u8; size_of::<i32>()] = exported
            .as_slice()
            .try_into()
            .map_err(|_| PyValueError::new_err("CUDA descriptor export has an invalid size"))?;
        // CUDA exports an owned POSIX descriptor. Mapping takes its own physical
        // allocation reference; this export closes on every return path.
        Some(unsafe { OwnedFd::from_raw_fd(i32::from_ne_bytes(bytes)) })
    };
    let size: usize = group.getattr("size")?.extract()?;
    if size == 1 {
        return Ok(allocation
            .call_method1(
                "map_peers",
                (PyList::new(py, [PyBytes::new(py, &exported)])?,),
            )?
            .unbind());
    }

    let endpoint = descriptor
        .as_ref()
        .map(|_| Uuid::new_v4().simple().to_string())
        .unwrap_or_default();
    let grants = descriptor
        .as_ref()
        .map(|descriptor| {
            py.detach(|| {
                let grants = DescriptorGrants::new(&endpoint)?;
                grants.register(&endpoint, descriptor.as_raw_fd())?;
                Ok::<_, std::io::Error>(grants)
            })
        })
        .transpose()?;
    let result = (|| {
        let handle = if fabric {
            PyBytes::new(py, &exported).into_any()
        } else {
            endpoint.as_str().into_pyobject(py)?.into_any()
        };
        let records = gather_allocations(py, group, shape, dtype, &handle)?;
        let mut received = Vec::new();
        let handles = PyList::empty(py);
        for record in records {
            if fabric {
                handles.append(record)?;
            } else {
                let endpoint: String = record.extract()?;
                let descriptor = py.detach(|| fetch_descriptor(&endpoint, &endpoint))?;
                handles.append(PyBytes::new(py, &descriptor.as_raw_fd().to_ne_bytes()))?;
                received.push(descriptor);
            }
        }
        if !fabric {
            // All consumers now own descriptors. Revoke grants only after this
            // host exchange; CUDA import can proceed independently on each rank.
            let options = PyDict::new(py);
            options.set_item("group", group.call_method0("_require")?)?;
            py.import("torch.distributed")?
                .call_method("barrier", (), Some(&options))?;
        }
        Ok(allocation.call_method1("map_peers", (handles,))?.unbind())
    })();
    // Descriptor service threads do not execute Python. Join outside the GIL.
    py.detach(|| drop(grants));
    result
}

fn gather_allocations<'py>(
    py: Python<'py>,
    group: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyTuple>,
    dtype: &Bound<'py, PyAny>,
    handle: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let size: usize = group.getattr("size")?.extract()?;
    let hostname = py.import("socket")?.call_method0("gethostname")?;
    let dtype = dtype.str()?;
    let local = (&hostname, shape, &dtype, handle);
    let records = PyList::new(py, (0..size).map(|_| py.None()))?;
    let options = PyDict::new(py);
    options.set_item("group", group.call_method0("_require")?)?;
    py.import("torch.distributed")?.call_method(
        "all_gather_object",
        (&records, local),
        Some(&options),
    )?;
    let mut handles = Vec::with_capacity(size);
    for record in &records {
        if !record.get_item(0)?.eq(&hostname)?
            || !record.get_item(1)?.eq(shape)?
            || !record.get_item(2)?.eq(&dtype)?
        {
            return Err(PyValueError::new_err(
                "peer tensors require matching shapes on one host",
            ));
        }
        handles.push(record.get_item(3)?);
    }
    logical_order(group, handles)
}

/// Allocate VMM backing suitable for NCCL registration, without peer mapping.
#[pyfunction]
#[pyo3(signature = (shape, *, dtype, device))]
pub(super) fn allocate_collective_buffer(
    py: Python<'_>,
    shape: &Bound<'_, PyTuple>,
    dtype: &Bound<'_, PyAny>,
    device: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let storage = py.import("torch.distributed._symmetric_memory")?;
    if storage
        .call_method1("get_backend", (device,))?
        .extract::<Option<String>>()?
        .as_deref()
        != Some("NCCL")
    {
        storage.call_method1("set_backend", ("NCCL",))?;
    }
    let options = PyDict::new(py);
    options.set_item("dtype", dtype)?;
    options.set_item("device", device)?;
    Ok(storage
        .call_method("empty", (shape,), Some(&options))?
        .unbind())
}

/// Retain local allocation, backend handle and peer views as one lifetime.
#[pyfunction]
#[pyo3(signature = (group, shape, *, dtype))]
pub(super) fn allocate_symmetric_storage(
    py: Python<'_>,
    group: &Bound<'_, PyAny>,
    shape: &Bound<'_, PyTuple>,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<Py<SymmetricStorage>> {
    let device = group.getattr("device")?;
    let rank = group.getattr("rank")?.extract()?;
    let size = group.getattr("size")?.extract()?;
    let (local, peers, handle) = if size == 1 {
        let options = PyDict::new(py);
        options.set_item("dtype", dtype)?;
        options.set_item("device", &device)?;
        let local = py
            .import("torch")?
            .call_method("empty", (shape,), Some(&options))?;
        let peers = PyTuple::new(py, [&local])?;
        (local.unbind(), peers.unbind(), py.None())
    } else {
        let backend = group.call_method0("_require")?;
        let config: String = py
            .import("torch.distributed")?
            .call_method1("get_backend_config", (&backend,))?
            .extract()?;
        if !config.split(',').any(|part| part == "cuda:nccl") {
            return Err(PyRuntimeError::new_err(
                "symmetric peer storage requires the NCCL backend",
            ));
        }
        let local = allocate_collective_buffer(py, shape, dtype, &device)?;
        let handle = py
            .import("torch.distributed._symmetric_memory")?
            .call_method1("rendezvous", (&local, backend))?;
        let peers = (0..size)
            .map(|rank| handle.call_method1("get_buffer", (rank, shape, dtype)))
            .collect::<PyResult<Vec<_>>>()?;
        let peers = logical_order(group, peers)?;

        (local, PyTuple::new(py, peers)?.unbind(), handle.unbind())
    };
    Py::new(
        py,
        SymmetricStorage {
            coordinator: group.clone().unbind(),
            local,
            peers,
            handle,
            rank,
            size,
        },
    )
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SymmetricStorage>()?;
    module.add_function(wrap_pyfunction!(allocate_peer_tensor, module)?)?;
    module.add_function(wrap_pyfunction!(allocate_collective_buffer, module)?)?;
    module.add_function(wrap_pyfunction!(allocate_symmetric_storage, module)?)
}

fn logical_order<'py>(
    group: &Bound<'py, PyAny>,
    values: Vec<Bound<'py, PyAny>>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let order: Vec<usize> = group.getattr("backend_order")?.extract()?;
    let mut values: Vec<_> = order.into_iter().zip(values).collect();
    values.sort_by_key(|(rank, _)| *rank);
    Ok(values.into_iter().map(|(_, value)| value).collect())
}
