//! Native tensor locations shared by transport owners and numerical readers.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use uniserve_worker_ipc::{Locator as NativeLocator, TransferTransport, WorkerEndpoint};

use super::error::invalid;

/// A physical tensor view and the transport handle used to read it.
/// Native readers share the immutable description; Python receives field
/// views only when a numerical operation needs them.
#[derive(PartialEq, Eq)]
#[pyclass(frozen, eq, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct Locator {
    pub(crate) inner: Arc<NativeLocator>,
}

impl Locator {
    pub(crate) fn wrap(py: Python<'_>, value: NativeLocator) -> PyResult<Bound<'_, Self>> {
        Bound::new(
            py,
            Self {
                inner: Arc::new(value),
            },
        )
    }

    pub(super) fn endpoint(&self) -> &str {
        match &self.inner.transport {
            TransferTransport::Local { endpoint, .. }
            | TransferTransport::PosixShm { endpoint, .. }
            | TransferTransport::CudaVmm { endpoint, .. }
            | TransferTransport::Channel { endpoint, .. } => endpoint,
        }
    }
}

#[pymethods]
impl Locator {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        source: &Bound<'_, PyAny>,
        transport: &Bound<'_, PyAny>,
        nbytes: u64,
        dtype: String,
        shape: Vec<u64>,
        offset: Vec<u64>,
        device: String,
    ) -> PyResult<Self> {
        let py = source.py();
        let value = NativeLocator {
            source: endpoint_from_py(source)?,
            transport: transport_from_py(transport)?,
            nbytes,
            dtype,
            shape,
            offset,
            device,
        };
        value
            .validate()
            .map_err(|error| invalid(py, error.to_string()))?;

        Ok(Self {
            inner: Arc::new(value),
        })
    }

    #[getter]
    pub(super) fn backend(&self) -> &'static str {
        match &self.inner.transport {
            TransferTransport::Local { .. } => "local",
            TransferTransport::PosixShm { .. } => "shm",
            TransferTransport::CudaVmm { .. } => "cuda_vmm",
            TransferTransport::Channel { .. } => "channel",
        }
    }

    #[getter]
    pub(super) fn source<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        endpoint_to_py(py, &self.inner.source)
    }

    #[getter]
    fn transport<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (fields, name) = transport_fields(py, &self.inner.transport)?;
        types(py)?.getattr(name)?.call((), Some(&fields))
    }

    #[getter]
    fn nbytes(&self) -> u64 {
        self.inner.nbytes
    }

    #[getter]
    fn dtype(&self) -> &str {
        &self.inner.dtype
    }

    #[getter]
    fn device(&self) -> &str {
        &self.inner.device
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.shape)
    }

    #[getter]
    fn offset<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.offset)
    }

    /// Decode the flattened mapping used by the worker's Python IPC records.
    #[staticmethod]
    #[pyo3(signature = (value, where_="transfer locator"))]
    fn from_mapping(value: &Bound<'_, PyAny>, where_: &str) -> PyResult<Self> {
        let py = value.py();
        let value = crate::convert::transfer_locator_from_py(value)
            .ok_or_else(|| invalid(py, format!("{where_} has invalid tensor coordinates")))?;
        value
            .validate()
            .map_err(|error| invalid(py, error.to_string()))?;

        Ok(Self {
            inner: Arc::new(value),
        })
    }

    /// Encode tensor metadata and transport fields for Python IPC consumers.
    fn to_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (fields, _) = transport_fields(py, &self.inner.transport)?;
        let tag = if self.backend() == "shm" {
            "posix_shm"
        } else {
            self.backend()
        };
        fields.set_item("transport", tag)?;
        fields.set_item("source", pythonize::pythonize(py, &self.inner.source)?)?;
        fields.set_item("nbytes", self.inner.nbytes)?;
        fields.set_item("dtype", &self.inner.dtype)?;
        fields.set_item("device", &self.inner.device)?;
        fields.set_item("shape", &self.inner.shape)?;
        fields.set_item("offset", &self.inner.offset)?;

        // Python descriptors expose immutable tuples; the IPC mapping uses lists.
        for (name, value) in fields.iter() {
            if let Ok(values) = value.cast::<PyTuple>() {
                fields.set_item(name, PyList::new(py, values.iter())?)?;
            }
        }
        Ok(fields)
    }
}

pub(super) fn endpoint_from_py(value: &Bound<'_, PyAny>) -> PyResult<WorkerEndpoint> {
    Ok(WorkerEndpoint {
        worker_id: value.getattr("worker_id")?.extract()?,
        rank: value.getattr("rank")?.extract()?,
        node: value.getattr("node")?.extract()?,
        address_space: value.getattr("address_space")?.extract()?,
        incarnation: value.getattr("incarnation")?.extract()?,
    })
}

fn transport_from_py(value: &Bound<'_, PyAny>) -> PyResult<TransferTransport> {
    let py = value.py();
    let module = types(py)?;
    let endpoint = value.getattr("endpoint")?.extract()?;

    if value.is_instance(&module.getattr("LocalTransfer")?)? {
        Ok(TransferTransport::Local {
            endpoint,
            key: value.getattr("key")?.extract()?,
        })
    } else if value.is_instance(&module.getattr("PosixShmTransfer")?)? {
        Ok(TransferTransport::PosixShm {
            endpoint,
            name: value.getattr("name")?.extract()?,
        })
    } else if value.is_instance(&module.getattr("ChannelTransfer")?)? {
        Ok(TransferTransport::Channel {
            endpoint,
            payload: value.getattr("payload")?.extract()?,
        })
    } else if value.is_instance(&module.getattr("CudaVmmTransfer")?)? {
        Ok(TransferTransport::CudaVmm {
            endpoint,
            export_id: value.getattr("export_id")?.extract()?,
            storage_size_bytes: value.getattr("storage_size_bytes")?.extract()?,
            storage_offsets_bytes: value.getattr("storage_offsets_bytes")?.extract()?,
            span_lengths: value.getattr("span_lengths")?.extract()?,
            span_counts: value.getattr("span_counts")?.extract()?,
            tensor_stride: value.getattr("tensor_stride")?.extract()?,
            ready_event_handle: value.getattr("ready_event_handle")?.extract()?,
            allocation_handle: value.getattr("allocation_handle")?.extract()?,
            acknowledgment_offset: value.getattr("acknowledgment_offset")?.extract()?,
        })
    } else {
        Err(invalid(py, "tensor location names an unknown transport"))
    }
}

fn types(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.protocol.transfer")
}

pub(super) fn endpoint_to_py<'py>(
    py: Python<'py>,
    value: &WorkerEndpoint,
) -> PyResult<Bound<'py, PyAny>> {
    let source = PyDict::new(py);
    source.set_item("worker_id", &value.worker_id)?;
    source.set_item("rank", value.rank)?;
    source.set_item("node", &value.node)?;
    source.set_item("address_space", &value.address_space)?;
    source.set_item("incarnation", &value.incarnation)?;
    types(py)?
        .getattr("WorkerEndpoint")?
        .call((), Some(&source))
}

fn transport_fields<'py>(
    py: Python<'py>,
    value: &TransferTransport,
) -> PyResult<(Bound<'py, PyDict>, &'static str)> {
    let handle = PyDict::new(py);
    let name = match value {
        TransferTransport::Local { endpoint, key } => {
            handle.set_item("endpoint", endpoint.as_str())?;
            handle.set_item("key", key)?;
            "LocalTransfer"
        }
        TransferTransport::PosixShm { endpoint, name } => {
            handle.set_item("endpoint", endpoint.as_str())?;
            handle.set_item("name", name.as_str())?;
            "PosixShmTransfer"
        }
        TransferTransport::CudaVmm {
            endpoint,
            export_id,
            storage_size_bytes,
            storage_offsets_bytes,
            span_lengths,
            span_counts,
            tensor_stride,
            ready_event_handle,
            allocation_handle,
            acknowledgment_offset,
        } => {
            handle.set_item("endpoint", endpoint.as_str())?;
            handle.set_item("export_id", export_id.as_str())?;
            handle.set_item("storage_size_bytes", storage_size_bytes)?;
            handle.set_item(
                "storage_offsets_bytes",
                PyTuple::new(py, storage_offsets_bytes)?,
            )?;
            handle.set_item("span_lengths", PyTuple::new(py, span_lengths)?)?;
            handle.set_item("span_counts", PyTuple::new(py, span_counts)?)?;
            handle.set_item("tensor_stride", PyTuple::new(py, tensor_stride)?)?;
            handle.set_item("ready_event_handle", PyBytes::new(py, ready_event_handle))?;
            handle.set_item("allocation_handle", PyBytes::new(py, allocation_handle))?;
            handle.set_item("acknowledgment_offset", acknowledgment_offset)?;
            "CudaVmmTransfer"
        }
        TransferTransport::Channel { endpoint, payload } => {
            handle.set_item("endpoint", endpoint.as_str())?;
            handle.set_item("payload", PyBytes::new(py, payload))?;
            "ChannelTransfer"
        }
    };
    Ok((handle, name))
}
