//! Native request, call and buffer values used by execution and storage.

use pyo3::prelude::*;
use uniserve_core::RequestId;
use uniserve_worker_ipc::{
    BufferId as NativeBufferId, CallId as NativeCallId, RequestKey as NativeRequestKey,
};

use crate::worker::error::invalid;

/// One admission epoch. Equality and hashing use all three coordinates.
#[derive(Clone, PartialEq, Eq, Hash)]
#[pyclass(
    frozen,
    eq,
    hash,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
pub(crate) struct RequestKey {
    pub(crate) inner: NativeRequestKey,
}

#[pymethods]
impl RequestKey {
    #[new]
    fn new(
        #[pyo3(from_py_with = request_coordinate)] engine_id: u64,
        #[pyo3(from_py_with = request_coordinate)] request_id: u64,
        #[pyo3(from_py_with = request_coordinate)] request_epoch: u64,
    ) -> Self {
        Self {
            inner: NativeRequestKey::new(engine_id, RequestId(request_id), request_epoch),
        }
    }

    #[getter]
    fn engine_id(&self) -> u64 {
        self.inner.engine_id
    }

    #[getter]
    fn request_id(&self) -> u64 {
        self.inner.request_id.0
    }

    #[getter]
    fn request_epoch(&self) -> u64 {
        self.inner.request_epoch
    }

    #[staticmethod]
    #[pyo3(signature = (value, where_="request_key"))]
    fn from_mapping(value: &Bound<'_, PyAny>, where_: &str) -> PyResult<Self> {
        Ok(Self {
            inner: crate::convert::request_key_from_py(value).ok_or_else(|| {
                invalid(
                    value.py(),
                    format!("{where_} has invalid request coordinates"),
                )
            })?,
        })
    }

    fn to_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize::pythonize(py, &self.inner)?)
    }

    fn __getnewargs__(&self) -> (u64, u64, u64) {
        (self.engine_id(), self.request_id(), self.request_epoch())
    }

    fn __repr__(&self) -> String {
        format!(
            "RequestKey(engine_id={}, request_id={}, request_epoch={})",
            self.engine_id(),
            self.request_id(),
            self.request_epoch()
        )
    }
}

fn request_coordinate(value: &Bound<'_, PyAny>) -> PyResult<u64> {
    crate::convert::u64_of(value)
        .ok_or_else(|| invalid(value.py(), "request coordinate must be a uint64 integer"))
}

/// Scheduler batch and selection ordinal, ordered before physical row packing.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[pyclass(
    frozen,
    eq,
    ord,
    hash,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
pub(crate) struct CallId {
    pub(crate) inner: NativeCallId,
}

#[pymethods]
impl CallId {
    #[new]
    fn new(py: Python<'_>, batch_id: u64, request_index: u32) -> PyResult<Self> {
        // (0, 0) names the admission root; real calls use positive batches.
        if batch_id == 0 && request_index != 0 {
            return Err(invalid(
                py,
                "admission identity requires request index zero",
            ));
        }

        Ok(Self {
            inner: NativeCallId::new(batch_id, request_index),
        })
    }

    #[getter]
    fn batch_id(&self) -> u64 {
        self.inner.batch_id
    }

    #[getter]
    fn request_index(&self) -> u32 {
        self.inner.request_index
    }

    #[staticmethod]
    #[pyo3(signature = (value, where_="computation_id"))]
    fn from_mapping(value: &Bound<'_, PyAny>, where_: &str) -> PyResult<Self> {
        let inner = crate::convert::computation_id_from_py(value)
            .ok_or_else(|| invalid(value.py(), format!("{where_} has invalid call coordinates")))?;
        Self::new(value.py(), inner.batch_id, inner.request_index)
    }

    fn to_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize::pythonize(py, &self.inner)?)
    }

    fn __getnewargs__(&self) -> (u64, u32) {
        (self.batch_id(), self.request_index())
    }

    fn __repr__(&self) -> String {
        format!(
            "CallId(batch_id={}, request_index={})",
            self.batch_id(),
            self.request_index()
        )
    }
}

/// One allocation generation of a call output; physical location is separate.
#[derive(Clone, PartialEq, Eq, Hash)]
#[pyclass(
    frozen,
    eq,
    hash,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
pub(crate) struct BufferId {
    pub(crate) inner: NativeBufferId,
}

#[pymethods]
impl BufferId {
    #[new]
    fn new(
        py: Python<'_>,
        owner: PyRef<'_, RequestKey>,
        producer_call_id: PyRef<'_, CallId>,
        output_index: u16,
        generation: u32,
    ) -> PyResult<Self> {
        let inner = NativeBufferId {
            owner: owner.inner,
            producer_call_id: producer_call_id.inner,
            output_index,
            generation,
        };
        inner
            .validate()
            .map_err(|error| invalid(py, error.to_string()))?;

        Ok(Self { inner })
    }

    #[getter]
    fn owner(&self) -> RequestKey {
        RequestKey {
            inner: self.inner.owner,
        }
    }

    #[getter]
    fn producer_call_id(&self) -> CallId {
        CallId {
            inner: self.inner.producer_call_id,
        }
    }

    #[getter]
    fn output_index(&self) -> u16 {
        self.inner.output_index
    }

    #[getter]
    fn generation(&self) -> u32 {
        self.inner.generation
    }

    #[staticmethod]
    #[pyo3(signature = (value, where_="buffer_id"))]
    fn from_mapping(value: &Bound<'_, PyAny>, where_: &str) -> PyResult<Self> {
        let inner = crate::convert::buffer_id_mapping_from_py(value).ok_or_else(|| {
            invalid(
                value.py(),
                format!("{where_} has invalid buffer coordinates"),
            )
        })?;
        CallId::new(
            value.py(),
            inner.producer_call_id.batch_id,
            inner.producer_call_id.request_index,
        )?;
        inner
            .validate()
            .map_err(|error| invalid(value.py(), error.to_string()))?;

        Ok(Self { inner })
    }

    fn to_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize::pythonize(py, &self.inner)?)
    }

    // Multiprocess transport and direct callers pickle the enclosing records.
    fn __getnewargs__(&self) -> (RequestKey, CallId, u16, u32) {
        (
            self.owner(),
            self.producer_call_id(),
            self.output_index(),
            self.generation(),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "BufferId(owner={}, producer_call_id={}, output_index={}, generation={})",
            self.owner().__repr__(),
            self.producer_call_id().__repr__(),
            self.output_index(),
            self.generation()
        )
    }
}
