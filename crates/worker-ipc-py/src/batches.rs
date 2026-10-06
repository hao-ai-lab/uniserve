//! Immutable scheduler batches shared by IPC and direct execution.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple};
use uniserve_worker_ipc::{Batch as NativeBatch, CanvasSampling as NativeCanvasSampling};

use crate::calls::Call;
use crate::convert::{self, RequestConversion, cached, records_from_py};
use crate::worker::error::invalid;

/// Block-diffusion sampling uses the IPC's FP32 scalars in both deployment and
/// request parameters, so serving compares the values the sampler receives.
#[pyclass(
    frozen,
    eq,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
#[derive(PartialEq)]
pub(crate) struct CanvasSampling {
    pub(crate) inner: NativeCanvasSampling,
}

impl CanvasSampling {
    fn checked(py: Python<'_>, inner: NativeCanvasSampling) -> PyResult<Self> {
        if let Some(parameter) = inner.invalid_parameter() {
            return Err(invalid(py, format!("invalid canvas sampling {parameter}")));
        }

        Ok(Self { inner })
    }
}

#[pymethods]
impl CanvasSampling {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        canvas_length: u32,
        max_steps: u32,
        entropy_bound: f32,
        t_min: f32,
        t_max: f32,
        confidence_threshold: f32,
        stability_threshold: u32,
    ) -> PyResult<Self> {
        Self::checked(
            py,
            NativeCanvasSampling {
                canvas_length,
                max_steps,
                entropy_bound,
                t_min,
                t_max,
                confidence_threshold,
                stability_threshold,
            },
        )
    }

    #[staticmethod]
    fn from_mapping(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::checked(value.py(), convert::mapping_from_py(value)?)
    }

    fn to_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize::pythonize(py, &self.inner).map_err(Into::into)
    }

    #[pyo3(signature = (**fields))]
    fn replace(&self, py: Python<'_>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut sampling = self.inner;
        if let Some(fields) = fields {
            for (name, value) in fields {
                match name.extract::<&str>()? {
                    "canvas_length" => sampling.canvas_length = value.extract()?,
                    "max_steps" => sampling.max_steps = value.extract()?,
                    "entropy_bound" => sampling.entropy_bound = value.extract()?,
                    "t_min" => sampling.t_min = value.extract()?,
                    "t_max" => sampling.t_max = value.extract()?,
                    "confidence_threshold" => sampling.confidence_threshold = value.extract()?,
                    "stability_threshold" => sampling.stability_threshold = value.extract()?,
                    name => {
                        return Err(PyTypeError::new_err(format!(
                            "unknown CanvasSampling field {name:?}"
                        )));
                    }
                }
            }
        }

        Self::checked(py, sampling)
    }

    #[getter]
    fn canvas_length(&self) -> u32 {
        self.inner.canvas_length
    }

    #[getter]
    fn max_steps(&self) -> u32 {
        self.inner.max_steps
    }

    #[getter]
    fn entropy_bound(&self) -> f32 {
        self.inner.entropy_bound
    }

    #[getter]
    fn t_min(&self) -> f32 {
        self.inner.t_min
    }

    #[getter]
    fn t_max(&self) -> f32 {
        self.inner.t_max
    }

    #[getter]
    fn confidence_threshold(&self) -> f32 {
        self.inner.confidence_threshold
    }

    #[getter]
    fn stability_threshold(&self) -> u32 {
        self.inner.stability_threshold
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyAny>,))> {
        Ok((
            py.get_type::<Self>().getattr("from_mapping")?,
            (self.to_mapping(py)?,),
        ))
    }
}

/// One submitted computation and its request commands and resource assignments.
/// Native consumers share this description; Python materializes only the views
/// its numerical backend reads.
#[pyclass(
    frozen,
    eq,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
pub(crate) struct Batch {
    pub(crate) inner: Arc<NativeBatch>,
    calls: PyOnceLock<Py<PyAny>>,
    block_tables: PyOnceLock<Py<PyAny>>,
    new_cache_units: PyOnceLock<Py<PyAny>>,
    forward_call_indices: PyOnceLock<Py<PyAny>>,
    request_pool_indices: PyOnceLock<Py<PyAny>>,
    seq_lens: PyOnceLock<Py<PyAny>>,
    query_lens: PyOnceLock<Py<PyAny>>,
    write_kv: PyOnceLock<Py<PyAny>>,
    latent_params: PyOnceLock<Py<PyAny>>,
    decode_ranges: PyOnceLock<Py<PyAny>>,
    buffer_allocations: PyOnceLock<Py<PyAny>>,
    commands: PyOnceLock<Py<PyAny>>,
    input_products: PyOnceLock<Py<PyAny>>,
    kv_inputs: PyOnceLock<Py<PyAny>>,
    admissions: PyOnceLock<Py<PyAny>>,
}

impl From<Arc<NativeBatch>> for Batch {
    fn from(inner: Arc<NativeBatch>) -> Self {
        Self {
            inner,
            calls: PyOnceLock::new(),
            block_tables: PyOnceLock::new(),
            new_cache_units: PyOnceLock::new(),
            forward_call_indices: PyOnceLock::new(),
            request_pool_indices: PyOnceLock::new(),
            seq_lens: PyOnceLock::new(),
            query_lens: PyOnceLock::new(),
            write_kv: PyOnceLock::new(),
            latent_params: PyOnceLock::new(),
            decode_ranges: PyOnceLock::new(),
            buffer_allocations: PyOnceLock::new(),
            commands: PyOnceLock::new(),
            input_products: PyOnceLock::new(),
            kv_inputs: PyOnceLock::new(),
            admissions: PyOnceLock::new(),
        }
    }
}

impl PartialEq for Batch {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[pymethods]
impl Batch {
    #[new]
    #[pyo3(signature = (batch_id, collective_seq=1, **fields))]
    fn new(
        py: Python<'_>,
        batch_id: u64,
        collective_seq: u64,
        fields: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut batch = NativeBatch::new(batch_id, Vec::new(), Vec::new());
        batch.collective_seq = collective_seq;
        apply_fields(&mut batch, fields)?;
        batch
            .validate()
            .map_err(|error| invalid(py, error.to_string()))?;

        Ok(Self::from(Arc::new(batch)))
    }

    /// Build a new submission from this batch and explicitly changed fields.
    #[pyo3(signature = (**fields))]
    fn replace(&self, py: Python<'_>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut batch = self.inner.as_ref().clone();
        apply_fields(&mut batch, fields)?;
        batch
            .validate()
            .map_err(|error| invalid(py, error.to_string()))?;

        Ok(Self::from(Arc::new(batch)))
    }

    #[staticmethod]
    fn from_mapping(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let batch = convert::batch_from_py(value)
            .map_err(|error| invalid(value.py(), error.to_string()))?;
        batch
            .validate()
            .map_err(|error| invalid(value.py(), error.to_string()))?;

        Ok(Self::from(Arc::new(batch)))
    }

    fn to_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mapping = pythonize::pythonize(py, self.inner.as_ref())?;
        let mut views = RequestConversion::new(py)?;
        // Transfer locators use the same flattened Python representation as
        // outputs. Ordinary fields already have the IPC serde representation.
        let products = self
            .inner
            .input_products
            .iter()
            .map(|value| {
                convert::tensor_export_to_py(py, value, &mut views)?.call_method0("to_mapping")
            })
            .collect::<PyResult<Vec<_>>>()?;
        let imports = self
            .inner
            .kv_inputs
            .iter()
            .map(|value| convert::kv_transfer_to_py(py, value)?.call_method0("to_mapping"))
            .collect::<PyResult<Vec<_>>>()?;
        mapping.set_item("input_products", PyList::new(py, products)?)?;
        mapping.set_item("kv_inputs", PyList::new(py, imports)?)?;

        Ok(mapping)
    }

    #[getter]
    fn batch_id(&self) -> u64 {
        self.inner.batch_id
    }

    #[getter]
    fn collective_seq(&self) -> u64 {
        self.inner.collective_seq
    }

    #[getter]
    fn forward_call_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.forward_call_indices, || {
            PyTuple::new(py, &self.inner.forward.call_indices).map(Bound::into_any)
        })
    }

    #[getter]
    fn request_pool_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.request_pool_indices, || {
            PyTuple::new(py, &self.inner.forward.request_pool_indices).map(Bound::into_any)
        })
    }

    #[getter]
    fn seq_lens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.seq_lens, || {
            PyTuple::new(py, &self.inner.forward.seq_lens).map(Bound::into_any)
        })
    }

    #[getter]
    fn query_lens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.query_lens, || {
            PyTuple::new(py, &self.inner.forward.query_lens).map(Bound::into_any)
        })
    }

    #[getter]
    fn write_kv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.write_kv, || {
            PyTuple::new(py, &self.inner.forward.write_kv).map(Bound::into_any)
        })
    }

    #[getter]
    fn calls<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.calls, || {
            let views = RequestConversion::new(py)?;
            convert::record_tuple(py, &self.inner.calls, |value| views.call(value))
                .map(Bound::into_any)
        })
    }

    #[getter]
    fn block_tables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.block_tables, || {
            let views = RequestConversion::new(py)?;
            convert::record_tuple(py, &self.inner.block_tables, |value| {
                views.block_table(value)
            })
            .map(Bound::into_any)
        })
    }

    #[getter]
    fn new_cache_units<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.new_cache_units, || {
            let views = RequestConversion::new(py)?;
            convert::record_tuple(py, &self.inner.new_cache_units, |value| {
                views.cache_unit_allocation(value)
            })
            .map(Bound::into_any)
        })
    }

    #[getter]
    fn commands<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.commands, || {
            let mut views = RequestConversion::new(py)?;
            convert::record_tuple(py, &self.inner.commands, |value| views.command(value))
                .map(Bound::into_any)
        })
    }

    #[getter]
    fn latent_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.latent_params, || {
            let mut views = RequestConversion::new(py)?;
            convert::record_tuple(py, &self.inner.latent_params, |value| {
                convert::latent_params_with_context(py, value, &mut views)
            })
            .map(Bound::into_any)
        })
    }

    #[getter]
    fn decode_ranges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.decode_ranges, || {
            let mut views = RequestConversion::new(py)?;
            convert::record_tuple(py, &self.inner.decode_ranges, |value| {
                convert::decode_range_to_py(py, value, &mut views)
            })
            .map(Bound::into_any)
        })
    }

    #[getter]
    fn buffer_allocations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.buffer_allocations, || {
            let mut views = RequestConversion::new(py)?;
            convert::record_tuple(py, &self.inner.buffer_allocations, |value| {
                convert::buffer_allocation_to_py(py, value, &mut views)
            })
            .map(Bound::into_any)
        })
    }

    #[getter]
    fn input_products<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.input_products, || {
            let mut views = RequestConversion::new(py)?;
            convert::record_tuple(py, &self.inner.input_products, |value| {
                convert::tensor_export_to_py(py, value, &mut views)
            })
            .map(Bound::into_any)
        })
    }

    #[getter]
    fn kv_inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.kv_inputs, || {
            convert::record_tuple(py, &self.inner.kv_inputs, |value| {
                convert::kv_transfer_to_py(py, value)
            })
            .map(Bound::into_any)
        })
    }

    #[getter]
    fn admissions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.admissions, || {
            let mut views = RequestConversion::new(py)?;
            let values = self
                .inner
                .admissions()
                .map(|value| convert::admission_to_py(py, value, &mut views))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, values).map(Bound::into_any)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Batch(batch_id={}, calls={}, commands={})",
            self.inner.batch_id,
            self.inner.calls.len(),
            self.inner.commands.len()
        )
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyAny>,))> {
        Ok((
            py.get_type::<Self>().getattr("from_mapping")?,
            (self.to_mapping(py)?,),
        ))
    }
}

fn apply_fields(batch: &mut NativeBatch, fields: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    let Some(fields) = fields else {
        return Ok(());
    };

    for (name, value) in fields {
        match name.extract::<&str>()? {
            "batch_id" => batch.batch_id = value.extract()?,
            "collective_seq" => batch.collective_seq = value.extract()?,
            "calls" => {
                batch.calls = value
                    .try_iter()?
                    .map(|call| Ok(call?.extract::<PyRef<'_, Call>>()?.inner.as_ref().clone()))
                    .collect::<PyResult<_>>()?;
            }
            "commands" => {
                let views = RequestConversion::new(value.py())?;
                batch.commands = value
                    .try_iter()?
                    .map(|command| views.command_from_py(&command?))
                    .collect::<PyResult<_>>()?;
            }
            "block_tables" => batch.block_tables = records_from_py(&value)?,
            "new_cache_units" => batch.new_cache_units = records_from_py(&value)?,
            "latent_params" => batch.latent_params = records_from_py(&value)?,
            "decode_ranges" => batch.decode_ranges = records_from_py(&value)?,
            "buffer_allocations" => batch.buffer_allocations = records_from_py(&value)?,
            "forward_call_indices" => batch.forward.call_indices = value.extract()?,
            "request_pool_indices" => batch.forward.request_pool_indices = value.extract()?,
            "seq_lens" => batch.forward.seq_lens = value.extract()?,
            "query_lens" => batch.forward.query_lens = value.extract()?,
            "write_kv" => batch.forward.write_kv = value.extract()?,
            "input_products" => {
                batch.input_products = value
                    .try_iter()?
                    .map(|item| {
                        convert::tensor_export_from_py(&item?.call_method0("to_mapping")?)
                            .ok_or_else(|| invalid(value.py(), "invalid batch tensor product"))
                    })
                    .collect::<PyResult<_>>()?;
            }
            "kv_inputs" => {
                batch.kv_inputs = value
                    .try_iter()?
                    .map(|item| {
                        convert::kv_transfer_from_py(&item?.call_method0("to_mapping")?)
                            .ok_or_else(|| invalid(value.py(), "invalid batch KV transfer"))
                    })
                    .collect::<PyResult<_>>()?;
            }
            name => {
                return Err(PyTypeError::new_err(format!(
                    "unknown Batch field {name:?}"
                )));
            }
        }
    }
    Ok(())
}
