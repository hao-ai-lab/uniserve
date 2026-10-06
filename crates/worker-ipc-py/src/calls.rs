//! Immutable native calls exposed to numerical preparation.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple};
use serde::de::DeserializeOwned;
use uniserve_worker_ipc::{Call as NativeCall, CallKind};

use crate::convert::{
    RequestConversion, cached, mapping_from_py as mapping, record_from_py as record,
    records_from_py as records,
};
use crate::ids::{BufferId, CallId, RequestKey};
use crate::worker::error::invalid;

/// A scheduler computation. Native consumers share its immutable description;
/// Python reads numerical parameters and tensor references as needed.
#[pyclass(
    frozen,
    eq,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
pub(crate) struct Call {
    pub(crate) inner: Arc<NativeCall>,
    // Numerical views are immutable. Materialize them once on first use,
    // avoiding repeated Python tensor-reference and token-tuple construction.
    coordinates: PyOnceLock<Py<PyAny>>,
    bounds: PyOnceLock<Py<PyAny>>,
    rng: PyOnceLock<Py<PyAny>>,
    sampling_state: PyOnceLock<Py<PyAny>>,
    readout: PyOnceLock<Py<PyAny>>,
    canvas: PyOnceLock<Py<PyAny>>,
    token_input: PyOnceLock<Py<PyAny>>,
    token_output: PyOnceLock<Py<PyAny>>,
    latent_feature_input: PyOnceLock<Py<PyAny>>,
    encoder_output: PyOnceLock<Py<PyAny>>,
    latent_input: PyOnceLock<Py<PyAny>>,
    latent_output: PyOnceLock<Py<PyAny>>,
    image_input: PyOnceLock<Py<PyAny>>,
    image_output: PyOnceLock<Py<PyAny>>,
    completion_output: PyOnceLock<Py<PyAny>>,
    transition_output: PyOnceLock<Py<PyAny>>,
    predicate: PyOnceLock<Py<PyAny>>,
    inputs: PyOnceLock<Py<PyAny>>,
    outputs: PyOnceLock<Py<PyAny>>,
    vision_inputs: PyOnceLock<Py<PyAny>>,
    input_token_ids: PyOnceLock<Py<PyAny>>,
    input_image: PyOnceLock<Py<PyAny>>,
    consumer_slots: PyOnceLock<Py<PyAny>>,
}

impl From<NativeCall> for Call {
    fn from(inner: NativeCall) -> Self {
        Self {
            inner: Arc::new(inner),
            coordinates: PyOnceLock::new(),
            bounds: PyOnceLock::new(),
            rng: PyOnceLock::new(),
            sampling_state: PyOnceLock::new(),
            readout: PyOnceLock::new(),
            canvas: PyOnceLock::new(),
            token_input: PyOnceLock::new(),
            token_output: PyOnceLock::new(),
            latent_feature_input: PyOnceLock::new(),
            encoder_output: PyOnceLock::new(),
            latent_input: PyOnceLock::new(),
            latent_output: PyOnceLock::new(),
            image_input: PyOnceLock::new(),
            image_output: PyOnceLock::new(),
            completion_output: PyOnceLock::new(),
            transition_output: PyOnceLock::new(),
            predicate: PyOnceLock::new(),
            inputs: PyOnceLock::new(),
            outputs: PyOnceLock::new(),
            vision_inputs: PyOnceLock::new(),
            input_token_ids: PyOnceLock::new(),
            input_image: PyOnceLock::new(),
            consumer_slots: PyOnceLock::new(),
        }
    }
}

impl PartialEq for Call {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for Call {}

#[pymethods]
impl Call {
    #[new]
    #[pyo3(signature = (request_key, call_id, coordinates, kind, bounds, component="model", **fields))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        request_key: PyRef<'_, RequestKey>,
        call_id: PyRef<'_, CallId>,
        coordinates: &Bound<'_, PyAny>,
        kind: &str,
        bounds: &Bound<'_, PyAny>,
        component: &str,
        fields: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut call = NativeCall {
            request_key: request_key.inner,
            call_id: call_id.inner,
            coordinates: record(coordinates)?,
            code: call_kind(py, kind)?,
            bounds: record(bounds)?,
            component: component.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            consumer_slots: Vec::new(),
            token_input: None,
            token_output: None,
            vision_inputs: Vec::new(),
            latent_feature_input: None,
            encoder_output: None,
            latent_input: None,
            latent_output: None,
            image_input: None,
            image_output: None,
            completion_output: None,
            transition_output: None,
            predicate: None,
            rng: None,
            sampling_state: None,
            input_token_ids: Vec::new(),
            readout: None,
            canvas: None,
            input_image: None,
            kv_input: None,
            kv_output: None,
        };
        apply_fields(&mut call, fields)?;

        Ok(Self::from(call))
    }

    /// Derive a call while retaining every field the caller does not change.
    #[pyo3(signature = (**fields))]
    fn replace(&self, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut call = self.inner.as_ref().clone();
        apply_fields(&mut call, fields)?;

        Ok(Self::from(call))
    }

    #[staticmethod]
    #[pyo3(signature = (value, where_="call"))]
    fn from_mapping(value: &Bound<'_, PyAny>, where_: &str) -> PyResult<Self> {
        let call: NativeCall =
            mapping(value).map_err(|error| invalid(value.py(), format!("{where_}: {error}")))?;
        call.validate()
            .map_err(|error| invalid(value.py(), error.to_string()))?;

        Ok(Self::from(call))
    }

    fn to_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize::pythonize(py, self.inner.as_ref())?)
    }

    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        self.inner
            .validate()
            .map_err(|error| invalid(py, error.to_string()))
    }

    fn writes_context(&self) -> bool {
        self.inner.writes_context()
    }

    #[getter]
    fn advances_state(&self) -> bool {
        self.inner.advances_state()
    }

    #[getter]
    fn request_key(&self) -> RequestKey {
        RequestKey {
            inner: self.inner.request_key,
        }
    }

    #[getter]
    fn call_id(&self) -> CallId {
        CallId {
            inner: self.inner.call_id,
        }
    }

    #[getter]
    fn component(&self) -> &str {
        &self.inner.component
    }

    #[getter]
    fn kind<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(RequestConversion::new(py)?.kind(self.inner.code))
    }

    #[getter]
    fn input_image<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.input_image, || {
            self.inner
                .input_image
                .as_deref()
                .into_pyobject(py)
                .map(Bound::into_any)
                .map_err(Into::into)
        })
    }

    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.coordinates, || {
            RequestConversion::new(py)?.coordinates(&self.inner.coordinates)
        })
    }

    #[getter]
    fn bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.bounds, || {
            RequestConversion::new(py)?.bounds(&self.inner.bounds)
        })
    }

    #[getter]
    fn rng<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.rng, || {
            self.inner
                .rng
                .as_ref()
                .map(|value| RequestConversion::new(py)?.rng(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn sampling_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.sampling_state, || {
            self.inner
                .sampling_state
                .as_ref()
                .map(|value| RequestConversion::new(py)?.sampling_state(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn readout<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.readout, || {
            self.inner
                .readout
                .as_ref()
                .map(|value| RequestConversion::new(py)?.readout(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn canvas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.canvas, || {
            self.inner
                .canvas
                .as_ref()
                .map(|value| RequestConversion::new(py)?.canvas(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn token_input<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.token_input, || {
            self.inner
                .token_input
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn token_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.token_output, || {
            self.inner
                .token_output
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn latent_feature_input<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.latent_feature_input, || {
            self.inner
                .latent_feature_input
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn encoder_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.encoder_output, || {
            self.inner
                .encoder_output
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn latent_input<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.latent_input, || {
            self.inner
                .latent_input
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn latent_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.latent_output, || {
            self.inner
                .latent_output
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn image_input<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.image_input, || {
            self.inner
                .image_input
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn image_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.image_output, || {
            self.inner
                .image_output
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn completion_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.completion_output, || {
            self.inner
                .completion_output
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn transition_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.transition_output, || {
            self.inner
                .transition_output
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn predicate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.predicate, || {
            self.inner
                .predicate
                .as_ref()
                .map(|value| RequestConversion::new(py)?.tensor_ref(value))
                .transpose()
                .map(|value| value.unwrap_or_else(|| py.None().into_bound(py)))
        })
    }

    #[getter]
    fn inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.inputs, || {
            RequestConversion::new(py)?
                .tensor_refs(self.inner.inputs.iter())
                .map(Bound::into_any)
        })
    }

    #[getter]
    fn outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.outputs, || {
            RequestConversion::new(py)?
                .tensor_refs(self.inner.outputs.iter())
                .map(Bound::into_any)
        })
    }

    #[getter]
    fn vision_inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.vision_inputs, || {
            RequestConversion::new(py)?
                .vision_inputs(&self.inner.vision_inputs)
                .map(Bound::into_any)
        })
    }

    #[getter]
    fn input_token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.input_token_ids, || {
            PyTuple::new(py, &self.inner.input_token_ids).map(Bound::into_any)
        })
    }

    #[getter]
    fn consumer_slots<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        cached(py, &self.consumer_slots, || {
            PyTuple::new(py, &self.inner.consumer_slots).map(Bound::into_any)
        })
    }

    #[getter]
    fn kv_input(&self) -> Option<BufferId> {
        self.inner.kv_input.map(|inner| BufferId { inner })
    }

    #[getter]
    fn kv_output(&self) -> Option<BufferId> {
        self.inner.kv_output.map(|inner| BufferId { inner })
    }

    fn tensor_inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        RequestConversion::new(py)?.tensor_refs(self.inner.tensor_inputs())
    }

    fn tensor_outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        RequestConversion::new(py)?.tensor_refs(self.inner.tensor_outputs())
    }

    fn buffer_inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        RequestConversion::new(py)?.tensor_refs(self.inner.buffer_inputs())
    }

    fn buffer_outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        RequestConversion::new(py)?.tensor_refs(self.inner.buffer_outputs())
    }

    fn __repr__(&self) -> String {
        format!(
            "Call(component={:?}, kind={:?}, call_id={:?})",
            self.inner.component, self.inner.code, self.inner.call_id
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

fn call_kind(py: Python<'_>, value: &str) -> PyResult<CallKind> {
    serde_json::from_value(serde_json::Value::String(value.into()))
        .map_err(|error| invalid(py, error.to_string()))
}

fn optional<T: DeserializeOwned>(value: &Bound<'_, PyAny>) -> PyResult<Option<T>> {
    if value.is_none() {
        Ok(None)
    } else {
        record(value).map(Some)
    }
}

fn apply_fields(call: &mut NativeCall, fields: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    let Some(fields) = fields else {
        return Ok(());
    };

    for (name, value) in fields {
        match name.extract::<&str>()? {
            "request_key" => call.request_key = value.extract::<PyRef<'_, RequestKey>>()?.inner,
            "call_id" => call.call_id = value.extract::<PyRef<'_, CallId>>()?.inner,
            "coordinates" => call.coordinates = record(&value)?,
            "kind" => call.code = call_kind(value.py(), value.extract()?)?,
            "bounds" => call.bounds = record(&value)?,
            "component" => call.component = value.extract()?,
            "inputs" => call.inputs = records(&value)?,
            "outputs" => call.outputs = records(&value)?,
            "vision_inputs" => call.vision_inputs = records(&value)?,
            "token_input" => call.token_input = optional(&value)?,
            "token_output" => call.token_output = optional(&value)?,
            "latent_feature_input" => call.latent_feature_input = optional(&value)?,
            "encoder_output" => call.encoder_output = optional(&value)?,
            "latent_input" => call.latent_input = optional(&value)?,
            "latent_output" => call.latent_output = optional(&value)?,
            "image_input" => call.image_input = optional(&value)?,
            "image_output" => call.image_output = optional(&value)?,
            "completion_output" => call.completion_output = optional(&value)?,
            "transition_output" => call.transition_output = optional(&value)?,
            "predicate" => call.predicate = optional(&value)?,
            "rng" => call.rng = optional(&value)?,
            "sampling_state" => call.sampling_state = optional(&value)?,
            "readout" => call.readout = optional(&value)?,
            "canvas" => call.canvas = optional(&value)?,
            "input_token_ids" => call.input_token_ids = value.extract()?,
            "input_image" => call.input_image = value.extract::<Option<String>>()?.map(Arc::from),
            "consumer_slots" => call.consumer_slots = value.extract()?,
            "kv_input" => {
                call.kv_input = value
                    .extract::<Option<PyRef<'_, BufferId>>>()?
                    .map(|id| id.inner)
            }
            "kv_output" => {
                call.kv_output = value
                    .extract::<Option<PyRef<'_, BufferId>>>()?
                    .map(|id| id.inner)
            }
            name => return Err(PyTypeError::new_err(format!("unknown Call field {name:?}"))),
        }
    }

    Ok(())
}
