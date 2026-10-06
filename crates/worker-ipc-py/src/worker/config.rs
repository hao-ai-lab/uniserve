//! Python views of native rank configuration and launch normalization.

use std::sync::Arc;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyTuple};
use uniserve_worker::WorkerConfig as NativeConfig;
use uniserve_worker_ipc::LaneConfig as NativeLane;

use super::error::native_error;
use crate::batches::CanvasSampling;

#[pyclass(frozen, eq, module = "uniserve_worker._uniserve_ipc")]
#[derive(PartialEq)]
pub(crate) struct LaneConfig {
    pub(super) inner: NativeLane,
}

#[pymethods]
impl LaneConfig {
    #[new]
    #[pyo3(signature = (lane_id, sm_budget, call_kinds, max_batch_calls=None, max_batch_tokens=None, max_inflight=None))]
    fn new(
        lane_id: String,
        sm_budget: usize,
        call_kinds: &Bound<'_, PyAny>,
        max_batch_calls: Option<usize>,
        max_batch_tokens: Option<usize>,
        max_inflight: Option<usize>,
    ) -> PyResult<Self> {
        let inner = NativeLane {
            lane_id,
            sm_budget,
            call_kinds: pythonize::depythonize(call_kinds)?,
            max_batch_calls,
            max_batch_tokens,
            max_inflight,
        };
        inner.validate().map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn lane_id(&self) -> &str {
        &self.inner.lane_id
    }

    #[getter]
    fn sm_budget(&self) -> usize {
        self.inner.sm_budget
    }

    #[getter]
    fn call_kinds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let module = py.import("uniserve_worker.protocol.call")?;
        let kinds = self
            .inner
            .call_kinds
            .iter()
            .map(|kind| {
                module
                    .getattr(match kind {
                        uniserve_worker_ipc::CallKind::Forward(_) => "ForwardMode",
                        uniserve_worker_ipc::CallKind::Media(_) => "MediaCall",
                        uniserve_worker_ipc::CallKind::Transfer(_) => "TransferMode",
                    })?
                    .call1((kind.as_str(),))
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, kinds)
    }

    #[getter]
    fn max_batch_calls(&self) -> Option<usize> {
        self.inner.max_batch_calls
    }

    #[getter]
    fn max_batch_tokens(&self) -> Option<usize> {
        self.inner.max_batch_tokens
    }

    #[getter]
    fn max_inflight(&self) -> Option<usize> {
        self.inner.max_inflight
    }
}

/// Host settings have one native representation. FlashInfer's numerical
/// options stay with that backend and are only borrowed during binding.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct WorkerConfig {
    pub(super) inner: Arc<NativeConfig>,
    #[pyo3(get)]
    flashinfer: Py<PyAny>,
}

#[pymethods]
impl WorkerConfig {
    #[new]
    #[pyo3(signature = (**fields))]
    fn new(py: Python<'_>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut inner = NativeConfig::default();
        apply_fields(&mut inner, fields)?;
        inner.validate().map_err(|error| native_error(py, error))?;
        let flashinfer = match fields
            .map(|fields| fields.get_item("flashinfer"))
            .transpose()?
            .flatten()
        {
            Some(value) => value,
            None => py
                .import("uniserve.runtime.backends.attention.flashinfer")?
                .call_method0("Config")?,
        };
        Ok(Self {
            inner: Arc::new(inner),
            flashinfer: flashinfer.unbind(),
        })
    }

    /// Normalize one launch descriptor; numerical backend options are built
    /// once here and retained independently from native execution settings.
    #[staticmethod]
    #[pyo3(signature = (fields, *, device, generation_device=None))]
    fn from_launch(
        fields: &Bound<'_, PyDict>,
        device: String,
        generation_device: Option<String>,
    ) -> PyResult<Self> {
        from_launch(fields, device, generation_device).map_err(|error| {
            let py = fields.py();
            if error.is_instance_of::<PyTypeError>(py)
                || error.is_instance_of::<PyOverflowError>(py)
            {
                PyValueError::new_err(error.to_string())
            } else {
                error
            }
        })
    }

    #[pyo3(signature = (**fields))]
    fn replace(&self, py: Python<'_>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut inner = (*self.inner).clone();
        apply_fields(&mut inner, fields)?;
        inner.validate().map_err(|error| native_error(py, error))?;
        let flashinfer = fields
            .map(|fields| fields.get_item("flashinfer"))
            .transpose()?
            .flatten()
            .map(Bound::unbind)
            .unwrap_or_else(|| self.flashinfer.clone_ref(py));
        Ok(Self {
            inner: Arc::new(inner),
            flashinfer,
        })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let py = other.py();
        let other = other.borrow();
        Ok(self.inner == other.inner && self.flashinfer.bind(py).eq(other.flashinfer.bind(py))?)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.flashinfer)
    }

    #[getter]
    fn device(&self) -> &str {
        &self.inner.device
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank
    }

    #[getter]
    fn world_size(&self) -> usize {
        self.inner.world_size
    }

    #[getter]
    fn role(&self) -> &str {
        &self.inner.role
    }

    #[getter]
    fn expert_exchange(&self) -> &str {
        &self.inner.expert_exchange
    }

    #[getter]
    fn expert_microbatches(&self) -> usize {
        self.inner.expert_microbatches
    }

    #[getter]
    fn block_size(&self) -> Option<usize> {
        self.inner.block_size
    }

    #[getter]
    fn kv_token_capacity(&self) -> Option<usize> {
        self.inner.kv_token_capacity
    }

    #[getter]
    fn attention_backend(&self) -> Option<&str> {
        self.inner.attention_backend.as_deref()
    }

    #[getter]
    fn max_batch_calls(&self) -> usize {
        self.inner.max_batch_calls
    }

    #[getter]
    fn max_batch_tokens(&self) -> usize {
        self.inner.max_batch_tokens
    }

    #[getter]
    fn max_sequence_tokens(&self) -> usize {
        self.inner.max_sequence_tokens
    }

    #[getter]
    fn max_video_seconds(&self) -> f64 {
        self.inner.max_video_seconds
    }

    #[getter]
    fn max_condition_rows(&self) -> usize {
        self.inner.max_condition_rows
    }

    #[getter]
    fn ffmpeg(&self) -> &str {
        &self.inner.ffmpeg
    }

    #[getter]
    fn min_video_seconds(&self) -> Option<f64> {
        self.inner.min_video_seconds
    }

    #[getter]
    fn video_text_capacities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.video_text_capacities)
    }

    #[getter]
    fn deployment_components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.deployment_components)
    }

    #[getter]
    fn canvas_sampling(&self) -> Option<CanvasSampling> {
        self.inner
            .canvas_sampling
            .map(|inner| CanvasSampling { inner })
    }

    #[getter]
    fn max_request_pool_size(&self) -> usize {
        self.inner.max_request_pool_size
    }

    #[getter]
    fn encoder_cache_entries(&self) -> usize {
        self.inner.encoder_cache_entries
    }

    #[getter]
    fn generation_device(&self) -> Option<&str> {
        self.inner.generation_device.as_deref()
    }

    #[getter]
    fn min_request_pool_size(&self) -> usize {
        self.inner.min_request_pool_size
    }

    #[getter]
    fn pool_storage_bytes(&self) -> Option<usize> {
        self.inner.pool_storage_bytes
    }

    #[getter]
    fn model_dtype(&self) -> &str {
        &self.inner.model_dtype
    }

    #[getter]
    fn kv_cache_dtype(&self) -> Option<&str> {
        self.inner.kv_cache_dtype.as_deref()
    }

    #[getter]
    fn kv_storage_fraction(&self) -> f64 {
        self.inner.kv_storage_fraction
    }

    #[getter]
    fn lanes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner
                .lanes
                .iter()
                .cloned()
                .map(|inner| Py::new(py, LaneConfig { inner }))
                .collect::<PyResult<Vec<_>>>()?,
        )
    }

    #[getter]
    fn graph_policy(&self) -> &str {
        &self.inner.graph_policy
    }

    #[getter]
    fn decode_graph_batch_sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.decode_graph_batch_sizes)
    }

    #[getter]
    fn prefill_cuda_graph(&self) -> bool {
        self.inner.prefill_cuda_graph
    }

    #[getter]
    fn prefill_outputs(&self) -> bool {
        self.inner.prefill_outputs
    }

    #[getter]
    fn prefill_graph_token_sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.prefill_graph_token_sizes)
    }

    #[getter]
    fn flow_cuda_graph(&self) -> bool {
        self.inner.flow_cuda_graph
    }

    #[getter]
    fn flow_graph_batch_sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.flow_graph_batch_sizes)
    }

    #[getter]
    fn flow_graph_shapes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.flow_graph_shapes)
    }
}

/// Return a Python view after native capacity fitting. Numerical backend
/// options are immutable and keep the same owner.
pub(super) fn updated<'py>(
    config: &Bound<'py, PyAny>,
    inner: NativeConfig,
) -> PyResult<Bound<'py, PyAny>> {
    let py = config.py();
    let flashinfer = config
        .cast::<WorkerConfig>()?
        .borrow()
        .flashinfer
        .clone_ref(py);
    Ok(Bound::new(
        py,
        WorkerConfig {
            inner: Arc::new(inner),
            flashinfer,
        },
    )?
    .into_any())
}

/// Borrow the immutable settings directly from their Python view.
pub(super) fn native(config: &Bound<'_, PyAny>) -> PyResult<Arc<NativeConfig>> {
    Ok(Arc::clone(&config.cast::<WorkerConfig>()?.borrow().inner))
}

fn apply_fields(inner: &mut NativeConfig, fields: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    let Some(fields) = fields else {
        return Ok(());
    };

    for (name, value) in fields {
        let name: String = name.extract()?;
        match name.as_str() {
            // PyTorch callers may supply torch.device; the host keeps its name.
            "device" => inner.device = value.str()?.extract()?,
            "rank" => inner.rank = value.extract()?,
            "world_size" => inner.world_size = value.extract()?,
            "role" => inner.role = value.extract()?,
            "expert_exchange" => inner.expert_exchange = value.extract()?,
            "expert_microbatches" => {
                inner.expert_microbatches = value.cast_exact::<PyInt>()?.extract()?;
            }
            "block_size" => inner.block_size = value.extract()?,
            "kv_token_capacity" => inner.kv_token_capacity = value.extract()?,
            "attention_backend" => inner.attention_backend = value.extract()?,
            "max_batch_calls" => inner.max_batch_calls = value.extract()?,
            "max_batch_tokens" => inner.max_batch_tokens = value.extract()?,
            "max_sequence_tokens" => inner.max_sequence_tokens = value.extract()?,
            "max_video_seconds" => inner.max_video_seconds = value.extract()?,
            "max_condition_rows" => inner.max_condition_rows = value.extract()?,
            "ffmpeg" => inner.ffmpeg = value.extract()?,
            "min_video_seconds" => inner.min_video_seconds = value.extract()?,
            "video_text_capacities" => inner.video_text_capacities = value.extract()?,
            "deployment_components" => inner.deployment_components = value.extract()?,
            "canvas_sampling" => {
                inner.canvas_sampling = if value.is_none() {
                    None
                } else {
                    Some(value.cast::<CanvasSampling>()?.borrow().inner)
                }
            }

            "max_request_pool_size" => inner.max_request_pool_size = value.extract()?,
            "encoder_cache_entries" => inner.encoder_cache_entries = value.extract()?,
            "generation_device" => {
                inner.generation_device = if value.is_none() {
                    None
                } else {
                    Some(value.str()?.extract()?)
                };
            }
            "min_request_pool_size" => inner.min_request_pool_size = value.extract()?,
            "pool_storage_bytes" => inner.pool_storage_bytes = value.extract()?,
            "model_dtype" => inner.model_dtype = value.extract()?,
            "kv_cache_dtype" => inner.kv_cache_dtype = value.extract()?,
            "kv_storage_fraction" => inner.kv_storage_fraction = value.extract()?,
            "lanes" => {
                inner.lanes = value
                    .try_iter()?
                    .map(|lane| Ok(lane?.cast::<LaneConfig>()?.borrow().inner.clone()))
                    .collect::<PyResult<_>>()?
            }

            "graph_policy" => inner.graph_policy = value.extract()?,
            "decode_graph_batch_sizes" => inner.decode_graph_batch_sizes = value.extract()?,
            "prefill_cuda_graph" => inner.prefill_cuda_graph = value.extract()?,
            "prefill_outputs" => inner.prefill_outputs = value.extract()?,
            "prefill_graph_token_sizes" => inner.prefill_graph_token_sizes = value.extract()?,
            "flow_cuda_graph" => inner.flow_cuda_graph = value.extract()?,
            "flow_graph_batch_sizes" => inner.flow_graph_batch_sizes = value.extract()?,
            "flow_graph_shapes" => inner.flow_graph_shapes = value.extract()?,
            "flashinfer" => {}
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unknown WorkerConfig field {name:?}"
                )));
            }
        }
    }
    Ok(())
}

#[pyfunction]
pub(crate) fn graph_padding_block_count(block_size: usize) -> usize {
    uniserve_worker::config::graph_padding_block_count(block_size)
}

fn from_launch(
    fields: &Bound<'_, PyDict>,
    device: String,
    generation_device: Option<String>,
) -> PyResult<WorkerConfig> {
    let py = fields.py();
    let mut inner = NativeConfig {
        device,
        generation_device,
        ..NativeConfig::default()
    };
    inner.rank = required(fields, "rank")?.extract()?;
    inner.world_size = required(fields, "world_size")?.extract()?;
    inner.max_batch_calls = required(fields, "max_batch_calls")?.extract()?;
    inner.max_batch_tokens = required(fields, "max_batch_tokens")?.extract()?;
    inner.max_request_pool_size = required(fields, "max_request_pool_size")?.extract()?;
    inner.max_sequence_tokens = required(fields, "max_model_len")?.extract()?;
    inner.max_video_seconds = required(fields, "max_video_seconds")?.extract()?;
    inner.max_condition_rows = required(fields, "max_condition_rows")?.extract()?;

    inner.ffmpeg = required(fields, "ffmpeg")?.extract()?;
    inner.deployment_components = required(fields, "deployment_components")?.extract()?;
    inner.model_dtype = required(fields, "model_dtype")?.extract()?;
    inner.kv_storage_fraction = required(fields, "kv_memory_fraction")?.extract()?;
    inner.prefill_cuda_graph = required(fields, "prefill_cuda_graph")?.extract()?;
    inner.attention_backend = Some(required(fields, "attention_backend")?.extract()?);

    inner.block_size = positive(fields, "block_size")?;
    inner.kv_token_capacity = positive(fields, "kv_token_capacity")?;
    inner.min_video_seconds = optional(fields, "min_video_seconds")?
        .map(|value| value.extract())
        .transpose()?;
    inner.kv_cache_dtype = optional(fields, "kv_cache_dtype")?
        .map(|value| {
            let value: String = value.extract()?;
            let value = value.trim().to_owned();
            if value.is_empty() {
                return Err(PyValueError::new_err(
                    "an explicitly provided string setting must not be empty",
                ));
            }
            Ok(value)
        })
        .transpose()?;
    if let Some(value) = optional(fields, "role")? {
        inner.role = value.extract()?;
    }
    if let Some(value) = optional(fields, "expert_microbatches")? {
        inner.expert_microbatches = value.cast_exact::<PyInt>()?.extract()?;
    }
    if let Some(value) = optional(fields, "graph_policy")? {
        inner.graph_policy = value.extract()?;
    }
    if let Some(value) = optional(fields, "prefill_outputs")? {
        inner.prefill_outputs = value.extract()?;
    }
    if let Some(value) = optional(fields, "flow_cuda_graph")? {
        inner.flow_cuda_graph = value.extract()?;
    }

    let experts = optional(fields, "expert_parallel")?;
    if let Some(experts) = &experts {
        let experts = experts.cast::<PyDict>()?;
        if let Some(exchange) = optional(experts, "exchange")? {
            inner.expert_exchange = exchange.extract()?;
        }
        if inner.expert_microbatches > 1
            && optional(experts, "attention_ranks")?
                .map(|value| value.extract::<usize>())
                .transpose()?
                .unwrap_or(0)
                == 0
        {
            return Err(PyValueError::new_err(
                "expert microbatches require disaggregated experts",
            ));
        }
        if !experts.is_empty() && inner.expert_exchange == "alltoall" {
            // Each local graph also captures larger transfer capacities.
            // Geometric small buckets bound that product; the largest gap
            // remains within the ordinary default KV padding reservation.
            inner.prefill_graph_token_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
                .into_iter()
                .chain((1024..=16384).step_by(512))
                .collect();
        }
    } else if inner.expert_microbatches > 1 {
        return Err(PyValueError::new_err(
            "expert microbatches require disaggregated experts",
        ));
    }

    for (name, values) in [
        (
            "decode_graph_batch_sizes",
            &mut inner.decode_graph_batch_sizes,
        ),
        (
            "prefill_graph_token_sizes",
            &mut inner.prefill_graph_token_sizes,
        ),
        ("flow_graph_batch_sizes", &mut inner.flow_graph_batch_sizes),
        ("video_text_capacities", &mut inner.video_text_capacities),
    ] {
        if let Some(value) = optional(fields, name)? {
            *values = positive_csv(&value.extract::<String>()?)?;
        }
    }
    if let Some(value) = optional(fields, "flow_graph_shapes")? {
        inner.flow_graph_shapes.clear();
        for shape in value.extract::<String>()?.split(',') {
            let shape = shape.trim().to_lowercase();
            let (height, width) = shape
                .split_once('x')
                .ok_or_else(|| PyValueError::new_err("flow graph shapes must use HEIGHTxWIDTH"))?;
            let shape = (positive_number(height)?, positive_number(width)?);
            if inner.flow_graph_shapes.contains(&shape) {
                return Err(PyValueError::new_err(
                    "flow graph shapes must be positive and unique",
                ));
            }
            inner.flow_graph_shapes.push(shape);
        }
    }
    if let Some(value) = optional(fields, "lane")? {
        inner.lanes = pythonize::depythonize(&value)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        for lane in &inner.lanes {
            lane.validate().map_err(PyValueError::new_err)?;
        }
    }
    if let Some(value) = optional(fields, "canvas_sampling")? {
        let sampling: uniserve_worker_ipc::CanvasSampling = pythonize::depythonize(&value)?;
        if let Some(parameter) = sampling.invalid_parameter() {
            return Err(PyValueError::new_err(format!(
                "invalid canvas sampling {parameter}"
            )));
        }
        inner.canvas_sampling = Some(sampling);
    }
    inner
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    // The service uses a 24-fps clock; model windows may extend it by
    // sixteen frames. Keep the resolved count within the IPC frame index.
    let frames = (inner.max_video_seconds * 24.0).round_ties_even();
    if !(1.0..=f64::from(u32::MAX - 16)).contains(&frames) {
        return Err(PyValueError::new_err(
            "max-video-seconds must resolve to a supported frame count",
        ));
    }

    let options = PyDict::new(py);
    for field in [
        "workspace_size",
        "decode_backend",
        "prefill_backend",
        "disable_split_kv",
    ] {
        options.set_item(field, required(fields, &format!("flashinfer_{field}"))?)?;
    }
    for field in ["decode_split_tile_size", "prefill_split_tile_size"] {
        options.set_item(field, optional(fields, &format!("flashinfer_{field}"))?)?;
    }
    let tensor_core = optional(fields, "flashinfer_use_tensor_core")?
        .map(|value| {
            if let Ok(value) = value.extract::<bool>() {
                return Ok(value);
            }
            match value.extract::<String>()?.trim().to_lowercase().as_str() {
                "true" => Ok(true),
                "false" => Ok(false),
                _ => Err(PyValueError::new_err("expected an optional bool token")),
            }
        })
        .transpose()?;
    options.set_item("use_tensor_core", tensor_core)?;
    let flashinfer = py
        .import("uniserve.runtime.backends.attention.flashinfer")?
        .getattr("Config")?
        .call((), Some(&options))?
        .unbind();
    Ok(WorkerConfig {
        inner: Arc::new(inner),
        flashinfer,
    })
}

fn required<'py>(fields: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    fields
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("launch descriptor omits {name}")))
}

fn optional<'py>(fields: &Bound<'py, PyDict>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(fields.get_item(name)?.filter(|value| !value.is_none()))
}

fn positive(fields: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<usize>> {
    optional(fields, name)?
        .map(|value| {
            let value = value.extract::<usize>()?;
            if value == 0 {
                return Err(PyValueError::new_err(
                    "optional integer tuning values must be positive when set",
                ));
            }
            Ok(value)
        })
        .transpose()
}

fn positive_number(value: &str) -> PyResult<usize> {
    value
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| PyValueError::new_err("bucket sizes must be positive integers"))
}

fn positive_csv(value: &str) -> PyResult<Vec<usize>> {
    let values = value
        .split(',')
        .map(positive_number)
        .collect::<PyResult<Vec<_>>>()?;
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(PyValueError::new_err(
            "integer bucket lists must be strictly increasing",
        ));
    }
    Ok(values)
}
