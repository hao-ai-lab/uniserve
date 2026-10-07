//! Request-owned diffusion schedules and cached positional tensor views.

use std::collections::HashMap;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::{CallKind, MediaCall};

use super::model_inputs::{AttentionRow, DiffusionRow, InputRow};

/// Schedules and guidance remain mathematical objects. This owner retains
/// their tensors for the request; accepted progress belongs to Request.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct DiffusionState {
    #[pyo3(get)]
    pub(super) size: Py<PyAny>,
    #[pyo3(get)]
    pub(super) schedules: Py<PyAny>,
    #[pyo3(get)]
    pub(super) guidance: Option<Py<PyAny>>,
    positions: HashMap<u64, Py<PyAny>>,
}

#[pymethods]
impl DiffusionState {
    #[new]
    #[pyo3(signature = (size, schedules, guidance=None))]
    fn new(size: Py<PyAny>, schedules: Py<PyAny>, guidance: Option<Py<PyAny>>) -> Self {
        Self {
            size,
            schedules,
            guidance,
            positions: HashMap::new(),
        }
    }

    /// Construct the denoiser's schedules once for the admitted numerical size.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (denoiser, size, *, steps, shift, device, guidance=None))]
    fn open(
        py: Python<'_>,
        denoiser: &Bound<'_, PyAny>,
        size: Py<PyAny>,
        steps: usize,
        shift: Option<f64>,
        device: &Bound<'_, PyAny>,
        guidance: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let options = PyDict::new(py);
        options.set_item("shift", shift)?;
        options.set_item("device", device)?;
        let schedules = denoiser.call_method("make_schedules", (steps,), Some(&options))?;
        let schedules = py.get_type::<PyDict>().call1((schedules,))?;
        Ok(Self::new(size, schedules.unbind(), guidance))
    }

    /// Bind guidance branches to executor-selected slot, KV and temporal coordinates.
    #[pyo3(signature = (builder, current, timestep, coordinates, *, device))]
    pub(super) fn prepare_inputs(
        &mut self,
        py: Python<'_>,
        builder: &Bound<'_, PyAny>,
        current: &Bound<'_, PyAny>,
        timestep: &Bound<'_, PyAny>,
        coordinates: Vec<(u32, u64, u64)>,
        device: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyTuple>> {
        let (current, timestep) = py
            .import("uniserve_worker.execution.diffusion")?
            .call_method1("diffusion_values", (current, timestep, device))?
            .extract::<(Py<PyAny>, Py<PyAny>)>()?;
        let size = self.size.bind(py);
        let height = size.getattr("height")?.extract()?;
        let width = size.getattr("width")?.extract()?;
        let query = builder
            .call_method1("sequence_length", (size,))?
            .extract()?;
        let options = PyDict::new(py);
        options.set_item("device", device)?;
        let mut rows = Vec::with_capacity(coordinates.len());
        for (slot, visible, temporal) in coordinates {
            let positions = match self.positions.entry(temporal) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => entry.insert(
                    builder
                        .call_method("positions", (size, temporal), Some(&options))?
                        .unbind(),
                ),
            };
            let input = InputRow {
                kind: CallKind::Media(MediaCall::Denoising),
                request_pool_idx: slot,
            };
            let attention = AttentionRow {
                positions: Some(positions.clone_ref(py)),
                seq_len: visible as i64,
                write_kv: false,
                causal: Some(false),
            };
            let row = DiffusionRow {
                latent: current.clone_ref(py),
                timestep: timestep.clone_ref(py),
                image_tokens: query,
                image_height: height,
                image_width: width,
            };
            rows.push(Py::new(py, attention.initializer(input).add_subclass(row))?);
        }
        Ok(PyTuple::new(py, rows)?.unbind())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.size)?;
        visit.call(&self.schedules)?;
        visit.call(&self.guidance)?;
        for positions in self.positions.values() {
            visit.call(positions)?;
        }
        Ok(())
    }
}
