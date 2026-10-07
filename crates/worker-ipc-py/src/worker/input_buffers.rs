//! Fixed input storage and host preparation for homogeneous model calls.

mod attention;
mod canvas;
mod prepare;

use std::cell::Cell;

use pyo3::buffer::{Element, PyBuffer};
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice, PyTuple};
use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};

use super::execution::close_all;
use super::host_buffers::HostBuffers;
use super::tensor_buffers::{TensorBuffers, mapping};

pub(super) const ROW_SECTIONS: usize = 6;

/// One lane's stable numerical columns and reusable host copy sources.
/// Readers and captured graphs retire before this owner is closed.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct InputBuffers {
    kind: CallKind,
    #[pyo3(get)]
    config: Py<PyAny>,
    #[pyo3(get)]
    device: Py<PyAny>,
    #[pyo3(get)]
    max_rows: usize,
    #[pyo3(get)]
    max_tokens: usize,
    #[pyo3(get)]
    max_text_tokens: usize,
    #[pyo3(get)]
    hidden_size: usize,
    table_widths: Vec<usize>,
    #[pyo3(get)]
    image_builder: Py<PyAny>,
    row_types: Py<PyAny>,
    backing: Py<TensorBuffers>,
    columns: Py<PyDict>,
    requests: Py<HostBuffers>,
    rows: Option<Py<HostBuffers>>,
    finish: Option<Py<HostBuffers>>,
    steps: Option<Py<HostBuffers>>,
    #[pyo3(get)]
    canvas_slots: Py<PyAny>,
    canvas_backing: Option<Py<TensorBuffers>>,
    #[pyo3(get)]
    candidate_storage: Py<PyAny>,
    closed: bool,
}

#[pymethods]
impl InputBuffers {
    #[new]
    #[pyo3(signature = (kind, *, config, device, max_inflight=1, image_builder=None))]
    pub(super) fn new(
        py: Python<'_>,
        kind: &Bound<'_, PyAny>,
        config: Py<PyAny>,
        device: &Bound<'_, PyAny>,
        max_inflight: isize,
        image_builder: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let kind = pythonize::depythonize::<CallKind>(kind)?;
        let token = matches!(kind, CallKind::Forward(mode) if mode != ForwardMode::TokenDenoising);
        let canvas = kind == CallKind::Forward(ForwardMode::TokenDenoising);
        let attention = token || canvas || kind == CallKind::Media(MediaCall::Denoising);
        let model_inputs = py.import("uniserve_worker.model_executor.input_batch")?;
        let row_types = match kind {
            CallKind::Forward(ForwardMode::TokenDenoising) => PyTuple::new(
                py,
                [
                    model_inputs.getattr("CanvasRow")?,
                    model_inputs.getattr("CanvasStepRow")?,
                ],
            )?
            .into_any(),
            CallKind::Forward(_) => model_inputs.getattr("TokenRow")?,
            CallKind::Media(MediaCall::Denoising) => py
                .import("uniserve_worker.model_executor.diffusion_inputs")?
                .getattr("DiffusionRow")?,
            CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding) => py
                .import("uniserve_worker.model_executor.image_inputs")?
                .getattr("VisionRow")?,
            CallKind::Media(MediaCall::ImageDecoding) => py
                .import("uniserve_worker.model_executor.image_inputs")?
                .getattr("DecodeRow")?,
            _ => {
                return Err(PyValueError::new_err(
                    "computation has no fixed input buffers",
                ));
            }
        };
        let cfg = config.bind(py);
        let max_rows = cfg.getattr("max_rows")?.extract()?;
        let max_tokens = if attention {
            cfg.getattr("max_tokens")?.extract()?
        } else {
            0
        };
        let table_widths = if attention {
            cfg.getattr("table_widths")?.extract()?
        } else {
            Vec::new()
        };
        let max_text_tokens = if token {
            cfg.getattr("max_text_tokens")?.extract()?
        } else {
            0
        };
        let hidden_size = if token {
            cfg.getattr("hidden_size")?.extract()?
        } else {
            0
        };
        let torch = py.import("torch")?;
        let device = torch.call_method1("device", (device,))?;
        let fields = cfg.call_method0("buffers")?;
        let backing = TensorBuffers::allocate(py, &fields, &device, false, None)?;
        let columns = mapping(backing.view(py, &fields)?.bind(py))?;
        for (_, tensor) in &columns {
            tensor.call_method0("zero_")?;
        }
        if attention {
            columns
                .as_any()
                .get_item("write_indices")?
                .call_method1("fill_", (-1,))?;
        }
        if token {
            columns
                .as_any()
                .get_item("input_ids")?
                .call_method1("fill_", (1,))?;
            if columns.get_item("input_embeddings")?.is_none() {
                columns.set_item("input_embeddings", py.None())?;
            }
        }

        let ring = |name| -> PyResult<Py<HostBuffers>> {
            let tensor = columns.as_any().get_item(name)?;
            Py::new(
                py,
                HostBuffers::new(
                    py,
                    &tensor.getattr("shape")?,
                    &tensor.getattr("dtype")?,
                    max_inflight,
                    &device,
                )?,
            )
        };
        let requests = ring("request_pool_indices")?;
        let rows = attention.then(|| ring("row_columns")).transpose()?;
        let finish = token.then(|| ring("decode_force_finish")).transpose()?;
        let steps = canvas.then(|| ring("step_columns")).transpose()?;
        let candidate_storage = if canvas {
            let options = PyDict::new(py);
            options.set_item("dtype", torch.getattr("int64")?)?;
            options.set_item("device", &device)?;
            torch.call_method("empty", (0,), Some(&options))?.unbind()
        } else {
            py.None()
        };

        Ok(Self {
            kind,
            config,
            device: device.unbind(),
            max_rows,
            max_tokens,
            max_text_tokens,
            hidden_size,
            table_widths,
            image_builder: image_builder.unwrap_or_else(|| py.None()),
            row_types: row_types.unbind(),
            backing: Py::new(py, backing)?,
            columns: columns.unbind(),
            requests,
            rows,
            finish,
            steps,
            canvas_slots: py.None(),
            canvas_backing: None,
            candidate_storage,
            closed: false,
        })
    }

    #[getter]
    fn table_widths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.table_widths)
    }

    /// Borrow a fixed numerical column; names come from the input config.
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        self.columns
            .bind(py)
            .get_item(name)?
            .map(Bound::unbind)
            .ok_or_else(|| {
                PyAttributeError::new_err(format!("input buffers have no column {name:?}"))
            })
    }

    fn validate_rows(&self, rows: &Bound<'_, PyTuple>) -> PyResult<()> {
        self.validate(rows)
    }

    /// Copy host controls and bind borrowed numerical inputs on the current stream.
    #[pyo3(signature = (rows, *, forward_mode, attention=None, cache=None, tables=None, states=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_inputs(
        slf: &Bound<'_, Self>,
        rows: &Bound<'_, PyTuple>,
        forward_mode: &Bound<'_, PyAny>,
        attention: Option<&Bound<'_, PyAny>>,
        cache: Option<&Bound<'_, PyAny>>,
        tables: Option<&Bound<'_, PyAny>>,
        states: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Self::prepare(slf, rows, forward_mode, attention, cache, tables, states)
    }

    /// Bind resident canvas state and allocate this lane's gathered input views.
    fn bind_canvas_slots(slf: &Bound<'_, Self>, slots: Py<PyAny>) -> PyResult<()> {
        Self::bind_canvas(slf, slots)
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.closed {
            return Ok(());
        }

        let rings = std::iter::once(&self.requests)
            .chain(self.rows.iter())
            .chain(self.finish.iter())
            .chain(self.steps.iter());
        close_all(py, rings.map(|ring| ring.borrow(py).close(py)))?;
        self.backing.borrow_mut(py).close(py);
        if let Some(backing) = self.canvas_backing.take() {
            backing.borrow_mut(py).close(py);
        }
        self.columns.bind(py).clear();
        self.candidate_storage = py.None();
        self.canvas_slots = py.None();
        self.image_builder = py.None();
        self.closed = true;
        Ok(())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for value in [
            &self.config,
            &self.device,
            &self.image_builder,
            &self.row_types,
            &self.canvas_slots,
            &self.candidate_storage,
        ] {
            visit.call(value)?;
        }
        visit.call(&self.backing)?;
        visit.call(&self.columns)?;
        visit.call(&self.requests)?;
        visit.call(&self.rows)?;
        visit.call(&self.finish)?;
        visit.call(&self.steps)?;
        visit.call(&self.canvas_backing)
    }
}

impl InputBuffers {
    fn open(&self) -> PyResult<()> {
        if self.closed {
            Err(PyRuntimeError::new_err("input buffers are closed"))
        } else {
            Ok(())
        }
    }

    fn column<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.columns.bind(py).as_any().get_item(name)
    }
}

fn numerical(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("uniserve_worker.model_executor.input_buffers")
}

fn prefix<'py>(value: &Bound<'py, PyAny>, count: usize) -> PyResult<Bound<'py, PyAny>> {
    value.get_item(PySlice::new(value.py(), 0, count as isize, 1))
}

/// The ring has retired its previous DMA reader before these CPU writes.
/// Buffer exports keep the allocation alive and check the element representation.
fn with_host<T: Element>(
    tensor: &Bound<'_, PyAny>,
    count: usize,
    write: impl FnOnce(&[Cell<T>]) -> PyResult<()>,
) -> PyResult<()> {
    let py = tensor.py();
    let array = tensor.call_method0("numpy")?;
    let buffer = PyBuffer::<T>::get(&array)?;
    let destination = buffer
        .as_mut_slice(py)
        .and_then(|slice| slice.get(..count))
        .ok_or_else(|| PyValueError::new_err("host input column exceeds contiguous backing"))?;
    write(destination)
}

fn fill<T: Element + Copy>(tensor: &Bound<'_, PyAny>, values: &[T]) -> PyResult<()> {
    with_host(tensor, values.len(), |destination| {
        for (cell, value) in destination.iter().zip(values) {
            cell.set(*value);
        }
        Ok(())
    })
}

fn copy(destination: &Bound<'_, PyAny>, source: &Bound<'_, PyAny>) -> PyResult<()> {
    let options = PyDict::new(destination.py());
    options.set_item("non_blocking", true)?;
    destination.call_method("copy_", (source,), Some(&options))?;
    Ok(())
}

fn host_tensor<'py>(py: Python<'py>, values: &[i64]) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let options = PyDict::new(py);
    options.set_item("dtype", torch.getattr("int64")?)?;
    options.set_item("device", "cpu")?;
    let tensor = torch.call_method("empty", (values.len(),), Some(&options))?;
    fill(&tensor, values)?;
    Ok(tensor)
}
