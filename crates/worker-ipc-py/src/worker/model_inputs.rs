//! Native call coordinates with borrowed numerical tensor views.

mod batch;
mod canvas;
mod media;
mod token;

pub(super) use batch::InputBatch;
pub(super) use canvas::{CanvasRow, CanvasStepRow};
pub(super) use media::{DecodeRow, DiffusionRow, VisionRow};
pub(super) use token::{TokenRow, parse_selections, selection_to_py, selections_to_py};

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use uniserve_worker_ipc::CallKind;

use crate::convert::RequestConversion;

/// The destination slot and operation belong to execution, independently of
/// the model's numerical input. Slot zero is reserved for graph padding.
#[pyclass(
    subclass,
    skip_from_py_object,
    module = "uniserve_worker._uniserve_ipc"
)]
#[derive(Clone)]
pub(crate) struct InputRow {
    pub(super) kind: CallKind,
    #[pyo3(get)]
    pub(super) request_pool_idx: u32,
}

#[pymethods]
impl InputRow {
    #[getter]
    fn forward_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(RequestConversion::new(py)?.kind(self.kind).unbind())
    }
}

impl InputRow {
    fn field(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match name {
            "forward_mode" => self.kind = pythonize::depythonize(value)?,
            "request_pool_idx" => self.request_pool_idx = value.extract()?,
            _ => return Err(unknown(name)),
        }
        Ok(())
    }
}

/// Shared attention coordinates; the prefix is measured in tokens, while
/// positions may contain several mathematical axes (for example M-RoPE).
#[pyclass(subclass, extends = InputRow, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct AttentionRow {
    #[pyo3(get)]
    pub(super) positions: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(super) seq_len: i64,
    #[pyo3(get)]
    pub(super) write_kv: bool,
    #[pyo3(get)]
    pub(super) causal: Option<bool>,
}

#[pymethods]
impl AttentionRow {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.positions)
    }
}

impl AttentionRow {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            positions: self.positions.as_ref().map(|value| value.clone_ref(py)),
            seq_len: self.seq_len,
            write_kv: self.write_kv,
            causal: self.causal,
        }
    }

    fn field(&mut self, row: &mut InputRow, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match name {
            "positions" => self.positions = value.extract()?,
            "seq_len" => self.seq_len = value.extract()?,
            "write_kv" => self.write_kv = value.extract()?,
            "causal" => self.causal = value.extract()?,
            _ => return row.field(name, value),
        }
        Ok(())
    }

    pub(super) fn initializer(self, row: InputRow) -> PyClassInitializer<Self> {
        PyClassInitializer::from(row).add_subclass(self)
    }
}

/// Borrow concrete rows once; host preparation reads Rust fields directly.
pub(super) enum Row<'py> {
    Token(PyRef<'py, TokenRow>),
    Canvas(PyRef<'py, CanvasRow>),
    CanvasStep(PyRef<'py, CanvasStepRow>),
    Diffusion(PyRef<'py, DiffusionRow>),
    Vision(PyRef<'py, VisionRow>),
    Decode(PyRef<'py, DecodeRow>),
}

impl<'py> Row<'py> {
    pub(super) fn borrow(value: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(row) = value.cast::<TokenRow>() {
            Ok(Self::Token(row.borrow()))
        } else if let Ok(row) = value.cast::<CanvasRow>() {
            Ok(Self::Canvas(row.borrow()))
        } else if let Ok(row) = value.cast::<CanvasStepRow>() {
            Ok(Self::CanvasStep(row.borrow()))
        } else if let Ok(row) = value.cast::<DiffusionRow>() {
            Ok(Self::Diffusion(row.borrow()))
        } else if let Ok(row) = value.cast::<VisionRow>() {
            Ok(Self::Vision(row.borrow()))
        } else if let Ok(row) = value.cast::<DecodeRow>() {
            Ok(Self::Decode(row.borrow()))
        } else {
            Err(PyTypeError::new_err("unsupported model input row"))
        }
    }

    pub(super) fn input(&self) -> &InputRow {
        match self {
            Self::Token(row) => row.as_super().as_super(),
            Self::Canvas(row) => row.as_super().as_super(),
            Self::CanvasStep(row) => row.as_super().as_super(),
            Self::Diffusion(row) => row.as_super().as_super(),
            Self::Vision(row) => row.as_super(),
            Self::Decode(row) => row.as_super(),
        }
    }

    pub(super) fn attention(&self) -> Option<&AttentionRow> {
        match self {
            Self::Token(row) => Some(row.as_super()),
            Self::Canvas(row) => Some(row.as_super()),
            Self::CanvasStep(row) => Some(row.as_super()),
            Self::Diffusion(row) => Some(row.as_super()),
            Self::Vision(_) | Self::Decode(_) => None,
        }
    }

    pub(super) fn query_tokens(&self, py: Python<'_>) -> PyResult<usize> {
        match self {
            Self::Token(row) => row.query_tokens(py),
            Self::Canvas(row) => row.query_tokens(py),
            Self::CanvasStep(row) => Ok(row.canvas_length),
            Self::Diffusion(row) => Ok(row.image_tokens),
            Self::Vision(_) | Self::Decode(_) => Ok(0),
        }
    }
}

fn tensor_len(py: Python<'_>, tensor: &Option<Py<PyAny>>) -> PyResult<usize> {
    tensor
        .as_ref()
        .map_or(Ok(0), |value| value.call_method0(py, "numel")?.extract(py))
}

fn unknown(name: &str) -> PyErr {
    PyTypeError::new_err(format!("unknown input field {name:?}"))
}
