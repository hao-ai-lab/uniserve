//! Readout selections and resident token-denoising coordinates.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::{AttentionRow, InputRow, tensor_len};

/// Candidate intervals follow slot order within this noncausal canvas.
#[pyclass(extends = AttentionRow, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct CanvasRow {
    #[pyo3(get)]
    pub(in crate::worker) token_ids: Option<Py<PyAny>>,
    pub(in crate::worker) slot_tokens: Vec<usize>,
    pub(in crate::worker) candidate_offsets: Vec<usize>,
    pub(in crate::worker) candidate_ids: Vec<i64>,
}

#[pymethods]
impl CanvasRow {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (forward_mode, request_pool_idx=0, positions=None, seq_len=0, write_kv=false, causal=Some(true), token_ids=None, slot_tokens=Vec::new(), candidate_offsets=vec![0], candidate_ids=Vec::new()))]
    fn new(
        forward_mode: &Bound<'_, PyAny>,
        request_pool_idx: u32,
        positions: Option<Py<PyAny>>,
        seq_len: i64,
        write_kv: bool,
        causal: Option<bool>,
        token_ids: Option<Py<PyAny>>,
        slot_tokens: Vec<usize>,
        candidate_offsets: Vec<usize>,
        candidate_ids: Vec<i64>,
    ) -> PyResult<PyClassInitializer<Self>> {
        Self {
            token_ids,
            slot_tokens,
            candidate_offsets,
            candidate_ids,
        }
        .initializer(
            forward_mode.py(),
            InputRow {
                kind: pythonize::depythonize(forward_mode)?,
                request_pool_idx,
            },
            AttentionRow {
                positions,
                seq_len,
                write_kv,
                causal,
            },
        )
    }

    #[getter]
    pub(in crate::worker) fn query_tokens(&self, py: Python<'_>) -> PyResult<usize> {
        tensor_len(py, &self.token_ids)
    }

    #[getter]
    fn slot_tokens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.slot_tokens)
    }

    #[getter]
    fn candidate_offsets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.candidate_offsets)
    }

    #[getter]
    fn candidate_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.candidate_ids)
    }

    #[pyo3(signature = (**fields))]
    fn replace(slf: PyRef<'_, Self>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let mut input = InputRow::clone(slf.as_super().as_super());
        let mut attention = slf.as_super().clone_ref(py);
        let mut row = Self {
            token_ids: slf.token_ids.as_ref().map(|value| value.clone_ref(py)),
            slot_tokens: slf.slot_tokens.clone(),
            candidate_offsets: slf.candidate_offsets.clone(),
            candidate_ids: slf.candidate_ids.clone(),
        };
        if let Some(fields) = fields {
            for (name, value) in fields {
                match name.extract::<&str>()? {
                    "token_ids" => row.token_ids = value.extract()?,
                    "slot_tokens" => row.slot_tokens = value.extract()?,
                    "candidate_offsets" => row.candidate_offsets = value.extract()?,
                    "candidate_ids" => row.candidate_ids = value.extract()?,
                    name => attention.field(&mut input, name, &value)?,
                }
            }
        }
        Py::new(py, row.initializer(py, input, attention)?)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.token_ids)
    }
}

impl CanvasRow {
    pub(in crate::worker) fn initializer(
        self,
        py: Python<'_>,
        input: InputRow,
        attention: AttentionRow,
    ) -> PyResult<PyClassInitializer<Self>> {
        let tokens = self.query_tokens(py)?;
        if !canvas_shape(py, &attention, tokens)? {
            return Err(PyValueError::new_err(
                "a canvas row is a read-only noncausal token sequence",
            ));
        }
        let offsets = &self.candidate_offsets;
        if self.slot_tokens.is_empty()
            || offsets.len() != self.slot_tokens.len() + 1
            || offsets.first() != Some(&0)
            || offsets.last() != Some(&self.candidate_ids.len())
            || offsets.windows(2).any(|pair| pair[1] <= pair[0])
            || self.slot_tokens.iter().any(|&token| token >= tokens)
        {
            return Err(PyValueError::new_err(
                "canvas slots must lie in the row and each read candidates",
            ));
        }
        Ok(attention.initializer(input).add_subclass(self))
    }
}

/// A generation step borrows its canvas from the resident request slot.
/// Seeds use the signed int64 bit pattern consumed by Philox device columns.
#[pyclass(extends = AttentionRow, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct CanvasStepRow {
    #[pyo3(get)]
    pub(in crate::worker) canvas_length: usize,
    #[pyo3(get)]
    pub(in crate::worker) seed: i64,
    #[pyo3(get)]
    pub(in crate::worker) block: i64,
    #[pyo3(get)]
    pub(in crate::worker) step: i64,
    #[pyo3(get)]
    pub(in crate::worker) sampling: Py<PyAny>,
}

#[pymethods]
impl CanvasStepRow {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (forward_mode, request_pool_idx=0, positions=None, seq_len=0, write_kv=false, causal=Some(true), canvas_length=0, seed=0, block=0, step=0, *, sampling))]
    fn new(
        forward_mode: &Bound<'_, PyAny>,
        request_pool_idx: u32,
        positions: Option<Py<PyAny>>,
        seq_len: i64,
        write_kv: bool,
        causal: Option<bool>,
        canvas_length: usize,
        seed: i64,
        block: i64,
        step: i64,
        sampling: Py<PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        Self {
            canvas_length,
            seed,
            block,
            step,
            sampling,
        }
        .initializer(
            forward_mode.py(),
            InputRow {
                kind: pythonize::depythonize(forward_mode)?,
                request_pool_idx,
            },
            AttentionRow {
                positions,
                seq_len,
                write_kv,
                causal,
            },
        )
    }

    #[getter]
    fn query_tokens(&self) -> usize {
        self.canvas_length
    }

    #[pyo3(signature = (**fields))]
    fn replace(slf: PyRef<'_, Self>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let mut input = InputRow::clone(slf.as_super().as_super());
        let mut attention = slf.as_super().clone_ref(py);
        let mut row = Self {
            canvas_length: slf.canvas_length,
            seed: slf.seed,
            block: slf.block,
            step: slf.step,
            sampling: slf.sampling.clone_ref(py),
        };
        if let Some(fields) = fields {
            for (name, value) in fields {
                match name.extract::<&str>()? {
                    "canvas_length" => row.canvas_length = value.extract()?,
                    "seed" => row.seed = value.extract()?,
                    "block" => row.block = value.extract()?,
                    "step" => row.step = value.extract()?,
                    "sampling" => row.sampling = value.unbind(),
                    name => attention.field(&mut input, name, &value)?,
                }
            }
        }
        Py::new(py, row.initializer(py, input, attention)?)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.sampling)
    }
}

impl CanvasStepRow {
    pub(in crate::worker) fn initializer(
        self,
        py: Python<'_>,
        input: InputRow,
        attention: AttentionRow,
    ) -> PyResult<PyClassInitializer<Self>> {
        if input.request_pool_idx == 0 || !canvas_shape(py, &attention, self.canvas_length)? {
            return Err(PyValueError::new_err(
                "a canvas step is a read-only noncausal canvas of its slot",
            ));
        }
        if self.block < 0
            || self.step < 0
            || self.step >= self.sampling.getattr(py, "steps")?.extract::<i64>(py)?
        {
            return Err(PyValueError::new_err(
                "a canvas step runs within its request's canvas sampling",
            ));
        }
        Ok(attention.initializer(input).add_subclass(self))
    }
}

fn canvas_shape(py: Python<'_>, attention: &AttentionRow, length: usize) -> PyResult<bool> {
    if attention.write_kv || attention.causal == Some(true) || length == 0 {
        return Ok(false);
    }
    let Some(positions) = &attention.positions else {
        return Ok(false);
    };
    Ok(positions
        .bind(py)
        .getattr("shape")?
        .get_item(-1)?
        .extract::<usize>()?
        == length)
}
