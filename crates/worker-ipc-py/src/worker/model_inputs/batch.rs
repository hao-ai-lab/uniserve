//! Numerical batch views aligned with native output controls.

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use uniserve_worker_ipc::{CallKind, ForwardMode};

use super::{parse_selections, selections_to_py, unknown};
use crate::convert::RequestConversion;
use uniserve_worker::TokenSelection;

/// Fixed numerical inputs and one output destination per live or padded row.
/// The tensor backend owns mathematical layouts; execution owns call mode and
/// output selection. Tensor views keep their backing independently of runners.
#[pyclass(module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct InputBatch {
    pub(in crate::worker) kind: CallKind,
    #[pyo3(get)]
    pub(in crate::worker) inputs: Py<PyAny>,
    #[pyo3(get)]
    pub(in crate::worker) request_pool_indices: Py<PyAny>,
    pub(in crate::worker) token_selections: Vec<TokenSelection>,
    #[pyo3(get)]
    pub(in crate::worker) decode_force_finish: Option<Py<PyAny>>,
    #[pyo3(get)]
    pub(in crate::worker) row_count: usize,
    #[pyo3(get)]
    pub(in crate::worker) query_tokens: Option<usize>,
}

#[pymethods]
impl InputBatch {
    #[new]
    #[pyo3(signature = (forward_mode, inputs, request_pool_indices, token_selections=None, decode_force_finish=None))]
    fn new(
        py: Python<'_>,
        forward_mode: &Bound<'_, PyAny>,
        inputs: Py<PyAny>,
        request_pool_indices: Py<PyAny>,
        token_selections: Option<&Bound<'_, PyTuple>>,
        decode_force_finish: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Self::build(
            py,
            pythonize::depythonize(forward_mode)?,
            inputs,
            request_pool_indices,
            token_selections
                .map(|values| parse_selections(values.as_any()))
                .transpose()?
                .unwrap_or_default(),
            decode_force_finish,
        )
    }

    #[getter]
    fn token_selections(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        selections_to_py(py, &self.token_selections)
    }

    #[getter]
    fn forward_mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(RequestConversion::new(py)?.kind(self.kind).unbind())
    }

    /// Preserve borrowed numerical storage when deriving a padded or rebound call.
    #[pyo3(signature = (**fields))]
    fn replace(&self, py: Python<'_>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut result = self.clone_ref(py);
        let mut inputs_changed = false;
        if let Some(fields) = fields {
            for (name, value) in fields {
                match name.extract::<&str>()? {
                    "forward_mode" => result.kind = pythonize::depythonize(&value)?,
                    "inputs" => {
                        result.inputs = value.unbind();
                        inputs_changed = true;
                    }
                    "request_pool_indices" => result.request_pool_indices = value.unbind(),
                    "token_selections" => result.token_selections = parse_selections(&value)?,
                    "decode_force_finish" => result.decode_force_finish = value.extract()?,
                    name => return Err(unknown(name)),
                }
            }
        }
        if inputs_changed {
            result.query_tokens = query_tokens(result.inputs.bind(py))?;
        }
        result.validate(py)?;
        Ok(result)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.inputs)?;
        visit.call(&self.request_pool_indices)?;
        visit.call(&self.decode_force_finish)
    }
}

impl InputBatch {
    pub(in crate::worker) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            kind: self.kind,
            inputs: self.inputs.clone_ref(py),
            request_pool_indices: self.request_pool_indices.clone_ref(py),
            token_selections: self.token_selections.clone(),
            decode_force_finish: self
                .decode_force_finish
                .as_ref()
                .map(|value| value.clone_ref(py)),
            row_count: self.row_count,
            query_tokens: self.query_tokens,
        }
    }

    pub(in crate::worker) fn build(
        py: Python<'_>,
        kind: CallKind,
        inputs: Py<PyAny>,
        request_pool_indices: Py<PyAny>,
        token_selections: Vec<TokenSelection>,
        decode_force_finish: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let query_tokens = query_tokens(inputs.bind(py))?;
        let mut result = Self {
            kind,
            inputs,
            request_pool_indices,
            token_selections,
            decode_force_finish,
            row_count: 0,
            query_tokens,
        };
        result.validate(py)?;
        Ok(result)
    }

    fn validate(&mut self, py: Python<'_>) -> PyResult<()> {
        let slots = self.request_pool_indices.bind(py);
        self.row_count = slots.call_method0("numel")?.extract()?;
        if slots.getattr("ndim")?.extract::<usize>()? != 1 || self.row_count == 0 {
            return Err(PyValueError::new_err(
                "execution requires a nonempty vector of request slots",
            ));
        }
        if matches!(self.kind, CallKind::Forward(mode) if mode != ForwardMode::TokenDenoising)
            && self.token_selections.len() != self.row_count
        {
            return Err(PyValueError::new_err(
                "text output selections must align with request slots",
            ));
        }
        if let Some(finish) = &self.decode_force_finish {
            let finish = finish.bind(py);
            if !finish.getattr("shape")?.eq(slots.getattr("shape")?)?
                || !finish
                    .getattr("dtype")?
                    .is(&py.import("torch")?.getattr("bool")?)
            {
                return Err(PyValueError::new_err(
                    "decode completion controls must align with request slots",
                ));
            }
        }
        Ok(())
    }
}

fn query_tokens(inputs: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    let Some(attention) = inputs.getattr_opt("attention")? else {
        return Ok(None);
    };
    let Some(queries) = attention.getattr_opt("queries")? else {
        return Ok(None);
    };
    if queries.is_none() {
        return Ok(None);
    }
    let host = queries.getattr("host")?;
    if host.is_none() {
        return Ok(None);
    }
    Ok(Some(host.extract::<Vec<usize>>()?.into_iter().sum()))
}
